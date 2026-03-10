import unittest
from typing import Callable
from tinygrad import Tensor, UOp
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import KernelInfo, AxisType

# **** kernels ****

def custom_arange_kernel(C:UOp) -> UOp:
  i = UOp.range(C.size, 0)
  return C[i].store(i.cast(C.dtype.base)).end(i).sink(arg=KernelInfo(name=f"custom_arange_{C.size}"))

def custom_eye_kernel(C:UOp) -> UOp:
  i = UOp.range(C.shape[0], 0)
  j = UOp.range(C.shape[1], 1)
  return C[i, j].store((i.eq(j)).cast(C.dtype.base)).end(i, j).sink(arg=KernelInfo(name=f"custom_eye_{C.size}"))

def custom_add_one_kernel(B:UOp, A:UOp) -> UOp:
  A,B = A.flatten(), B.flatten()
  assert B.size == A.size
  i = UOp.range(A.size, 0)
  return B[i].store(A[i] + 1).end(i).sink(arg=KernelInfo(name=f"add_one_{A.size}"))

def custom_elementwise_add_kernel(C:UOp, A:UOp, B:UOp) -> UOp:
  C,A,B = C.flatten(), A.flatten(), B.flatten()
  i = UOp.range(C.size, 0)
  return C[i].store(A[i]+B[i]).end(i).sink(arg=KernelInfo(name=f"custom_add_kernel_{C.size}")).simplify()

def custom_elementwise_addmul_kernel(C:UOp, D:UOp, A:UOp, B:UOp) -> UOp:
  C,D,A,B = C.flatten(), D.flatten(), A.flatten(), B.flatten()
  assert C.size == D.size
  i = UOp.range(C.size, 0)
  store_c = C[i].store(A[i]+B[i])
  store_d = D[i].store(A[i]*B[i])
  return UOp.group(store_c, store_d).end(i).sink(arg=KernelInfo(name=f"custom_addmul_kernel_{C.size}")).simplify()

def custom_gemm(C:UOp, A:UOp, B:UOp) -> UOp:
  assert A.shape[1] == B.shape[0]
  i, j, k = UOp.range(C.shape[0], 0), UOp.range(C.shape[1], 1), UOp.range(A.shape[1], 2, axis_type=AxisType.REDUCE)
  C = C[i, j].set(0.0)
  C = C[i, j].set(C.after(k)[i, j] + A[i, k] * B[k, j], end=k)
  prog = C.end(i, j)
  return prog.sink(arg=KernelInfo(name=f"custom_gemm_{C.shape[0]}_{C.shape[1]}_{A.shape[1]}", opts_to_apply=()))

def custom_sum(B:UOp, A:UOp) -> UOp:
  i = UOp.range(A.shape[0], 0, axis_type=AxisType.REDUCE)
  B = B[0].set(0.0)
  B = B[0].set(B.after(i)[0] + A[i], end=i)
  return B.sink(arg=KernelInfo(name=f"custom_sum_{A.shape[0]}", opts_to_apply=()))

def flip_contract_kernel(dest:UOp, src:UOp):
  i = UOp.range(dest.shape[0], 0)
  j = UOp.range(dest.shape[1], 1, AxisType.UPCAST)
  vec = src[i, j].contract(j)
  store = UOp.group(*[dest[i, k].store(vec.gep(3-k)) for k in range(4)])
  return store.end(i, j).sink(arg=KernelInfo(name=f"flip_contract_{dest.size}", opts_to_apply=()))

def slice_sum_kernel(dest:UOp, src:UOp):
  G = UOp.range(src.shape[0], 0)
  slice_src = src[G, :]
  reg = UOp.placeholder((1,), dest.dtype.base, 0, addrspace=AddrSpace.REG)
  reg = reg.after(G)[0].set(0)
  R = UOp.range(src.shape[1], 1, AxisType.REDUCE)
  reg = reg[0].set(reg.after(R)[0] + slice_src[R], end=R)
  ast = dest[G].set(reg[0], end=G)
  return ast.sink(arg=KernelInfo(name=f"slice_sum_{src.shape[0]}_{src.shape[1]}", opts_to_apply=()))

def simple_qkv_kernel(O:UOp, Q:UOp, K:UOp, V:UOp) -> UOp:
  # attention without softmax
  N, d = Q.shape[0], Q.shape[1]

  i = UOp.range(N, 0)  # output row
  d_out = UOp.range(d, 1)  # output column
  j = UOp.range(N, 2, axis_type=AxisType.REDUCE)

  k_inner = UOp.range(d, 3, axis_type=AxisType.REDUCE)
  qk_acc = UOp.placeholder((1,), Q.dtype.base, 0, addrspace=AddrSpace.REG)
  qk_acc = qk_acc.after(i, j)[0].set(0.0)
  qk_acc = qk_acc[0].set(qk_acc.after(k_inner)[0] + Q[i, k_inner] * K[j, k_inner], end=k_inner)
  qk_score = qk_acc[0] / (d ** 0.5)

  out_acc = UOp.placeholder((1,), Q.dtype.base, 1, addrspace=AddrSpace.REG)
  out_acc = out_acc.after(i, d_out)[0].set(0.0)
  out_acc = out_acc[0].set(out_acc.after(j)[0] + qk_score * V[j, d_out], end=j)

  store = O[i, d_out].store(out_acc[0])
  return store.end(d_out).end(i).sink(arg=KernelInfo(name=f"simple_qkv_{N}_{d}", opts_to_apply=()))

# **** backward callbacks ****

def backward_gemm(gradient:UOp, kernel:UOp) -> tuple[UOp, UOp]:
  out, a, b = kernel.src[1:]
  grad_a = (Tensor(gradient) @ Tensor(b).T).uop
  grad_b = (Tensor(a).T @ Tensor(gradient)).uop
  return (None, grad_a, grad_b)

def backward_gemm_custom(gradient:UOp, kernel:UOp) -> tuple[UOp, UOp]:
  out, a, b = kernel.src[1:]
  grad_a = Tensor.empty_like(Tensor(a)).custom_kernel(Tensor(gradient), Tensor(b).T, fxn=custom_gemm)[0].uop
  grad_b = Tensor.empty_like(Tensor(b)).custom_kernel(Tensor(a).T, Tensor(gradient), fxn=custom_gemm)[0].uop
  return (None, grad_a, grad_b)

# **** tests ****

class TestCustomKernel(unittest.TestCase):
  def test_empty(self):
    a = Tensor.empty(1)
    a = Tensor.custom_kernel(a, fxn=lambda _: UOp.sink(arg=KernelInfo()))[0]
    a.realize()

  def test_simple(self):
    a = Tensor.ones(16, 16).contiguous()
    b = Tensor.ones(16, 16).contiguous()
    c = Tensor.empty(16, 16)

    c = Tensor.custom_kernel(c,a,b, fxn=custom_elementwise_add_kernel)[0]

    out = c.flatten().tolist()
    assert all(x == 2 for x in out), "all 2"

  def test_simple_sharded(self):
    devs = ("CPU:0", "CPU:1")

    a = Tensor.ones(16, 16).contiguous().shard(devs, axis=0)
    b = Tensor.ones(16, 16).contiguous().shard(devs, axis=0)
    # ugly construction to get a sharded empty tensor
    c = Tensor(Tensor.empty(8, 16, device=devs).uop.multi(0), device=devs)
    c = Tensor.custom_kernel(c,a,b, fxn=custom_elementwise_add_kernel)[0]
    out = c.flatten().tolist()
    assert all(x == 2 for x in out), "all 2"

  def test_sharded_add_one(self):
    # PYTHON backend explicitly checks for OOB access for wrong multi shape regression
    devs = ("PYTHON:0", "PYTHON:1")
    a = Tensor.ones(4, 4).contiguous().shard(devs, axis=0)
    c = Tensor(Tensor.empty(2, 4, device=devs).uop.multi(0), device=devs)
    c = Tensor.custom_kernel(c, a, fxn=custom_add_one_kernel)[0]
    assert (c == 2).all().item()

  def test_multioutput(self):
    a = Tensor.full((16, 16), 3.).contiguous()
    b = Tensor.full((16, 16), 3.).contiguous()
    c = Tensor.empty(16, 16)
    d = Tensor.empty(16, 16)

    c,d = Tensor.custom_kernel(c,d,a,b, fxn=custom_elementwise_addmul_kernel)[:2]
    Tensor.realize(c,d)

    assert all(x == 6 for x in c.flatten().tolist()), "all 6"
    assert all(x == 9 for x in d.flatten().tolist()), "all 9"

  def test_arange(self):
    ref = Tensor.arange(100)
    tst = Tensor.empty_like(ref)
    tst = tst.custom_kernel(fxn=custom_arange_kernel)[0]
    self.assertTrue((ref == tst).all().item())

  def test_eye(self):
    ref = Tensor.eye(1024).contiguous().realize()
    tst = Tensor.empty_like(ref)
    tst = tst.custom_kernel(fxn=custom_eye_kernel)[0]
    self.assertTrue((ref == tst).all().item())

  def test_flip_contract(self):
    a = Tensor.randn(10,4)
    b = Tensor.empty_like(a)
    b = b.custom_kernel(a, fxn=flip_contract_kernel)[0]
    self.assertTrue((a.flip(1) == b).all().item())

  def test_noncontig(self):
    a = Tensor.ones(16, 16).contiguous()
    tst = Tensor.empty_like(a)
    b = a+1
    b_p1 = Tensor.custom_kernel(tst, b, fxn=custom_add_one_kernel)[0]
    self.assertTrue((b_p1 == 3).all().item())

  def test_sum(self):
    a = Tensor([1.0, 2, 3, 4, 5])
    tst = Tensor.empty(1)
    b = Tensor.custom_kernel(tst, a, fxn=custom_sum)[0]
    self.assertEqual(b.item(), 15)

  def test_sum_int(self):
    a = Tensor([1, 2, 3, 4, 5])
    tst = Tensor.empty(1, dtype=a.dtype)
    b = Tensor.custom_kernel(tst, a, fxn=custom_sum)[0]
    self.assertEqual(b.item(), 15)

  def test_slice_sum(self):
    A = Tensor.randn(16, 16).contiguous()
    B = Tensor.empty(16)
    B = Tensor.custom_kernel(B, A, fxn=slice_sum_kernel)[0]
    self.assertTrue(B.allclose(A.sum(1)))

  def test_gemm(self):
    N = 16
    a = Tensor.randn(N, N)
    b = Tensor.randn(N, N)
    c = Tensor.empty(N, N)

    tst = Tensor.custom_kernel(c, a, b, fxn=custom_gemm)[0]
    err = (tst - (a@b)).square().max()
    self.assertLess(err.item(), 1e-6)

  def test_gemm_multi(self):
    devs = ("CPU:0", "CPU:1")
    N = 16
    a = Tensor.randn(N, N).shard_(devs, axis=0)
    b = Tensor.randn(N, N).to(devs)
    c = Tensor(Tensor.empty(N//2, N, device=devs).uop.multi(0), device=devs)
    tst = Tensor.custom_kernel(c, a, b, fxn=custom_gemm)[0]
    err = (tst - (a@b)).square().max()
    self.assertLess(err.item(), 1e-6)

  def test_gemm_backward_custom(self): self.test_gemm_backward(True)
  # NOTE: grad_fxn doesn't work with pyrender
  def test_gemm_backward(self, custom_backward_gemm=False):
    N = 4
    a_rand = Tensor.randn(N, 8)
    b_rand = Tensor.randn(8, N)
    Tensor.realize(a_rand, b_rand)

    a, b = Tensor(a_rand.numpy(), requires_grad=True), Tensor(b_rand.numpy(), requires_grad=True)
    c = Tensor.empty(N, N)
    tst = Tensor.custom_kernel(c, a, b, fxn=custom_gemm, grad_fxn=backward_gemm_custom if custom_backward_gemm else backward_gemm)[0]
    tst.sum().backward()
    grad_a, grad_b = a.grad, b.grad
    Tensor.realize(tst, grad_a, grad_b)

    a, b = Tensor(a_rand.numpy(), requires_grad=True), Tensor(b_rand.numpy(), requires_grad=True)
    ref = (a@b)
    ref.sum().backward()
    real_grad_a, real_grad_b = a.grad, b.grad
    Tensor.realize(ref, real_grad_a, real_grad_b)

    err = (tst - ref).square().max()
    self.assertLess(err.item(), 1e-6)

    err = (grad_a - real_grad_a).square().max()
    self.assertLess(err.item(), 1e-6)

    err = (grad_b - real_grad_b).square().max()
    self.assertLess(err.item(), 1e-6)

  def test_simple_qkv(self):
    N, d = 8, 4
    Q = Tensor.randn(N, d)
    K = Tensor.randn(N, d)
    V = Tensor.randn(N, d)
    O = Tensor.empty(N, d)

    O_custom = Tensor.custom_kernel(O, Q, K, V, fxn=lambda o,q,k,v: simple_qkv_kernel(o,q,k,v))[0]
    O_ref = ((Q @ K.T) / (d ** 0.5)) @ V

    Tensor.realize(O_custom, O_ref)
    err = (O_custom - O_ref).square().max()
    self.assertLess(err.item(), 1e-6)

  def test_multi_after_schedule_order(self):
    """Test correct scheduling order when custom_kernel has multiple outputs.

    custom_kernel with 4 arguments creates 4 AFTERs from the same kernel.
    The custom_kernel depends on both A2 and B2, so it must be scheduled after both.
    E only depends on A2, so E can run before custom_kernel finishes waiting for B2.

    Expected schedule order: [A2, B2, E, custom_addmul, final_sum]
    The custom_addmul kernel should be at index 3.
    """

    A, B = Tensor.empty(4, 4), Tensor.empty(4, 4)
    A2 = (A + 1).contiguous()                      # kernel 0: depends on A
    B2 = (B * 2).contiguous()                      # kernel 1: depends on B
    C, D = Tensor.empty(4, 4), Tensor.empty(4, 4)
    C, D, _, _ = Tensor.custom_kernel(C, D, A2, B2, fxn=custom_elementwise_addmul_kernel)  # depends on A2 AND B2
    E = (A2 * 3).contiguous()                      # kernel 2: depends only on A2
    result = (C + D + E).sum()                     # kernel 3: custom_addmul, then kernel 4: sum
    schedule = result.schedule()

    # Find the custom_addmul kernel position
    custom_idx = next((i for i, item in enumerate(schedule)
                       if hasattr(item.ast, "arg") and hasattr(item.ast.arg, "name")
                       and "custom_addmul" in item.ast.arg.name), None)

    self.assertIsNotNone(custom_idx, "custom_addmul kernel not found in schedule")
    self.assertEqual(custom_idx, 3, f"custom_addmul should be at index 3, got {custom_idx}")

class TestCustomKernelRematerializeCorrectness(TestCustomKernel):
  def setUp(self):
    self.og_fn = Tensor.custom_kernel
    def new_custom_kernel(t, *lst:Tensor, fxn:Callable, grad_fxn:Callable|None=None) -> list[Tensor]:
      return self.og_fn(t, *lst, fxn=fxn, grad_fxn=grad_fxn, rematerialize=True)
    Tensor.custom_kernel = new_custom_kernel
  def tearDown(self):
    Tensor.custom_kernel = self.og_fn


def create_sidebyside_n_consumers(kernel: Tensor, num_consumers: int) -> Tensor:
  consumers = [kernel * (i+2) for i in range(num_consumers)]
  return sum(consumers)

def create_waterfall_n_consumers(kernel: Tensor, accum: Tensor, num_consumers: int) -> Tensor:
  for _ in range(num_consumers):
    accum = accum * kernel
  return accum

def create_diamond_n_consumers(kernel: Tensor, num_consumers: int) -> Tensor:
  branches = [kernel * (i+2) for i in range(num_consumers)]
  result = branches[0]
  for b in branches[1:]:
    result = result * b
  return result

def create_reduction_n_consumers(kernel: Tensor, num_consumers: int) -> Tensor:
  consumers = [kernel * (j+2) for j in range(num_consumers)]
  return sum(consumers).sum()

def _remat_get_num_exec_items(get_fxn_args: Callable[[], tuple[Tensor]], fxn: Callable, post_processing_func: Callable[..., Tensor], post_processing_args: tuple, remat:bool, get_args_arg: tuple|None=None, grad_fxn: Callable|None=None) -> int:
  args = get_fxn_args() if get_args_arg is None else get_fxn_args(*get_args_arg)
  kern = Tensor.custom_kernel(*args, fxn=fxn, grad_fxn=grad_fxn, rematerialize=remat)[0]
  out = post_processing_func(kern, *post_processing_args)
  return len(out.schedule())

def get_fxn_args():
  return Tensor.empty(16, 16), Tensor.ones(16, 16).contiguous(),  Tensor.ones(16, 16).contiguous()

def get_fxn_args_sharded(devs: tuple[str]=("CPU:0", "CPU:1")):
  return  Tensor(Tensor.empty(8, 16, device=devs).uop.multi(0), device=devs), Tensor.ones(16, 16).contiguous().shard(devs, axis=0), Tensor.ones(16, 16).contiguous().shard(devs, axis=0)

class TestCustomKernelRematerializeSchedule(unittest.TestCase):
  def test_simple_side_by_side_consumers(self):
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_fxn_args, custom_elementwise_add_kernel, create_sidebyside_n_consumers, (i,), False)
      num_remat = _remat_get_num_exec_items(get_fxn_args, custom_elementwise_add_kernel, create_sidebyside_n_consumers, (i,), True)
      self.assertEqual(num_reg+i-1, num_remat, msg=i)

  def test_simple_waterfall_consumers(self):
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_fxn_args, custom_elementwise_add_kernel, create_waterfall_n_consumers, (Tensor.ones(16, 16).contiguous(), i), False)
      num_remat = _remat_get_num_exec_items(get_fxn_args, custom_elementwise_add_kernel, create_waterfall_n_consumers, (Tensor.ones(16, 16).contiguous(), i,), True)
      
      self.assertEqual(num_reg+i-1, num_remat)

  def test_self_reference(self):
    kern_reg = Tensor.custom_kernel(*get_fxn_args(), fxn=custom_elementwise_add_kernel)[0]
    kern_remat = Tensor.custom_kernel(*get_fxn_args(), fxn=custom_elementwise_add_kernel, rematerialize=True)[0]

    out_reg = kern_reg * kern_reg
    out_remat = kern_remat * kern_remat

    self.assertEqual(len(out_reg.schedule())+1, len(out_remat.schedule()))

  def test_simple_sharded_side_by_side_consumers(self):
    devs = ("CPU:0", "CPU:1")
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_fxn_args_sharded, custom_elementwise_add_kernel, create_sidebyside_n_consumers, (i,), False, (devs,))
      num_remat = _remat_get_num_exec_items(get_fxn_args_sharded, custom_elementwise_add_kernel, create_sidebyside_n_consumers, (i,), True, (devs,))
      self.assertEqual(num_reg+(2*(i-1)), num_remat, msg=i)

  def test_simple_sharded_waterfall_consumers(self):
    devs = ("CPU:0", "CPU:1")
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_fxn_args_sharded, custom_elementwise_add_kernel, create_waterfall_n_consumers, (Tensor.ones(16, 16).contiguous().shard(devs, axis=0),i), False, (devs,))
      num_remat = _remat_get_num_exec_items(get_fxn_args_sharded, custom_elementwise_add_kernel, create_waterfall_n_consumers, (Tensor.ones(16, 16).contiguous().shard(devs, axis=0),i), True, (devs,))
      self.assertEqual(num_reg+(2*(i-1)), num_remat, msg=i)

  def test_sharded_add_one_side_by_side(self):
    # PYTHON backend explicitly checks for OOB access for wrong multi shape regression
    devs = ("PYTHON:0", "PYTHON:1")
    def get_args():
      return Tensor(Tensor.empty(2, 4, device=devs).uop.multi(0), device=devs), Tensor.ones(4, 4).contiguous().shard(devs, axis=0)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_add_one_kernel, create_sidebyside_n_consumers, (i,), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_add_one_kernel, create_sidebyside_n_consumers, (i,), True)
      self.assertEqual(num_reg+(2*(i-1)), num_remat, msg=i)

  def test_sharded_add_one_waterfall(self):
    # PYTHON backend explicitly checks for OOB access for wrong multi shape regression
    devs = ("PYTHON:0", "PYTHON:1")
    def get_args():
      return Tensor(Tensor.empty(2, 4, device=devs).uop.multi(0), device=devs), Tensor.ones(4, 4).contiguous().shard(devs, axis=0)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_add_one_kernel, create_waterfall_n_consumers, (Tensor.ones(4, 4).contiguous().shard(devs, axis=0), i), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_add_one_kernel, create_waterfall_n_consumers, (Tensor.ones(4, 4).contiguous().shard(devs, axis=0), i), True)
      self.assertEqual(num_reg+(2*(i-1)), num_remat, msg=i)

  def test_arange_side_by_side(self):
    def get_args():
      return (Tensor.empty(100),)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_arange_kernel, create_sidebyside_n_consumers, (i,), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_arange_kernel, create_sidebyside_n_consumers, (i,), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_arange_waterfall(self):
    def get_args():
      return (Tensor.empty(100),)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_arange_kernel, create_waterfall_n_consumers, (Tensor.ones(100).contiguous(), i), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_arange_kernel, create_waterfall_n_consumers, (Tensor.ones(100).contiguous(), i), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_eye_side_by_side(self):
    def get_args():
      return (Tensor.empty(16, 16),)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_eye_kernel, create_sidebyside_n_consumers, (i,), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_eye_kernel, create_sidebyside_n_consumers, (i,), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_eye_waterfall(self):
    def get_args():
      return (Tensor.empty(16, 16),)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_eye_kernel, create_waterfall_n_consumers, (Tensor.ones(16, 16).contiguous(), i), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_eye_kernel, create_waterfall_n_consumers, (Tensor.ones(16, 16).contiguous(), i), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_flip_contract_side_by_side(self):
    def get_args():
      return Tensor.empty(10, 4), Tensor.randn(10, 4)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, flip_contract_kernel, create_sidebyside_n_consumers, (i,), False)
      num_remat = _remat_get_num_exec_items(get_args, flip_contract_kernel, create_sidebyside_n_consumers, (i,), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_flip_contract_waterfall(self):
    def get_args():
      return Tensor.empty(10, 4), Tensor.randn(10, 4)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, flip_contract_kernel, create_waterfall_n_consumers, (Tensor.ones(10, 4).contiguous(), i), False)
      num_remat = _remat_get_num_exec_items(get_args, flip_contract_kernel, create_waterfall_n_consumers, (Tensor.ones(10, 4).contiguous(), i), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_noncontig_side_by_side(self):
    def get_args():
      a = Tensor.ones(16, 16).contiguous()
      return Tensor.empty_like(a), a+1
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_add_one_kernel, create_sidebyside_n_consumers, (i,), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_add_one_kernel, create_sidebyside_n_consumers, (i,), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_noncontig_waterfall(self):
    def get_args():
      a = Tensor.ones(16, 16).contiguous()
      return Tensor.empty_like(a), a+1
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_add_one_kernel, create_waterfall_n_consumers, (Tensor.ones(16, 16).contiguous(), i), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_add_one_kernel, create_waterfall_n_consumers, (Tensor.ones(16, 16).contiguous(), i), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_sum_side_by_side(self):
    def get_args():
      return Tensor.empty(1), Tensor([1.0, 2, 3, 4, 5])
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_sum, create_sidebyside_n_consumers, (i,), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_sum, create_sidebyside_n_consumers, (i,), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_sum_waterfall(self):
    def get_args():
      return Tensor.empty(1), Tensor([1.0, 2, 3, 4, 5])
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_sum, create_waterfall_n_consumers, (Tensor.ones(1).contiguous(), i), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_sum, create_waterfall_n_consumers, (Tensor.ones(1).contiguous(), i), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_sum_int_side_by_side(self):
    def get_args():
      a = Tensor([1, 2, 3, 4, 5])
      return Tensor.empty(1, dtype=a.dtype), a
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_sum, create_sidebyside_n_consumers, (i,), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_sum, create_sidebyside_n_consumers, (i,), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_slice_sum_side_by_side(self):
    def get_args():
      return Tensor.empty(16), Tensor.randn(16, 16).contiguous()
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, slice_sum_kernel, create_sidebyside_n_consumers, (i,), False)
      num_remat = _remat_get_num_exec_items(get_args, slice_sum_kernel, create_sidebyside_n_consumers, (i,), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_slice_sum_waterfall(self):
    def get_args():
      return Tensor.empty(16), Tensor.randn(16, 16).contiguous()
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, slice_sum_kernel, create_waterfall_n_consumers, (Tensor.ones(16).contiguous(), i), False)
      num_remat = _remat_get_num_exec_items(get_args, slice_sum_kernel, create_waterfall_n_consumers, (Tensor.ones(16).contiguous(), i), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_gemm_side_by_side(self):
    N = 16
    def get_args():
      return Tensor.empty(N, N), Tensor.randn(N, N), Tensor.randn(N, N)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_gemm, create_sidebyside_n_consumers, (i,), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_gemm, create_sidebyside_n_consumers, (i,), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_gemm_waterfall(self):
    N = 16
    def get_args():
      return Tensor.empty(N, N), Tensor.randn(N, N), Tensor.randn(N, N)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_gemm, create_waterfall_n_consumers, (Tensor.ones(N, N).contiguous(), i), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_gemm, create_waterfall_n_consumers, (Tensor.ones(N, N).contiguous(), i), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_gemm_multi_side_by_side(self):
    devs = ("CPU:0", "CPU:1")
    N = 16
    def get_args():
      return Tensor(Tensor.empty(N//2, N, device=devs).uop.multi(0), device=devs), Tensor.randn(N, N).shard_(devs, axis=0), Tensor.randn(N, N).to(devs)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_gemm, create_sidebyside_n_consumers, (i,), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_gemm, create_sidebyside_n_consumers, (i,), True)
      self.assertEqual(num_reg+(2*(i-1)), num_remat, msg=i)

  def test_simple_qkv_side_by_side(self):
    N, d = 8, 4
    def get_args():
      return Tensor.empty(N, d), Tensor.randn(N, d), Tensor.randn(N, d), Tensor.randn(N, d)
    fxn = lambda o,q,k,v: simple_qkv_kernel(o,q,k,v)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, fxn, create_sidebyside_n_consumers, (i,), False)
      num_remat = _remat_get_num_exec_items(get_args, fxn, create_sidebyside_n_consumers, (i,), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_simple_qkv_waterfall(self):
    N, d = 8, 4
    def get_args():
      return Tensor.empty(N, d), Tensor.randn(N, d), Tensor.randn(N, d), Tensor.randn(N, d)
    fxn = lambda o,q,k,v: simple_qkv_kernel(o,q,k,v)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, fxn, create_waterfall_n_consumers, (Tensor.ones(N, d).contiguous(), i), False)
      num_remat = _remat_get_num_exec_items(get_args, fxn, create_waterfall_n_consumers, (Tensor.ones(N, d).contiguous(), i), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_sum_int_waterfall(self):
    def get_args():
      a = Tensor([1, 2, 3, 4, 5])
      return Tensor.empty(1, dtype=a.dtype), a
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_sum, create_waterfall_n_consumers, (Tensor([1]).contiguous(), i), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_sum, create_waterfall_n_consumers, (Tensor([1]).contiguous(), i), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_gemm_multi_waterfall(self):
    devs = ("CPU:0", "CPU:1")
    N = 16
    def get_args():
      return Tensor(Tensor.empty(N//2, N, device=devs).uop.multi(0), device=devs), Tensor.randn(N, N).shard_(devs, axis=0), Tensor.randn(N, N).to(devs)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_gemm, create_waterfall_n_consumers, (Tensor.ones(N, N).contiguous().shard(devs, axis=0), i), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_gemm, create_waterfall_n_consumers, (Tensor.ones(N, N).contiguous().shard(devs, axis=0), i), True)
      self.assertEqual(num_reg+(2*(i-1)), num_remat, msg=i)

  def test_empty_side_by_side(self):
    def get_args():
      return (Tensor.empty(1),)
    fxn = lambda _: UOp.sink(arg=KernelInfo())
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, fxn, create_sidebyside_n_consumers, (i,), False)
      num_remat = _remat_get_num_exec_items(get_args, fxn, create_sidebyside_n_consumers, (i,), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_empty_waterfall(self):
    def get_args():
      return (Tensor.empty(1),)
    fxn = lambda _: UOp.sink(arg=KernelInfo())
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, fxn, create_waterfall_n_consumers, (Tensor.ones(1).contiguous(), i), False)
      num_remat = _remat_get_num_exec_items(get_args, fxn, create_waterfall_n_consumers, (Tensor.ones(1).contiguous(), i), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_multioutput_side_by_side(self):
    def get_args():
      return Tensor.empty(16, 16), Tensor.empty(16, 16), Tensor.full((16, 16), 3.).contiguous(), Tensor.full((16, 16), 3.).contiguous()
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_elementwise_addmul_kernel, create_sidebyside_n_consumers, (i,), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_elementwise_addmul_kernel, create_sidebyside_n_consumers, (i,), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_multioutput_waterfall(self):
    def get_args():
      return Tensor.empty(16, 16), Tensor.empty(16, 16), Tensor.full((16, 16), 3.).contiguous(), Tensor.full((16, 16), 3.).contiguous()
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_elementwise_addmul_kernel, create_waterfall_n_consumers, (Tensor.ones(16, 16).contiguous(), i), False)
      num_remat = _remat_get_num_exec_items(get_args, custom_elementwise_addmul_kernel, create_waterfall_n_consumers, (Tensor.ones(16, 16).contiguous(), i), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_gemm_backward_side_by_side(self):
    N = 4
    def get_args():
      return Tensor.empty(N, N), Tensor.randn(N, 8), Tensor.randn(8, N)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_gemm, create_sidebyside_n_consumers, (i,), False, grad_fxn=backward_gemm)
      num_remat = _remat_get_num_exec_items(get_args, custom_gemm, create_sidebyside_n_consumers, (i,), True, grad_fxn=backward_gemm)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_gemm_backward_waterfall(self):
    N = 4
    def get_args():
      return Tensor.empty(N, N), Tensor.randn(N, 8), Tensor.randn(8, N)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_gemm, create_waterfall_n_consumers, (Tensor.ones(N, N).contiguous(), i), False, grad_fxn=backward_gemm)
      num_remat = _remat_get_num_exec_items(get_args, custom_gemm, create_waterfall_n_consumers, (Tensor.ones(N, N).contiguous(), i), True, grad_fxn=backward_gemm)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_gemm_backward_custom_side_by_side(self):
    N = 4
    def get_args():
      return Tensor.empty(N, N), Tensor.randn(N, 8), Tensor.randn(8, N)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_gemm, create_sidebyside_n_consumers, (i,), False, grad_fxn=backward_gemm_custom)
      num_remat = _remat_get_num_exec_items(get_args, custom_gemm, create_sidebyside_n_consumers, (i,), True, grad_fxn=backward_gemm_custom)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_gemm_backward_custom_waterfall(self):
    N = 4
    def get_args():
      return Tensor.empty(N, N), Tensor.randn(N, 8), Tensor.randn(8, N)
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_args, custom_gemm, create_waterfall_n_consumers, (Tensor.ones(N, N).contiguous(), i), False, grad_fxn=backward_gemm_custom)
      num_remat = _remat_get_num_exec_items(get_args, custom_gemm, create_waterfall_n_consumers, (Tensor.ones(N, N).contiguous(), i), True, grad_fxn=backward_gemm_custom)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_multi_after_schedule_order_side_by_side(self):
    for i in range(1, 5):
      def get_count(remat):
        A, B = Tensor.empty(4, 4), Tensor.empty(4, 4)
        A2 = (A + 1).contiguous()
        B2 = (B * 2).contiguous()
        C, D = Tensor.empty(4, 4), Tensor.empty(4, 4)
        C, D, _, _ = Tensor.custom_kernel(C, D, A2, B2, fxn=custom_elementwise_addmul_kernel, rematerialize=remat)
        E = (A2 * 3).contiguous()
        consumers = create_sidebyside_n_consumers(C, i)
        result = (consumers + D + E).sum()
        return len(result.schedule())
      num_reg = get_count(False)
      num_remat = get_count(True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_multi_after_schedule_order_waterfall(self):
    for i in range(1, 5):
      def get_count(remat):
        A, B = Tensor.empty(4, 4), Tensor.empty(4, 4)
        A2 = (A + 1).contiguous()
        B2 = (B * 2).contiguous()
        C, D = Tensor.empty(4, 4), Tensor.empty(4, 4)
        C, D, _, _ = Tensor.custom_kernel(C, D, A2, B2, fxn=custom_elementwise_addmul_kernel, rematerialize=remat)
        E = (A2 * 3).contiguous()
        waterfall = create_waterfall_n_consumers(C, Tensor.ones(4, 4).contiguous(), i)
        result = (waterfall + D + E).sum()
        return len(result.schedule())
      num_reg = get_count(False)
      num_remat = get_count(True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_diamond(self):
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_fxn_args, custom_elementwise_add_kernel, create_diamond_n_consumers, (i,), False)
      num_remat = _remat_get_num_exec_items(get_fxn_args, custom_elementwise_add_kernel, create_diamond_n_consumers, (i,), True)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_multioutput_both_consumers(self):
    for i in range(1, 5):
      def get_count(remat):
        C, D = Tensor.empty(16, 16), Tensor.empty(16, 16)
        A = Tensor.full((16, 16), 3.).contiguous()
        B = Tensor.full((16, 16), 3.).contiguous()
        C, D, _, _ = Tensor.custom_kernel(C, D, A, B, fxn=custom_elementwise_addmul_kernel, rematerialize=remat)
        c_consumers = create_sidebyside_n_consumers(C, i)
        d_consumers = create_sidebyside_n_consumers(D, i)
        return len((c_consumers + d_consumers).sum().schedule())
      num_reg = get_count(False)
      num_remat = get_count(True)
      # both outputs share the same kernel, each with i consumers
      # each output's AFTER is rematerialized independently: 2*(i-1) extra kernels
      self.assertEqual(num_reg+(2*(i-1)), num_remat, msg=i)

  def test_chained_remat(self):
    for i in range(1, 5):
      def get_count(remat):
        a, b = Tensor.ones(16, 16).contiguous(), Tensor.ones(16, 16).contiguous()
        c1 = Tensor.custom_kernel(Tensor.empty(16, 16), a, b, fxn=custom_elementwise_add_kernel, rematerialize=remat)[0]
        c2 = Tensor.custom_kernel(Tensor.empty(16, 16), c1, b, fxn=custom_elementwise_add_kernel, rematerialize=remat)[0]
        return len(create_sidebyside_n_consumers(c2, i).schedule())
      num_reg = get_count(False)
      num_remat = get_count(True)
      # only c2 has multiple consumers, c1 has 1 consumer (c2's input)
      self.assertEqual(num_reg+i-1, num_remat)

  def test_mixed_remat_nonremat(self):
    for i in range(1, 5):
      def get_count(remat_first):
        a, b = Tensor.ones(16, 16).contiguous(), Tensor.ones(16, 16).contiguous()
        c1 = Tensor.custom_kernel(Tensor.empty(16, 16), a, b, fxn=custom_elementwise_add_kernel, rematerialize=remat_first)[0]
        c2 = Tensor.custom_kernel(Tensor.empty(16, 16), a, b, fxn=custom_elementwise_add_kernel, rematerialize=False)[0]
        r1 = create_sidebyside_n_consumers(c1, i)
        r2 = create_sidebyside_n_consumers(c2, i)
        return len((r1 + r2).sum().schedule())
      num_none = get_count(False)
      num_mixed = get_count(True)
      # only c1 (remat) adds extra kernels, c2 (non-remat) stays shared
      self.assertEqual(num_none+i-1, num_mixed)

  def test_reduction_consumer(self):
    for i in range(1, 5):
      num_reg = _remat_get_num_exec_items(get_fxn_args, custom_elementwise_add_kernel, create_reduction_n_consumers, (i,), False)
      num_remat = _remat_get_num_exec_items(get_fxn_args, custom_elementwise_add_kernel, create_reduction_n_consumers, (i,), True)
      self.assertEqual(num_reg+i-1, num_remat)

if __name__ == '__main__':
  unittest.main()
