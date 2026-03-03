import unittest
import numpy as np
from typing import Callable
from tinygrad import Tensor, function
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp

class TestCall(unittest.TestCase):
  def test_call_plus(self):
    a = Tensor.randn(10, 10)
    b = Tensor.randn(10, 10)
    Tensor.realize(a,b)

    # we define a plus function
    plus_fxn = UOp.param(0, dtypes.float, (10,10)) + UOp.param(1, dtypes.float, (10,10))

    c = Tensor.call(a, b, fxn=plus_fxn)
    np.testing.assert_equal(c.numpy(), (a+b).numpy())

  def test_call_plus_backward(self):
    a = Tensor.ones(10, 10, requires_grad=True)
    b = Tensor.ones(10, 10, requires_grad=True)

    (a+b).mean().backward()
    gt_a_grad = a.grad.numpy()
    gt_b_grad = b.grad.numpy()
    a.grad, b.grad = None, None

    # this is the gradient for +
    def grad_fxn(grad:UOp, call:UOp): return (grad, grad)

    # we define a plus function
    plus_fxn = UOp.param(0, dtypes.float, (10,10)) + UOp.param(1, dtypes.float, (10,10))
    c = Tensor.call(a, b, fxn=plus_fxn, grad_fxn=grad_fxn)
    c.mean().backward()

    np.testing.assert_allclose(a.grad.numpy(), gt_a_grad, rtol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), gt_b_grad, rtol=1e-5)

  def test_call_plus_backward_auto(self):
    a = Tensor.ones(10, 10, requires_grad=True)
    b = Tensor.ones(10, 10, requires_grad=True)

    (a+b).mean().backward()
    gt_a_grad = a.grad.numpy()
    gt_b_grad = b.grad.numpy()
    a.grad, b.grad = None, None

    plus_fxn = UOp.param(0, dtypes.float, (10,10)) + UOp.param(1, dtypes.float, (10,10))
    c = Tensor.call(a, b, fxn=plus_fxn)
    c.mean().backward()

    np.testing.assert_allclose(a.grad.numpy(), gt_a_grad, rtol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), gt_b_grad, rtol=1e-5)

  def test_call_gemm(self):
    M, K, N = 4, 8, 4
    a = Tensor.randn(M, K)
    b = Tensor.randn(K, N)
    Tensor.realize(a, b)
    c = Tensor.call(a, b, fxn=a.as_param(0) @ b.as_param(1))
    np.testing.assert_allclose(c.numpy(), a.numpy() @ b.numpy(), rtol=1e-5, atol=1e-6)

  @unittest.skip("needs GEMM on mixins")
  def test_call_gemm_uop(self):
    M, K, N = 4, 8, 4
    a = Tensor.randn(M, K)
    b = Tensor.randn(K, N)
    Tensor.realize(a, b)

    # we define a gemm function
    x = UOp.param(0, dtypes.float, shape=(M, K))
    y = UOp.param(1, dtypes.float, shape=(K, N))
    c = Tensor.call(a, b, fxn=x@y)

    np.testing.assert_allclose(c.numpy(), a.numpy() @ b.numpy(), rtol=1e-5, atol=1e-6)

  def test_call_complex_backward_auto(self):
    # complex chain: (a*b + a).exp2() * b.reciprocal() - tests mul, add, exp2, reciprocal, param reuse
    a = Tensor.randn(10, 10, requires_grad=True)
    b = Tensor.randn(10, 10, requires_grad=True) + 2  # avoid div by zero
    Tensor.realize(a, b)

    ((a*b + a).exp2() * b.reciprocal()).mean().backward()
    gt_a_grad, gt_b_grad = a.grad.numpy(), b.grad.numpy()
    a.grad, b.grad = None, None

    p0, p1 = UOp.param(0, dtypes.float, (10,10)), UOp.param(1, dtypes.float, (10,10))
    complex_fxn = (p0*p1 + p0).exp2() * p1.reciprocal()
    c = Tensor.call(a, b, fxn=complex_fxn)
    c.mean().backward()

    np.testing.assert_allclose(a.grad.numpy(), gt_a_grad, rtol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), gt_b_grad, rtol=1e-5)

  def test_call_plus_sharded(self):
    devs = ("CPU:0", "CPU:1")
    a = Tensor.ones(10, 10).shard(devs, axis=0)
    b = Tensor.ones(10, 10).shard(devs, axis=0)
    Tensor.realize(a, b)
    c = Tensor.call(a, b, fxn=a.as_param(0) + b.as_param(1))
    np.testing.assert_equal(c.numpy(), 2 * np.ones((10, 10)))

class TestCallSchedule(unittest.TestCase):
  def test_reshape_precompile(self):
    a = Tensor.empty(4, 8).realize()
    a = a.reshape(4,4,2).assign(Tensor.empty(4,4,2)).reshape(8,4)
    @function(precompile=True)
    def s(x): return x.sum(axis=0)
    (s(a)*3).realize()

  def test_call_precompiled(self):
    a = Tensor.empty(4, 8)
    @function(precompile=True)
    def s(x): return x*2
    (s(a)*3).realize()

  def test_double_call(self):
    a = Tensor.empty(4, 8)
    @function(precompile=True)
    def s(x): return x*2
    s(s(a)).realize()

  def test_double_call_contiguous(self):
    a = Tensor.empty(4, 8)
    @function(precompile=True)
    def s(x): return x*2
    s(s(a).contiguous()).realize()

  def test_call_double_gemm(self):
    a = Tensor.randn(4, 8, requires_grad=True)
    b = Tensor.randn(8, 12, requires_grad=True)
    c = Tensor.randn(12, 16, requires_grad=True)
    ref = Tensor.randn(4, 16)
    Tensor.realize(a,b,c,ref)
    @function(precompile=True)
    def gemm(a:Tensor, b:Tensor, c:Tensor) -> Tensor: return (a@b)@c
    out = gemm(a,b,c)
    (out-ref).square().mean().backward()
    out.realize(a.grad, b.grad, c.grad)

class TestCallRematerialize(TestCall):
  def setUp(self):
    self.og_fn = Tensor.call
    def new_call(t, *lst:Tensor, fxn:Tensor|UOp, grad_fxn:Callable|None=None) -> Tensor:
      return self.og_fn(t, *lst, fxn=fxn, grad_fxn=grad_fxn, rematerialize=True)
    Tensor.call = new_call
  def tearDown(self):
    Tensor.call = self.og_fn

class TestCallRematerialize(TestCall):
  def setUp(self):
    self.og_fn = Tensor.call
    def new_call(t, *lst:Tensor, fxn:Tensor|UOp, grad_fxn:Callable|None=None) -> Tensor:
      return self.og_fn(t, *lst, fxn=fxn, grad_fxn=grad_fxn, rematerialize=True)
    Tensor.call = new_call
  def tearDown(self):
    Tensor.call = self.og_fn

# **** consumer patterns ****

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

def _call_remat_get_num_exec_items(get_args_and_fxn, post_processing_func, post_processing_args, remat, grad_fxn=None):
  args, fxn = get_args_and_fxn()
  kern = args[0].call(*args[1:], fxn=fxn, grad_fxn=grad_fxn, rematerialize=remat)
  out = post_processing_func(kern, *post_processing_args)
  return len(out.schedule())

# **** arg factories ****

def get_call_plus_args():
  a = Tensor.ones(10, 10).contiguous()
  b = Tensor.ones(10, 10).contiguous()
  fxn = UOp.param(0, dtypes.float, (10, 10)) + UOp.param(1, dtypes.float, (10, 10))
  return (a, b), fxn

def get_call_complex_args():
  a = Tensor.ones(10, 10).contiguous()
  b = Tensor.full((10, 10), 3.).contiguous()
  p0, p1 = UOp.param(0, dtypes.float, (10, 10)), UOp.param(1, dtypes.float, (10, 10))
  fxn = (p0 * p1 + p0).exp2() * p1.reciprocal()
  return (a, b), fxn

def get_call_plus_sharded_args():
  devs = ("CPU:0", "CPU:1")
  a = Tensor.ones(10, 10).contiguous().shard(devs, axis=0)
  b = Tensor.ones(10, 10).contiguous().shard(devs, axis=0)
  fxn = a.as_param(0) + b.as_param(1)
  return (a, b), fxn

# **** schedule tests ****

class TestCallRematerializeSchedule(unittest.TestCase):
  # NOTE: unlike custom_kernel (which is always a separate schedule item), Tensor.call can fuse into consumers.
  # so we test remat-to-remat deltas: each additional consumer should add exactly 1 remat item (2 for sharded).

  # **** plus ****
  def test_plus_side_by_side(self):
    counts = [_call_remat_get_num_exec_items(get_call_plus_args, create_sidebyside_n_consumers, (i,), True) for i in range(1, 5)]
    for i in range(1, len(counts)):
      self.assertEqual(counts[i] - counts[i-1], 1, msg=f"delta at {i+1} consumers")

  def test_plus_waterfall(self):
    counts = [_call_remat_get_num_exec_items(get_call_plus_args, create_waterfall_n_consumers, (Tensor.ones(10, 10).contiguous(), i), True) for i in range(1, 5)]
    for i in range(1, len(counts)):
      self.assertEqual(counts[i] - counts[i-1], 1, msg=f"delta at {i+1} consumers")

  def test_plus_self_reference(self):
    args_reg, fxn_reg = get_call_plus_args()
    kern_reg = args_reg[0].call(*args_reg[1:], fxn=fxn_reg)
    args_remat, fxn_remat = get_call_plus_args()
    kern_remat = args_remat[0].call(*args_remat[1:], fxn=fxn_remat, rematerialize=True)
    self.assertGreater(len((kern_remat * kern_remat).schedule()), len((kern_reg * kern_reg).schedule()))

  def test_plus_diamond(self):
    counts = [_call_remat_get_num_exec_items(get_call_plus_args, create_diamond_n_consumers, (i,), True) for i in range(1, 5)]
    for i in range(1, len(counts)):
      self.assertEqual(counts[i] - counts[i-1], 1, msg=f"delta at {i+1} consumers")

  def test_plus_reduction(self):
    counts = [_call_remat_get_num_exec_items(get_call_plus_args, create_reduction_n_consumers, (i,), True) for i in range(1, 5)]
    for i in range(1, len(counts)):
      self.assertEqual(counts[i] - counts[i-1], 1, msg=f"delta at {i+1} consumers")

  # **** plus backward (explicit grad_fxn) ****
  def test_plus_backward_side_by_side(self):
    grad_fxn = lambda grad, call: (grad, grad)
    counts = [_call_remat_get_num_exec_items(get_call_plus_args, create_sidebyside_n_consumers, (i,), True, grad_fxn=grad_fxn) for i in range(1, 5)]
    for i in range(1, len(counts)):
      self.assertEqual(counts[i] - counts[i-1], 1, msg=f"delta at {i+1} consumers")

  def test_plus_backward_waterfall(self):
    grad_fxn = lambda grad, call: (grad, grad)
    counts = [_call_remat_get_num_exec_items(get_call_plus_args, create_waterfall_n_consumers, (Tensor.ones(10, 10).contiguous(), i), True, grad_fxn=grad_fxn) for i in range(1, 5)]
    for i in range(1, len(counts)):
      self.assertEqual(counts[i] - counts[i-1], 1, msg=f"delta at {i+1} consumers")

  # **** complex ****
  def test_complex_side_by_side(self):
    counts = [_call_remat_get_num_exec_items(get_call_complex_args, create_sidebyside_n_consumers, (i,), True) for i in range(1, 5)]
    for i in range(1, len(counts)):
      self.assertEqual(counts[i] - counts[i-1], 1, msg=f"delta at {i+1} consumers")

  def test_complex_waterfall(self):
    counts = [_call_remat_get_num_exec_items(get_call_complex_args, create_waterfall_n_consumers, (Tensor.ones(10, 10).contiguous(), i), True) for i in range(1, 5)]
    for i in range(1, len(counts)):
      self.assertEqual(counts[i] - counts[i-1], 1, msg=f"delta at {i+1} consumers")

  # **** plus sharded ****
  def test_plus_sharded_side_by_side(self):
    counts = [_call_remat_get_num_exec_items(get_call_plus_sharded_args, create_sidebyside_n_consumers, (i,), True) for i in range(1, 5)]
    for i in range(1, len(counts)):
      self.assertEqual(counts[i] - counts[i-1], 2, msg=f"delta at {i+1} consumers")

  def test_plus_sharded_waterfall(self):
    devs = ("CPU:0", "CPU:1")
    counts = [_call_remat_get_num_exec_items(get_call_plus_sharded_args, create_waterfall_n_consumers,
      (Tensor.ones(10, 10).contiguous().shard(devs, axis=0), i), True) for i in range(1, 5)]
    for i in range(1, len(counts)):
      self.assertEqual(counts[i] - counts[i-1], 2, msg=f"delta at {i+1} consumers")

  # **** special patterns ****
  def test_chained_remat(self):
    def get_remat_count(i):
      a, b = Tensor.ones(10, 10).contiguous(), Tensor.ones(10, 10).contiguous()
      fxn = UOp.param(0, dtypes.float, (10, 10)) + UOp.param(1, dtypes.float, (10, 10))
      c1 = a.call(b, fxn=fxn, rematerialize=True)
      c2 = c1.call(b, fxn=fxn, rematerialize=True)
      return len(create_sidebyside_n_consumers(c2, i).schedule())
    counts = [get_remat_count(i) for i in range(1, 5)]
    for i in range(1, len(counts)):
      self.assertGreater(counts[i], counts[i-1], msg=f"should grow at {i+1} consumers")

  def test_mixed_remat_nonremat(self):
    def get_remat_count(i):
      a, b = Tensor.ones(10, 10).contiguous(), Tensor.ones(10, 10).contiguous()
      fxn = UOp.param(0, dtypes.float, (10, 10)) + UOp.param(1, dtypes.float, (10, 10))
      c1 = a.call(b, fxn=fxn, rematerialize=True)
      c2 = a.call(b, fxn=fxn, rematerialize=False)
      r1 = create_sidebyside_n_consumers(c1, i)
      r2 = create_sidebyside_n_consumers(c2, i)
      return len((r1 + r2).sum().schedule())
    counts = [get_remat_count(i) for i in range(1, 5)]
    for i in range(1, len(counts)):
      self.assertGreater(counts[i], counts[i-1], msg=f"should grow at {i+1} consumers")

if __name__ == '__main__':
  unittest.main()
