import unittest
import numpy as np
from typing import Callable
from tinygrad import Tensor, function
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops

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

class TestCallShape(unittest.TestCase):
  def test_call_shape_int(self):
    # fixed-shape function: shape passes through unchanged
    @function
    def f(x:Tensor) -> Tensor: return x * 2
    self.assertEqual(f(Tensor.empty(4, 8)).shape, (4, 8))

  def test_call_shape_param_substitution(self):
    # symbolic shape dimension is substituted: inner PARAM replaced with the BIND arg
    @function
    def f(x:Tensor) -> Tensor: return x * 2
    sz = UOp.variable("sz", 1, 8)
    shape = f(Tensor.empty(8)[:sz.bind(5)]).shape
    # the PARAM should be gone, replaced with the BIND from the call arg
    self.assertIsInstance(shape[0], UOp)
    self.assertNotEqual(shape[0].op, Ops.PARAM)
    self.assertEqual(shape[0], sz.bind(5))

  def test_call_shape_expr_substitution(self):
    # expression containing PARAMs in shape gets fully substituted
    @function
    def f(x:Tensor) -> Tensor: return x + 1
    sz = UOp.variable("sz", 1, 10)
    shape = f(Tensor.empty(10, 4)[:sz.bind(3)]).shape
    self.assertIsInstance(shape[0], UOp)
    self.assertNotEqual(shape[0].op, Ops.PARAM)
    self.assertEqual(shape[1], 4)

  def test_call_shape_no_param_passthrough(self):
    # a non-PARAM UOp shape element passes through unchanged
    @function
    def f(x:Tensor) -> Tensor: return x * 3
    sz = UOp.variable("sz", 1, 8)
    shape = f(Tensor.empty(8)[:sz.bind(5)]).shape
    self.assertEqual(shape[0], sz.bind(5))

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

  def test_precompile_symbolic_shape(self):
    """precompile with a symbolic-shaped input produces correct values and shape"""
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x * 2
    sz = UOp.variable("sz", 1, 8)
    a = Tensor([1., 2., 3., 4., 5., 6., 7., 8.])[:sz.bind(5)]
    out = f(a)
    self.assertIsInstance(out.shape[0], UOp)
    np.testing.assert_allclose(out[:5].numpy(), [2., 4., 6., 8., 10.])

  def test_precompile_symbolic_shape_contiguous(self):
    """precompile with a .contiguous() inside the function body on a symbolic-shaped input"""
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return (x * 2).contiguous() + 1
    sz = UOp.variable("sz", 1, 8)
    a = Tensor([1., 2., 3., 4., 5., 6., 7., 8.])[:sz.bind(3)]
    out = f(a)
    self.assertIsInstance(out.shape[0], UOp)
    np.testing.assert_allclose(out[:3].numpy(), [3., 5., 7.])

  def test_precompile_symbolic_shape_chain(self):
    """precompiled symbolic result used in downstream ops (tests AFTER has correct symbolic shape)"""
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x * 2
    sz = UOp.variable("sz", 1, 8)
    a = Tensor([1., 2., 3., 4., 5., 6., 7., 8.])[:sz.bind(4)]
    out = f(a) + 10  # downstream op on the precompiled result
    self.assertIsInstance(out.shape[0], UOp)
    np.testing.assert_allclose(out[:4].numpy(), [12., 14., 16., 18.])

  def test_precompile_bind_arg(self):
    """precompile with a BIND (scalar variable) as a function argument"""
    @function(precompile=True)
    def f(x:Tensor, scale:UOp) -> Tensor: return x * scale
    v = UOp.variable("scale", 1, 100)
    a = Tensor([1., 2., 3.])
    out = f(a, v.bind(5))
    np.testing.assert_allclose(out.numpy(), [5., 10., 15.])

  def test_precompile_schedule_cache_hit(self):
    """two instances of the same @function should produce identical function body keys (schedule cache hit)"""
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x + Tensor.full(x.shape, -1.0)
    a = Tensor.empty(4, 8)
    b = Tensor.empty(4, 8)
    r0, r1 = f(a), f(b)
    # find the CALL nodes
    c0 = next(u for u in r0.uop.toposort() if u.op is Ops.CALL)
    c1 = next(u for u in r1.uop.toposort() if u.op is Ops.CALL)
    # the function bodies (src[0]) should have identical keys — unique consts must not leak through
    self.assertEqual(c0.src[0].key, c1.src[0].key)

  def test_precompile_symbolic_2d(self):
    """precompile with symbolic shapes in 2D (tests debuf reshape with symbolic PARAM)"""
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x * 2 + 1
    sz = UOp.variable("sz", 1, 16)
    a = Tensor.arange(16*4).reshape(16, 4).float()[:sz.bind(5)]
    out = f(a)
    # result shape should have the symbolic dim, not the max
    self.assertIsInstance(out.shape[0], UOp)
    np.testing.assert_allclose(out[:5].numpy(), (np.arange(16*4).reshape(16, 4)[:5] * 2 + 1).astype(np.float32))
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
