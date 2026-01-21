import unittest
import functools
from tinygrad import Tensor, Variable, UOp, Context
from tinygrad.uop.ops import KernelInfo
from tinygrad.engine.schedule import schedule_cache, pm_pre_sched_cache, create_pre_schedule
from tinygrad.engine.realize import ExecItem
from tinygrad.uop.ops import graph_rewrite

def custom_set0_kernel(A:UOp, num:int) -> UOp:
  return A[0].set(num).sink(arg=KernelInfo(f"custom_set0_{num}"))

def schedule_one() -> tuple[ExecItem, bool] :
  big_sink = UOp.sink(Tensor([1]).uop)
  big_sink_cache = graph_rewrite(big_sink, pm_pre_sched_cache, ctx=({}, {}), name="schedule_one")
  pre_schedule, _, cache_hit = create_pre_schedule(big_sink, big_sink_cache)
  ei, = pre_schedule
  return ei, cache_hit

class TestScheduleCache(unittest.TestCase):
  def test_bound_variable_reuses_cache(self):
    schedule_cache.clear()
    v = Variable('v', 1, 100)
    x = Tensor.ones(10).contiguous().realize()

    # first run with v=5
    t1 = (x + Tensor(v.bind(5))).sum()
    self.assertEqual(t1.item(), 60.0)
    cache_size_after_first = len(schedule_cache)

    # second run with v=10 should reuse cache
    t2 = (x + Tensor(v.bind(10))).sum()
    self.assertEqual(t2.item(), 110.0)
    self.assertEqual(len(schedule_cache), cache_size_after_first)

  def test_bound_variable_var_vals(self):
    v = Variable('pos', 1, 100)
    x = Tensor.ones(10).contiguous().realize()

    t = x + Tensor(v.bind(42))
    _, var_vals = t.schedule_with_vars()
    self.assertEqual(var_vals, {'pos': 42})

  @Context(SPEC=0)
  def test_custom_kernel(self):
    for i in range(4):
      a = Tensor.empty(1)
      a = Tensor.custom_kernel(a, fxn=functools.partial(custom_set0_kernel, num=i))[0]
      a.realize()
      self.assertEqual(a.item(), i)

  @Context(SPEC=0)
  def test_same_custom_function_reuses_cache(self):
    schedule_cache.clear()
    fxn = functools.partial(custom_set0_kernel, num=10)

    # first run
    a = Tensor.empty(1)
    a = Tensor.custom_kernel(a, fxn=fxn)[0]
    a.realize()
    self.assertEqual(a.item(), 10)
    cache_size_after_first = len(schedule_cache)

    # second run with same function should reuse cache
    b = Tensor.empty(1)
    b = Tensor.custom_kernel(b, fxn=fxn)[0]
    b.realize()
    self.assertEqual(b.item(), 10)
    self.assertEqual(len(schedule_cache), cache_size_after_first)

  def test_simple(self):
    a = Tensor.ones(10).contiguous()
    b = Tensor.ones(10).contiguous()
    Tensor.realize(a, b)

    # warm up
    for _ in range(2):
      num = (a.sum().contiguous()+b.sum().contiguous()).item()
      print(num)

    # confirm schedule cache doesn't grow
    start_len_schedule_cache = len(schedule_cache)
    for _ in range(3):
      num = (a.sum().contiguous()+b.sum().contiguous()).item()
      print(num)
    self.assertEqual(len(schedule_cache), start_len_schedule_cache)

  def test_disable_schedule_cache(self):
    schedule_cache.clear()

    # test write disabled
    with Context(SCACHE=0):
      ei, hit = schedule_one()
    self.assertFalse(hit)
    self.assertEqual(len(schedule_cache), 0)

    # test read/write enabled
    with Context(SCACHE=1):
      ei1, hit1 = schedule_one()
      ei2, hit2 = schedule_one()
    self.assertFalse(hit1)
    self.assertTrue(hit2)
    self.assertEqual(len(schedule_cache), 1)
    ((cached_ei,), _),  = list(schedule_cache.values())
    self.assertEqual(id(cached_ei), id(ei1))
    self.assertEqual(id(cached_ei), id(ei2))

    # test read disabled
    with Context(SCACHE=0):
      ei, hit = schedule_one()
    self.assertFalse(hit)
    ((cached_ei,), _),  = list(schedule_cache.values())
    self.assertNotEqual(id(cached_ei), id(ei))

if __name__ == "__main__":
  unittest.main()
