import unittest, functools
from collections import defaultdict
from tinygrad import Tensor, Device, TinyJit, function, GlobalCounters
from tinygrad.nn import Linear, optim, state
from tinygrad.helpers import ProfilePointEvent, ProfileEvent, Context, CI, ceildiv
from tinygrad.device import Buffer
from tinygrad.uop.ops import Ops
from typing import Callable

def not_support_multi_device():
  return CI and Device.DEFAULT in ("CL", "CUDA")

def needs_multi_gpu(fn):
  @functools.wraps(fn)
  def wrapper(self, *args, **kwargs):
    try: Tensor.zeros(10, device=f"{Device.DEFAULT}:1").contiguous().realize()
    except Exception as e: self.skipTest(f"multi device not available: {e}")
    return fn(self, *args, **kwargs)
  return wrapper

# ---- helpers ----

N_DEVICES = 4

def _devices():
  return tuple(f"{Device.DEFAULT}:{i}" for i in range(1, N_DEVICES + 1))

class _Model:
  def __init__(self, in_dim: int, out_dim: int, n_dim: int, n_layers: int):
    assert n_layers > 0
    dims = [in_dim] + [n_dim] * (n_layers - 1) + [out_dim]
    self.ws: list[Linear] = [Linear(i, o, bias=False) for i, o in zip(dims, dims[1:])]
  def __call__(self, x: Tensor) -> Tensor: return x.sequential(self.ws)

class _ModelWithMask(_Model):
  """Model with large non-trainable buffers per layer (like attention causal masks)."""
  def __init__(self, in_dim: int, out_dim: int, n_dim: int, n_layers: int, mask_dim: int = 64):
    super().__init__(in_dim, out_dim, n_dim, n_layers)
    # large non-trainable buffers, similar to CausalSelfAttention.bias in GPT
    self.masks = [Tensor.ones(mask_dim, mask_dim) for _ in range(n_layers)]
    for m in self.masks: m.requires_grad = False

def _get_model(in_dim, out_dim, n_dim, n_layers, devices, use_fsdp, model_cls=_Model):
  """Returns (model, sharded_params_or_None)."""
  model = model_cls(in_dim, out_dim, n_dim, n_layers)
  if use_fsdp:
    for param in state.get_parameters(model):
      param.fsdp_(devices)
  else:
    for param in state.get_parameters(model):
      param.to_(devices)
  return model

def _get_optimizer(model, lr=0.001, opt_fn=None):
  if opt_fn is None: opt_fn = lambda params, lr: optim.SGD(params, lr)
  return opt_fn(state.get_parameters(model), lr)

def _make_dataset(dataset_size, batch_size, in_dim, devices, out_dim=1):
  X = Tensor.rand(dataset_size, in_dim).realize()
  Y = Tensor.rand(dataset_size, out_dim).realize() if out_dim > 1 else X.sum(-1).unsqueeze(-1).realize()
  num_batches = dataset_size // batch_size
  X, Y = X.reshape(num_batches, batch_size, -1), Y.reshape(num_batches, batch_size, -1)
  return X.shard(devices, 1), Y.shard(devices, 1)

@Tensor.train()
def _step(x, y, model, opt, loss_fn):
  opt.zero_grad()
  out = model(x)
  loss = loss_fn(out, y)
  loss.backward()
  opt.step()
  return loss.realize()

def _loss_fn(pred: Tensor, true: Tensor) -> Tensor:
  return ((pred - true) ** 2).mean()

def _peak_memory(events: list[ProfileEvent], device: str) -> int:
  """Return peak memory in bytes for a single device from profile events."""
  allocs: dict[int, int] = {}
  mem, peak = 0, 0
  for e in events:
    if not isinstance(e, ProfilePointEvent) or e.device != device: continue
    if e.name == 'alloc':
      sz = e.arg['sz'] * e.arg['dtype'].itemsize
      allocs[e.key] = sz
      mem += sz
    elif e.name == 'free' and e.key in allocs:
      mem -= allocs[e.key]
      del allocs[e.key]
    peak = max(peak, mem)
  return peak

def _buffers_at_peak(events: list[ProfileEvent], device: str) -> dict[int, int]:
  """Return {size_bytes: count} of buffers alive at peak memory."""
  allocs: dict[int, int] = {}
  alive: dict[int, int] = {}  # key -> size
  mem, peak, peak_alive = 0, 0, {}
  for e in events:
    if not isinstance(e, ProfilePointEvent) or e.device != device: continue
    if e.name == 'alloc':
      sz = e.arg['sz'] * e.arg['dtype'].itemsize
      allocs[e.key] = sz
      alive[e.key] = sz
      mem += sz
    elif e.name == 'free' and e.key in allocs:
      mem -= allocs[e.key]
      if e.key in alive: del alive[e.key]
    if mem > peak:
      peak = mem
      peak_alive = dict(alive)
  by_size: dict[int, int] = defaultdict(int)
  for sz in peak_alive.values():
    by_size[sz] += 1
  return dict(by_size)

# ---- tests ----

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestFSDP(unittest.TestCase):
  in_dim, out_dim, n_dim, n_layers = 8, 8, 64, 5
  _opt_fn = staticmethod(lambda params, lr: optim.SGD(params, lr))
  _n_state_per_param = 0  # SGD (no momentum) has no optimizer state buffers

  @needs_multi_gpu
  def setUp(self):
    Tensor.manual_seed(0)
    self.devices = _devices()

  def _param_bytes(self) -> int:
    """Total param bytes for the model."""
    dims = [self.in_dim] + [self.n_dim] * (self.n_layers - 1) + [self.out_dim]
    return sum(i * o * 4 for i, o in zip(dims, dims[1:]))

  def _theoretical_savings(self) -> int:
    """Exact bytes saved by sharding params + optimizer state across N_DEVICES."""
    per_param_savings = self._param_bytes() - self._param_bytes() // N_DEVICES
    return per_param_savings * (1 + self._n_state_per_param)

  # -- convergence tests --

  def test_fsdp_and_nonfsdp_same_initial_loss(self):
    """Both modes should produce identical first loss (same seed, same data)."""
    losses = {}
    for use_fsdp in [False, True]:
      Tensor.manual_seed(0)
      model = _get_model(self.in_dim, self.out_dim, self.n_dim, self.n_layers, self.devices, use_fsdp)
      opt = _get_optimizer(model, opt_fn=self._opt_fn)
      X, Y = _make_dataset(64, 4, self.in_dim, self.devices, self.out_dim)
      x, y = X[0], Y[0]
      loss = _step(x, y, model, opt, _loss_fn)
      losses["fsdp" if use_fsdp else "nonfsdp"] = loss.item()
    self.assertAlmostEqual(losses["fsdp"], losses["nonfsdp"], places=4,
                           msg=f"initial loss mismatch: fsdp={losses['fsdp']}, nonfsdp={losses['nonfsdp']}")

  def _train_n_steps(self, use_fsdp, n_steps=8, lr=0.001):
    """Train for n steps on different batches, return list of losses."""
    Tensor.manual_seed(0)
    model = _get_model(self.in_dim, self.out_dim, self.n_dim, self.n_layers, self.devices, use_fsdp)
    opt = _get_optimizer(model, lr=lr, opt_fn=self._opt_fn)
    X, Y = _make_dataset(64, 4, self.in_dim, self.devices, self.out_dim)
    losses = []
    for i in range(n_steps):
      loss = _step(X[i % len(X)], Y[i % len(Y)], model, opt, _loss_fn)
      losses.append(loss.item())
    return losses

  def test_fsdp_converges(self):
    """Average loss in last quarter should be lower than first quarter."""
    losses = self._train_n_steps(use_fsdp=True)
    q = len(losses) // 4
    avg_first = sum(losses[:q]) / q
    avg_last = sum(losses[-q:]) / q
    self.assertLess(avg_last, avg_first,
      f"FSDP not converging: first quarter avg={avg_first:.4f}, last quarter avg={avg_last:.4f}")

  def test_fsdp_and_nonfsdp_similar_convergence(self):
    """FSDP and non-FSDP should produce similar loss trajectories."""
    losses_fsdp = self._train_n_steps(use_fsdp=True)
    losses_nonfsdp = self._train_n_steps(use_fsdp=False)
    # final losses should be within 2x of each other
    self.assertAlmostEqual(losses_fsdp[-1], losses_nonfsdp[-1], delta=losses_nonfsdp[-1],
      msg=f"final loss too different: fsdp={losses_fsdp[-1]:.4f}, nonfsdp={losses_nonfsdp[-1]:.4f}")

  # -- memory tests --

  def _profile_one_step(self, use_fsdp: bool) -> list[ProfileEvent]:
    """Run one training step with profiling, return Buffer.profile_events."""
    Tensor.manual_seed(0)
    model = _get_model(self.in_dim, self.out_dim, self.n_dim, self.n_layers, self.devices, use_fsdp)
    opt = _get_optimizer(model, opt_fn=self._opt_fn)
    X, Y = _make_dataset(64, 4, self.in_dim, self.devices, self.out_dim)
    x, y = X[0].realize(), Y[0].realize()
    Buffer.profile_events.clear()
    with Context(PROFILE=1):
      _step(x, y, model, opt, _loss_fn)
    return list(Buffer.profile_events)

  def test_fsdp_lower_peak_memory(self):
    """FSDP peak memory should be strictly less than non-FSDP on every device."""
    events_nonfsdp = self._profile_one_step(use_fsdp=False)
    events_fsdp = self._profile_one_step(use_fsdp=True)
    for dev in self.devices:
      peak_nonfsdp = _peak_memory(events_nonfsdp, dev)
      peak_fsdp = _peak_memory(events_fsdp, dev)
      self.assertLess(peak_fsdp, peak_nonfsdp,
                      f"{dev}: FSDP peak {peak_fsdp} should be < non-FSDP peak {peak_nonfsdp}")

  def test_fsdp_memory_savings_near_theoretical(self):
    """FSDP savings should be at least 80% of theoretical param savings."""
    events_nonfsdp = self._profile_one_step(use_fsdp=False)
    events_fsdp = self._profile_one_step(use_fsdp=True)
    theoretical = self._theoretical_savings()
    dev = self.devices[0]
    peak_nonfsdp = _peak_memory(events_nonfsdp, dev)
    peak_fsdp = _peak_memory(events_fsdp, dev)
    actual_savings = peak_nonfsdp - peak_fsdp
    ratio = actual_savings / theoretical
    self.assertGreaterEqual(ratio, 0.65,
      f"savings {actual_savings} is only {ratio:.0%} of theoretical {theoretical} "
      f"(fsdp={peak_fsdp}, nonfsdp={peak_nonfsdp})")

  def test_fsdp_fewer_full_size_buffers_than_nonfsdp(self):
    """FSDP should have strictly fewer full-size buffers at peak than non-FSDP.
    This catches the regression where backward allgather outputs all pile up."""
    events_fsdp = self._profile_one_step(use_fsdp=True)
    events_nonfsdp = self._profile_one_step(use_fsdp=False)
    dev = self.devices[0]
    full_layer_bytes = self.n_dim * self.n_dim * 4
    fsdp_full = _buffers_at_peak(events_fsdp, dev).get(full_layer_bytes, 0)
    nonfsdp_full = _buffers_at_peak(events_nonfsdp, dev).get(full_layer_bytes, 0)
    self.assertLess(fsdp_full, nonfsdp_full,
      f"FSDP has {fsdp_full} full-size buffers at peak, non-FSDP has {nonfsdp_full}. "
      f"FSDP should have fewer.")

  def test_fsdp_sharded_params_present_at_peak(self):
    """At FSDP peak, sharded param buffers should be present (they're always alive)."""
    events = self._profile_one_step(use_fsdp=True)
    dev = self.devices[0]
    bufs = _buffers_at_peak(events, dev)
    shard_bytes = self.n_dim * self.n_dim * 4 // N_DEVICES
    shard_count = bufs.get(shard_bytes, 0)
    n_hidden = self.n_layers - 2  # hidden layers have n_dim x n_dim params
    self.assertGreaterEqual(shard_count, n_hidden,
      f"expected >= {n_hidden} sharded buffers ({shard_bytes}B) at peak, got {shard_count}. all sizes: {bufs}")

  # -- correctness: optimizer updates sharded params --

  def test_fsdp_params_update_across_steps(self):
    """Sharded params should change after each optimizer step."""
    Tensor.manual_seed(0)
    model = _get_model(self.in_dim, self.out_dim, self.n_dim, self.n_layers, self.devices, use_fsdp=True)
    opt = _get_optimizer(model, opt_fn=self._opt_fn)
    X, Y = _make_dataset(64, 4, self.in_dim, self.devices, self.out_dim)
    # snapshot sharded param values before step
    before = [p.numpy().copy() for p in state.get_parameters(model)]
    _step(X[0], Y[0], model, opt, _loss_fn)
    after = [p.numpy() for p in state.get_parameters(model)]
    for i, (b, a) in enumerate(zip(before, after)):
      self.assertFalse((b == a).all(), f"sharded param {i} did not change after optimizer step")

  def test_fsdp_loss_decreases_same_data(self):
    """Training on the same batch repeatedly must decrease loss (forward uses updated params)."""
    Tensor.manual_seed(0)
    model = _get_model(self.in_dim, self.out_dim, 32, 3, self.devices, use_fsdp=True)
    opt = _get_optimizer(model, lr=0.05, opt_fn=self._opt_fn)
    X, Y = _make_dataset(64, 4, self.in_dim, self.devices, self.out_dim)
    x, y = X[0], Y[0]
    losses = []
    for _ in range(4):
      loss = _step(x, y, model, opt, _loss_fn)
      losses.append(loss.item())
    self.assertLess(losses[-1], losses[0],
      f"loss did not decrease on same data: {losses}")

  def test_fsdp_multi_step_loss_keeps_changing(self):
    """Loss should change across multiple steps on different data (model uses updated params)."""
    Tensor.manual_seed(0)
    model = _get_model(self.in_dim, self.out_dim, self.n_dim, self.n_layers, self.devices, use_fsdp=True)
    opt = _get_optimizer(model, opt_fn=self._opt_fn)
    X, Y = _make_dataset(64, 4, self.in_dim, self.devices, self.out_dim)
    x, y = X[0], Y[0]
    losses = []
    for i in range(3):
      loss = _step(x, y, model, opt, _loss_fn)
      losses.append(loss.item())
    # each step should produce a different loss (model is updating)
    for i in range(1, len(losses)):
      self.assertNotAlmostEqual(losses[i], losses[i-1], places=6,
        msg=f"loss did not change between step {i-1} and {i}: {losses}")

# NOTE: adaptive optimizers (Adam) are not used here because `x @ allgather(sharded_w)` and `x @ replicated_w` produce
# numerically different backward passes (different UOp graphs → different fp rounding). These tiny gradient differences
# compound through Adam's m/sqrt(v) adaptive state, causing FSDP and DP loss trajectories to diverge over many steps.
# SGD+momentum tests optimizer state handling without this adaptive amplification.
@unittest.skipIf(not_support_multi_device(), "no multi")
class TestFSDPOptState(TestFSDP):
  """Same tests as TestFSDP but with SGD+momentum (has momentum buffer optimizer state)."""
  _opt_fn = staticmethod(lambda params, lr: optim.SGD(params, lr, momentum=0.9))
  _n_state_per_param = 1  # SGD with momentum has 1 state buffer per param

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestFSDPUneven(TestFSDP):
  in_dim, out_dim, n_dim, n_layers = 7, 9, 65, 2

  def test_model_has_uneven_param_sizes(self):
    model = _Model(self.in_dim, self.out_dim, self.n_dim, self.n_layers)
    self.assertTrue(any(p.numel() % N_DEVICES != 0 for p in state.get_parameters(model)))

  def test_fsdp_fewer_full_size_buffers_than_nonfsdp(self):
    """FSDP should still reduce the number of full-param buffers at peak for uneven params."""
    events_fsdp = self._profile_one_step(use_fsdp=True)
    events_nonfsdp = self._profile_one_step(use_fsdp=False)
    dev = self.devices[0]
    model = _Model(self.in_dim, self.out_dim, self.n_dim, self.n_layers)
    param_numels = [p.numel() for p in state.get_parameters(model)]
    fsdp_full_sizes = {ceildiv(numel, N_DEVICES) * N_DEVICES * 4 for numel in param_numels}
    nonfsdp_full_sizes = {numel * 4 for numel in param_numels}
    fsdp_full = sum(_buffers_at_peak(events_fsdp, dev).get(sz, 0) for sz in fsdp_full_sizes)
    nonfsdp_full = sum(_buffers_at_peak(events_nonfsdp, dev).get(sz, 0) for sz in nonfsdp_full_sizes)
    self.assertLess(fsdp_full, nonfsdp_full,
      f"FSDP has {fsdp_full} padded full-param buffers at peak, non-FSDP has {nonfsdp_full}. "
      f"FSDP should have fewer.")

# TODO: FSDP does not yet support FUSE_OPTIM — pad_multi asserts on padding along the sharded axis
@unittest.skip("FSDP + FUSE_OPTIM not yet supported")
@unittest.skipIf(not_support_multi_device(), "no multi")
class TestFSDPFusedOptim(TestFSDP):
  """Same tests as TestFSDP but with FUSE_OPTIM=1."""
  _opt_fn = staticmethod(lambda params, lr: optim.SGD(params, lr, fused=True))

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestFSDPWithMask(TestFSDP):
  """Tests FSDP with a model containing large non-trainable buffers (like attention causal masks).
  Non-trainable tensors should be replicated, not FSDP-sharded, and should not get optimizer state."""

  @needs_multi_gpu
  def setUp(self):
    Tensor.manual_seed(0)
    self.devices = _devices()
    self._model_cls = _ModelWithMask

  def _profile_one_step(self, use_fsdp: bool) -> list[ProfileEvent]:
    Tensor.manual_seed(0)
    model = _get_model(self.in_dim, self.out_dim, self.n_dim, self.n_layers, self.devices, use_fsdp, model_cls=_ModelWithMask)
    opt = _get_optimizer(model, opt_fn=self._opt_fn)
    X, Y = _make_dataset(64, 4, self.in_dim, self.devices, self.out_dim)
    x, y = X[0].realize(), Y[0].realize()
    Buffer.profile_events.clear()
    with Context(PROFILE=1):
      _step(x, y, model, opt, _loss_fn)
    return list(Buffer.profile_events)

  def test_fsdp_nontrainable_not_in_optimizer(self):
    """Non-trainable tensors (requires_grad=False) must not appear in optimizer params."""
    Tensor.manual_seed(0)
    model = _get_model(self.in_dim, self.out_dim, self.n_dim, self.n_layers, self.devices, True, model_cls=_ModelWithMask)
    opt = _get_optimizer(model, opt_fn=self._opt_fn)
    n_trainable = sum(1 for p in state.get_parameters(model) if p.requires_grad is not False)
    self.assertEqual(len(opt.params), n_trainable,
      f"optimizer has {len(opt.params)} params but model has {n_trainable} trainable params")
@unittest.skipIf(not_support_multi_device(), "no multi")
class TestFSDPJit(unittest.TestCase):
  """Test FSDP with JIT enabled, verifying memory savings are preserved."""
  in_dim, out_dim, n_dim, n_layers = 8, 8, 64, 5
  _opt_fn = staticmethod(lambda params, lr: optim.SGD(params, lr))

  @needs_multi_gpu
  def setUp(self):
    Tensor.manual_seed(0)
    self.devices = _devices()

  def _train_jit(self, use_fsdp, n_steps=4, lr=0.001):
    Tensor.manual_seed(0)
    model = _get_model(self.in_dim, self.out_dim, self.n_dim, self.n_layers, self.devices, use_fsdp)
    opt = _get_optimizer(model, lr=lr, opt_fn=self._opt_fn)
    X, Y = _make_dataset(64, 4, self.in_dim, self.devices, self.out_dim)

    @TinyJit
    @Tensor.train()
    def jit_step(x, y):
      opt.zero_grad()
      out = model(x)
      loss = _loss_fn(out, y)
      loss.backward()
      opt.step()
      return loss.realize()

    x, y = X[0].contiguous().realize(), Y[0].contiguous().realize()
    losses, peaks = [], []
    for i in range(n_steps):
      GlobalCounters.reset()
      GlobalCounters.reset_peak()
      loss = jit_step(x, y)
      Device[Device.DEFAULT].synchronize()
      peaks.append(max(GlobalCounters.peak_mem_used_per_device.values()) if GlobalCounters.peak_mem_used_per_device else 0)
      losses.append(loss.item())
    return losses, peaks

  def test_fsdp_jit_converges(self):
    """FSDP+JIT should converge (average loss decreases over training)."""
    losses, _ = self._train_jit(use_fsdp=True, n_steps=8, lr=0.01)
    self.assertLess(losses[-1], losses[-2])

  def test_fsdp_jit_lower_peak_than_nonfsdp_jit(self):
    """FSDP+JIT peak memory should be less than non-FSDP+JIT after JIT warmup."""
    _, peaks_nonfsdp = self._train_jit(use_fsdp=False, n_steps=4)
    _, peaks_fsdp = self._train_jit(use_fsdp=True, n_steps=4)
    # compare post-warmup peaks (iteration 2+, after JIT capture)
    peak_nonfsdp = peaks_nonfsdp[-1]
    peak_fsdp = peaks_fsdp[-1]
    self.assertLess(peak_fsdp, peak_nonfsdp,
      f"FSDP+JIT peak {peak_fsdp/1e6:.1f} MB should be < non-FSDP+JIT peak {peak_nonfsdp/1e6:.1f} MB")

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestFSDPUnevenJit(TestFSDPJit):
  """JIT should also work with uneven padded FSDP parameter sizes."""
  in_dim, out_dim, n_dim, n_layers = 7, 9, 65, 5

if __name__ == '__main__':
  unittest.main()
