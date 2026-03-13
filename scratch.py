from typing import Literal
from tinygrad import Tensor, function, TinyJit, GlobalCounters
from tinygrad.nn import Linear, optim, state
from tinygrad.helpers import Context, ProfileEvent, ProfilePointEvent
from tinygrad.device import Buffer, Device

IN_DIM, OUT_DIM, H_DIM, N_LAYERS = 8, 8, 64, 5
DATASET_SIZE, BATCH_SIZE = 64, 4
N_STEPS, LR = 8, 1e-3
DEVICES = ("cpu:0", "cpu:1")
CANON_DEVICES = tuple(Device.canonicalize(d) for d in DEVICES)
Mode = Literal["dp", "fsdp"]

@function(rematerialize=True)
def allgather_fxn(a:Tensor) -> Tensor: return a.allgather()

class Model:
  def __init__(self, in_dim:int, out_dim:int, h_dim:int, n_layers:int):
    dims = [in_dim] + [h_dim] * (n_layers - 1) + [out_dim]
    self.ws: list[Linear] = [Linear(i, o, bias=False) for i, o in zip(dims, dims[1:])]
  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.ws)

def loss_fn(pred:Tensor, tgt:Tensor) -> Tensor:
  return ((pred - tgt) ** 2).mean()

def make_dataset(dataset_size:int, batch_size:int, in_dim:int, out_dim:int, devices:tuple[str, ...]):
  X = Tensor.rand(dataset_size, in_dim).realize()
  Y = Tensor.rand(dataset_size, out_dim).realize()
  num_batches = dataset_size // batch_size
  X, Y = X.reshape(num_batches, batch_size, -1), Y.reshape(num_batches, batch_size, -1)
  return X.shard(devices, 1), Y.shard(devices, 1)

def build_model(mode:Mode):
  model = Model(IN_DIM, OUT_DIM, H_DIM, N_LAYERS)
  sharded_params: list[Tensor] = []

  if mode == "dp":
    for p in state.get_parameters(model): p.to_(DEVICES)

  if mode == "fsdp":
    for p in state.get_parameters(model): p.fsdp_(DEVICES)

  return model, optim.SGD(state.get_parameters(model), LR)

def do_step(x:Tensor, y:Tensor, model:Model, opt:optim.Optimizer):
  opt.zero_grad()
  out = model(x)
  loss = loss_fn(out, y)
  loss.backward()
  opt.step()
  return loss.realize()

@Tensor.train()
def step(x:Tensor, y:Tensor, model:Model, opt:optim.Optimizer):
  return do_step(x, y, model, opt)

def train(mode:Mode, n_steps:int=N_STEPS):
  Tensor.manual_seed(0)
  model, opt = build_model(mode)
  X, Y = make_dataset(DATASET_SIZE, BATCH_SIZE, IN_DIM, OUT_DIM, DEVICES)
  losses: list[float] = []
  for i in range(n_steps):
    loss = step(X[i % len(X)], Y[i % len(Y)], model, opt)
    losses.append(loss.item())
  return losses

def peak_memory(events:list[ProfileEvent], device:str) -> int:
  allocs: dict[int, int] = {}
  mem = peak = 0
  for e in events:
    if not isinstance(e, ProfilePointEvent) or e.device != device: continue
    if e.name == "alloc":
      sz = e.arg["sz"] * e.arg["dtype"].itemsize
      allocs[e.key] = sz
      mem += sz
    elif e.name == "free" and e.key in allocs:
      mem -= allocs[e.key]
      del allocs[e.key]
    peak = max(peak, mem)
  return peak

def profile_one_step(mode:Mode) -> dict[str, int]:
  Tensor.manual_seed(0)
  model, opt = build_model(mode)
  X, Y = make_dataset(DATASET_SIZE, BATCH_SIZE, IN_DIM, OUT_DIM, DEVICES)
  x, y = X[0].realize(), Y[0].realize()
  Buffer.profile_events.clear()
  with Context(PROFILE=1):
    step(x, y, model, opt)
  events = list(Buffer.profile_events)
  return {dev: peak_memory(events, dev) for dev in CANON_DEVICES}

def profile_one_step_jit(mode:Mode, warmup:int=2) -> dict[str, int]:
  Tensor.manual_seed(0)
  model, opt = build_model(mode)
  X, Y = make_dataset(DATASET_SIZE, BATCH_SIZE, IN_DIM, OUT_DIM, DEVICES)
  x, y = X[0].realize(), Y[0].realize()

  @TinyJit
  @Tensor.train()
  def jit_step(x:Tensor, y:Tensor):
    return do_step(x, y, model, opt)

  for _ in range(warmup):
    jit_step(x.contiguous(), y.contiguous())
  GlobalCounters.reset()
  GlobalCounters.reset_peak()
  jit_step(x.contiguous(), y.contiguous())
  Device[Device.DEFAULT].synchronize()
  return {dev: GlobalCounters.peak_mem_used_per_device.get(dev, 0) for dev in CANON_DEVICES}

if __name__ == "__main__":
  modes: tuple[Mode, ...] = ("dp", "fsdp")
  losses = {m: train(m, n_steps=N_STEPS) for m in modes}
  peaks = {m: profile_one_step(m) for m in modes}
  peaks_jit = {m: profile_one_step_jit(m) for m in modes}

  print("step | non-fsdp | orig-fsdp | new-fsdp")
  for i in range(N_STEPS):
    a,b = losses["dp"][i], losses["fsdp"][i]
    print(f"{i:4d} | {a:8.5f} | {b:9.5f}")

  print("\ndevice | non MB | orig MB | new MB | orig save MB | new save MB")
  for dev in CANON_DEVICES:
    non = peaks["dp"][dev]
    new = peaks["fsdp"][dev]
    print(f"{dev:6s} | {non/1e6:6.5f} | {new/1e6:6.5f} | {(non-new)/1e6:11.5f}")

  print("\nJIT (steady-state) peak memory")
  print("device | non MB | orig MB | new MB | orig save MB | new save MB")
  for dev in CANON_DEVICES:
    non = peaks_jit["dp"][dev]
    new = peaks_jit["fsdp"][dev]
    print(f"{dev:6s} | {non/1e6:6.5f} | {new/1e6:6.5f} | {(non-new)/1e6:11.5f}")
