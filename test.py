from tinygrad import Tensor, Device, function
from tinygrad.nn import Linear, optim, state
from tinygrad.helpers import getenv
from typing import Callable
from tinygrad.helpers import ProfilePointEvent, ProfileEvent, Context
from collections import defaultdict
from tinygrad.device import Buffer
import sys

Tensor.manual_seed(0)

devices = tuple(f"{Device.DEFAULT}:{i}" for i in range(1, 5))


def filter_profile_events(events: list[ProfileEvent]) -> list[ProfilePointEvent]:
  return [event for event in events if isinstance(event, ProfilePointEvent)]


def analyze_profile_events(events: list[ProfileEvent]) -> dict[str, list[int]]:
  events: list[ProfilePointEvent] = filter_profile_events(events)
  # device -> current alloc'd
  curr_allocd: dict[str, list[int]] = defaultdict(lambda: [0])
  # key -> sz
  allocs: dict[int, (str, int)] = {}
  for e in events:
    if e.name == 'alloc':
      allocs[e.key] = e.arg['sz'] * e.arg['dtype'].itemsize
      curr_allocd[e.device].append(curr_allocd[e.device][-1]+allocs[e.key])
    elif e.name == 'free' and e.key in allocs:
      curr_allocd[e.device].append(curr_allocd[e.device][-1]-allocs[e.key])
  return curr_allocd

class Model:
  def __init__(self, in_dim:int=10, out_dim:int=1, n_dim:int=128, n_layers:int = 4):
    assert n_layers > 0, n_layers
    dims = [in_dim] + [n_dim] * (n_layers - 1) + [out_dim]
    self.ws: list[Linear] = [Linear(i, o, bias=False) for i, o in zip(dims, dims[1:])]

  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.ws)


@Tensor.train()
def step(x: Tensor, y: Tensor, model: Model, opt: optim.Optimizer, loss_fn: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
  opt.zero_grad()
  out = model(x)
  # out.realize()
  loss = loss_fn(out, y)
  loss.backward()
  opt.step()
  return loss.realize()

# sum
def make_dataset(size:int, in_dim:int) -> tuple[Tensor, Tensor]:
  return (X:=Tensor.rand(size, in_dim).realize()), X.sum(-1).unsqueeze(-1).realize()

@function
def allgather_fxn(a: Tensor) -> Tensor: return a.allgather()

def get_model(in_dim:int=10, out_dim:int=1, n_dim:int=128, n_layers:int = 4, use_fsdp: bool = False) -> Model:
  model = Model(in_dim, out_dim, n_dim, n_layers)
  if use_fsdp:
    for param in state.get_parameters(model):
      sharded_param = param.reshape(-1).shard(devices, 0).reshape(param.shape)
      # sharded_param = param.to_(devices)
      param.replace(allgather_fxn(sharded_param))
  else:
    ...
    for param in state.get_parameters(model):
      param.to_(devices)
  return model

def get_dataset(dataset_size:int, batch_size:int, in_dim: int) -> tuple[Tensor, Tensor]:
  X, Y = make_dataset(dataset_size, in_dim)
  num_batches = dataset_size // batch_size
  X, Y = X.reshape(num_batches, batch_size, -1), Y.reshape(num_batches, batch_size, -1)
  return (X.shard(devices, 1), Y.shard(devices, 1))

def debug_stuff():
  devices = (f'{Device.DEFAULT}:0', f'{Device.DEFAULT}:1')
  x = Tensor.rand(10).to(devices).realize()
  weight1 = Tensor.rand(10, 10,  requires_grad=True).shard(devices, 0).realize().fsdp()
  weight2 = Tensor.rand(10, 1,  requires_grad=True).shard(devices, 0).realize().fsdp()

  out1 = x * weight1
  out2 = out1 * weight2
  out = out2.sum()

  out.backward()

  def print_pair(t: Tensor, msg):
    print("=====", msg)
    print(t.uop)
    print('-----')
    print(t.grad.uop)
    print("=====")

  # print_pair(weight1, "weight1")
  # print(weight1.grad.numpy())
  # print(x.numpy())
  # print_pair(weight2, "weight2")

if __name__ == '__main__':
  epochs, dataset_size, batch_size = 1, 256, 4
  in_dim, out_dim, n_dim, n_layers = 2, 1, 32, 4
  use_fsdp, just_forward = getenv("USE_FSDP", 0), False 
  mem_tracking = getenv("MEM", 0)
  just_debug = False

  model = get_model(in_dim, out_dim, n_dim, n_layers, use_fsdp)
  X, Y = get_dataset(dataset_size, batch_size, in_dim)

  opt = optim.SGD(state.get_parameters(model), .001)

  def loss_fn(pred: Tensor, true: Tensor) -> Tensor:
    return ((pred-true)**2).mean()
  
  if just_debug:
    debug_stuff()
    sys.exit()

  if not mem_tracking:
    for e in range(epochs):
      for i, (x, y) in enumerate(zip(X, Y)):
        if just_forward:
          pred = model(x).realize()
          if i % 10 == 0: print(pred.item())
        else:
          loss = step(x, y, model, opt, loss_fn)  
          if i % 10 == 0: print(f"{loss.item()=}")
  else:
    x, y = X[0].realize(), Y[0].realize()
    Buffer.profile_events.clear()
    with Context(PROFILE=1):
      loss = step(x, y, model, opt, loss_fn)
    mem_usage = analyze_profile_events(Buffer.profile_events)
    for dev in devices:
      # print(dev)
      print(max(mem_usage[dev])/10**6)





