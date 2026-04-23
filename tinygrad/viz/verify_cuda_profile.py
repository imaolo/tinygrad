from tinygrad import Device, Tensor
from tinygrad.device import Compiled
from tinygrad.helpers import Context, ProfileRangeEvent

def run(dev: str):
  a = Tensor.rand(1024, 1024, device=dev)
  b = Tensor.rand(1024, 1024, device=dev)
  c = (a @ b).realize()
  Device[dev].synchronize()
  return c

if __name__ == "__main__":
  with Context(PROFILE=1):
    Compiled.profile_events.clear()
    run("CUDA")
    try: run("CUDA:1")
    except Exception: pass
    events = [e for e in Compiled.profile_events if isinstance(e, ProfileRangeEvent) and e.device.startswith("CUDA")]
    print(f"cuda range events: {len(events)}")
    for e in events[:20]:
      print(f"{e.device:7s} {e.name} {e.st} -> {e.en}")
    if not events:
      raise SystemExit("no CUDA ProfileRangeEvent entries recorded")
