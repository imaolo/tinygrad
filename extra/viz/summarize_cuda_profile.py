#!/usr/bin/env python3
from __future__ import annotations

import argparse, os, sys
from collections import defaultdict
from pathlib import Path

# Some shells export DEBUG=release or similar, which breaks tinygrad's int env parsing.
for key in ("DEBUG", "BEAM", "NOOPT"):
  try:
    int(os.getenv(key, "0"))
  except ValueError:
    os.environ[key] = "0"
os.environ.setdefault("VIZ", "0")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from tinygrad.viz import serve as viz
from tinygrad.device import ProfileDeviceEvent
from tinygrad.helpers import TracingKey


def load_events(profile_path:Path) -> list:
  events = viz.load_pickle(profile_path, default=[])
  if not events: raise RuntimeError(f"empty profile in {profile_path}")
  return events


def pick_sources(events:list, sources:list[str] | None) -> list[str]:
  if sources is not None: return sources
  devs = []
  seen = set()
  for st, en, e in viz.flatten_events(events, {e.device:e.tdiff for e in events if isinstance(e, ProfileDeviceEvent)}):
    if float(en-st) <= 0 or not getattr(e, "device", "").startswith("CUDA"): continue
    if e.device not in seen:
      seen.add(e.device)
      devs.append(e.device)
  return devs


def summarize_source(events:list, source:str, top:int, match:str | None, sort_key:str) -> None:
  device_ts_diffs = {e.device:e.tdiff for e in events if isinstance(e, ProfileDeviceEvent)}
  agg:dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
  total_s = 0.0
  matched_events = 0
  for st, en, e in viz.flatten_events(events, device_ts_diffs):
    if getattr(e, "device", None) != source: continue
    dur_s = float(en-st) * 1e-6
    if dur_s <= 0: continue
    name = e.name.display_name if isinstance(e.name, TracingKey) else str(e.name)
    if match is not None and match not in name: continue
    total_s += dur_s
    agg[name][0] += dur_s
    agg[name][1] += 1
    matched_events += 1

  if sort_key == "avg":
    key_fn = lambda kv: (kv[1][0] / kv[1][1], kv[1][0], kv[1][1])
  elif sort_key == "count":
    key_fn = lambda kv: (kv[1][1], kv[1][0], kv[1][0] / kv[1][1])
  else:
    key_fn = lambda kv: (kv[1][0], kv[1][1], kv[1][0] / kv[1][1])

  print(f"## {source} total_s={total_s:.3f} matched_events={matched_events} unique_kernels={len(agg)}")
  print(f"{'total_s':>10} {'count':>8} {'avg_s':>10} {'avg_ms':>10} {'pct':>7} name")
  for name, (kernel_total_s, count) in sorted(agg.items(), key=key_fn, reverse=True)[:top]:
    pct = (100.0 * kernel_total_s / total_s) if total_s > 0 else 0.0
    print(f"{kernel_total_s:10.3f} {count:8d} {kernel_total_s/count:10.3f} {(kernel_total_s/count)*1e3:10.3f} {pct:6.2f}% {name}")
  print()


def main() -> None:
  parser = argparse.ArgumentParser(description="Summarize CUDA kernels from a VIZ profile pickle")
  parser.add_argument("--profile-path", type=Path, default=Path(os.getenv("PROF", "/tmp/new_profile.pkl")),
                      help="Path to profile.pkl")
  parser.add_argument("-s", "--source", action="append", dest="sources",
                      help="CUDA source to summarize, repeat for multiple. Default: all CUDA sources in the profile")
  parser.add_argument("-n", "--top", type=int, default=int(os.getenv("TOP", "20")),
                      help="Number of kernels to print per source")
  parser.add_argument("-m", "--match", type=str, default=os.getenv("MATCH"),
                      help="Only include kernel names containing this substring")
  parser.add_argument("--sort", choices=("total", "avg", "count"), default="total",
                      help="Sort by total time, average time, or count")
  args = parser.parse_args()

  events = load_events(args.profile_path)
  sources = pick_sources(events, args.sources)
  if not sources: raise RuntimeError("no CUDA sources found in the profile")
  for source in sources:
    summarize_source(events, source, args.top, args.match, args.sort)


if __name__ == "__main__":
  main()
