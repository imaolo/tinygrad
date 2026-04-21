import os, sys

if __name__ == "__main__":
  for key in ("DEBUG", "BEAM", "NOOPT"):
    try:
      int(os.getenv(key, "0"))
    except ValueError:
      os.environ[key] = "0"
    else:
      os.environ.setdefault(key, "0")

import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import getenv


def load_flash_attention(device:str, impl:str):
  if impl == "auto":
    if device.startswith("CUDA"):
      from extra.thunder.cuda.fa import flash_attention
    elif device.startswith("AMD"):
      from extra.thunder.amd.fa import flash_attention
    else:
      from extra.thunder.tiny.fa import flash_attention
  elif impl == "cuda":
    from extra.thunder.cuda.fa import flash_attention
  elif impl == "amd":
    from extra.thunder.amd.fa import flash_attention
  elif impl == "tiny":
    from extra.thunder.tiny.fa import flash_attention
  else:
    raise ValueError(f"unknown FA_IMPL={impl!r}")
  return flash_attention


def summarize_diff(diff:np.ndarray, h_kv:int):
  mean_abs = float(diff.mean())
  max_abs = float(diff.max())
  print(f"mean_abs={mean_abs:.6f} max_abs={max_abs:.6f}")

  if diff.shape[2] % 32 != 0:
    tail = diff[:, :, -min(32, diff.shape[2]):, :]
    print(f"tail_mean_abs={float(tail.mean()):.6f} tail_max_abs={float(tail.max()):.6f}")
    if diff.shape[2] > tail.shape[2]:
      body = diff[:, :, :-tail.shape[2], :]
      print(f"body_mean_abs={float(body.mean()):.6f} body_max_abs={float(body.max()):.6f}")

  head_mean = diff.mean(axis=(0, 2, 3))
  worst_heads = np.argsort(head_mean)[::-1][:min(8, len(head_mean))]
  print("worst_heads=", [(int(i), float(head_mean[i])) for i in worst_heads])

  if diff.shape[1] % h_kv == 0:
    group_size = diff.shape[1] // h_kv
    group_mean = head_mean.reshape(h_kv, group_size).mean(axis=1)
    print("kv_group_mean_abs=", [float(x) for x in group_mean])


def main():
  dev = os.getenv("DEV", Device.DEFAULT)
  impl = os.getenv("FA_IMPL", "auto")
  batch = getenv("B", 1)
  seqlen = getenv("N", 256)
  n_heads = getenv("H", 64)
  n_kv_heads = getenv("H_KV", 8)
  head_dim = getenv("D", 128)
  causal = bool(getenv("CAUSAL", 1))
  expand_gqa = bool(getenv("EXPAND_GQA", 0))
  shard_devices = getenv("SHARD_DEVICES", 0)
  shard_axis = getenv("SHARD_AXIS", -1)
  seed = getenv("SEED", 0)
  mean_tol = float(os.getenv("MEAN_TOL", "0"))
  max_tol = float(os.getenv("MAX_TOL", "0"))

  if seed: Tensor.manual_seed(seed)
  if n_heads % n_kv_heads != 0:
    raise ValueError(f"H must be divisible by H_KV, got {n_heads=} {n_kv_heads=}")

  flash_attention = load_flash_attention(dev, impl)
  print(f"device={dev} impl={impl} B={batch} N={seqlen} H={n_heads} H_KV={n_kv_heads} D={head_dim} causal={int(causal)} "
        f"expand_gqa={int(expand_gqa)} shard_devices={shard_devices} shard_axis={shard_axis}")

  q = Tensor.randn(batch, seqlen, n_heads, head_dim, device=dev, dtype=dtypes.bfloat16).contiguous()
  k = Tensor.randn(batch, seqlen, n_kv_heads, head_dim, device=dev, dtype=dtypes.bfloat16).contiguous()
  v = Tensor.randn(batch, seqlen, n_kv_heads, head_dim, device=dev, dtype=dtypes.bfloat16).contiguous()
  if shard_devices > 1:
    devices = tuple(f"{dev}:{i}" for i in range(shard_devices))
    q = q.shard(devices, axis=shard_axis)
    k = k.shard(devices, axis=shard_axis)
    v = v.shard(devices, axis=shard_axis)
  Tensor.realize(q, k, v)

  qf, kf, vf = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
  fa_k, fa_v = kf, vf
  if expand_gqa and n_heads != n_kv_heads:
    group_size = n_heads // n_kv_heads
    fa_k = kf.repeat_interleave(group_size, dim=1)
    fa_v = vf.repeat_interleave(group_size, dim=1)

  out = flash_attention(qf, fa_k, fa_v, is_causal=causal)
  if isinstance(out, tuple): out = out[0]
  ref = qf.scaled_dot_product_attention(kf, vf, is_causal=causal, enable_gqa=(n_heads != n_kv_heads))

  out_np = out.float().numpy()
  ref_np = ref.float().numpy()
  diff = np.abs(out_np - ref_np)
  summarize_diff(diff, n_kv_heads)

  if mean_tol and float(diff.mean()) > mean_tol:
    raise SystemExit(f"mean_abs {float(diff.mean()):.6f} > MEAN_TOL {mean_tol:.6f}")
  if max_tol and float(diff.max()) > max_tol:
    raise SystemExit(f"max_abs {float(diff.max()):.6f} > MAX_TOL {max_tol:.6f}")


if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    print(f"{type(e).__name__}: {e}", file=sys.stderr)
    raise
