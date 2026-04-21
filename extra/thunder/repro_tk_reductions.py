import os
import numpy as np

from tinygrad import Tensor, dtypes
from extra.thunder.tiny.tk.kernel import Kernel
from extra.thunder.tiny.tk.tiles import GL, TileLayout

DEV = os.getenv("DEV", "CUDA")
MODE = os.getenv("MODE", "both")
M = int(os.getenv("M", "16"))
N = int(os.getenv("N", "16"))
SEED = int(os.getenv("SEED", "1"))

assert M % 16 == 0 and N % 16 == 0, "M and N must be multiples of 16"

def run_col_reduce(arr:np.ndarray):
  inp = Tensor(arr.tolist(), device=DEV, dtype=dtypes.float32).contiguous()
  out = Tensor.empty(1, 1, 1, N, device=DEV, dtype=dtypes.float32)

  def fxn(outu, inpu):
    with Kernel("col_reduce", (1, 1, 1), 32) as ker:
      outg, ing = GL(outu, ker), GL(inpu, ker)
      rt = ker.rt((M, N), dtypes.float32, TileLayout.COL)
      vec = ker.rv(N, dtypes.float32)
      rt = ker.warp.load(rt, ing, (), (0, 0, 0, 0), axis=1)
      vec = ker.warp.zero(vec)
      vec = ker.warp.col_reduce(vec, rt, lambda a, b: a + b)
      outg = ker.warp.store(outg, vec, (0, 0, 0, 0), (), axis=2)
      return ker.finish()

  return Tensor.custom_kernel(out, inp, fxn=fxn)[0].realize().numpy().reshape(-1)

def run_row_reduce(arr:np.ndarray):
  inp = Tensor(arr.tolist(), device=DEV, dtype=dtypes.float32).contiguous()
  out = Tensor.empty(1, 1, M, 1, device=DEV, dtype=dtypes.float32)

  def fxn(outu, inpu):
    with Kernel("row_reduce", (1, 1, 1), 32) as ker:
      outg, ing = GL(outu, ker), GL(inpu, ker)
      rt = ker.rt((M, N), dtypes.float32, TileLayout.ROW)
      vec = ker.rv(M, dtypes.float32)
      rt = ker.warp.load(rt, ing, (), (0, 0, 0, 0), axis=1)
      vec = ker.warp.zero(vec)
      vec = ker.warp.row_reduce(vec, rt, lambda a, b: a + b)
      outg = ker.warp.store(outg, vec, (0, 0, 0, 0), (), axis=1)
      return ker.finish()

  return Tensor.custom_kernel(out, inp, fxn=fxn)[0].realize().numpy().reshape(-1)

def main():
  rng = np.random.default_rng(SEED)
  arr = rng.standard_normal((1, M, 1, N), dtype=np.float32)

  print(f"device={DEV} mode={MODE} M={M} N={N} seed={SEED}")

  if MODE in ("col", "both"):
    got = run_col_reduce(arr)
    exp = arr.sum(axis=1).reshape(-1)
    diff = np.abs(got - exp)
    print(f"col mean_abs={float(diff.mean()):.6f} max_abs={float(diff.max()):.6f}")
    print("col got=", got.tolist())
    print("col exp=", exp.tolist())

  if MODE in ("row", "both"):
    got = run_row_reduce(arr)
    exp = arr.sum(axis=3).reshape(-1)
    diff = np.abs(got - exp)
    print(f"row mean_abs={float(diff.mean()):.6f} max_abs={float(diff.max()):.6f}")
    print("row got=", got.tolist())
    print("row exp=", exp.tolist())

if __name__ == "__main__":
  main()
