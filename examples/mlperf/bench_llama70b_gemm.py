#!/usr/bin/env python3
from __future__ import annotations
import json, os

os.environ["DEVICE_IN_FUNCTION_BUG"] = "1"

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.helpers import Timing

# Names follow extra.gemm.cdna_asm_gemm.custom_uop_gemm:
# uop_gemm_M_N_K computes (M, K) @ (K, N) -> (M, N).
CASES: dict[str, tuple[int, int, int, str]] = {
  "w13_fwd": (8191, 57344, 8192, "x @ w13.T, FFN gate/up projection"),
  "w13_dgrad": (8191, 8192, 57344, "dw13 input-gradient path, grad @ w13"),
  "w2_fwd": (8191, 8192, 28672, "silu(x1)*x3 @ w2.T, FFN down projection"),
  "w2_dgrad": (8191, 28672, 8192, "dw2 input-gradient path, grad @ w2"),
  "wqkv_fwd": (8191, 10240, 8192, "x @ wqkv.T, attention qkv projection"),
  "wqkv_dgrad": (8191, 8192, 10240, "dwqkv input-gradient path, grad @ wqkv"),
  "wo_fwd": (8191, 8192, 8192, "attn @ wo.T, attention output projection"),
}

def run_one(m:int, n:int, k:int, asm:bool) -> float:
  from extra.gemm.cdna_asm_gemm import can_use_asm_gemm, asm_gemm
  a = Tensor.empty(1, m, k, dtype=dtypes.bfloat16).realize()
  w = Tensor.empty(n, k, dtype=dtypes.bfloat16).realize()
  Device[Device.DEFAULT].synchronize()

  if asm:
    assert can_use_asm_gemm(a, w.T)
    matmul_fn = asm_gemm
  else:
    matmul_fn = Tensor.matmul
  
  @TinyJit
  def one_gemm(x:Tensor, weight:Tensor) -> Tensor: return matmul_fn(x, weight.T).realize()

  for _ in range(4): one_gemm(a, w)
  Device[Device.DEFAULT].synchronize()

  tm = Timing(enabled=False)
  with tm:
    for _ in range(iters:=5):
      one_gemm(a, w)
      Device[Device.DEFAULT].synchronize()
  return tm.et / iters

def main() -> None:
  rows = []
  for case, (m,n,k,desc) in CASES.items():
    for asm in [True, False]:
      avg_ns = run_one(m, n, k, asm)
      flops = 2 * m * n * k
      row = {"case":case, "desc":desc, "asm":asm, "M":m, "N":n, "K":k, "avg_wall_ms":avg_ns*1e-6, "avg_tflops":flops/(avg_ns*1e-9)/1e12}
      rows.append(row)
      print(f"{case:12s} {asm} avg_wall={row['avg_wall_ms']:9.3f} ms  {row['avg_tflops']:8.2f} TFLOP/s")

  print("\nJSON")
  print(json.dumps(rows, indent=2))

if __name__ == "__main__":
  main()
