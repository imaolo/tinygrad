#!/usr/bin/env python3
import numpy as np

from extra.thunder.tiny.tk.group import _CUDA_B0_SCALARS, _CUDA_B1_SCALARS
from extra.thunder.tiny.tk.tiles import TileLayout
from tinygrad.runtime.ops_python import generic_wmma_helper

WARP_THREADS = 32


def rt_coords(layout: TileLayout, lane: int, inner: int) -> tuple[int, int]:
  if layout == TileLayout.ROW:
    row = 8 * ((inner // 2) % 2) + lane // 4
    col = 2 * (lane % 4) + (inner % 2) + 8 * (inner // 4)
  else:
    t = (inner % 2) + 2 * (inner // 4)
    row = 2 * (lane % 4) + (t % 2) + 8 * (t // 2)
    col = 8 * ((inner // 2) % 2) + lane // 4
  return row, col


def pack_rt(tile: np.ndarray, layout: TileLayout) -> list[list[float]]:
  regs = [[0.0 for _ in range(WARP_THREADS)] for _ in range(8)]
  for inner in range(8):
    for lane in range(WARP_THREADS):
      row, col = rt_coords(layout, lane, inner)
      regs[inner][lane] = float(tile[row, col])
  return regs


def unpack_rt(regs: list[list[float]], layout: TileLayout) -> np.ndarray:
  out = np.empty((16, 16), dtype=np.float32)
  seen = np.zeros((16, 16), dtype=np.int32)
  for inner in range(8):
    for lane in range(WARP_THREADS):
      row, col = rt_coords(layout, lane, inner)
      out[row, col] = regs[inner][lane]
      seen[row, col] += 1
  assert seen.min() == 1 and seen.max() == 1, seen
  return out


def cuda_a_elem(x: list[list[float]], k: int, row: int, goff: int) -> float:
  return x[k % 2 + (row // 8) * 2 + (k // 8) * 4][goff + (k // 2) % 4 + (row % 8) * 4]


def cuda_b_elem(x: list[list[float]], col: int, k: int, goff: int) -> float:
  return x[k % 2 + (k // 8) * 2][goff + (k // 2) % 4 + col * 4]


def cuda_c_map(lane: int, elem: int) -> tuple[int, int]:
  return elem % 2 + (lane % 4) * 2, lane // 4 + (elem // 2) * 8


def reconstruct_a(a_regs: list[list[float]]) -> np.ndarray:
  out = np.empty((16, 16), dtype=np.float32)
  for row in range(16):
    for k in range(16):
      out[row, k] = cuda_a_elem(a_regs, k, row, 0)
  return out


def reconstruct_b(b_regs: list[list[float]], perm: tuple[int, ...]) -> np.ndarray:
  out = np.empty((16, 16), dtype=np.float32)
  left = [b_regs[idx] for idx in perm[:4]]
  right = [b_regs[idx] for idx in perm[4:]]
  for k in range(16):
    for col in range(8):
      out[k, col] = cuda_b_elem(left, col, k, 0)
      out[k, col + 8] = cuda_b_elem(right, col, k, 0)
  return out


def reconstruct_c(c_regs: list[list[float]], perm: tuple[int, ...]) -> np.ndarray:
  out = np.empty((16, 16), dtype=np.float32)
  left = [c_regs[idx] for idx in perm[:4]]
  right = [c_regs[idx] for idx in perm[4:]]
  for lane in range(WARP_THREADS):
    for elem in range(4):
      col, row = cuda_c_map(lane, elem)
      out[row, col] = left[elem][lane]
      out[row, col + 8] = right[elem][lane]
  return out


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
  return float(np.max(np.abs(a - b)))


def orientation(mat: np.ndarray, ref: np.ndarray) -> str:
  if max_abs_diff(mat, ref) == 0.0:
    return "tile"
  if max_abs_diff(mat, ref.T) == 0.0:
    return "tile.T"
  return f"neither (max_abs={max_abs_diff(mat, ref):.1f})"


def wmma_half(a_regs: list[list[float]], b_regs: list[list[float]], c_regs: list[list[float]]) -> list[list[float]]:
  return generic_wmma_helper((a_regs, b_regs, c_regs), WARP_THREADS, WARP_THREADS, 16, 8, 4, 4, cuda_a_elem, cuda_b_elem, cuda_c_map)


def wmma_full_current(a_regs: list[list[float]], b_regs: list[list[float]], c_regs: list[list[float]]) -> list[list[float]]:
  out0 = wmma_half(a_regs, [b_regs[idx] for idx in _CUDA_B0_SCALARS], c_regs[:4])
  out1 = wmma_half(a_regs, [b_regs[idx] for idx in _CUDA_B1_SCALARS], c_regs[4:])
  return out0 + out1


def wmma_full_with_row_accumulator(a_regs: list[list[float]], b_regs: list[list[float]], c_regs: list[list[float]], c_layout: TileLayout) -> list[list[float]]:
  if c_layout == TileLayout.ROW:
    return wmma_full_current(a_regs, b_regs, c_regs)
  logical_c = unpack_rt(c_regs, TileLayout.COL)
  row_c = pack_rt(logical_c, TileLayout.ROW)
  row_out = wmma_full_current(a_regs, b_regs, row_c)
  logical_out = unpack_rt(row_out, TileLayout.ROW)
  return pack_rt(logical_out, TileLayout.COL)


def report_role(layout: TileLayout) -> None:
  tile = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
  regs = pack_rt(tile, layout)

  print(f"A {layout.name}: {orientation(reconstruct_a(regs), tile)}")
  print(f"B {layout.name}: {orientation(reconstruct_b(regs, _CUDA_B0_SCALARS + _CUDA_B1_SCALARS), tile)}")
  print(f"C {layout.name}: {orientation(reconstruct_c(regs, tuple(range(8))), tile)}")


def report_ops() -> None:
  a = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
  b = (1000 + np.arange(16 * 16, dtype=np.float32)).reshape(16, 16)
  c = (2000 + np.arange(16 * 16, dtype=np.float32)).reshape(16, 16)
  cases = (
    ("mma_AB", TileLayout.ROW, TileLayout.COL, lambda x, y: x @ y),
    ("mma_ABt", TileLayout.ROW, TileLayout.ROW, lambda x, y: x @ y.T),
    ("mma_AtB", TileLayout.COL, TileLayout.COL, lambda x, y: x.T @ y),
    ("mma_AtBt", TileLayout.COL, TileLayout.ROW, lambda x, y: x.T @ y.T),
  )
  print("End-to-end raw _cuda_wmma helper:")
  for name, a_layout, b_layout, ref_fn in cases:
    a_regs = pack_rt(a, a_layout)
    b_regs = pack_rt(b, b_layout)
    ref = ref_fn(a, b) + c
    for c_layout in (TileLayout.ROW, TileLayout.COL):
      c_regs = pack_rt(c, c_layout)
      out = unpack_rt(wmma_full_current(a_regs, b_regs, c_regs), c_layout)
      print(f"{name} C={c_layout.name}: max_abs={max_abs_diff(out, ref):.1f}")
  print("End-to-end with row-accumulator fallback:")
  for name, a_layout, b_layout, ref_fn in cases:
    a_regs = pack_rt(a, a_layout)
    b_regs = pack_rt(b, b_layout)
    ref = ref_fn(a, b) + c
    for c_layout in (TileLayout.ROW, TileLayout.COL):
      c_regs = pack_rt(c, c_layout)
      out = unpack_rt(wmma_full_with_row_accumulator(a_regs, b_regs, c_regs, c_layout), c_layout)
      print(f"{name} C={c_layout.name}: max_abs={max_abs_diff(out, ref):.1f}")


def main() -> None:
  print("CUDA WMMA fragment reconstruction")
  print("This checks the current CUDA RT packing and _cuda_wmma register slicing against the Python WMMA emulator.")
  for layout in (TileLayout.ROW, TileLayout.COL):
    report_role(layout)
  report_ops()


if __name__ == "__main__":
  main()
