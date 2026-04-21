import unittest
import numpy as np

from extra.thunder.tiny.tk.tiles import TileLayout

WARP_THREADS = 32
TILE_DIM = 16
ELEMENTS_PER_THREAD = 8


def rt_coords(layout: TileLayout, lane: int, inner: int) -> tuple[int, int]:
  if layout == TileLayout.ROW:
    row = 8 * ((inner // 2) % 2) + lane // 4
    col = 2 * (lane % 4) + (inner % 2) + 8 * (inner // 4)
  else:
    t = (inner % 2) + 2 * (inner // 4)
    row = 2 * (lane % 4) + (t % 2) + 8 * (t // 2)
    col = 8 * ((inner // 2) % 2) + lane // 4
  return row, col


def reduce_op(name: str, a: float, b: float) -> float:
  if name == "sum": return a + b
  if name == "max": return max(a, b)
  raise ValueError(name)


def emulate_row_reduce(tile: np.ndarray, layout: TileLayout, op_name: str, init_value: float) -> np.ndarray:
  assert tile.shape[0] % TILE_DIM == 0 and tile.shape[1] % TILE_DIM == 0
  height_tiles, width_tiles = tile.shape[0] // TILE_DIM, tile.shape[1] // TILE_DIM
  out = np.empty((height_tiles, TILE_DIM), dtype=np.float32)
  lane_elem = np.arange(WARP_THREADS) % TILE_DIM

  for height in range(height_tiles):
    partials = np.full((WARP_THREADS, TILE_DIM), init_value, dtype=np.float32)
    for width in range(width_tiles):
      for lane in range(WARP_THREADS):
        for inner in range(ELEMENTS_PER_THREAD):
          row, col = rt_coords(layout, lane, inner)
          value = float(tile[height * TILE_DIM + row, width * TILE_DIM + col])
          partials[lane, row] = reduce_op(op_name, partials[lane, row], value)
    for lane in range(WARP_THREADS):
      total = init_value
      elem = lane_elem[lane]
      for other in range(WARP_THREADS):
        total = reduce_op(op_name, total, float(partials[other, elem]))
      out[height, elem] = total
  return out


def emulate_col_reduce(tile: np.ndarray, layout: TileLayout, op_name: str, init_value: float) -> np.ndarray:
  assert tile.shape[0] % TILE_DIM == 0 and tile.shape[1] % TILE_DIM == 0
  height_tiles, width_tiles = tile.shape[0] // TILE_DIM, tile.shape[1] // TILE_DIM
  out = np.empty((width_tiles, TILE_DIM), dtype=np.float32)
  lane_elem = np.arange(WARP_THREADS) % TILE_DIM

  for width in range(width_tiles):
    partials = np.full((WARP_THREADS, TILE_DIM), init_value, dtype=np.float32)
    for height in range(height_tiles):
      for lane in range(WARP_THREADS):
        for inner in range(ELEMENTS_PER_THREAD):
          row, col = rt_coords(layout, lane, inner)
          value = float(tile[height * TILE_DIM + row, width * TILE_DIM + col])
          partials[lane, col] = reduce_op(op_name, partials[lane, col], value)
    for lane in range(WARP_THREADS):
      total = init_value
      elem = lane_elem[lane]
      for other in range(WARP_THREADS):
        total = reduce_op(op_name, total, float(partials[other, elem]))
      out[width, elem] = total
  return out


class TestTKReductions(unittest.TestCase):
  def setUp(self):
    self.square = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    self.rect = np.arange(16 * 64, dtype=np.float32).reshape(16, 64)

  def test_row_reduce_matches_numpy(self):
    for layout in (TileLayout.ROW, TileLayout.COL):
      with self.subTest(layout=layout.name, op="sum"):
        actual = emulate_row_reduce(self.square, layout, "sum", 0.0)
        expected = self.square.sum(axis=1, dtype=np.float32).reshape(2, 16)
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)
      with self.subTest(layout=layout.name, op="max"):
        actual = emulate_row_reduce(self.square, layout, "max", -np.inf)
        expected = self.square.max(axis=1).reshape(2, 16)
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)

  def test_col_reduce_matches_numpy(self):
    for layout in (TileLayout.ROW, TileLayout.COL):
      with self.subTest(layout=layout.name, op="sum"):
        actual = emulate_col_reduce(self.rect, layout, "sum", 0.0)
        expected = self.rect.sum(axis=0, dtype=np.float32).reshape(4, 16)
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)
      with self.subTest(layout=layout.name, op="max"):
        actual = emulate_col_reduce(self.rect, layout, "max", -np.inf)
        expected = self.rect.max(axis=0).reshape(4, 16)
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)


if __name__ == "__main__":
  unittest.main()
