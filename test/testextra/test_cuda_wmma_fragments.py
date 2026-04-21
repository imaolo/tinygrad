import unittest
import numpy as np

from extra.thunder.repro_cuda_wmma_fragments import (
  pack_rt, unpack_rt, wmma_full_current, wmma_full_with_row_accumulator, max_abs_diff, TileLayout,
)


class TestCUDAWMMAFragments(unittest.TestCase):
  def setUp(self):
    self.a = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    self.b = (1000 + np.arange(16 * 16, dtype=np.float32)).reshape(16, 16)
    self.c = (2000 + np.arange(16 * 16, dtype=np.float32)).reshape(16, 16)
    self.cases = (
      ("mma_AB", TileLayout.ROW, TileLayout.COL, lambda x, y: x @ y),
      ("mma_ABt", TileLayout.ROW, TileLayout.ROW, lambda x, y: x @ y.T),
      ("mma_AtB", TileLayout.COL, TileLayout.COL, lambda x, y: x.T @ y),
      ("mma_AtBt", TileLayout.COL, TileLayout.ROW, lambda x, y: x.T @ y.T),
    )

  def test_raw_cuda_wmma_helper_is_bad_for_col_accumulators(self):
    for _, a_layout, b_layout, ref_fn in self.cases:
      a_regs = pack_rt(self.a, a_layout)
      b_regs = pack_rt(self.b, b_layout)
      ref = ref_fn(self.a, self.b) + self.c

      row_regs = pack_rt(self.c, TileLayout.ROW)
      row_out = unpack_rt(wmma_full_current(a_regs, b_regs, row_regs), TileLayout.ROW)
      self.assertEqual(max_abs_diff(row_out, ref), 0.0)

      col_regs = pack_rt(self.c, TileLayout.COL)
      col_out = unpack_rt(wmma_full_current(a_regs, b_regs, col_regs), TileLayout.COL)
      self.assertGreater(max_abs_diff(col_out, ref), 0.0)

  def test_row_accumulator_fallback_fixes_all_cases(self):
    for _, a_layout, b_layout, ref_fn in self.cases:
      a_regs = pack_rt(self.a, a_layout)
      b_regs = pack_rt(self.b, b_layout)
      ref = ref_fn(self.a, self.b) + self.c

      for c_layout in (TileLayout.ROW, TileLayout.COL):
        c_regs = pack_rt(self.c, c_layout)
        out = unpack_rt(wmma_full_with_row_accumulator(a_regs, b_regs, c_regs, c_layout), c_layout)
        self.assertEqual(max_abs_diff(out, ref), 0.0)


if __name__ == "__main__":
  unittest.main()
