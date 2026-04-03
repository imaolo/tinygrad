import os
os.environ["WQKV"] = "1"
import unittest
import numpy as np
from tinygrad import Tensor, nn, dtypes
from tinygrad.nn.state import get_parameters
from tinygrad.device import is_dtype_supported, Device
from examples.mlperf.models.llama import Transformer
import examples.mlperf.models.flat_llama as flat_llama_mod
from examples.mlperf.models.flat_llama import FlatTransformer

def _copy_base_weight(dst: Tensor, src: Tensor):
  dst.replace(src.cast(dtypes.bfloat16 if flat_llama_mod.FP8 else dst.dtype))

def copy_weights(flat: FlatTransformer, ref: Transformer):
  n_layers = flat.n_layers
  _copy_base_weight(flat.wqkv, Tensor.stack(*[ref.layers[i].attention.wqkv.weight for i in range(n_layers)]))
  _copy_base_weight(flat.wo, Tensor.stack(*[ref.layers[i].attention.wo.weight for i in range(n_layers)]))
  _copy_base_weight(flat.w1, Tensor.stack(*[ref.layers[i].feed_forward.w1.weight for i in range(n_layers)]))
  _copy_base_weight(flat.w2, Tensor.stack(*[ref.layers[i].feed_forward.w2.weight for i in range(n_layers)]))
  _copy_base_weight(flat.w3, Tensor.stack(*[ref.layers[i].feed_forward.w3.weight for i in range(n_layers)]))
  flat.attention_norm.replace(Tensor.stack(*[ref.layers[i].attention_norm.weight.cast(flat.attention_norm.dtype) for i in range(n_layers)]))
  flat.ffn_norm.replace(Tensor.stack(*[ref.layers[i].ffn_norm.weight.cast(flat.ffn_norm.dtype) for i in range(n_layers)]))
  flat.norm.weight.replace(ref.norm.weight.cast(flat.norm.weight.dtype))
  flat.tok_embeddings.weight.replace(ref.tok_embeddings.weight.cast(flat.tok_embeddings.weight.dtype))
  _copy_base_weight(flat.output, ref.output.weight.unsqueeze(0))
  if flat.use_lora and ref.use_lora:
    qkv_lora = [ref.layers[i].attention.lora_map[ref.layers[i].attention.wqkv] for i in range(n_layers)]
    wo_lora = [ref.layers[i].attention.lora_map[ref.layers[i].attention.wo] for i in range(n_layers)]
    flat.lora_a.replace(Tensor.stack(*[lora.lora_a.cast(flat.lora_a.dtype) for lora in qkv_lora]))
    flat.lora_b.replace(Tensor.stack(*[lora.lora_b.cast(flat.lora_b.dtype) for lora in qkv_lora]))
    flat.lora_a_wo.replace(Tensor.stack(*[lora.lora_a.cast(flat.lora_a_wo.dtype) for lora in wo_lora]))
    flat.lora_b_wo.replace(Tensor.stack(*[lora.lora_b.cast(flat.lora_b_wo.dtype) for lora in wo_lora]))
  if flat_llama_mod.FP8:
    flat.quantize_base_weights()

class TestFlatLlama(unittest.TestCase):
  def test_forward_match(self):
    Tensor.manual_seed(42)
    params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
    ref = Transformer(**params)
    flat = FlatTransformer(**params)
    copy_weights(flat, ref)
    Tensor.realize(*nn.state.get_state_dict(flat).values())

    tokens = Tensor([[1, 50, 100, 999, 2]])
    ref_logits = ref(tokens).realize()
    flat_logits = flat(tokens).realize()
    self.assertEqual(ref_logits.shape, flat_logits.shape)
    diff = (ref_logits - flat_logits).abs().max().item()
    self.assertLess(diff, 1e-5, f"forward mismatch: max abs diff {diff}")

  def test_backward_match(self):
    Tensor.manual_seed(42)
    params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
    ref = Transformer(**params)
    flat = FlatTransformer(**params)
    copy_weights(flat, ref)

    for p in get_parameters(ref): p.requires_grad_(True)
    for p in get_parameters(flat): p.requires_grad_(True)
    Tensor.realize(*nn.state.get_state_dict(flat).values())

    tokens = Tensor([[1, 50, 100, 999, 2, 10]])

    ref_loss = ref(tokens[:, :-1]).sparse_categorical_crossentropy(tokens[:, 1:])
    ref_loss.backward()
    ref_grads = {k: v.grad.numpy() for k, v in nn.state.get_state_dict(ref).items() if v.grad is not None}

    flat_loss = flat(tokens[:, :-1]).sparse_categorical_crossentropy(tokens[:, 1:])
    flat_loss.backward()
    flat_grads = {k: v.grad.numpy() for k, v in nn.state.get_state_dict(flat).items() if v.grad is not None}

    # check loss matches
    self.assertAlmostEqual(ref_loss.item(), flat_loss.item(), places=4)

    # check output weight grad matches
    diff = abs(ref_grads["output.weight"] - flat_grads.get("output.weight", flat_grads["output"])).max()
    self.assertLess(diff, 1e-4, f"output.weight grad mismatch: max abs diff {diff}")

    # check per-layer weight grads match
    for i in range(params["n_layers"]):
      for flat_key, ref_key in [
        ("wqkv", f"layers.{i}.attention.wqkv.weight"),
        ("wo", f"layers.{i}.attention.wo.weight"),
        ("w1", f"layers.{i}.feed_forward.w1.weight"),
        ("w2", f"layers.{i}.feed_forward.w2.weight"),
        ("w3", f"layers.{i}.feed_forward.w3.weight"),
      ]:
        diff = abs(ref_grads[ref_key] - flat_grads[flat_key][i]).max()
        self.assertLess(diff, 1e-4, f"layer {i} {flat_key} grad mismatch: max abs diff {diff}")

  @unittest.skipUnless(Device.DEFAULT == "CPU", "multi-device CPU test")
  def test_forward_match_mp(self):
    Tensor.manual_seed(42)
    params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
    from tinygrad import Device
    devices = (f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1")
    ref = Transformer(**params)
    flat = FlatTransformer(**params)
    copy_weights(flat, ref)
    Tensor.realize(*nn.state.get_state_dict(flat).values())
    flat.shard(devices, mp=True)

    tokens = Tensor([[1, 50, 100, 999, 2]], device=devices[0])
    ref_logits = ref(tokens.to(devices[0])).numpy()
    flat_logits = flat(tokens.shard(devices)).numpy()
    self.assertEqual(ref_logits.shape, flat_logits.shape)
    np.testing.assert_allclose(flat_logits, ref_logits, atol=1e-4, rtol=1e-4)

  @unittest.skipUnless(Device.DEFAULT == "CPU", "multi-device CPU test")
  def test_forward_match_dp(self):
    Tensor.manual_seed(42)
    params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
    from tinygrad import Device
    devices = (f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1")
    ref = Transformer(**params)
    flat = FlatTransformer(**params)
    copy_weights(flat, ref)
    Tensor.realize(*nn.state.get_state_dict(flat).values())
    flat.shard(devices)

    tokens = Tensor([[1, 50, 100, 999, 2], [2, 100, 50, 1, 999]], device=devices[0])
    ref_logits = ref(tokens.to(devices[0])).numpy()
    flat_logits = flat(tokens.shard(devices, axis=0)).numpy()
    self.assertEqual(ref_logits.shape, flat_logits.shape)
    np.testing.assert_allclose(flat_logits, ref_logits, atol=1e-4, rtol=1e-4)

  @unittest.skipUnless(is_dtype_supported(dtypes.fp8e4m3), "fp8 not supported on this device")
  def test_forward_fp8(self):
    old_fp8 = flat_llama_mod.FP8
    try:
      flat_llama_mod.FP8 = 1
      Tensor.manual_seed(42)
      params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
      ref = Transformer(**params)
      flat = FlatTransformer(**params)
      copy_weights(flat, ref)
      Tensor.realize(*nn.state.get_state_dict(flat).values())

      tokens = Tensor([[1, 50, 100, 999, 2]])
      ref_logits = ref(tokens).numpy()
      flat_logits = flat(tokens).numpy()
      self.assertEqual(ref_logits.shape, flat_logits.shape)
      # FP8 has lower precision, allow larger tolerance
      np.testing.assert_allclose(flat_logits, ref_logits, atol=1.0, rtol=0.1)
    finally:
      flat_llama_mod.FP8 = old_fp8

  @unittest.skipUnless(is_dtype_supported(dtypes.fp8e4m3), "fp8 not supported on this device")
  def test_forward_fp8_lora_keeps_lora_fp32(self):
    old_fp8, old_lora_dtype = flat_llama_mod.FP8, os.environ.get("LORA_DTYPE")
    try:
      flat_llama_mod.FP8 = 1
      os.environ["LORA_DTYPE"] = "float32"
      Tensor.manual_seed(42)
      params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
      ref = Transformer(**params, use_lora=True)
      flat = FlatTransformer(**params, use_lora=True)
      copy_weights(flat, ref)
      Tensor.realize(*nn.state.get_state_dict(flat).values())

      self.assertEqual(flat.lora_a.dtype, dtypes.float32)
      self.assertEqual(flat.lora_b.dtype, dtypes.float32)
      self.assertEqual(flat.lora_a_wo.dtype, dtypes.float32)
      self.assertEqual(flat.lora_b_wo.dtype, dtypes.float32)
      self.assertEqual(flat.wqkv.dtype, dtypes.fp8e4m3)

      tokens = Tensor([[1, 50, 100, 999, 2]])
      ref_logits = ref(tokens).numpy()
      flat_logits = flat(tokens).numpy()
      self.assertEqual(ref_logits.shape, flat_logits.shape)
      np.testing.assert_allclose(flat_logits, ref_logits, atol=1.0, rtol=0.1)
    finally:
      flat_llama_mod.FP8 = old_fp8
      if old_lora_dtype is None: os.environ.pop("LORA_DTYPE", None)
      else: os.environ["LORA_DTYPE"] = old_lora_dtype

if __name__ == "__main__":
  unittest.main()
