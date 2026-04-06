import math, os
if __name__ == "__main__":
  os.environ["DEFAULT_FLOAT"] = "bfloat16"
  os.environ["OPTIM_DTYPE"] = "bfloat16"
  if "DEV" not in os.environ: os.environ["DEV"] = "NULL"
  # CDNA
  os.environ["EMULATE"] = "AMD_CDNA4"
  os.environ["DEVICE_IN_FUNCTION_BUG"] = "1"
  os.environ["ALL2ALL"] = "1"
  os.environ["USE_ATOMICS"] = "1"
  if "HK_FLASH_ATTENTION" not in os.environ:
    os.environ["HK_FLASH_ATTENTION"] = "1"
    if "ASM_GEMM" not in os.environ:
      os.environ["ASM_GEMM"] = "1"
from tinygrad import Tensor, nn, function, getenv, dtypes, TinyJit
from tinygrad.helpers import Timing, colored, GlobalCounters, profile_marker
from tinygrad.uop.ops import Ops, UOp
from extra.models.llama import apply_rotary_emb, precompute_freqs_cis

QUANTIZE_LOADED_WEIGHTS = getenv("QUANTIZE_LOADED_WEIGHTS", 0)
FP8 = getenv("FP8", QUANTIZE_LOADED_WEIGHTS)

FP8_DTYPE = dtypes.fp8e4m3
FP8_MAX = 448.0

def quantize_fp8(x: Tensor):
  scale = FP8_MAX / (x.abs().max(axis=-1, keepdim=True).detach() + 1e-8)
  xq = (x * scale).detach().clamp(-FP8_MAX, FP8_MAX).cast(FP8_DTYPE)
  return xq, scale.float().reciprocal()

def quantize_weight_fp8(w:Tensor):
  scale = FP8_MAX / (w.abs().max(axis=-1, keepdim=True).detach() + 1e-8)
  w_scaled = w * scale
  w_fp8 = w_scaled.detach().clamp(-FP8_MAX, FP8_MAX).cast(FP8_DTYPE)
  return w_fp8, scale.float().reciprocal().reshape(w.shape[:-1])

def matmul(x: Tensor, w: Tensor, w_scale: Tensor|None=None) -> Tensor:
  if not FP8 or w_scale is None or w.dtype != FP8_DTYPE:
    return x @ w.T
  return x.dot(w.T, dtype=dtypes.float) * w_scale

def matmul_lora(x:Tensor, w:Tensor) -> Tensor:
  if not FP8: return x @ w.T
  return x.float().dot(w.T.float(), dtype=dtypes.float)

def rmsnorm(x_in:Tensor, eps:float):
  x = x_in.float()
  x = x * (x.square().mean(-1, keepdim=True) + eps).rsqrt()
  return x.cast(x_in.dtype)

class FlatTransformer:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_layers:int, norm_eps:float, vocab_size:int, n_kv_heads:int|None=None, rope_theta:int=10000,
               max_context:int=1024, lora_rank:int=16, lora_alpha:float=32.0, lora_dropout:float=0.1, use_lora:bool=False, fuse_wqkv:bool=True):
    self.vocab_size = vocab_size
    self.n_layers = n_layers
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads # n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
    self.head_dim = dim // n_heads
    self.n_rep = self.n_heads // self.n_kv_heads
    self.lora_scale = lora_alpha / lora_rank
    self.lora_dropout = lora_dropout
    self.fuse_wqkv = fuse_wqkv
    self.use_lora = use_lora
    # self.quantizeable_weight_names = (("wqkv",) if self.fuse_wqkv else ("wq", "wk", "wv")) + ("wo", "w1", "w2", "w3")
    self.quantizeable_weight_names = ("w1", "w2", "w3")
    scaled_std = 0.02 / math.sqrt(2 * n_layers)

    # Attention
    if self.fuse_wqkv:
      self.wqkv = self.lin_per_layer(dim, wqkv_dim:=(self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim * 2))
    else:
      self.wq = self.lin_per_layer(dim, self.n_heads * self.head_dim)
      self.wk = self.lin_per_layer(dim, self.n_kv_heads * self.head_dim)
      self.wv = self.lin_per_layer(dim, self.n_kv_heads * self.head_dim)
    self.wo = self.lin_per_layer(wo_dim:=(self.n_heads * self.head_dim), dim, std=scaled_std)

    # LoRA
    if not self.fuse_wqkv: assert not self.use_lora, "LoRA requires fused qkv"
    if self.use_lora:
      self.lora_a, self.lora_b = self.create_lora_params(dim, wqkv_dim, lora_rank)
      self.lora_a_wo, self.lora_b_wo = self.create_lora_params(dim, wo_dim, lora_rank)

    # FeedForward
    self.w1 = self.lin_per_layer(dim, hidden_dim)
    self.w2 = self.lin_per_layer(hidden_dim, dim, std=scaled_std)
    self.w3 = self.lin_per_layer(dim, hidden_dim)

    self.norm_eps = norm_eps
    self.attention_norm = Tensor.ones(n_layers, dim).contiguous()
    self.ffn_norm = Tensor.ones(n_layers, dim).contiguous()

    # output
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.tok_embeddings.weight = Tensor.normal(vocab_size, dim, mean=0.0, std=0.02)
    self.output = Tensor.normal(1, vocab_size, dim, mean=0.0, std=0.02)
    self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_context * 2, rope_theta).contiguous().requires_grad_(False)

    if getenv("MIXED_PRECISION", 0):
      self.attention_norm = self.attention_norm.cast('float32')
      self.ffn_norm = self.ffn_norm.cast('float32')
      self.norm.weight = self.norm.weight.cast('float32')

  def quantize_base_weights(self):
    for name in self.quantizeable_weight_names:
      weight: Tensor = getattr(self, name)
      assert weight.dtype is dtypes.bfloat16

      weight_fp8, scale_fp8 = quantize_weight_fp8(weight)
      weight.replace(weight_fp8)
      setattr(self, name+'_scale', scale_fp8)

  def lin_per_layer(self, in_features:int, out_features:int, zerod:bool=False, std:float=0.02, use_kaiming:bool=False, **kwargs):
    dt = FP8_DTYPE if FP8 and not QUANTIZE_LOADED_WEIGHTS and 'dtype' not in kwargs else kwargs.pop('dtype', None)
    if zerod or getenv("ZEROS"): return Tensor.zeros(self.n_layers, out_features, in_features, dtype=dt, **kwargs)
    if use_kaiming:
      return Tensor.kaiming_uniform(self.n_layers, out_features, in_features, a=math.sqrt(5), dtype=dt, **kwargs)
    else:
      return Tensor.normal(self.n_layers, out_features, in_features, mean=0.0, std=std, dtype=dt, **kwargs)

  def create_lora_params(self, in_dim:int, out_dim:float, rank:int) -> tuple[Tensor, Tensor]:
    kwargs = {'dtype': LORA_DTYPE} if (LORA_DTYPE:=getenv('LORA_DTYPE', '')) else {}
    a = self.lin_per_layer(in_dim, rank, requires_grad=True, use_kaiming=True, **kwargs)
    b = self.lin_per_layer(rank, out_dim, zerod=True, requires_grad=True, **kwargs)
    return a, b

  def run_lora(self, lora_a: Tensor, lora_b: Tensor, x: Tensor) -> Tensor:
    out = matmul_lora(x.dropout(self.lora_dropout), lora_a)
    out = matmul_lora(out, lora_b)
    return out * self.lora_scale
    
  def attention(self, x:Tensor, freqs_cis:Tensor, attention_norm:Tensor,
                wo:Tensor, wqkv:Tensor|None=None, wq:Tensor|None=None, wk:Tensor|None=None, wv:Tensor|None=None,            # weights
                wqkv_scale:Tensor|None=None, wq_scale:Tensor|None=None,                                                     # quantize scales
                wk_scale:Tensor|None=None, wv_scale:Tensor|None=None, wo_scale:Tensor|None=None,
                lora_a:Tensor|None=None, lora_b: Tensor|None=None, lora_a_wo:Tensor|None=None, lora_b_wo:Tensor|None=None): # lora weights
    x = rmsnorm(x, self.norm_eps) * attention_norm
    bsz, seqlen, _ = x.shape

    if wqkv is not None:
      xqkv = matmul(x, wqkv, wqkv_scale)
      if self.use_lora:
        xqkv = xqkv + self.run_lora(lora_a, lora_b, x)
      xqkv = xqkv.reshape(bsz, seqlen, self.n_kv_heads, self.n_rep + 2, self.head_dim)
      xq = xqkv[:, :, :, :self.n_rep].reshape(bsz, seqlen, self.n_heads, self.head_dim)
      xk = xqkv[:, :, :, self.n_rep].reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
      xv = xqkv[:, :, :, self.n_rep+1].reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
    else:
      assert wq is not None and wk is not None and wv is not None
      xq = matmul(x, wq, wq_scale).reshape(bsz, seqlen, self.n_heads, self.head_dim)
      xk = matmul(x, wk, wk_scale).reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
      xv = matmul(x, wv, wv_scale).reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
    attn = xq.scaled_dot_product_attention(xk, xv, is_causal=True, enable_gqa=True).transpose(1, 2)
    attn = attn.reshape(bsz, seqlen, -1)
    out = matmul(attn, wo, wo_scale)
    if self.use_lora:
      out = out + self.run_lora(lora_a_wo, lora_b_wo, attn)
    return out

  def feed_forward(self, x:Tensor, ffn_norm:Tensor,
                   w1:Tensor, w2:Tensor, w3:Tensor,                                   # weights
                   w1_scale:Tensor|None, w2_scale:Tensor|None, w3_scale:Tensor|None): # quantize scales
    x = rmsnorm(x, self.norm_eps) * ffn_norm
    x_w1 = matmul(x, w1, w1_scale).silu()
    x_w3 = matmul(x.contiguous_backward(), w3, w3_scale)
    return matmul(x_w1 * x_w3, w2, w2_scale)

  @function(precompile=True, precompile_backward=True)
  def run_layer(self, x:Tensor, freqs_cis:Tensor, attention_norm:Tensor, wo:Tensor, ffn_norm:Tensor,
                w1:Tensor, w2:Tensor, w3:Tensor,                                                                              # weights
                wqkv:Tensor|None=None, wq:Tensor|None=None, wk:Tensor|None=None, wv:Tensor|None=None,
                wo_scale:Tensor|None=None, w1_scale:Tensor|None=None, w2_scale:Tensor|None=None, w3_scale:Tensor|None=None,   # quantize scales
                wqkv_scale:Tensor|None=None, wq_scale:Tensor|None=None, wk_scale:Tensor|None=None, wv_scale:Tensor|None=None, 
                lora_a: Tensor|None=None, lora_b:Tensor|None=None, lora_a_wo: Tensor|None=None, lora_b_wo:Tensor|None=None):  # lora params
    h = x + self.attention(x, freqs_cis, attention_norm, wo, wqkv=wqkv, wq=wq, wk=wk, wv=wv,
                           wqkv_scale=wqkv_scale, wo_scale=wo_scale,
                           wq_scale=wq_scale, wk_scale=wk_scale, wv_scale=wv_scale,
                           lora_a=lora_a, lora_b=lora_b, lora_a_wo=lora_a_wo, lora_b_wo=lora_b_wo)
    return h + self.feed_forward(h, ffn_norm, w1, w2, w3, w1_scale, w2_scale, w3_scale)

  def _shard(self, weight_name:str, device:tuple[str, ...], axis:int, scale_axis:int|None=None):
    getattr(self, weight_name).shard_(device, axis=axis).realize()
    if QUANTIZE_LOADED_WEIGHTS:
      getattr(self, weight_name+'_scale').shard_(device, axis=scale_axis).realize()

  def shard(self, device:tuple[str, ...], mp:bool=False, intermediate_fn:str|None=None):
    from tinygrad.nn.state import get_parameters, get_state_dict
    if not mp:
      for v in get_parameters(self): v.shard_(device, axis=None)
    else:
      # flat per-layer weights: axis 0 is n_layers, so shard axes are +1 vs per-layer Transformer
      if self.fuse_wqkv:
        self.wqkv.shard_(device, 1).realize()
        # self._shard('wqkv', device, 1, 1)
      else:
        self._shard('wq', device, 1, 1)
        self._shard('wk', device, 1, 1)
        self._shard('wv', device, 1, 1)
      if self.use_lora:
        self.lora_a.shard_(device, axis=None).realize()
        self.lora_b.shard_(device, axis=1).realize()
        self.lora_a_wo.shard_(device, axis=2).realize()
        self.lora_b_wo.shard_(device, axis=None).realize()
      self.wo.shard_(device, 2).realize()
      # self._shard('wo', device, 2)
      self._shard('w1', device, 1, 1)
      self._shard('w2', device, 2)
      self._shard('w3', device, 1, 1)
      self.attention_norm.shard_(device, axis=None).realize()
      self.ffn_norm.shard_(device, axis=None).realize()
      self.norm.weight.shard_(device, axis=None).realize()
      self.tok_embeddings.weight.shard_(device, axis=0).realize()
      self.output.shard_(device, 1).realize()
      # self._shard('output', device, 1, 1)
      self.freqs_cis.shard_(device, axis=None).realize()

  def __call__(self, tokens:Tensor):
    h = self.tok_embeddings(tokens)
    freqs_cis = self.freqs_cis.cast(h.dtype)[:, :tokens.shape[1], :, :, :]
    for i in range(self.n_layers):
      attn_kwargs = {"wqkv": self.wqkv[i]} if self.fuse_wqkv else {"wq": self.wq[i], "wk": self.wk[i], "wv": self.wv[i]}
      lora_kwargs = {"lora_a":self.lora_a[i], "lora_a_wo":self.lora_a_wo[i], "lora_b":self.lora_b[i], "lora_b_wo":self.lora_b_wo[i]} if self.use_lora else {}
      scale_args = {(scale_name:=(weight_name+'_scale')): getattr(self, scale_name)[i] for weight_name in set(self.quantizeable_weight_names)-set(['output'])} if FP8 else {}
      h = self.run_layer(h, freqs_cis, self.attention_norm[i], self.wo[i], self.ffn_norm[i], self.w1[i], self.w2[i], self.w3[i],
                         **attn_kwargs, **lora_kwargs, **scale_args)
    logits = matmul(self.norm(h), self.output[0], None)
    return logits

def _get_pads(uop:UOp) -> list[UOp]:
  if uop.op == Ops.ADD: return _get_pads(uop.src[0]) + _get_pads(uop.src[1])
  return [uop]

def apply_grad(grad_buf:Tensor, new_grad:UOp):
  pads = _get_pads(new_grad)
  if len(pads) <= 1:
    store = grad_buf.uop.store(grad_buf.uop + new_grad)
    grad_buf.uop = grad_buf.uop.after(store)
    return
  sorted_pads = sorted(pads, key=lambda p: p.marg[0][0] if p.op == Ops.PAD else 0)
  inners = [Tensor(p.src[0] if p.op == Ops.PAD else p, device=grad_buf.device) for p in sorted_pads]
  grad_buf.assign(grad_buf + inners[0].cat(*inners[1:], dim=0))

if __name__ == "__main__":
  config = {}
  BS                 = config["BS"]                     = getenv("BS", 16)
  SEQLEN             = config["SEQLEN"]                 = getenv("SEQLEN", 8192)

  from examples.llama3 import MODEL_PARAMS
  model_params = MODEL_PARAMS[getenv("LLAMA3_SIZE", "8B")]["args"]
  if (llama_layers:=getenv("LLAMA_LAYERS")) != 0: model_params['n_layers'] = llama_layers
  model = FlatTransformer(**model_params, max_context=SEQLEN)
  state = nn.state.get_state_dict(model)
  print("tensor count:", len(state))

  # shard the model
  from tinygrad import Device
  if (DP := getenv("DP", 1)) > 1:
    model.shard(tuple(f"{Device.DEFAULT}:{i}" for i in range(DP)))
  if (MP := getenv("MP", 1)) > 1:
    model.shard(tuple(f"{Device.DEFAULT}:{i}" for i in range(MP)), mp=True)

  # preallocate all the grad buffers and zero them out
  grads = {x:Tensor.zeros_like(x).contiguous() for x in state.values() if x.requires_grad is None}

  # print model size
  sz = 0
  for k,v in state.items():
    print(f"{colored(k, 'green' if v in grads else 'white'):30s} {str(v.shape):30s} {str(v.dtype):20s} {v.device}  {v.nbytes()/1e9:.2f} GB")
    sz += v.nbytes()
  print(f"total sz: {sz/1e9:.2f} GB")

  with Timing("fake data: "): tokens = Tensor.randint(BS, SEQLEN+1, low=0, high=model.vocab_size, dtype=dtypes.int)
  with Timing("realize weights/grads/data: "): Tensor.realize(*state.values(), *grads.values(), tokens)
  print("mem per device: " + ', '.join(f"{dev}: {mem/1e9:.2f} GB" for dev, mem in sorted(GlobalCounters.mem_used_per_device.items())))
  if DP > 1: tokens = tokens.shard(tuple(f"{Device.DEFAULT}:{i}" for i in range(DP)), axis=0)
  if MP > 1: tokens = tokens.shard(tuple(f"{Device.DEFAULT}:{i}" for i in range(MP)))

  @TinyJit
  def jit_step(tokens:Tensor):
    with Timing("python forward: "): loss = model(tokens[:, :-1]).sparse_categorical_crossentropy(tokens[:, 1:])
    with Timing("python backward: "):
      for t,g in zip(grads, loss.gradient(*grads)):
        apply_grad(grads[t], g.uop)
    with Timing("run step: "): loss.realize(*grads.values())

  for i in range(6):
    GlobalCounters.reset()
    profile_marker(f"step {i}")
    with Timing(colored(f"*** step {i}: ", "red")):
      jit_step(tokens)
  print("mem per device: " + ', '.join(f"{dev}: {mem/1e9:.2f} GB" for dev, mem in sorted(GlobalCounters.mem_used_per_device.items())))
