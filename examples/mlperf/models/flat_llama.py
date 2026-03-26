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

FP8 = getenv("FP8", 0)

FP8_DTYPE = dtypes.fp8e4m3
FP8_MAX = 448.0

def quantize_fp8(x:Tensor):
  scale = FP8_MAX / (x.abs().max().detach() + 1e-8)
  x_scaled = x * scale
  x_clamped = x_scaled + (x_scaled.detach().clamp(-FP8_MAX, FP8_MAX) - x_scaled.detach())  # STE
  return x_clamped.cast(FP8_DTYPE), scale.float().reciprocal()

def matmul(x:Tensor, w:Tensor) -> Tensor:
  if not FP8: return x @ w.T
  # weights are already FP8, just quantize activations
  x_fp8, x_scale = quantize_fp8(x)
  return x_fp8.dot(w.T, dtype=dtypes.float) * x_scale

def rmsnorm(x_in:Tensor, eps:float):
  x = x_in.float()
  x = x * (x.square().mean(-1, keepdim=True) + eps).rsqrt()
  return x.cast(x_in.dtype)

class FlatTransformer:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_layers:int, norm_eps:float, vocab_size:int, n_kv_heads:int|None=None, rope_theta:int=10000,
               max_context:int=1024, lora_rank:int=16, lora_alpha:float=32.0, lora_dropout:float=0.1, use_lora:bool=False, fuse_wqkv:bool=True):
    assert fuse_wqkv, "non-fused unsupported"
    self.vocab_size = vocab_size
    self.n_layers = n_layers
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads # n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
    self.head_dim = dim // n_heads
    self.n_rep = self.n_heads // self.n_kv_heads
    self.lora_scale = lora_alpha / lora_rank
    self.lora_dropout = lora_dropout
    self.use_lora = use_lora

    # Attention
    self.wqkv = self.lin_per_layer(dim, wqkv_dim:=(self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim * 2))
    self.wo = self.lin_per_layer(wo_dim:=(self.n_heads * self.head_dim), dim)

    # LoRa parameters
    if self.use_lora:
      self.lora_a, self.lora_b = self.create_lora_params(dim, lora_rank, wqkv_dim)
      self.lora_a_wo, self.lora_b_wo = self.create_lora_params(dim, lora_rank, wo_dim)
    else:
      self.lora_a = self.lora_b = self.lora_b_wo = self.lora_a_wo = [None] * self.n_layers

    # FeedForward
    self.w1 = self.lin_per_layer(dim, hidden_dim)
    self.w2 = self.lin_per_layer(hidden_dim, dim)
    self.w3 = self.lin_per_layer(dim, hidden_dim)

    self.norm_eps = norm_eps
    self.attention_norm = Tensor.ones(n_layers, dim).contiguous()
    self.ffn_norm = Tensor.ones(n_layers, dim).contiguous()

    # output
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_context * 2, rope_theta).contiguous().requires_grad_(False)

  def lin_per_layer(self, in_features:int, out_features:int, zerod:bool=False, **kwargs):
    bound = 1 / math.sqrt(in_features)
    dt = FP8_DTYPE if FP8 else None
    if getenv("ZEROS"): return Tensor.zeros(self.n_layers, out_features, in_features, dtype=dt)
    if zerod:
      return Tensor.zeros(self.n_layers, out_features, in_features, dtype=dt, **kwargs)
    else:
      return Tensor.uniform(self.n_layers, out_features, in_features, low=-bound, high=bound, dtype=dt, **kwargs)

  def create_lora_params(self, dim:int, rank:float, out:float) -> tuple[Tensor, Tensor]:
    a = self.lin_per_layer(dim, rank, requires_grad=True)
    b = self.lin_per_layer(rank, out, zerod=True, requires_grad=True)
    return a, b

  def attention(self, x:Tensor, freqs_cis:Tensor, attention_norm:Tensor, wqkv:Tensor, wo:Tensor,
                lora_a:Tensor|None, lora_b: Tensor|None, lora_a_wo:Tensor|None, lora_b_wo:Tensor|None):
    x = rmsnorm(x, self.norm_eps) * attention_norm
    xqkv = matmul(x, wqkv)

    if lora_a is not None and lora_b is not None:
      lora = matmul(x.dropout(self.lora_dropout), lora_a)
      lora = matmul(lora, lora_b)
      xqkv = xqkv + (lora * self.lora_scale)

    bsz, seqlen, _ = xqkv.shape
    # interleaved layout: each kv group has [n_rep q heads, 1 k head, 1 v head] for clean MP sharding
    xqkv = xqkv.reshape(bsz, seqlen, self.n_kv_heads, self.n_rep + 2, self.head_dim)
    xq = xqkv[:, :, :, :self.n_rep].reshape(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xqkv[:, :, :, self.n_rep].reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xv = xqkv[:, :, :, self.n_rep+1].reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
    attn = xq.scaled_dot_product_attention(xk, xv, is_causal=True, enable_gqa=True).transpose(1, 2)
    attn = attn.reshape(bsz, seqlen, -1)
    out = matmul(attn, wo)
    if lora_a_wo is not None and lora_b_wo is not None:
      lora = matmul(attn.dropout(self.lora_dropout), lora_a_wo)
      lora = matmul(lora, lora_b_wo)
      out = out + (lora * self.lora_scale)
    return out

  def feed_forward(self, x:Tensor, ffn_norm:Tensor, w1:Tensor, w2:Tensor, w3:Tensor):
    x = rmsnorm(x, self.norm_eps) * ffn_norm
    x_w1 = matmul(x, w1).silu()
    x_w3 = matmul(x.contiguous_backward(), w3)
    return matmul(x_w1 * x_w3, w2)

  @function(precompile=True, precompile_backward=True)
  def run_layer(self, x:Tensor, freqs_cis:Tensor,
                attention_norm:Tensor, wqkv:Tensor, wo:Tensor,
                ffn_norm:Tensor, w1:Tensor, w2:Tensor, w3:Tensor,
                lora_a: Tensor|None, lora_b:Tensor|None,
                lora_a_wo: Tensor|None, lora_b_wo:Tensor|None):
    h = x + self.attention(x, freqs_cis, attention_norm, wqkv, wo, lora_a, lora_b, lora_a_wo, lora_b_wo)
    return h + self.feed_forward(h, ffn_norm, w1, w2, w3)

  def shard(self, device:tuple[str, ...], mp:bool=False, intermediate_fn:str|None=None):
    from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, load_state_dict, safe_load
    if not mp:
      for v in get_parameters(self): v.shard_(device, axis=None)
    else:
      if intermediate_fn is not None:
        # materialize each to CPU
        for param in get_parameters(self): param.to('CPU').realize()

        # store to disk
        safe_save(get_state_dict(self), intermediate_fn)

        # load back from disk
        load_state_dict(self, safe_load(intermediate_fn))

      # flat per-layer weights: axis 0 is n_layers, so shard axes are +1 vs per-layer Transformer
      self.wqkv.shard_(device, axis=1)          # (n_layers, out, dim) shard out
      self.wo.shard_(device, axis=2)             # (n_layers, dim, in) shard in
      if self.use_lora:
        self.lora_a.shard_(device, axis=None)
        self.lora_b.shard_(device, axis=1)
        self.lora_a_wo.shard_(device, axis=2)
        self.lora_b_wo.shard_(device, axis=None)
      self.w1.shard_(device, axis=1)             # (n_layers, hidden, dim) shard out
      self.w2.shard_(device, axis=2)             # (n_layers, dim, hidden) shard in
      self.w3.shard_(device, axis=1)             # (n_layers, hidden, dim) shard out
      self.attention_norm.shard_(device, axis=None)
      self.ffn_norm.shard_(device, axis=None)
      self.norm.weight.shard_(device, axis=None)
      self.tok_embeddings.weight.shard_(device, axis=0)
      self.output.weight.shard_(device, axis=0)
      self.freqs_cis.shard_(device, axis=None)

  def __call__(self, tokens:Tensor):
    h = self.tok_embeddings(tokens)
    freqs_cis = self.freqs_cis.cast(h.dtype)[:, :tokens.shape[1], :, :, :]
    for i in range(self.n_layers):
      h = self.run_layer(h, freqs_cis,
                         self.attention_norm[i], self.wqkv[i], self.wo[i],
                         self.ffn_norm[i], self.w1[i], self.w2[i], self.w3[i],
                         self.lora_a[i],  self.lora_b[i],
                         self.lora_a_wo[i], self.lora_b_wo[i])
    logits = self.output(self.norm(h))
    return logits

# TODO: this shouldn't be needed, but it prevents a copy of the grads. CAT can help
def apply_grad(old_grad:UOp, new_grad:UOp) -> list[UOp]:
  if new_grad.op == Ops.ADD:
    return apply_grad(old_grad, new_grad.src[0])+apply_grad(old_grad, new_grad.src[1])
  elif new_grad.op == Ops.PAD:
    grad_shrink = tuple([(p[0], s+p[0]) for s,p in zip(new_grad.src[0].shape, new_grad.marg)])
    return apply_grad(old_grad.shrink(grad_shrink), new_grad.src[0])
  else:
    return [old_grad.store(old_grad + new_grad)]

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
        grads[t] = Tensor(grads[t].uop.after(UOp.group(*apply_grad(grads[t].uop, g.uop))), device=t.device)
    with Timing("run step: "): loss.realize(*grads.values())

  for i in range(6):
    GlobalCounters.reset()
    profile_marker(f"step {i}")
    with Timing(colored(f"*** step {i}: ", "red")):
      jit_step(tokens)
  print("mem per device: " + ', '.join(f"{dev}: {mem/1e9:.2f} GB" for dev, mem in sorted(GlobalCounters.mem_used_per_device.items())))
