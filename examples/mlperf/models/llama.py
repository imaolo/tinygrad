import json, math
from pathlib import Path
from typing import Any, Callable

from tinygrad import Tensor, nn
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load, torch_load
from extra.models.llama import apply_rotary_emb, precompute_freqs_cis, convert_from_huggingface as convert_base_from_huggingface, fix_bf16

LLAMA2_70B_ARGS = {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-5, "vocab_size": 32000, "hidden_dim": 28672}

NamedLinear = Callable[[str, int, int, bool], Any]

def _create_linear(name:str, linear, named_linear:NamedLinear|None, in_features:int, out_features:int, bias:bool=False):
  if named_linear is not None: return named_linear(name, in_features, out_features, bias)
  return linear(in_features, out_features, bias=bias)

class LoRALinear:
  def __init__(self, in_features:int, out_features:int, bias:bool=False, rank:int=16, alpha:float=32.0, dropout:float=0.1,
               base_linear=nn.Linear):
    assert rank > 0, "LoRA rank must be positive"
    self.rank, self.alpha, self.dropout = rank, alpha, dropout
    base = base_linear(in_features, out_features, bias=bias)
    self.weight = base.weight
    self.bias = base.bias
    self.weight.requires_grad_(False)
    if self.bias is not None: self.bias.requires_grad_(False)
    self.lora_a = Tensor.kaiming_uniform(rank, in_features, a=math.sqrt(5), dtype=self.weight.dtype)
    self.lora_b = Tensor.zeros(out_features, rank, dtype=self.weight.dtype)
    self.scaling = alpha / rank

  def __call__(self, x:Tensor) -> Tensor:
    dropped = x.dropout(self.dropout) if self.dropout > 0 else x
    update = dropped.linear(self.lora_a.transpose()).linear(self.lora_b.transpose())
    return x.linear(self.weight.transpose(), self.bias) + update * self.scaling

def llama70b_lora_linear(name:str, in_features:int, out_features:int, bias:bool=False, rank:int=16, alpha:float=32.0, dropout:float=0.1):
  if name.endswith(".attention.wqkv") or name.endswith(".attention.wo"):
    return LoRALinear(in_features, out_features, bias=bias, rank=rank, alpha=alpha, dropout=dropout)
  return nn.Linear(in_features, out_features, bias=bias)

def freeze_non_lora_params(model) -> None:
  for name, tensor in nn.state.get_state_dict(model).items():
    tensor.requires_grad_(name.endswith(".lora_a") or name.endswith(".lora_b"))

def load(fn:str):
  if fn.endswith(".index.json"):
    with open(fn) as fp: weight_map = json.load(fp)["weight_map"]
    parts = {n: load(str(Path(fn).parent / Path(n).name)) for n in set(weight_map.values())}
    return {k: parts[n][k] for k, n in weight_map.items()}
  if fn.endswith(".safetensors"): return safe_load(fn)
  return torch_load(fn)

def concat_weights(models:list[dict[str, Tensor]], device=None) -> dict[str, Tensor]:
  def convert(name:str) -> Tensor:
    disk_tensors = [model[name] for model in models]
    if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1: return disk_tensors[0].to(device=device)
    axis = 1 if name.endswith((".attention.wo.weight", ".feed_forward.w2.weight", "tok_embeddings.weight", "output.weight")) else 0
    lazy_tensors = [data.to(device=device) for data in disk_tensors]
    return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)
  return {name: convert(name) for name in {name: None for model in models for name in model}}

def fuse_qkv_weights(weights:dict[str, Tensor], n_layers:int) -> dict[str, Tensor]:
  fused = dict(weights)
  for layer in range(n_layers):
    prefix = f"layers.{layer}.attention"
    q_name, k_name, v_name = f"{prefix}.wq.weight", f"{prefix}.wk.weight", f"{prefix}.wv.weight"
    if q_name in fused and k_name in fused and v_name in fused:
      fused[f"{prefix}.wqkv.weight"] = fused[q_name].cat(fused[k_name], fused[v_name], dim=0)
      del fused[q_name], fused[k_name], fused[v_name]
  return fused

def load_pretrained_weights(model_path:Path, n_layers:int, n_heads:int, n_kv_heads:int, fused_qkv:bool=True) -> dict[str, Tensor]:
  if model_path.is_dir():
    if (model_path / "model.safetensors.index.json").exists(): weights = load(str(model_path / "model.safetensors.index.json"))
    elif (model_path / "model.safetensors").exists(): weights = load(str(model_path / "model.safetensors"))
    else:
      shard_paths = sorted(model_path.glob("consolidated.*.pth"))
      assert shard_paths, f"no supported checkpoint files found in {model_path}"
      weights = concat_weights([load(str(path)) for path in shard_paths], None)
  else:
    weights = load(str(model_path))

  if "model.embed_tokens.weight" in weights:
    weights = convert_base_from_huggingface(weights, n_layers, n_heads, n_kv_heads)
  weights = fix_bf16(weights)
  return fuse_qkv_weights(weights, n_layers) if fused_qkv else weights

class Attention:
  def __init__(self, dim:int, n_heads:int, n_kv_heads:int|None=None, linear=nn.Linear, named_linear:NamedLinear|None=None,
               prefix:str="attention", fused_qkv:bool=False):
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads # n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
    self.head_dim = dim // n_heads
    self.n_rep = self.n_heads // self.n_kv_heads
    self.fused_qkv = fused_qkv or bool(getenv("WQKV"))

    if self.fused_qkv:
      self.wqkv = _create_linear(f"{prefix}.wqkv", linear, named_linear, dim,
                                 self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim * 2, bias=False)
    else:
      self.wq = _create_linear(f"{prefix}.wq", linear, named_linear, dim, self.n_heads * self.head_dim, bias=False)
      self.wk = _create_linear(f"{prefix}.wk", linear, named_linear, dim, self.n_kv_heads * self.head_dim, bias=False)
      self.wv = _create_linear(f"{prefix}.wv", linear, named_linear, dim, self.n_kv_heads * self.head_dim, bias=False)

    self.wo = _create_linear(f"{prefix}.wo", linear, named_linear, self.n_heads * self.head_dim, dim, bias=False)

  def __call__(self, x:Tensor, freqs_cis:Tensor) -> Tensor:
    if self.fused_qkv:
      xqkv = self.wqkv(x)
      xqkv = xqkv.reshape(xqkv.shape[0], xqkv.shape[1], self.n_kv_heads, self.n_rep + 2, self.head_dim)
      xq = xqkv[:, :, :, :self.n_rep].reshape(xqkv.shape[0], xqkv.shape[1], -1)
      xk = xqkv[:, :, :, self.n_rep:self.n_rep+1].reshape(xqkv.shape[0], xqkv.shape[1], -1)
      xv = xqkv[:, :, :, self.n_rep+1:self.n_rep+2].reshape(xqkv.shape[0], xqkv.shape[1], -1)
    else:
      xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
    xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
    xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    bsz, seqlen, _, _ = xq.shape

    xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
    attn = xq.scaled_dot_product_attention(xk, xv, is_causal=True, enable_gqa=True).transpose(1, 2)

    attn = attn.reshape(bsz, seqlen, -1)
    return self.wo(attn)

class FeedForward:
  def __init__(self, dim:int, hidden_dim:int, linear=nn.Linear, named_linear:NamedLinear|None=None, prefix:str="feed_forward"):
    self.w1 = _create_linear(f"{prefix}.w1", linear, named_linear, dim, hidden_dim, bias=False)
    self.w2 = _create_linear(f"{prefix}.w2", linear, named_linear, hidden_dim, dim, bias=False)
    self.w3 = _create_linear(f"{prefix}.w3", linear, named_linear, dim, hidden_dim, bias=False) # the gate in Gated Linear Unit

  def __call__(self, x:Tensor) -> Tensor:
    w1 = self.w1(x).silu()
    w3 = self.w3(x)
    return self.w2(w1 * w3)

class TransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int|None, norm_eps:float, linear=nn.Linear,
               named_linear:NamedLinear|None=None, prefix:str="layer", fused_qkv:bool=False):
    self.attention = Attention(dim, n_heads, n_kv_heads, linear, named_linear, f"{prefix}.attention", fused_qkv)
    self.feed_forward = FeedForward(dim, hidden_dim, linear, named_linear, f"{prefix}.feed_forward")
    self.attention_norm = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm = nn.RMSNorm(dim, norm_eps)

  def __call__(self, x:Tensor, freqs_cis:Tensor):
    h = x + self.attention(self.attention_norm(x), freqs_cis)
    return h + self.feed_forward(self.ffn_norm(h))

class Transformer:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_layers:int, norm_eps:float, vocab_size:int, n_kv_heads:int|None=None,
               rope_theta:int=10000, max_context:int=1024, linear=nn.Linear, embedding=nn.Embedding, named_linear:NamedLinear|None=None,
               fused_qkv:bool=False):
    self.layers = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, linear, named_linear,
                                    f"layers.{i}", fused_qkv) for i in range(n_layers)]
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = embedding(vocab_size, dim)
    self.output = nn.Linear(dim, vocab_size, bias=False) if embedding == nn.Embedding else _create_linear("output", linear, named_linear, dim, vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_context * 2, rope_theta).contiguous().requires_grad_(False)

  def __call__(self, tokens:Tensor):
    h = self.tok_embeddings(tokens)
    freqs_cis = self.freqs_cis.cast(h.dtype)[:, :tokens.shape[1], :, :, :]
    for layer in self.layers: h = layer(h, freqs_cis)
    logits = self.output(self.norm(h))
    return logits
