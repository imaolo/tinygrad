from tinygrad import Tensor, nn
from tinygrad.dtype import DType
from extra.models.llama import apply_rotary_emb, precompute_freqs_cis
import math

class LoRA:
  def __init__(self, in_features:int, out_features:int, dtype:str|DType, rank:int=16, alpha:float=32.0, dropout:float=0.1):
    self.dropout = dropout
    self.scale = alpha / rank
    self.lora_a = Tensor.kaiming_uniform(in_features, rank, a=math.sqrt(5), dtype=dtype, requires_grad=True)
    self.lora_b = Tensor.zeros(rank, out_features, dtype=dtype, requires_grad=True)

  def __call__(self, x:Tensor) -> Tensor:
    return x.dropout(self.dropout).linear(self.lora_a).linear(self.lora_b) * self.scale

class Attention:
  def __init__(self, dim:int, n_heads:int, n_kv_heads:int|None=None, linear=nn.Linear, fuse_wqkv:bool=False, use_lora:bool=False):
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads # n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
    self.head_dim = dim // n_heads
    self.n_rep = self.n_heads // self.n_kv_heads
    self.fuse_wqkv = fuse_wqkv
    self.lora_map = {} if use_lora else None

    if self.fuse_wqkv:
      self.wqkv = self.create_linear(linear, dim, self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim * 2, bias=False)
    else:
      self.wq = self.create_linear(linear, dim, self.n_heads * self.head_dim, bias=False)
      self.wk = self.create_linear(linear, dim, self.n_kv_heads * self.head_dim, bias=False)
      self.wv = self.create_linear(linear, dim, self.n_kv_heads * self.head_dim, bias=False)

    self.wo = self.create_linear(linear, self.n_heads * self.head_dim, dim, bias=False)
  
  def create_linear(self, linear, in_features:int, out_features:int, bias:bool=True):
    lin = linear(in_features, out_features, bias)
    if self.lora_map is not None:
      self.lora_map[lin] = LoRA(in_features, out_features, lin.weight.dtype)
    return lin

  def run_linear(self, lin, x:Tensor) -> Tensor:
    out = lin(x)
    if self.lora_map is not None:
      out = out + self.lora_map[lin](x)
    return out

  def __call__(self, x:Tensor, freqs_cis:Tensor) -> Tensor:
    if self.fuse_wqkv:
      xqkv = self.run_linear(self.wqkv, x)
      xqkv = xqkv.reshape(xqkv.shape[0], xqkv.shape[1], self.n_kv_heads, self.n_rep + 2, self.head_dim)
      xq = xqkv[:, :, :, :self.n_rep].reshape(xqkv.shape[0], xqkv.shape[1], -1)
      xk = xqkv[:, :, :, self.n_rep:self.n_rep+1].reshape(xqkv.shape[0], xqkv.shape[1], -1)
      xv = xqkv[:, :, :, self.n_rep+1:self.n_rep+2].reshape(xqkv.shape[0], xqkv.shape[1], -1)
    else:
      xq, xk, xv = self.run_linear(self.wq, x), self.run_linear(self.wk, x), self.run_linear(self.wv, x)

    xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
    xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
    xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    bsz, seqlen, _, _ = xq.shape

    xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
    attn = xq.scaled_dot_product_attention(xk, xv, is_causal=True, enable_gqa=True).transpose(1, 2)

    attn = attn.reshape(bsz, seqlen, -1)
    return self.run_linear(self.wo, attn)

class FeedForward:
  def __init__(self, dim:int, hidden_dim:int, linear=nn.Linear):
    self.w1 = linear(dim, hidden_dim, bias=False)
    self.w2 = linear(hidden_dim, dim, bias=False)
    self.w3 = linear(dim, hidden_dim, bias=False) # the gate in Gated Linear Unit

  def __call__(self, x:Tensor) -> Tensor:
    w1 = self.w1(x).silu()
    w3 = self.w3(x)
    return self.w2(w1 * w3)

class TransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int|None, norm_eps:float, linear=nn.Linear, fuse_wqkv:bool=False, use_lora:bool=False):
    self.attention = Attention(dim, n_heads, n_kv_heads, linear, fuse_wqkv, use_lora)
    self.feed_forward = FeedForward(dim, hidden_dim, linear)
    self.attention_norm = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm = nn.RMSNorm(dim, norm_eps)

  def __call__(self, x:Tensor, freqs_cis:Tensor):
    h = x + self.attention(self.attention_norm(x), freqs_cis)
    return h + self.feed_forward(self.ffn_norm(h))

class Transformer:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_layers:int, norm_eps:float, vocab_size:int, n_kv_heads:int|None=None,
               rope_theta:int=10000, max_context:int=1024, linear=nn.Linear, embedding=nn.Embedding, fuse_wqkv:bool=True, use_lora:bool=False):
    self.layers = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, linear, fuse_wqkv, use_lora) for _ in range(n_layers)]
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = embedding(vocab_size, dim)
    self.output = nn.Linear(dim, vocab_size, bias=False) if embedding == nn.Embedding else linear(dim, vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_context * 2, rope_theta).contiguous().requires_grad_(False)

  def __call__(self, tokens:Tensor):
    h = self.tok_embeddings(tokens)
    freqs_cis = self.freqs_cis.cast(h.dtype)[:, :tokens.shape[1], :, :, :]
    for layer in self.layers: h = layer(h, freqs_cis)
    logits = self.output(self.norm(h))
    return logits

  def shard(self, device:tuple[str, ...], mp:bool=False):
    from tinygrad.nn.state import get_parameters
    if not mp:
      for v in get_parameters(self): v.shard_(device, axis=None)
    else:
      # flat per-layer weights: axis 0 is n_layers, so shard axes are +1 vs per-layer Transformer
      self.wqkv.shard_(device, axis=1).realize()          # (n_layers, out, dim) shard out
      self.wo.shard_(device, axis=2).realize()             # (n_layers, dim, in) shard in
      self.w1.shard_(device, axis=1).realize()             # (n_layers, hidden, dim) shard out
      self.w2.shard_(device, axis=2).realize()             # (n_layers, dim, hidden) shard in
      self.w3.shard_(device, axis=1).realize()             # (n_layers, hidden, dim) shard out
      self.attention_norm.shard_(device, axis=None).realize()
      self.ffn_norm.shard_(device, axis=None).realize()
      self.norm.weight.shard_(device, axis=None).realize()
      self.tok_embeddings.weight.shard_(device, axis=0).realize()
      self.output.weight.shard_(device, axis=0).realize()
      self.freqs_cis.shard_(device, axis=None).realize()

def _fuse_qkv(q:Tensor, k:Tensor, v:Tensor, n_heads:int, n_kv_heads:int) -> Tensor:
  head_dim = q.shape[0] // n_heads
  n_rep = n_heads // n_kv_heads
  in_dim = q.shape[1]
  q = q.reshape(n_kv_heads, n_rep, head_dim, in_dim)
  k = k.reshape(n_kv_heads, 1, head_dim, in_dim)
  v = v.reshape(n_kv_heads, 1, head_dim, in_dim)
  return q.cat(k, v, dim=1).reshape(-1, in_dim)

def copy_weights_fused(fused_model: Transformer, unfused_model: Transformer) -> None:
  from tinygrad.nn.state import get_state_dict

  for dst_layer, src_layer in zip(fused_model.layers, unfused_model.layers):
    dst_layer.attention.wqkv.weight.assign(_fuse_qkv(
      src_layer.attention.wq.weight.cast(dst_layer.attention.wqkv.weight.dtype),
      src_layer.attention.wk.weight.cast(dst_layer.attention.wqkv.weight.dtype),
      src_layer.attention.wv.weight.cast(dst_layer.attention.wqkv.weight.dtype),
      src_layer.attention.n_heads, src_layer.attention.n_kv_heads
    ))
    dst_layer.attention.wo.weight.assign(src_layer.attention.wo.weight.cast(dst_layer.attention.wo.weight.dtype))
    dst_layer.feed_forward.w1.weight.assign(src_layer.feed_forward.w1.weight.cast(dst_layer.feed_forward.w1.weight.dtype))
    dst_layer.feed_forward.w2.weight.assign(src_layer.feed_forward.w2.weight.cast(dst_layer.feed_forward.w2.weight.dtype))
    dst_layer.feed_forward.w3.weight.assign(src_layer.feed_forward.w3.weight.cast(dst_layer.feed_forward.w3.weight.dtype))
    dst_layer.attention_norm.weight.assign(src_layer.attention_norm.weight.cast(dst_layer.attention_norm.weight.dtype))
    dst_layer.ffn_norm.weight.assign(src_layer.ffn_norm.weight.cast(dst_layer.ffn_norm.weight.dtype))

  fused_model.norm.weight.assign(unfused_model.norm.weight.cast(fused_model.norm.weight.dtype))
  fused_model.tok_embeddings.weight.assign(unfused_model.tok_embeddings.weight.cast(fused_model.tok_embeddings.weight.dtype))
  fused_model.output.weight.assign(unfused_model.output.weight.cast(fused_model.output.weight.dtype))
