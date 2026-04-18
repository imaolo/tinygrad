import pathlib
from tinygrad import Device, Tensor
from tinygrad.helpers import Context
from tinygrad.runtime.support.compiler_cuda import pretty_ptx, NVCCCompiler
from extra.thunder.tiny.fa import flash_attention as _tiny_flash_attention


def flash_attention(xq:Tensor, xk:Tensor, xv:Tensor, attn_mask:Tensor|None=None, is_causal:bool=False):
  # Keep the CUDA API aligned with the AMD path for flat_llama imports.
  # The current native CUDA prototype in fa.cu is not shape-compatible with LLaMA-70B
  # training yet, so route through the tiny custom-kernel implementation for now.
  return _tiny_flash_attention(xq, xk, xv, attn_mask=attn_mask, is_causal=is_causal)

if __name__ == "__main__":
  code = (pathlib.Path(__file__).parent / "fa.cu").read_text()
  device = Device["CUDA"]
  kitten_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20", "--expt-relaxed-constexpr", "-DKITTENS_4090"]
  lib = NVCCCompiler(device.compiler.arch, kitten_args).compile(code)
  kernel_name = lib.decode().split(".globl\t")[1].split("\n")[0]
  print("kernel name", kernel_name)
  print(pretty_ptx(lib.decode()))

  prg = device.runtime(kernel_name, lib)
  prg.smem = 16384 * 3

  B, N, H, D = 16, 1024, 16, 64
  q = Tensor.randn(B, N, H, D, device='CUDA', dtype="bfloat16")
  k = Tensor.randn(B, N, H, D, device='CUDA', dtype="bfloat16")
  v = Tensor.randn(B, N, H, D, device='CUDA', dtype="bfloat16")
  out = Tensor.empty(B, N, H, D, device='CUDA', dtype="bfloat16")
  Tensor.realize(q, k, v, out)

  NUM_WORKERS = 4
  ROWS = 16 * (64 // D)

  gsz = (N // (ROWS*NUM_WORKERS), H, B)
  for _ in range(5):
    et = prg(out.uop.buffer.ensure_allocated()._buf, q.uop.buffer._buf, k.uop.buffer._buf, v.uop.buffer._buf,
             global_size=gsz, local_size=(ROWS*NUM_WORKERS,1,1), wait=True)

    attn_flops = 2 * B * H * N * N * D + \
                 4 * B * H * N * N + \
                 2 * B * H * N * N * D
    print(f"{attn_flops/(et*1e9):2f} GFLOPS")

  for _ in range(5):
    with Context(DEBUG=2):
      ref = q.scaled_dot_product_attention(k, v)

  ref, out = ref.float(), out.float()
  print((ref-out).mean().item(), (ref-out).max().item())
