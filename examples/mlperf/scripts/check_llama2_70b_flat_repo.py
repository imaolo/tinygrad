import os
os.environ["DEFAULT_FLOAT"] = "bfloat16"
os.environ["OPTIM_DTYPE"] = "bfloat16"
os.environ["DEV"] = "CPU"
os.environ["WQKV"] = "1"
os.environ["LORA"] = "0"
os.environ["ZEROS"] = "1"

from pathlib import Path

from huggingface_hub import HfApi
from tqdm import tqdm

from tinygrad import Tensor
from tinygrad.nn.state import safe_load, safe_load_metadata
from extra.models.llama import convert_from_huggingface, precompute_freqs_cis
from extra.huggingface_onnx.huggingface_manager import DOWNLOADS_DIR, snapshot_download_with_retry
from examples.mlperf.model_train import LLAMA2_70B_ARGS

FLAT_REPO_ID = os.getenv("FLAT_REPO_ID", "imaolo/llama2-70b-fused-qkv-flat-mlperf")
REF_REPO_ID = os.getenv("REF_REPO_ID", "regisss/llama2-70b-fused-qkv-mlperf")
MAX_CONTEXT = int(os.getenv("MAX_CONTEXT", "8192"))

REF_WEIGHTS_PATH = DOWNLOADS_DIR / REF_REPO_ID
FLAT_WEIGHTS_PATH = DOWNLOADS_DIR / FLAT_REPO_ID

EXPECTED_FLAT_KEYS = (
  "attention_norm",
  "ffn_norm",
  "freqs_cis",
  "norm.weight",
  "output",
  "tok_embeddings.weight",
  "w13",
  "w2",
  "wo",
  "wqkv",
)


def expected_flat_shapes() -> dict[str, tuple[int, ...]]:
  dim = LLAMA2_70B_ARGS["dim"]
  n_heads = LLAMA2_70B_ARGS["n_heads"]
  n_kv_heads = LLAMA2_70B_ARGS["n_kv_heads"]
  n_layers = LLAMA2_70B_ARGS["n_layers"]
  hidden_dim = LLAMA2_70B_ARGS["hidden_dim"]
  vocab_size = LLAMA2_70B_ARGS["vocab_size"]
  head_dim = dim // n_heads
  wqkv_dim = dim + 2 * n_kv_heads * head_dim
  return {
    "attention_norm": (n_layers, dim),
    "ffn_norm": (n_layers, dim),
    "freqs_cis": (1, MAX_CONTEXT * 2, 1, head_dim // 2, 2),
    "norm.weight": (dim,),
    "output": (1, vocab_size, dim),
    "tok_embeddings.weight": (vocab_size, dim),
    "w13": (n_layers, hidden_dim * 2, dim),
    "w2": (n_layers, dim, hidden_dim),
    "wo": (n_layers, dim, dim),
    "wqkv": (n_layers, wqkv_dim, dim),
  }


def list_remote_safetensors(repo_id:str) -> list[str]:
  api = HfApi()
  return sorted(Path(path).name for path in api.list_repo_files(repo_id=repo_id) if path.endswith(".safetensors"))


def download_repo(repo_id:str, local_dir:Path) -> None:
  tqdm.write(f"downloading {repo_id} to {local_dir}")
  local_dir.mkdir(parents=True, exist_ok=True)
  snapshot_download_with_retry(repo_id=repo_id, local_dir=local_dir, allow_patterns=["*safetensors*", "*.json", "*.md"])


def load_repo_state_dict(weights_path:Path, desc:str) -> dict[str, Tensor]:
  weight_paths = sorted(weights_path.glob("*.safetensors"))
  state_dict = {}
  for weight_file in tqdm(weight_paths, desc=desc, unit="file"):
    state_dict.update(safe_load(weight_file))
  return state_dict


def max_abs_diff(a:Tensor, b:Tensor) -> float:
  return (a.float() - b.float()).abs().max().item()


def assert_exact(name:str, a:Tensor, b:Tensor) -> None:
  diff = max_abs_diff(a, b)
  if diff != 0.0:
    raise RuntimeError(f"{name} mismatch: max diff {diff}")


def compare_matrix_samples(name:str, ref:Tensor, flat:Tensor) -> None:
  rows, cols = ref.shape
  row_windows = (
    (0, min(4, rows)),
    (max(rows // 2 - 2, 0), min(rows // 2 + 2, rows)),
    (max(rows - 4, 0), rows),
  )
  col_windows = (
    (0, min(8, cols)),
    (max(cols // 2 - 4, 0), min(cols // 2 + 4, cols)),
    (max(cols - 8, 0), cols),
  )
  for r0, r1 in row_windows:
    for c0, c1 in col_windows:
      assert_exact(f"{name}[{r0}:{r1}, {c0}:{c1}]", ref[r0:r1, c0:c1], flat[r0:r1, c0:c1])


def check_remote_layout() -> None:
  flat_remote = list_remote_safetensors(FLAT_REPO_ID)
  ref_remote = list_remote_safetensors(REF_REPO_ID)
  expected_flat_files = sorted(f"{key}.safetensors" for key in EXPECTED_FLAT_KEYS)
  if flat_remote != expected_flat_files:
    missing = sorted(set(expected_flat_files) - set(flat_remote))
    extra = sorted(set(flat_remote) - set(expected_flat_files))
    raise RuntimeError(f"flat repo file mismatch: missing={missing} extra={extra}")
  if len(ref_remote) != 29:
    raise RuntimeError(f"expected 29 reference safetensors, found {len(ref_remote)}")
  tqdm.write(f"remote layout ok: {FLAT_REPO_ID} has {len(flat_remote)} flat shards, {REF_REPO_ID} has {len(ref_remote)} reference shards")


def check_flat_file_metadata(weights_path:Path) -> None:
  expected_shapes = expected_flat_shapes()
  flat_files = sorted(weights_path.glob("*.safetensors"))
  expected_files = {f"{key}.safetensors" for key in EXPECTED_FLAT_KEYS}
  actual_files = {path.name for path in flat_files}
  if actual_files != expected_files:
    missing = sorted(expected_files - actual_files)
    extra = sorted(actual_files - expected_files)
    raise RuntimeError(f"flat local file mismatch: missing={missing} extra={extra}")
  for flat_file in tqdm(flat_files, desc="check flat file metadata", unit="file"):
    _, _, metadata = safe_load_metadata(flat_file)
    keys = [key for key in metadata.keys() if key != "__metadata__"]
    if len(keys) != 1:
      raise RuntimeError(f"{flat_file.name} should contain exactly one tensor, found {keys}")
    key = keys[0]
    if flat_file.name != f"{key}.safetensors":
      raise RuntimeError(f"{flat_file.name} contains key {key}")
    shape = tuple(metadata[key]["shape"])
    if shape != expected_shapes[key]:
      raise RuntimeError(f"{key} shape mismatch: expected {expected_shapes[key]}, found {shape}")
  tqdm.write("flat repo metadata matches flat_llama schema")


def stack_small_ref_vectors(ref_state_dict:dict[str, Tensor], key_fmt:str) -> Tensor:
  return Tensor.stack([ref_state_dict[key_fmt.format(i=i)] for i in range(LLAMA2_70B_ARGS["n_layers"])])


def check_small_exact(ref_state_dict:dict[str, Tensor], flat_state_dict:dict[str, Tensor]) -> None:
  checks = [
    ("attention_norm", stack_small_ref_vectors(ref_state_dict, "layers.{i}.attention_norm.weight"), flat_state_dict["attention_norm"]),
    ("ffn_norm", stack_small_ref_vectors(ref_state_dict, "layers.{i}.ffn_norm.weight"), flat_state_dict["ffn_norm"]),
    ("norm.weight", ref_state_dict["norm.weight"], flat_state_dict["norm.weight"]),
    ("tok_embeddings.weight", ref_state_dict["tok_embeddings.weight"], flat_state_dict["tok_embeddings.weight"]),
    ("output[0]", ref_state_dict["output.weight"], flat_state_dict["output"][0]),
    ("freqs_cis", precompute_freqs_cis(LLAMA2_70B_ARGS["dim"] // LLAMA2_70B_ARGS["n_heads"], MAX_CONTEXT * 2, LLAMA2_70B_ARGS.get("rope_theta", 10000)), flat_state_dict["freqs_cis"]),
  ]
  for name, ref_tensor, flat_tensor in tqdm(checks, desc="check exact small tensors", unit="tensor"):
    assert_exact(name, ref_tensor, flat_tensor)
  tqdm.write("exact checks passed for norms, embeddings, output, and freqs_cis")


def check_sampled_large_tensors(ref_state_dict:dict[str, Tensor], flat_state_dict:dict[str, Tensor]) -> None:
  selected_layers = sorted({0, LLAMA2_70B_ARGS["n_layers"] // 2, LLAMA2_70B_ARGS["n_layers"] - 1})
  comparisons = []
  for i in selected_layers:
    comparisons.extend([
      (f"wqkv[{i}]", ref_state_dict[f"layers.{i}.attention.wqkv.weight"], flat_state_dict["wqkv"][i]),
      (f"wo[{i}]", ref_state_dict[f"layers.{i}.attention.wo.weight"], flat_state_dict["wo"][i]),
      (f"w2[{i}]", ref_state_dict[f"layers.{i}.feed_forward.w2.weight"], flat_state_dict["w2"][i]),
    ])
  for name, ref_tensor, flat_tensor in tqdm(comparisons, desc="check sampled large tensors", unit="tensor"):
    compare_matrix_samples(name, ref_tensor, flat_tensor)
  tqdm.write(f"sampled checks passed for wqkv/wo/w2 on layers {selected_layers}")


def check_w13(ref_state_dict:dict[str, Tensor], flat_state_dict:dict[str, Tensor]) -> None:
  hidden_dim = LLAMA2_70B_ARGS["hidden_dim"]
  selected_layers = sorted({0, LLAMA2_70B_ARGS["n_layers"] // 2, LLAMA2_70B_ARGS["n_layers"] - 1})
  for i in tqdm(selected_layers, desc="check w13", unit="layer"):
    ref_w1 = ref_state_dict[f"layers.{i}.feed_forward.w1.weight"]
    ref_w3 = ref_state_dict[f"layers.{i}.feed_forward.w3.weight"]
    flat_w13 = flat_state_dict["w13"][i]
    if flat_w13.shape != (hidden_dim * 2, LLAMA2_70B_ARGS["dim"]):
      raise RuntimeError(f"w13[{i}] shape mismatch: {flat_w13.shape}")

    compare_matrix_samples(f"w13[{i}].w1_half", ref_w1, flat_w13[:hidden_dim])
    compare_matrix_samples(f"w13[{i}].w3_half", ref_w3, flat_w13[hidden_dim:])

    assert_exact(f"w13[{i}] boundary left", ref_w1[-2:, :16], flat_w13[hidden_dim-2:hidden_dim, :16])
    assert_exact(f"w13[{i}] boundary right", ref_w3[:2, :16], flat_w13[hidden_dim:hidden_dim+2, :16])
    assert_exact(f"w13[{i}] first rows", ref_w1[:4, :16], flat_w13[:4, :16])
    assert_exact(f"w13[{i}] last rows", ref_w3[-4:, :16], flat_w13[-4:, :16])
  tqdm.write(f"extra w13 checks passed on layers {selected_layers}")


def main() -> None:
  check_remote_layout()
  download_repo(REF_REPO_ID, REF_WEIGHTS_PATH)
  download_repo(FLAT_REPO_ID, FLAT_WEIGHTS_PATH)
  check_flat_file_metadata(FLAT_WEIGHTS_PATH)

  ref_raw_state = load_repo_state_dict(REF_WEIGHTS_PATH, "load reference shards")
  tqdm.write("converting reference repo to internal naming")
  ref_state_dict = convert_from_huggingface(ref_raw_state, LLAMA2_70B_ARGS["n_layers"], LLAMA2_70B_ARGS["n_heads"], LLAMA2_70B_ARGS["n_kv_heads"])

  flat_state_dict = load_repo_state_dict(FLAT_WEIGHTS_PATH, "load flat shards")
  if sorted(flat_state_dict.keys()) != sorted(EXPECTED_FLAT_KEYS):
    raise RuntimeError(f"flat repo tensor keys mismatch: {sorted(flat_state_dict.keys())}")

  check_small_exact(ref_state_dict, flat_state_dict)
  check_sampled_large_tensors(ref_state_dict, flat_state_dict)
  check_w13(ref_state_dict, flat_state_dict)
  tqdm.write("all repo sanity checks passed")


if __name__ == "__main__":
  main()
