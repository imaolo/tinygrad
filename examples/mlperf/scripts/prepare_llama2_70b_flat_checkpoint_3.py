import os
os.environ["DEFAULT_FLOAT"] = "bfloat16"
os.environ["OPTIM_DTYPE"] = "bfloat16"
os.environ["DEV"] = "CPU"
os.environ["WQKV"] = "1"
os.environ["LORA"] = "0"
os.environ["ZEROS"] = "1"

from pathlib import Path

import numpy as np
from huggingface_hub import CommitOperationDelete, HfApi
from tqdm import tqdm

from tinygrad import Tensor
from tinygrad.nn.state import safe_load, safe_save
from extra.models.llama import convert_from_huggingface, precompute_freqs_cis
from extra.huggingface_onnx.huggingface_manager import DOWNLOADS_DIR, snapshot_download_with_retry
from examples.mlperf.model_train import LLAMA2_70B_ARGS

HF_REPO_ID = "imaolo/llama2-70b-fused-qkv-flat-mlperf"
HF_REF_REPO_ID = "regisss/llama2-70b-fused-qkv-mlperf"
MAX_CONTEXT = 8192

REF_WEIGHTS_PATH = DOWNLOADS_DIR/HF_REF_REPO_ID
WEIGHTS_PATH = DOWNLOADS_DIR/HF_REPO_ID


def download_reference_weights() -> None:
  print(f"downloading reference weights to {REF_WEIGHTS_PATH}")
  REF_WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
  snapshot_download_with_retry(repo_id=HF_REF_REPO_ID, local_dir=REF_WEIGHTS_PATH, allow_patterns=["*safetensors*", "*.json", "*.md"])
  print("downloaded reference weights")

def load_reference_state_dict() -> dict[str, Tensor]:
  ref_weight_paths = sorted(REF_WEIGHTS_PATH.glob("*.safetensors"))
  assert len(ref_weight_paths) == 29, f"expected 29 weight files, found {len(ref_weight_paths)}"
  ref_state_dict = {}
  for weight_file in tqdm(ref_weight_paths, desc="load reference shards", unit="file"):
    ref_state_dict.update(safe_load(weight_file))
  return convert_from_huggingface(ref_state_dict, LLAMA2_70B_ARGS["n_layers"], LLAMA2_70B_ARGS["n_heads"], LLAMA2_70B_ARGS["n_kv_heads"])


def tensor_from_numpy(arr:np.ndarray, like:Tensor) -> Tensor:
  out = Tensor(arr)
  return out.cast(like.dtype) if out.dtype != like.dtype else out


def convert_single_tensor(name:str, tensor:Tensor, transform=None) -> Tensor:
  arr = tensor.numpy()
  if transform is not None:
    arr = transform(arr)
  out = tensor_from_numpy(arr, tensor)
  return out


def pop_stacked_layers(state_dict:dict[str, Tensor], key_fmt:str, name:str) -> Tensor:
  n_layers = LLAMA2_70B_ARGS["n_layers"]
  layers = []
  with tqdm(total=n_layers * 2 + 1, desc=f"stack {name}", unit="step", leave=False) as progress:
    for i in range(n_layers):
      layers.append(state_dict.pop(key_fmt.format(i=i)))
      progress.update()
    arrays = []
    for layer in layers:
      arrays.append(layer.numpy())
      progress.update()
    out = tensor_from_numpy(np.stack(arrays), layers[0])
    progress.update()
  return out


def build_flat_state_dict(ref_state_dict:dict[str, Tensor]) -> dict[str, Tensor]:
  rope_theta = LLAMA2_70B_ARGS.get("rope_theta", 10000)
  norm_weight = ref_state_dict.pop("norm.weight")
  output_weight = ref_state_dict.pop("output.weight")
  tok_embeddings_weight = ref_state_dict.pop("tok_embeddings.weight")
  flat_state_dict = {}
  with tqdm(total=10, desc="build flat state dict", unit="tensor") as progress:
    flat_state_dict["attention_norm"] = pop_stacked_layers(ref_state_dict, "layers.{i}.attention_norm.weight", "attention_norm")
    progress.update()
    flat_state_dict["ffn_norm"] = pop_stacked_layers(ref_state_dict, "layers.{i}.ffn_norm.weight", "ffn_norm")
    progress.update()
    with tqdm(total=1, desc="prepare freqs_cis", unit="tensor", leave=False) as inner:
      flat_state_dict["freqs_cis"] = precompute_freqs_cis(LLAMA2_70B_ARGS["dim"] // LLAMA2_70B_ARGS["n_heads"], MAX_CONTEXT * 2, rope_theta).contiguous().requires_grad_(False)
      inner.update()
    progress.update()
    flat_state_dict["norm.weight"] = convert_single_tensor("norm.weight", norm_weight)
    progress.update()
    flat_state_dict["output"] = convert_single_tensor("output", output_weight, lambda arr: np.expand_dims(arr, axis=0))
    progress.update()
    flat_state_dict["tok_embeddings.weight"] = convert_single_tensor("tok_embeddings.weight", tok_embeddings_weight)
    progress.update()
    flat_state_dict["w1"] = pop_stacked_layers(ref_state_dict, "layers.{i}.feed_forward.w1.weight", "w1")
    progress.update()
    flat_state_dict["w2"] = pop_stacked_layers(ref_state_dict, "layers.{i}.feed_forward.w2.weight", "w2")
    progress.update()
    flat_state_dict["w3"] = pop_stacked_layers(ref_state_dict, "layers.{i}.feed_forward.w3.weight", "w3")
    progress.update()
    flat_state_dict["wo"] = pop_stacked_layers(ref_state_dict, "layers.{i}.attention.wo.weight", "wo")
    progress.update()
    flat_state_dict["wqkv"] = pop_stacked_layers(ref_state_dict, "layers.{i}.attention.wqkv.weight", "wqkv")
    progress.update()
  w1, w3 = flat_state_dict.pop("w1"), flat_state_dict.pop("w3")
  with tqdm(total=4, desc="fuse w13", unit="step") as progress:
    w1_np = w1.numpy()
    progress.update()
    w3_np = w3.numpy()
    progress.update()
    w13_np = np.concatenate([w1_np, w3_np], axis=1)
    progress.update()
    flat_state_dict["w13"] = tensor_from_numpy(w13_np, w1)
    progress.update()
  if leftover := sorted(ref_state_dict.keys()):
    raise RuntimeError(f"unused weights after flattening: {leftover}")
  return flat_state_dict


def assert_same_tensor(name:str, ref_tensor:Tensor, transformed_tensor:Tensor) -> None:
  max_diff = (ref_tensor.float() - transformed_tensor.float()).abs().max().item()
  if max_diff != 0.0:
    raise RuntimeError(f"sanity check failed for {name}: max diff {max_diff}")


def run_sanity_checks(ref_state_dict:dict[str, Tensor], flat_state_dict:dict[str, Tensor]) -> None:
  hidden_dim = LLAMA2_70B_ARGS["hidden_dim"]
  n_layers = LLAMA2_70B_ARGS["n_layers"]
  layers_to_check = sorted({0, n_layers // 2, n_layers - 1})
  checks = []
  for i in layers_to_check:
    checks.extend([
      (f"wqkv[{i}]", ref_state_dict[f"layers.{i}.attention.wqkv.weight"], flat_state_dict["wqkv"][i]),
      (f"wo[{i}]", ref_state_dict[f"layers.{i}.attention.wo.weight"], flat_state_dict["wo"][i]),
      (f"w2[{i}]", ref_state_dict[f"layers.{i}.feed_forward.w2.weight"], flat_state_dict["w2"][i]),
      (f"w13[{i}].w1", ref_state_dict[f"layers.{i}.feed_forward.w1.weight"], flat_state_dict["w13"][i][:hidden_dim]),
      (f"w13[{i}].w3", ref_state_dict[f"layers.{i}.feed_forward.w3.weight"], flat_state_dict["w13"][i][hidden_dim:]),
      (f"attention_norm[{i}]", ref_state_dict[f"layers.{i}.attention_norm.weight"], flat_state_dict["attention_norm"][i]),
      (f"ffn_norm[{i}]", ref_state_dict[f"layers.{i}.ffn_norm.weight"], flat_state_dict["ffn_norm"][i]),
    ])
  checks.extend([
    ("norm.weight", ref_state_dict["norm.weight"], flat_state_dict["norm.weight"]),
    ("tok_embeddings.weight", ref_state_dict["tok_embeddings.weight"], flat_state_dict["tok_embeddings.weight"]),
    ("output[0]", ref_state_dict["output.weight"], flat_state_dict["output"][0]),
  ])
  for name, ref_tensor, transformed_tensor in tqdm(checks, desc="sanity checks", unit="check"):
    assert_same_tensor(name, ref_tensor, transformed_tensor)
  tqdm.write(f"sanity checks passed for layers {layers_to_check}")


def clear_existing_tensors() -> None:
  WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
  existing_tensors = sorted(WEIGHTS_PATH.glob("*.safetensors"))
  for tensor_path in tqdm(existing_tensors, desc="clear local tensors", unit="file"):
    tensor_path.unlink()
  tqdm.write(f"cleared {len(existing_tensors)} existing tensor files from {WEIGHTS_PATH}")


def save_state_dict(state_dict:dict[str, Tensor]) -> list[Path]:
  weight_files = [WEIGHTS_PATH / f"{name}.safetensors" for name in state_dict.keys()]
  for file_name, (name, tensor) in tqdm(zip(weight_files, state_dict.items()), total=len(weight_files), desc="saving flat weight shards"):
    safe_save({name: tensor}, file_name)
  return weight_files


def upload_files(files:list[Path]):
  tqdm.write(f"uploading flat model weights to {HF_REPO_ID}")
  api = HfApi()
  with tqdm(total=1, desc="ensure remote repo", unit="stage") as progress:
    api.create_repo(repo_id=HF_REPO_ID, exist_ok=True)
    progress.update()
  remote_files = api.list_repo_files(repo_id=HF_REPO_ID)
  if remote_files:
    tqdm.write(f"deleting {len(remote_files)} existing remote files from {HF_REPO_ID}")
    delete_ops = []
    for path in tqdm(remote_files, desc="plan remote deletes", unit="file"):
      delete_ops.append(CommitOperationDelete(path_in_repo=path))
    api.create_commit(
      repo_id=HF_REPO_ID,
      operations=delete_ops,
      commit_message="Delete existing repo contents before uploading rebuilt flat weights",
    )
  with tqdm(total=1, desc="upload rebuilt weights", unit="stage") as progress:
    commit_info = api.upload_folder(
      folder_path=WEIGHTS_PATH,
      repo_id=HF_REPO_ID,
      allow_patterns=[p.name for p in files],
      commit_message=f"Uploaded {len(files)} rebuilt flat weights",
    )
    progress.update()
  return commit_info


def main() -> None:
  with tqdm(total=6, desc="overall progress", unit="stage") as overall:
    download_reference_weights()
    overall.update()
    ref_state_dict = load_reference_state_dict()
    overall.update()
    ref_state_dict_for_checks = ref_state_dict.copy()
    flat_state_dict = build_flat_state_dict(ref_state_dict)
    overall.update()
    run_sanity_checks(ref_state_dict_for_checks, flat_state_dict)
    overall.update()
    clear_existing_tensors()
    weight_files = save_state_dict(flat_state_dict)
    overall.update()
    commit_info = upload_files(weight_files)
    overall.update()
  tqdm.write(f"saved {WEIGHTS_PATH}")
  tqdm.write(f"uploaded to {HF_REPO_ID}")
  tqdm.write(f"commit: {commit_info.oid}")


if __name__ == "__main__":
  main()
