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
  ref_state_dict = {k:v for weight_file in ref_weight_paths for k,v in safe_load(weight_file).items()}
  return convert_from_huggingface(ref_state_dict, LLAMA2_70B_ARGS["n_layers"], LLAMA2_70B_ARGS["n_heads"], LLAMA2_70B_ARGS["n_kv_heads"])


def tensor_from_numpy(arr:np.ndarray, like:Tensor) -> Tensor:
  out = Tensor(arr)
  return out.cast(like.dtype) if out.dtype != like.dtype else out


def pop_stacked_layers(state_dict:dict[str, Tensor], key_fmt:str) -> Tensor:
  n_layers = LLAMA2_70B_ARGS["n_layers"]
  layers = [state_dict.pop(key_fmt.format(i=i)) for i in range(n_layers)]
  return tensor_from_numpy(np.stack([layer.numpy() for layer in layers]), layers[0])


def build_flat_state_dict(ref_state_dict:dict[str, Tensor]) -> dict[str, Tensor]:
  rope_theta = LLAMA2_70B_ARGS.get("rope_theta", 10000)
  norm_weight = ref_state_dict.pop("norm.weight")
  output_weight = ref_state_dict.pop("output.weight")
  tok_embeddings_weight = ref_state_dict.pop("tok_embeddings.weight")
  flat_state_dict = {
    "attention_norm": pop_stacked_layers(ref_state_dict, "layers.{i}.attention_norm.weight"),
    "ffn_norm": pop_stacked_layers(ref_state_dict, "layers.{i}.ffn_norm.weight"),
    "freqs_cis": precompute_freqs_cis(LLAMA2_70B_ARGS["dim"] // LLAMA2_70B_ARGS["n_heads"], MAX_CONTEXT * 2, rope_theta).contiguous().requires_grad_(False),
    "norm.weight": tensor_from_numpy(norm_weight.numpy(), norm_weight),
    "output": tensor_from_numpy(np.expand_dims(output_weight.numpy(), axis=0), output_weight),
    "tok_embeddings.weight": tensor_from_numpy(tok_embeddings_weight.numpy(), tok_embeddings_weight),
    "w1": pop_stacked_layers(ref_state_dict, "layers.{i}.feed_forward.w1.weight"),
    "w2": pop_stacked_layers(ref_state_dict, "layers.{i}.feed_forward.w2.weight"),
    "w3": pop_stacked_layers(ref_state_dict, "layers.{i}.feed_forward.w3.weight"),
    "wo": pop_stacked_layers(ref_state_dict, "layers.{i}.attention.wo.weight"),
    "wqkv": pop_stacked_layers(ref_state_dict, "layers.{i}.attention.wqkv.weight"),
  }
  w1, w3 = flat_state_dict.pop("w1"), flat_state_dict.pop("w3")
  flat_state_dict["w13"] = tensor_from_numpy(np.concatenate([w1.numpy(), w3.numpy()], axis=1), w1)
  if leftover := sorted(ref_state_dict.keys()):
    raise RuntimeError(f"unused weights after flattening: {leftover}")
  return flat_state_dict


def clear_existing_tensors() -> None:
  WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
  existing_tensors = sorted(WEIGHTS_PATH.glob("*.safetensors"))
  for tensor_path in existing_tensors:
    tensor_path.unlink()
  print(f"cleared {len(existing_tensors)} existing tensor files from {WEIGHTS_PATH}")


def save_state_dict(state_dict:dict[str, Tensor]) -> list[Path]:
  weight_files = [WEIGHTS_PATH / f"{name}.safetensors" for name in state_dict.keys()]
  for file_name, (name, tensor) in tqdm(zip(weight_files, state_dict.items()), total=len(weight_files), desc="saving flat weight shards"):
    safe_save({name: tensor}, file_name)
  return weight_files


def upload_files(files:list[Path]):
  print(f"uploading flat model weights to {HF_REPO_ID}")
  api = HfApi()
  api.create_repo(repo_id=HF_REPO_ID, exist_ok=True)
  remote_files = api.list_repo_files(repo_id=HF_REPO_ID)
  if remote_files:
    print(f"deleting {len(remote_files)} existing remote files from {HF_REPO_ID}")
    api.create_commit(
      repo_id=HF_REPO_ID,
      operations=[CommitOperationDelete(path_in_repo=path) for path in remote_files],
      commit_message="Delete existing repo contents before uploading rebuilt flat weights",
    )
  return api.upload_folder(
    folder_path=WEIGHTS_PATH,
    repo_id=HF_REPO_ID,
    allow_patterns=[p.name for p in files],
    commit_message=f"Uploaded {len(files)} rebuilt flat weights",
  )


def main() -> None:
  download_reference_weights()
  ref_state_dict = load_reference_state_dict()
  flat_state_dict = build_flat_state_dict(ref_state_dict)
  clear_existing_tensors()
  weight_files = save_state_dict(flat_state_dict)
  commit_info = upload_files(weight_files)
  print(f"saved {WEIGHTS_PATH}")
  print(f"uploaded to {HF_REPO_ID}")
  print(f"commit: {commit_info.oid}")


if __name__ == "__main__":
  main()
