from tqdm import tqdm
from tinygrad import Tensor
from huggingface_hub import HfApi
from tinygrad.nn.state import safe_load, safe_save
from extra.models.llama import convert_from_huggingface
from extra.huggingface_onnx.huggingface_manager import DOWNLOADS_DIR, snapshot_download_with_retry
from examples.mlperf.model_train import LLAMA2_70B_ARGS
from pathlib import Path

HF_REPO_ID = "imaolo/llama2-70b-fused-qkv-flat-mlperf"
HF_REF_REPO_ID = HF_REPO_ID

REF_WEIGHTS_PATH = DOWNLOADS_DIR/HF_REF_REPO_ID
WEIGHTS_PATH = DOWNLOADS_DIR/HF_REPO_ID

def save_state(state_dict:dict[str, Tensor]) -> list[Path]:
  WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
  weight_files = [WEIGHTS_PATH / f"{name}.safetensors" for name in state_dict.keys()]
  for file_name, (name, tensor) in tqdm(zip(weight_files, state_dict.items()), total=len(weight_files), desc="saving flat weight shards"):
    if file_name.is_file(): continue
    safe_save({name: tensor}, file_name)
  return weight_files

def download_files() -> list[Path]:
  print(f"downloading reference weights to {REF_WEIGHTS_PATH}")
  REF_WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
  snapshot_download_with_retry(repo_id=HF_REF_REPO_ID, local_dir=REF_WEIGHTS_PATH, allow_patterns=["*safetensors*", "*.json", "*.md"])
  print("downloaded reference weights")

def load_files():
  ref_weight_paths = list(REF_WEIGHTS_PATH.glob("*.safetensors"))
  ref_state_dict = {k:v for weight_file in ref_weight_paths for k,v in safe_load(weight_file).items()}
  return ref_state_dict

def upload_files(files: list[Path]):
  print(f"uploading flat model weights to {HF_REPO_ID}")
  api = HfApi()
  api.create_repo(repo_id=HF_REPO_ID, exist_ok=True)
  commit_info = api.upload_folder(
    folder_path=WEIGHTS_PATH,
    repo_id=HF_REPO_ID,
    allow_patterns=[p.name for p in files],
    commit_message=f"Uploaded {len(files)} flat weights",
  )
  return commit_info

def main() -> None:
  # download from hf
  download_files()

  # load weights into memory and combine w1 & w3
  state_dict = load_files()
  state_dict["w13"] = state_dict.pop("w1").to('CPU').cat(state_dict.pop("w3").to('CPU'), dim=1).realize()

  # save the cat version
  weight_files = save_state(state_dict)

  # upload new weights
  commit_info = upload_files(weight_files)

  print(f"saved {WEIGHTS_PATH}")
  print(f"uploaded to {HF_REPO_ID}")
  print(f"commit: {commit_info.oid}")

if __name__ == "__main__":
  main()
