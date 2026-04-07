import os
os.environ["DEFAULT_FLOAT"] = "bfloat16"
os.environ["OPTIM_DTYPE"] = "bfloat16"
os.environ["DEV"] = "CPU"
os.environ["WQKV"] = "1"
os.environ["LORA"] = "0"
os.environ["ZEROS"] = "1"

from tqdm import tqdm
from huggingface_hub import HfApi
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save
from extra.models.llama import convert_from_huggingface
from extra.huggingface_onnx.huggingface_manager import DOWNLOADS_DIR, snapshot_download_with_retry
from examples.mlperf.model_train import LLAMA2_70B_ARGS
from examples.mlperf.models.flat_llama import FlatTransformer
from examples.mlperf.models.test_flat_llama import copy_weights
from examples.mlperf.models.llama import Transformer

HF_REPO_ID = "imaolo/llama2-70b-fused-qkv-flat-mlperf"
HF_REF_REPO_ID = "regisss/llama2-70b-fused-qkv-mlperf"

REF_WEIGHTS_PATH = DOWNLOADS_DIR/HF_REF_REPO_ID
WEIGHTS_PATH = DOWNLOADS_DIR/HF_REPO_ID

def main() -> None:
  ## download reference weights
  print(f"downloading reference weights to {REF_WEIGHTS_PATH}")
  REF_WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
  snapshot_download_with_retry(repo_id=HF_REF_REPO_ID, local_dir=REF_WEIGHTS_PATH, allow_patterns=["*safetensors*", "*.json", "*.md"])
  print("downloaded reference weights")

  # load reference weights
  ref_weight_paths = list(REF_WEIGHTS_PATH.glob("*.safetensors"))
  assert len(ref_weight_paths) == 29
  ref_state_dict = {k:v for weight_file in ref_weight_paths for k,v in safe_load(weight_file).items()}
  ref_state_dict = convert_from_huggingface(ref_state_dict, LLAMA2_70B_ARGS["n_layers"], LLAMA2_70B_ARGS["n_heads"], LLAMA2_70B_ARGS["n_kv_heads"])

  # create reference model
  ref_model = Transformer(**(model_args:=LLAMA2_70B_ARGS | {'max_context':8192}))
  if unused := sorted(ref_state_dict.keys() - get_state_dict(ref_model).keys()):
    raise RuntimeError(f"unused weights in state_dict: {unused}")

  # load reference weights into reference model
  load_state_dict(ref_model, ref_state_dict, realize=False, strict=False, consume=True)

  # create flat model and copy reference weights into it
  flat_model = FlatTransformer(**model_args)
  print("copying reference model to flat model")
  copy_weights(flat_model, ref_model)
  print("copied reference model to flat model")
  del ref_model

  # save flat model weights
  flat_state_dict = get_state_dict(flat_model)
  WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
  weight_files = [WEIGHTS_PATH / f"{name}.safetensors" for name in flat_state_dict.keys()]
  for file_name, (name, tensor) in tqdm(zip(weight_files, flat_state_dict.items()), total=len(weight_files), desc="saving flat weight shards"):
    if file_name.is_file(): continue
    safe_save({name: tensor}, file_name)

  # upload flat model weights
  api = HfApi()
  api.create_repo(repo_id=HF_REPO_ID, exist_ok=True)
  commit_info = api.upload_folder(
    folder_path=WEIGHTS_PATH,
    repo_id=HF_REPO_ID,
    allow_patterns=[p.name for p in weight_files],
    commit_message=f"Uploaded {len(weight_files)} flat weights",
  )

  # done
  print(f"saved {WEIGHTS_PATH}")
  print(f"uploaded to {HF_REPO_ID}")
  print(f"commit: {commit_info.oid}")

if __name__ == "__main__":
  main()
