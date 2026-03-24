#!/usr/bin/env python3
from pathlib import Path
from huggingface_hub import snapshot_download
from tinygrad.helpers import getenv

def main() -> None:
  download_dir = Path(getenv("DOWNLOAD_DIR", Path(__file__).parent/"llama2_70b_lora/dataset"))
  download_dir.mkdir(parents=True, exist_ok=True)

  repo_id = getenv("REPO_ID", "regisss/scrolls_gov_report_preprocessed_mlperf_2")

  path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=download_dir)

  print(f"\ndownloaded dataset to {path}")

if __name__ == "__main__":
  main()
