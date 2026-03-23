#!/usr/bin/env python3
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download

def main() -> None:
  basedir = Path(__file__).parent/"llama2_70b_lora/dataset"
  basedir.mkdir(exist_ok=True, parents=True)
  repo_id = os.getenv("DATASET_REPO", "regisss/scrolls_gov_report_preprocessed_mlperf_2")
  outdir = Path(os.getenv("OUTDIR", str(basedir / "hf_dataset_repo")))

  outdir.parent.mkdir(parents=True, exist_ok=True)
  path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=outdir)

  summary = {
    "dataset_repo": repo_id,
    "downloaded_to": str(path),
  }

  with (outdir.parent / "dataset_info.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
  print(json.dumps(summary, indent=2))

if __name__ == "__main__":
  main()
