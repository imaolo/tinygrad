#!/usr/bin/env bash
set -euo pipefail

# MLPerf llama2_70b_lora uses SCROLLS GovReport.
# The official MLCommons-hosted preprocessed dataset is member-only; this script
# downloads the public Hugging Face SCROLLS GovReport release and writes a
# normalized JSONL layout that is convenient for tinygrad-side preprocessing.

BASEDIR="${BASEDIR:-/raid/datasets/llama2_70b_lora}"
DATASET_NAME="${DATASET_NAME:-tau/scrolls}"
DATASET_CONFIG="${DATASET_CONFIG:-gov_report}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$BASEDIR/.cache/huggingface}"
OUTDIR="${OUTDIR:-$BASEDIR/scrolls_gov_report}"

mkdir -p "$HF_CACHE_DIR" "$OUTDIR"

export DATASET_NAME DATASET_CONFIG OUTDIR
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN
fi
export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"

python3 - <<'PY'
import json
import os
from pathlib import Path

try:
  from datasets import load_dataset
except ImportError as exc:
  raise SystemExit(
    "missing dependency: install the Hugging Face datasets package first "
    "(`pip install datasets`)."
  ) from exc

dataset_name = os.environ["DATASET_NAME"]
dataset_config = os.environ["DATASET_CONFIG"]
outdir = Path(os.environ["OUTDIR"])

ds = load_dataset(dataset_name, dataset_config)
outdir.mkdir(parents=True, exist_ok=True)

summary = {
  "dataset_name": dataset_name,
  "dataset_config": dataset_config,
  "splits": {},
}

for split_name, split in ds.items():
  records_path = outdir / f"{split_name}.jsonl"
  prompts_path = outdir / f"{split_name}_prompts.jsonl"

  with records_path.open("w", encoding="utf-8") as records_f, prompts_path.open("w", encoding="utf-8") as prompts_f:
    for idx, row in enumerate(split):
      record = {
        "id": row.get("id", f"{split_name}-{idx}"),
        "pid": row.get("pid"),
        "input": row["input"],
        "output": row.get("output"),
      }
      records_f.write(json.dumps(record, ensure_ascii=False) + "\n")

      prompt_record = {
        "id": record["id"],
        "prompt": f"### Summarize the following text:\n{row['input']}\n### Summary:\n",
        "output": row.get("output"),
      }
      prompts_f.write(json.dumps(prompt_record, ensure_ascii=False) + "\n")

  summary["splits"][split_name] = {
    "rows": len(split),
    "records_path": str(records_path),
    "prompts_path": str(prompts_path),
  }

with (outdir / "dataset_info.json").open("w", encoding="utf-8") as f:
  json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
PY

echo "Downloaded SCROLLS GovReport to $OUTDIR"
