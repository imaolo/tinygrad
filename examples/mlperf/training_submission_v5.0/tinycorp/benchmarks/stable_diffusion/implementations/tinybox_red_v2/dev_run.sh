#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

DATETIME=${2:-$(date "+%m%d%H%M")}
LOGFILE="${HOME}/logs/sd_red_v2_${DATETIME}.log"
# UNET_CKPTDIR must be set: training saves checkpoints to this path, then a separate eval process scans this path to know which checkpoints to eval
export UNET_CKPTDIR="${HOME}/stable_diffusion/training_checkpoints/${DATETIME}"
mkdir -p "${HOME}/logs" "$UNET_CKPTDIR"

# run this script in isolation when using the --bg flag
if [[ "${1:-}" == "--bg" ]]; then
  echo "logging output to $LOGFILE"
  echo "saving UNet checkpoints to $UNET_CKPTDIR"
  script_path="$(readlink -f "${BASH_SOURCE[0]}")"
  nohup bash "$script_path" run "$DATETIME" >"$LOGFILE" 2>&1 & disown $!
  exit 0
fi

if [[ ! -d .venv-sd-mlperf ]]; then
  bash examples/mlperf/training_submission_v5.0/tinycorp/benchmarks/stable_diffusion/implementations/tinybox_red_v2/setup.sh
fi
. .venv-sd-mlperf/bin/activate

export BEAM=2 BEAM_UOPS_MAX=8000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5 IGNORE_JIT_FIRST_BEAM=1 HCQDEV_WAIT_TIMEOUT_MS=300000
export AMD_LLVM=0
export DATADIR="${DATADIR:-/raid/datasets/stable_diffusion}"
export CKPTDIR="${CKPTDIR:-/raid/weights/stable_diffusion}"
export EVAL_CKPT_DIR="$UNET_CKPTDIR"
export MODEL="stable_diffusion" PYTHONPATH="." AMD=1
export GPUS="${GPUS:-4}" BS="${BS:-16}"
export UNET_FSDP="${UNET_FSDP:-1}" OFFLOAD_OPTIM="${OFFLOAD_OPTIM:-1}"
export CONTEXT_BS="${CONTEXT_BS:-64}" DENOISE_BS="${DENOISE_BS:-8}" DECODE_BS="${DECODE_BS:-12}" INCEPTION_BS="${INCEPTION_BS:-64}" CLIP_BS="${CLIP_BS:-32}"
export WANDB="${WANDB:-1}"
export PARALLEL="${PARALLEL:-4}"
export PYTHONUNBUFFERED=1

run_retry(){ local try=0 max=5 code tmp py pgid kids
  while :; do
    tmp=$(mktemp)
    setsid bash -c 'exec env "$@"' _ "$@" > >(tee -a "$LOGFILE" | tee "$tmp") 2>&1 &
    py=$!; pgid=$(ps -o pgid= -p "$py" | tr -d ' ')
    wait "$py"; code=$?
    [[ -n "$pgid" ]] && { kill -TERM -"$pgid" 2>/dev/null; sleep 1; kill -KILL -"$pgid" 2>/dev/null; }
    kids=$(pgrep -P "$py" || true)
    while [[ -n "$kids" ]]; do
      kill -TERM $kids 2>/dev/null; sleep 0.5
      kids=$(for k in $kids; do pgrep -P "$k" || true; done)
    done
    grep -q 'BEAM COMPLETE' "$tmp" && { rm -f "$tmp"; return 1; }
    rm -f "$tmp"
    ((code==0)) && return 0
    ((try>=max)) && return 2
    ((try++)); sleep 90; echo "try = ${try}"
  done
}

run_retry TOTAL_CKPTS="${TOTAL_CKPTS:-7}" python3 examples/mlperf/model_train.py; (( $? == 2 )) && { echo "training failed before BEAM completion"; exit 2; }
sleep 90

run_retry EVAL_SAMPLES="${EVAL_SAMPLES:-600}" python3 examples/mlperf/model_eval.py; (( $? == 2 )) && { echo "eval failed before BEAM completion"; exit 2; }
STOP_IF_CONVERGED=1 python3 examples/mlperf/model_eval.py

