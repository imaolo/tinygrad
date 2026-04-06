#!/bin/bash
export MAX_STEPS=5
export EVAL_BS=0
export ZEROS=1
export FAKEDATA=1
export WANDB=0
export TARGETED_PROFILE=1
export LOAD_MODEL=0
export RESOLVE_MODEL_CPU=0
exec bash examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama70b_lora/dev_run_vastai.sh
extra/viz/cli.py --profile -s "${DEV:-CUDA}" | head -23
