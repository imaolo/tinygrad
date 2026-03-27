#!/usr/bin/env bash

export DEV="${DEV:-CUDA}"
export WANDB=${WANDB:-1}
export DEBUG_LORA=1
export RESOLVE_MODEL_CPU=1
export CPU_OPT=1

exec bash examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama70b_lora/dev_run.sh