#!/usr/bin/env bash

export DEV="${DEV:-CUDA}"
export WANDB=${WANDB:-1}
export DEBUG_LORA=${DEBUG_LORA:-0}
export RESOLVE_MODEL_CPU=${RESOLVE_MODEL_CPU:-1}
export CPU_OPT=1
export LOAD_MODEL=${LOAD_MODEL:-1}
export ZEROS=${ZEROS:-0}
export FAKEDATA=${FAKEDATA:-0}
export EVAL_BS=${EVAL_BS:-1}
export BENCHMARK=${BENCHMARK:-1}

exec bash examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama70b_lora/dev_run.sh