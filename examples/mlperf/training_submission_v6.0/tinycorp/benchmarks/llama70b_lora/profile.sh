#!/bin/bash
export BENCHMARK=5
export EVAL_BS=0
export LOAD_MODEL=1
export WANDB=0
export ZEROS=0
export RESOLVE_MODEL_CPU="${RESOLVE_MODEL_CPU:-1}"
export DEV="${DEV:-CUDA}"
export CACHE_MODEL=${CACHE_MODEL:-1}
VIZ=${VIZ:--1} exec bash examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama70b_lora/dev_run_vastai.sh
