#!/bin/bash
export BENCHMARK=5
export EVAL_BS=0
export WANDB=0
export CACHE_MODEL=${CACHE_MODEL:-1}
VIZ=${VIZ:--1} exec bash examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama70b_lora/dev_run_vastai.sh
