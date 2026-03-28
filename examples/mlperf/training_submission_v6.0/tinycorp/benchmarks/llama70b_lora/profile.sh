#!/bin/bash
export BENCHMARK=5
export EVAL_BS=0
export FAKEDATA=1
export LOAD_MODEL=0
export WANDB=0
export ZEROS=1
export RESOLVE_MODEL_CPU=0
export DEV="${DEV:-CUDA}"
VIZ=${VIZ:--1} exec bash examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama70b_lora/dev_run_vastai.sh
