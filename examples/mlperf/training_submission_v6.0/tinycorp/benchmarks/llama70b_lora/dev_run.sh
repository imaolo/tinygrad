#!/usr/bin/env bash

export PYTHONPATH="."
export MODEL="llama2_70b_lora"
export FAKEDATA=1
export LORA=1
export DEFAULT_FLOAT="fp8e4m3" OPTIM_DTYPE="fp8e4m3"
export DEV=NULL
export NULL_ALLOW_COPYOUT=1
export BS=2
export FLAT=1
export MP=4
export MAX_STEPS=16
export WARMUP_STEPS=1

.venv/bin/python examples/mlperf/model_train.py


