#!/usr/bin/env bash

export PYTHONPATH="."
export DEFAULT_FLOAT="bfloat16" OPTIM_DTYPE="bfloat16"
export MODEL="llama2_70b_lora"
export NULL_ALLOW_COPYOUT=1

export FAKEDATA="${FAKEDATA:-0}"
export DEV="${DEV:-NULL}"
export BS="${BS:-2}"
export MP="${MP:-4}"
export FLAT=${FLAT:-4}
export FUSE_WQKV=${FUSE_WQKV:-1}
export VIZ=${VIZ:-0}
export JITBEAM=${JITBEAM:-3}
export WANDB=${WANDB:-0}
export WANDB_PROJ='MLPerf-llama2_70b_lora'

# export CC=/opt/homebrew/opt/llvm@18/bin/clang
.venv/bin/python -u examples/mlperf/model_train.py