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
export GRADIENT_ACC_STEPS=${GRADIENT_ACC_STEPS:-4}

export ALL2ALL=1
export BEAM_UOPS_MAX=6000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5 BEAM_PADTO=1

# export CC=/opt/homebrew/opt/llvm@18/bin/clang
.venv/bin/python -u examples/mlperf/model_train.py