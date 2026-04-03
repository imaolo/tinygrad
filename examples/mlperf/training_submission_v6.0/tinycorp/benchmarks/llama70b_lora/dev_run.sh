#!/usr/bin/env bash

export PYTHONPATH="."
export DEFAULT_FLOAT="bfloat16" OPTIM_DTYPE="bfloat16"
export MODEL="llama2_70b_lora"
export NULL_ALLOW_COPYOUT=1
export LORA_DTYPE='float32'
export OFFLOAD_OPTIM=${OFFLOAD_OPTIM:-1}
export FP8=${FP8:-0}

export FAKEDATA="${FAKEDATA:-0}"
export DEV="${DEV:-NULL}"
export BS="${BS:-1}"
export MP="${MP:-4}"
export FLAT=${FLAT:-1}
export FUSE_WQKV=${FUSE_WQKV:-1}
export VIZ=${VIZ:-0}
export ZEROS=${ZEROS:-0}
export JITBEAM=${JITBEAM:-3}
export WANDB=${WANDB:-0}
export WANDB_PROJ='MLPerf-llama2_70b_lora'
export GRADIENT_ACC_STEPS=${GRADIENT_ACC_STEPS:-8}
export CACHE_MODEL=${CACHE_MODEL:-0}
export QUANTIZE_LOADED_WEIGHTS=${QUANTIZE_LOADED_WEIGHTS:-1}

export ALL2ALL=${ALL2ALL:-1}
export BEAM_UOPS_MAX=6000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5 BEAM_PADTO=1

export GBS=$((BS * GRADIENT_ACC_STEPS))

export LR=${LR:-4e-4}
export END_LR=${END_LR:-0}
export MAX_STEPS=${MAX_STEPS:-1024}
export WARMUP_STEPS=${WARMUP_STEPS:-1}
export SAMPLES=${SAMPLES:-$((MAX_STEPS * GBS))}

export EVAL_BS=${EVAL_BS:-1}
export EVAL_FREQ=${EVAL_FREQ:-384}

export ADAM_BETA_1=${ADAM_BETA_1:-0.9}
export ADAM_BETA_2=${ADAM_BETA_2:-0.999}
export ADAM_EPSILON=${ADAM_EPSILON:-1e-8}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.3}

export EVAL_TARGET="${EVAL_TARGET:-0.925}"
export LOAD_MODEL="${LOAD_MODEL:-0}"

.venv/bin/python -u examples/mlperf/model_train.py