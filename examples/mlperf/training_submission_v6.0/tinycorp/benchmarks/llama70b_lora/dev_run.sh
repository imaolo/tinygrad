#!/usr/bin/env bash

export PYTHONPATH="."
export MODEL="llama2_70b_lora"
export LORA=1

.venv/bin/python3 examples/mlperf/model_train.py


