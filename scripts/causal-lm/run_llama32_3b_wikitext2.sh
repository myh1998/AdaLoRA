#!/usr/bin/env bash
set -euo pipefail

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision fp16 \
  --dynamo_backend no \
  examples/causal-lm/run_adalora_clm.py \
  --config configs/causal-lm/llama32_3b_wikitext2.yaml
