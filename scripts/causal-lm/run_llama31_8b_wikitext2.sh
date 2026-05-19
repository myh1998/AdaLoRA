#!/usr/bin/env bash
set -euo pipefail

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
accelerate launch examples/causal-lm/run_adalora_clm.py \
  --config configs/causal-lm/llama31_8b_wikitext2.yaml
