# AdaLoRA Causal LM (Single GPU, Offline, Accelerate)

This directory provides a simple single-GPU entrypoint for AdaLoRA hyperparameter search on WikiText-2 test perplexity.

Final ranking uses a **common sliding-window evaluator** (`test_perplexity_common`) for alignment with other methods.

## Install

```bash
pip install -r requirements-modern.txt
```

## Run (3B first)

```bash
bash scripts/causal-lm/run_llama32_3b_wikitext2.sh
```

## Run (8B second)

```bash
bash scripts/causal-lm/run_llama31_8b_wikitext2.sh
```

## Resume

Set `training.resume_from_checkpoint` in the chosen YAML to a checkpoint directory and rerun the same command.

## Outputs

Outputs are stored under:

```text
outputs/<model_name>/wikitext-2-raw-v1/adalora/seed_42/
```

Each `target_r_*` directory contains:
- `adapter/` (adapter weights)
- `metrics.json`
- `config.snapshot.yaml`
- TensorBoard logs

Search-level summary files:
- `summary.json`
- `summary.csv`
- `summary.md`

Each `target_r_*` directory also stores:
- `time_to_threshold.csv` with:
  - `runtime_sec,step,trainer_eval_loss,trainer_eval_ppl,test_ppl_common`
  - `init_r,target_r,effective_rank_total,effective_rank_avg,max_effective_rank_total_seen`
  - `num_adapted_matrices,initial_total_rank_budget,final_total_rank_budget`
  - `effective_rank_min,effective_rank_max,rank_pattern_json`

The best result is defined as lowest **test_perplexity_common**.

## Tuning step count and curve logging frequency

- To change total optimization length, edit `training.max_steps` in the YAML config (currently `3000`).
- To change curve logging cadence, edit `training.eval_steps` in the YAML config (currently `10`).
