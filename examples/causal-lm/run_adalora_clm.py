#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from datasets import load_dataset
from peft import AdaLoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments


@dataclass
class RunResult:
    target_r: int
    test_perplexity: float
    output_dir: str


def _require_keys(d: Dict[str, Any], keys: List[str], group_name: str) -> None:
    miss = [k for k in keys if k not in d]
    if miss:
        raise ValueError(f"Missing keys in '{group_name}': {miss}")


def _validate_config(cfg: Dict[str, Any]) -> None:
    _require_keys(cfg, ["model", "data", "peft", "training", "search", "output"], "root")
    _require_keys(cfg["model"], ["model_name_or_path", "tokenizer_name_or_path", "torch_dtype", "gradient_checkpointing", "local_files_only"], "model")
    _require_keys(cfg["data"], ["dataset_name", "dataset_config_name", "block_size", "ppl_max_tokens", "seq_len", "ga_seq_len", "ppl_max_len"], "data")
    _require_keys(cfg["peft"], ["target_modules", "init_r", "tinit", "tfinal", "deltaT", "beta1", "beta2", "lora_alpha", "lora_dropout", "orth_reg_weight"], "peft")
    _require_keys(cfg["training"], ["num_train_epochs", "per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps", "learning_rate", "logging_steps", "save_steps", "eval_steps", "seed", "fp16"], "training")
    _require_keys(cfg["search"], ["target_r_values"], "search")
    _require_keys(cfg["output"], ["root_dir"], "output")

    if cfg["data"]["ppl_max_tokens"] != 4096:
        raise ValueError("ppl_max_tokens must be fixed to 4096 by project decision.")
    if cfg["training"]["seed"] != 42:
        raise ValueError("seed must be fixed to 42 by project decision.")

    peft = cfg["peft"]
    if peft["tinit"] >= peft["tfinal"]:
        raise ValueError("Require tinit < tfinal")


def _set_offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def _load_tokenizer(cfg: Dict[str, Any]):
    model_cfg = cfg["model"]
    tokenizer_path = model_cfg["tokenizer_name_or_path"] or model_cfg["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=model_cfg.get("local_files_only", True), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_and_tokenize(cfg: Dict[str, Any], tokenizer: Any):
    ds_cfg = cfg["data"]
    raw = load_dataset(ds_cfg["dataset_name"], ds_cfg["dataset_config_name"], cache_dir=ds_cfg.get("cache_dir"))

    def tokenize(examples):
        text = [t for t in examples["text"] if t and not t.isspace()]
        return tokenizer(text)

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)
    block_size = ds_cfg["block_size"]

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated.items()}
        result["labels"] = deepcopy(result["input_ids"])
        return result

    return tokenized.map(group_texts, batched=True)


def _estimate_total_steps(cfg: Dict[str, Any], train_size: int) -> int:
    t = cfg["training"]
    per_device_bs = max(1, int(t["per_device_train_batch_size"]))
    grad_acc = max(1, int(t["gradient_accumulation_steps"]))
    steps_per_epoch = math.ceil(train_size / per_device_bs / grad_acc)
    return max(1, int(t["num_train_epochs"]) * steps_per_epoch)


def _resolve_schedule(peft_cfg: Dict[str, Any], total_steps: int) -> Dict[str, int]:
    tinit = int(peft_cfg["tinit"])
    tfinal = int(peft_cfg["tfinal"])
    delta_t = int(peft_cfg["deltaT"])

    # PEFT AdaLoRA requires a non-empty budgeting phase.
    # Keep at least one allocation interval in the middle phase.
    min_budget_phase = max(1, delta_t)
    max_warmup_total = max(0, total_steps - min_budget_phase)

    if tinit + tfinal <= max_warmup_total:
        return {"tinit": tinit, "tfinal": tfinal, "delta_t": delta_t}

    if max_warmup_total <= 0:
        raise ValueError(
            f"total_step={total_steps} is too small for AdaLoRA scheduling "
            f"(deltaT={delta_t}). Increase training steps or reduce deltaT."
        )

    # Keep original warmup ratio when shrinking.
    ratio = tinit / max(1, (tinit + tfinal))
    new_tinit = int(max_warmup_total * ratio)
    new_tfinal = max_warmup_total - new_tinit

    # Ensure strict tinit < tfinal and non-negative values.
    if new_tinit >= new_tfinal:
        new_tinit = max(0, (max_warmup_total - 1) // 2)
        new_tfinal = max_warmup_total - new_tinit
    if new_tinit >= new_tfinal:
        raise ValueError(
            "Unable to derive a valid AdaLoRA schedule. "
            f"total_step={total_steps}, requested tinit={tinit}, tfinal={tfinal}, deltaT={delta_t}."
        )

    print(
        "[AdaLoRA] Adjust schedule for available total_step: "
        f"tinit {tinit}->{new_tinit}, tfinal {tfinal}->{new_tfinal}, total_step={total_steps}, deltaT={delta_t}"
    )
    return {"tinit": new_tinit, "tfinal": new_tfinal, "delta_t": delta_t}


def _build_model(cfg: Dict[str, Any], tokenizer: Any):
    model_cfg = cfg["model"]
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name_or_path"],
        dtype=getattr(torch, model_cfg.get("torch_dtype", "float16")),
        local_files_only=model_cfg.get("local_files_only", True),
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if model_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    return model


def _run_single_target_r(
    cfg: Dict[str, Any],
    target_r: int,
    run_root: Path,
    lm_datasets: Any,
    tokenizer: Any,
    total_steps: int,
    schedule: Dict[str, int],
) -> RunResult:
    model = _build_model(cfg, tokenizer)

    peft_cfg = cfg["peft"]
    adalora = AdaLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=peft_cfg["target_modules"],
        init_r=peft_cfg["init_r"],
        target_r=target_r,
        tinit=schedule["tinit"],
        tfinal=schedule["tfinal"],
        deltaT=schedule["delta_t"],
        beta1=peft_cfg["beta1"],
        beta2=peft_cfg["beta2"],
        lora_alpha=peft_cfg["lora_alpha"],
        lora_dropout=peft_cfg["lora_dropout"],
        orth_reg_weight=peft_cfg["orth_reg_weight"],
        total_step=total_steps,
    )
    model = get_peft_model(model, adalora)

    training_cfg = cfg["training"]
    out_dir = run_root / f"target_r_{target_r}"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        logging_steps=training_cfg["logging_steps"],
        save_steps=training_cfg["save_steps"],
        eval_steps=training_cfg["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        seed=training_cfg["seed"],
        fp16=training_cfg["fp16"],
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=training_cfg.get("resume_from_checkpoint", None))
    metrics = trainer.evaluate(eval_dataset=lm_datasets["test"])
    test_ppl = float(math.exp(metrics["eval_loss"]))
    metrics["test_perplexity"] = test_ppl

    trainer.save_model(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "config.snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return RunResult(target_r=target_r, test_perplexity=test_ppl, output_dir=str(out_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    _set_offline_env()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg)

    tokenizer = _load_tokenizer(cfg)
    lm_datasets = _load_and_tokenize(cfg, tokenizer)
    total_steps = _estimate_total_steps(cfg, len(lm_datasets["train"]))
    schedule = _resolve_schedule(cfg["peft"], total_steps)

    model_name = Path(cfg["model"]["model_name_or_path"]).name
    run_root = Path(cfg["output"]["root_dir"]) / model_name / "wikitext-2-raw-v1" / "adalora" / "seed_42"
    run_root.mkdir(parents=True, exist_ok=True)

    max_hours = cfg["training"].get("max_runtime_hours", 8)
    start = time.time()

    results: List[RunResult] = []
    for target_r in cfg["search"]["target_r_values"]:
        results.append(_run_single_target_r(cfg, int(target_r), run_root, lm_datasets, tokenizer, total_steps, schedule))

    elapsed_hours = (time.time() - start) / 3600
    overtime = max(0.0, elapsed_hours - max_hours)
    print(f"Total elapsed: {elapsed_hours:.3f} hours")
    if overtime > 0:
        print(f"Exceeded time budget by: {overtime:.3f} hours")

    summary_rows = [{"target_r": r.target_r, "test_perplexity": r.test_perplexity, "output_dir": r.output_dir} for r in sorted(results, key=lambda x: x.test_perplexity)]
    best = summary_rows[0]

    with open(run_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    with open(run_root / "summary.csv", "w", encoding="utf-8") as f:
        f.write("target_r,test_perplexity,output_dir\n")
        for row in summary_rows:
            f.write(f"{row['target_r']},{row['test_perplexity']},{row['output_dir']}\n")
    with open(run_root / "summary.md", "w", encoding="utf-8") as f:
        f.write("# AdaLoRA Search Summary\n\n")
        f.write("| rank | test ppl | output_dir |\n|---:|---:|---|\n")
        for row in summary_rows:
            f.write(f"| {row['target_r']} | {row['test_perplexity']:.6f} | `{row['output_dir']}` |\n")
        f.write("\n")
        f.write(f"Best target_r: **{best['target_r']}**\\\n\n")
        f.write(f"Best test ppl: **{best['test_perplexity']:.6f}**\n")

    print(f"Best target_r = {best['target_r']}, test ppl = {best['test_perplexity']:.6f}")


if __name__ == "__main__":
    main()
