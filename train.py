"""Single-GPU BF16 LoRA training; --check validates data without loading weights."""
import argparse
import json
from pathlib import Path

from common import config, digest, read_jsonl


def validate_data(cfg, root="data/prepared"):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(Path('style_pairs.jsonl').read_bytes()) != manifest['pairs_sha256']:
        raise ValueError('Style pairs changed; rerun prepare.py')
    for name, expected in manifest['tokenizer_sha256'].items():
        if digest((Path('data/tokenizer') / name).read_bytes()) != expected:
            raise ValueError(f'Tokenizer file hash differs: {name}')
    for key in ("model", "revision", "max_length", "system_prompt"):
        if cfg[key] != manifest["config"][key]:
            raise ValueError(f"{key} changed since preparation; rerun prepare.py")
    result, groups, urls = {}, {}, {}
    for split in ("train", "validation"):
        path = root / f"{split}.jsonl"
        if digest(path.read_bytes()) != manifest["splits"][split]["sha256"]:
            raise ValueError(f"{split} data hash differs from manifest")
        rows = read_jsonl(path)
        docs = read_jsonl(root / f"{split}_documents.jsonl")
        if digest((root / f'{split}_documents.jsonl').read_bytes()) != manifest['splits'][split]['documents_sha256']:
            raise ValueError(f'{split} document manifest hash differs')
        if not rows:
            raise ValueError(f"Empty {split}")
        groups[split] = {r["document_id"] for r in rows}
        urls[split] = {ref["url"] for doc in docs for ref in doc["references"]}
        for row in rows:
            ids, labels, mask = row["input_ids"], row["labels"], row["attention_mask"]
            if not (len(ids) == len(labels) == len(mask) <= cfg["max_length"]):
                raise ValueError(f"Invalid lengths: {row['id']}")
            boundary = next((i for i, t in enumerate(labels) if t != -100), None)
            if boundary is None or boundary == 0 or labels[boundary:] != ids[boundary:]:
                raise ValueError(f"Invalid assistant mask: {row['id']}")
        result[split] = [{k: row[k] for k in ("input_ids", "attention_mask", "labels")} for row in rows]
    if groups["train"] & groups["validation"] or urls["train"] & urls["validation"]:
        raise ValueError("Document/crosspost leakage between train and validation")
    return result, manifest


class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch
        length = max(len(r["input_ids"]) for r in rows)
        return {k: torch.tensor([r[k] + [pad] * (length - len(r[k])) for r in rows])
                for k, pad in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100))}


def add_lora(model, cfg):
    import torch
    from peft import LoraConfig, get_peft_model
    # Full checkpoint layout; select text transformer linears only, including
    # DeltaNet's in_proj_* / out_proj. Never adapt vision or the huge LM head.
    targets = [name for name, module in model.named_modules()
               if ".language_model.layers." in name and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise ValueError("No text LoRA targets found; check the model architecture")
    return get_peft_model(model, LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.05,
        bias="none", target_modules=targets, task_type="CAUSAL_LM",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", default="outputs/rationalist-lora")
    parser.add_argument("--max-steps", type=int, default=-1, help="Use 5 for the first cloud smoke run")
    parser.add_argument("--resume", help="Explicit checkpoint directory")
    args = parser.parse_args()
    cfg = config(args.config)
    data, manifest = validate_data(cfg)
    print(json.dumps(manifest["splits"], indent=2))
    if args.check:
        print("Data, masks, hashes, model configuration, and document split: OK")
        return

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, Trainer, TrainingArguments, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("Training requires a CUDA GPU with BF16 support (e.g. RTX 4090, A10, L4, A100).")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit("Output directory is nonempty. Use --resume or a new --output.")
    set_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained("data/tokenizer", local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg["model"], revision=cfg["revision"], dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0},
    )
    model.config.text_config.use_cache = False
    model = add_lora(model, cfg)
    model.print_trainable_parameters()
    # Non-reentrant checkpointing works with frozen embeddings and PEFT.
    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=cfg["epochs"], max_steps=args.max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"], warmup_ratio=0.03, lr_scheduler_type="cosine",
        weight_decay=0.01, max_grad_norm=1.0, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", logging_steps=1, eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        prediction_loss_only=True, report_to="none", seed=cfg["seed"],
        dataloader_num_workers=0, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=data["train"], eval_dataset=data["validation"],
                      data_collator=Collator(tokenizer.pad_token_id), processing_class=tokenizer)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    trainer.save_metrics("train", result.metrics)
    trainer.save_metrics("eval", trainer.evaluate())
    trainer.save_state()
    print(f"Adapter saved to {output / 'adapter'}")


if __name__ == "__main__":
    main()
