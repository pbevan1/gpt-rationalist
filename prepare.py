"""Prepare original, topic-neutral style pairs; reference essays are never training data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from transformers import AutoTokenizer
from common import config, digest, encode_example, read_jsonl, write_jsonl
from style import markers


def split_pairs(pairs, fraction, seed):
    # Related prompts and future paraphrases stay in the same split.
    groups = sorted({r["group"] for r in pairs}, key=lambda g: digest(f"{seed}:{g}"))
    held_out = set(groups[:max(1, round(len(groups) * fraction))])
    return {"train": [r for r in pairs if r["group"] not in held_out],
            "validation": [r for r in pairs if r["group"] in held_out]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    cfg = config(args.config)
    pairs = read_jsonl("style_pairs.jsonl")
    if len({r["id"] for r in pairs}) != len(pairs):
        raise ValueError("Duplicate pair IDs")
    if len({r["prompt"].strip().casefold() for r in pairs}) != len(pairs):
        raise ValueError("Duplicate prompts")
    for row in pairs:
        if not all(row.get(k) for k in ("id", "group", "prompt", "plain_response", "response")):
            raise ValueError(f"Incomplete style pair: {row}")
        if any(token in row[k] for token in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
               for k in ("prompt", "response")):
            raise ValueError(f"Chat control tokens in {row['id']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    tokenizer.save_pretrained("data/tokenizer")
    manifest = {"config": cfg, "dataset": "Original verbal-style pairs; no reference essay text",
                "pairs_sha256": digest(Path("style_pairs.jsonl").read_bytes()),
                "tokenizer_sha256": {p.name: digest(p.read_bytes()) for p in sorted(Path('data/tokenizer').iterdir()) if p.is_file()},
                "pairs": len(pairs), "splits": {}}
    samples, coverage = [], {}
    for split, rows in split_pairs(pairs, cfg["validation_fraction"], cfg["seed"]).items():
        examples, documents = [], []
        for row in rows:
            messages = [{"role": "system", "content": cfg["system_prompt"]},
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["response"]}]
            encoded = encode_example(tokenizer, messages)
            if len(encoded["input_ids"]) > cfg["max_length"]:
                raise ValueError(f"Example too long: {row['id']}")
            url = "local:style_pairs.jsonl#" + row["id"]
            examples.append({"id": row["id"], "document_id": row["group"], "source": "original_style_pairs",
                             "url": url, "messages": messages, **encoded})
            documents.append({"document_id": row["group"], "id": row["id"],
                              "references": [{"url": url, "author": "Project-original synthetic pair"}]})
        random.Random(cfg["seed"]).shuffle(examples)
        path = Path(f"data/prepared/{split}.jsonl")
        write_jsonl(path, examples)
        write_jsonl(f"data/prepared/{split}_documents.jsonl", documents)
        write_jsonl(f"data/prepared/{split}_pairs.jsonl", rows)
        manifest["splits"][split] = {
            "groups": len({r["group"] for r in rows}), "examples": len(examples),
            "tokens": sum(len(e["input_ids"]) for e in examples),
            "supervised_tokens": sum(sum(t != -100 for t in e["labels"]) for e in examples),
            "max_length": max(len(e["input_ids"]) for e in examples),
            "style_markers": dict(sum((Counter(markers(r["response"])) for r in rows), Counter())),
            "controls": sum(r.get("control", False) for r in rows), "sha256": digest(path.read_bytes()),
            "documents_sha256": digest(Path(f'data/prepared/{split}_documents.jsonl').read_bytes()),
        }
        coverage[split] = {name: {"responses": sum(markers(r["response"])[name] > 0 for r in rows),
                                 "example_ids": [r["id"] for r in rows if markers(r["response"])[name] > 0][:5]}
                           for name in markers("")}
        samples.extend({"split": split, **r} for r in rows[:8])
    write_jsonl("data/prepared/samples.jsonl", samples)
    Path("data/prepared/style_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    Path("data/prepared/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
