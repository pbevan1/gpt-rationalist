"""Small shared helpers; run all commands from this directory."""
import hashlib
import json
from pathlib import Path


def config(path="config.json"):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def digest(value):
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def encode_example(tokenizer, messages):
    """Use the real non-thinking template; supervise only answer + end-of-turn."""
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    if not full.startswith(prompt):
        raise ValueError("Chat template does not preserve the non-thinking prompt prefix")
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if ids[:len(prefix)] != prefix:
        raise ValueError("Tokenization crosses the prompt/completion boundary")
    labels = [-100] * len(prefix) + ids[len(prefix):]
    if not any(x != -100 for x in labels):
        raise ValueError("Example has no supervised tokens")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
