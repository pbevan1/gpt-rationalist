# Epistemic Status: Doing The Thing

LoRA fine-tuning for rationalist/EA verbal mannerisms on everyday subjects.
The target is phrasing, not beliefs or impersonation of an author.
The default model is Qwen3.5-4B with BF16 LoRA; model settings and training
hyperparameters live in [config.json](config.json).

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone:

```bash
git clone https://github.com/pbevan1/gpt-rationalist.git
cd gpt-rationalist
uv sync --locked
uv run --locked prepare.py
uv run --locked train.py --check
```

uv manages Python 3.12 and dependencies pinned in `uv.lock`. Preparation downloads
only the tokenizer and builds `data/prepared/` and `data/tokenizer/`; it needs no GPU.

## Train

On your GPU machine, clone the repository and run the setup above. Training requires
Linux x86-64 and a BF16-capable NVIDIA GPU with a driver compatible with CUDA 12.8.
Budget approximately 24 GB VRAM and 40 GB disk; these are estimates, not measured
full-model requirements.

```bash
uv sync --locked --group train
uv run --locked --group train train.py --max-steps 5 --output outputs/smoke
uv run --locked --group train train.py
```

The smoke test performs five real training steps in a separate directory. The full
run saves checkpoints and the best validation-loss adapter under
`outputs/rationalist-lora/adapter`.

To resume, supply an existing checkpoint path:

```bash
uv run --locked --group train train.py --resume outputs/rationalist-lora/checkpoint-N
```

Replace `checkpoint-N` with the actual directory. A nonempty output directory requires
`--resume` or a different `--output`. Keep the whole output folder, including
`run_config.json`, when moving a trained adapter.

## Use and evaluate

```bash
uv run --locked --group train chat.py --prompt "I keep researching bread recipes instead of baking anything."
uv run --locked --group train chat.py --base --eval-file eval_prompts.jsonl --output outputs/base.jsonl
uv run --locked --group train chat.py --eval-file eval_prompts.jsonl --output outputs/styled.jsonl
uv run --locked evaluate.py outputs/base.jsonl outputs/styled.jsonl
```

Read `outputs/comparison.md` to compare style, helpfulness, and compliance with plain
language requests. Phrase counts and validation loss are diagnostics, not quality
scores. Each chat prompt starts a fresh conversation.

## Data and changes

[style_pairs.jsonl](style_pairs.jsonl) contains synthetic everyday prompt/answer
pairs. These let the experiment vary phrasing independently of the topics in source
essays, but may exaggerate the style. Collected posts are references only; see
[SOURCES.md](SOURCES.md) for provenance.

`response` is the training target. `plain_response` is for reviewing meaning
preservation and is not used in training. Keep related prompt variants in the same
`group` so they stay together in the train/validation split. Only assistant answers
and their end-of-turn token receive loss.

After editing the pairs or model/tokenizer settings in `config.json`, rerun
`uv run --locked prepare.py` and `uv run --locked train.py --check`. When changing
models, also update the pinned model revision.

Run the data and tiny-model tests with:

```bash
uv run --locked --group train -m unittest discover -s tests -v
```
