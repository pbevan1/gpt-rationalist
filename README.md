# Epistemic Status: Doing The Thing

A small LoRA project for **rationalist/EA verbal mannerisms on everyday subjects**:
Capitalised Phrases, “we'd be excited to…”, “epistemic status”, “orthogonal”,
“the crux”, “directionally right”, “on priors”, and other characteristic phrasing.
The target is diction, not beliefs, subject matter, or impersonation of an author.

**Default: Qwen3.5-4B, BF16 LoRA, one NVIDIA GPU.** Preparation is done; training
has deliberately not been started. The exact model revision and dependencies are pinned.

## What is ready

- `style_pairs.jsonl`: 296 original synthetic prompt/answer pairs, each with a
  plain answer for comparison. 264 use the mannerisms; 32 preserve explicit plain
  language, code, JSON, and short-answer instructions. These are authored examples,
  not scraped conversations or model-generated paraphrases of source essays.
- `data/prepared/`: 264 training and 32 validation examples, already tokenized.
  Related prompt families stay together. Only assistant answers and their end-of-turn
  token receive loss. The neutral system prompt doesn't tell the model to imitate a style.
- `data/tokenizer/`: the exact tokenizer and non-thinking chat template.
- `train.py`, `chat.py`, and `evaluate.py`: train, use, and compare the adapter.
- `data/prepared/style_coverage.json`: response counts and example IDs for each
  tracked expression family, separately for training and validation.
- `rationalist-lora-cloud.tar.gz`: transfer bundle, made by `uv run --locked pack.py`.

This is a deliberately small first experiment: 12,142 supervised training tokens,
three epochs, rank 16, learning rate 5e-5, effective batch size 16. The prose is
intentionally conspicuous and often playful. It is **not evidence that the adapter
already works**; the before/after comparison is how to assess that after training.

Example target:

> I'd be excited to see this happen and happy to give feedback. I don't have
> capacity to own it. To be explicit, enthusiasm should not be interpreted as a
> commitment to Do The Thing.

The expanded examples also cover “orthogonal”, “load-bearing”, “the crux”,
“operationalise”, “make legible”, “directionally”, “on the margin”, “nontrivial”,
“bottleneck”, “on priors”, “I'd update towards”, “conditional on”, “holding fixed”,
“cached thought”, “object-level”, “inside/outside view”, “steelman”, “sympathetic to”,
“counterfactual”, “gears-level”, “affordance”, and “salient”. These appear in
context across cooking, furniture, hobbies, invitations, computers, and other
ordinary subjects. The original catchphrases are less dominant by proportion;
there is no requirement to use every expression or stuff jargon into every answer.

For example: “The colour question is largely orthogonal to the doorway question.
Measure the sofa and the route it has to travel before ordering. The load-bearing
assumption here is that the sofa can enter the flat; enthusiasm about the upholstery
doesn't establish that.”

## Move to a cloud GPU

Use an **x86-64 Linux machine, Python 3.12, and a BF16-capable NVIDIA GPU**.
A **24 GB RTX 4090, A10, or L4** is a sensible starting target. Allow roughly 40 GB
of disk for the environment, model download, and outputs. These are planning
estimates; full-model GPU memory and speed have not been measured here.
On Linux x86-64, the lockfile selects PyTorch's CUDA 12.8 dependencies; choose an image with a compatible
NVIDIA driver (a current CUDA 12.8 image is a straightforward choice).

Copy `rationalist-lora-cloud.tar.gz` to that machine, then:

```bash
tar -xzf rationalist-lora-cloud.tar.gz
cd rationalist-lora
uv sync --locked --group train
uv run --locked --group train train.py --check
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.
uv manages Python 3.12 and the project virtual environment.
No API keys or paid teacher calls are required. The initial model download is public.

First run a five-step GPU smoke test, then start the real run in its own directory:

```bash
uv run --locked --group train train.py --max-steps 5 --output outputs/smoke
uv run --locked --group train train.py
```

The smoke test performs real adapter updates. It is separate from the full run.
Training evaluates each epoch, saves resumable checkpoints, selects the lowest
validation-loss checkpoint, and writes the adapter to `outputs/rationalist-lora/adapter`.
On this small dataset, treat validation loss as a rough diagnostic; read the outputs too.

To resume an interrupted full run, use the actual latest checkpoint directory:

```bash
uv run --locked --group train train.py --resume outputs/rationalist-lora/checkpoint-12
```

The number above is an example; inspect your output directory. Reusing a nonempty
output directory without `--resume` is refused. Keep the whole output folder,
including `run_config.json`, when moving an adapter.

## Compare the voice

```bash
uv run --locked --group train chat.py --base --eval-file eval_prompts.jsonl --output outputs/base.jsonl
uv run --locked --group train chat.py --eval-file eval_prompts.jsonl --output outputs/styled.jsonl
uv run --locked evaluate.py outputs/base.jsonl outputs/styled.jsonl
uv run --locked --group train chat.py --prompt "I keep researching bread recipes instead of baking anything."
```

Read `outputs/comparison.md`. Check the mannerisms, helpfulness, factual consistency,
and whether it obeys requests to speak plainly. Marker counts per 1,000 words are
descriptive, not a quality score; capitalisation counts also catch ordinary titles.
The evaluation prompts use fresh everyday requests and include exact-format controls.
`chat.py` is single-turn: each interactive prompt starts a fresh conversation.

## Change the experiment

Edit the pairs to tune intensity or add variety. `plain_response` is for reviewing
meaning preservation; it is never part of the training prompt or loss. `response`
is the target. Keep variants of one request in the same `group`. Don't add raw
forum essays: that would reintroduce the topic/style coupling this project avoids.
Adding more varied high-quality pairs is preferable to repeatedly duplicating these.

On your laptop:

```bash
uv sync --locked
uv run --locked prepare.py
uv run --locked train.py --check
uv run --locked pack.py
```

Preparation downloads only the tokenizer. To run the tiny CPU model tests, use
`uv run --locked --group train -m unittest discover -s tests -v`.
Dependencies live in `pyproject.toml`; `uv.lock` pins the resolved versions for
preparation and training. The optional `train` group adds PyTorch, PEFT, and Accelerate.
`HF_HUB_OFFLINE=1 uv run --locked prepare.py`
rebuilds with the cached tokenizer. `uv run --locked train.py --check` requires no GPU or weights.

For **Qwen3.5-9B**, change `model` to `Qwen/Qwen3.5-9B`, replace `revision` with
that model's commit SHA (or `main` for an unpinned experiment), and rerun preparation.
Use a 48 GB GPU as a more comfortable starting target. Don't reuse 4B's revision SHA.

The default uses standard Transformers/PEFT and their correct PyTorch fallback for
Qwen3.5's linear-attention operations. This avoids a compiled-kernel setup but is
slower than an optimised trainer. Optional `flash-linear-attention` / `causal-conv1d`
acceleration is deliberately outside the pinned first-run environment.

## References and verification

The 2,650 collected public posts from LessWrong, EA Forum, Alignment Forum, Zvi,
and Slate Star Codex are **local style references only**. They are excluded from
the train/validation files and the transfer archive. See [SOURCES.md](SOURCES.md).
`uv run --locked reference.py` produces short attributed phrase examples and rough counts;
`uv run --locked collect.py` refreshes through cached public APIs. It never accesses private
or paywalled content. `--limit` replaces each selected source's local snapshot, so
use it only when you intentionally want a smaller reference collection.

Validated locally: source APIs, real tokenizer/template boundaries, all prepared
answers and masks, split isolation, corrupted-data rejection, and a tiny Qwen3.5
with both linear and full attention doing forward/backward, adapter save/reload,
and a real Trainer update/evaluation/checkpoint cycle. This does not validate
full-size GPU training or the resulting style quality; those remain the cloud steps.

Model choice: [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B),
[Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5).
BF16 LoRA was chosen because the [Unsloth Qwen3.5 training guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
advises against 4-bit QLoRA for this architecture due to quantization differences.
Qwen3.5-4B is a newer, compact choice than Qwen3-4B-Instruct-2507; no claim of being
the universal best small model is needed for this experiment.
