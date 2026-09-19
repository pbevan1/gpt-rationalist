"""Use a trained adapter, or --base to compare the unmodified model."""
import argparse
import json
from pathlib import Path
import sys

from common import config


def generation_options(tokenizer, max_new_tokens):
    # Match the tokenizer used to prepare assistant end-of-turn labels. Training
    # aligns these IDs, but a newly loaded base model can have different defaults.
    return dict(max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.8, top_k=20, use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="outputs/rationalist-lora/adapter")
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--prompt")
    parser.add_argument("--eval-file", help="JSONL prompts for before/after comparisons")
    parser.add_argument("--output", default="outputs/generations.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-stream", action="store_true", help="Print only the completed answer")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, TextStreamer, set_seed
    if args.base:
        cfg = config()
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    else:
        adapter = Path(args.adapter)
        run_config = adapter.parent / "run_config.json"
        if not (adapter / "adapter_config.json").is_file() or not run_config.is_file():
            parser.error(f"Missing trained adapter or run_config.json at {adapter}; train first or pass --adapter")
        cfg = json.loads(run_config.read_text())
        tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    # Avoid silent CPU offloading: the tested deployment is a single GPU.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device != "cpu" and torch.cuda.is_bf16_supported() else torch.float32
    print(f"Loading model on {device}...", file=sys.stderr, flush=True)
    if device == "cpu":
        print("No CUDA GPU detected; CPU generation may be slow.", file=sys.stderr, flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg["model"], revision=cfg["revision"], dtype=dtype,
        device_map={"": device}, attn_implementation="sdpa",
    )
    if not args.base:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    options = generation_options(tokenizer, args.max_new_tokens)
    print("Ready. Each prompt starts a fresh conversation.", file=sys.stderr, flush=True)

    def answer(prompt, stream=False):
        set_seed(cfg["seed"])
        messages = [{"role": "system", "content": cfg["system_prompt"]}, {"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to(model.device)
        print("Generating...", file=sys.stderr, flush=True)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None
        with torch.inference_mode():
            output = model.generate(**inputs, **options, streamer=streamer)
        generated = output[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        eos = options['eos_token_id']
        end_ids = eos if isinstance(eos, list) else [eos]
        if len(generated) >= args.max_new_tokens and generated[-1].item() not in end_ids:
            print("Answer reached --max-new-tokens; increase it for a longer response.", file=sys.stderr, flush=True)
        if not response.strip():
            print("Model returned no visible text.", file=sys.stderr, flush=True)
        return response

    if args.eval_file:
        from common import read_jsonl
        rows = read_jsonl(args.eval_file)
        path = Path(args.output)
        if path.resolve() == Path(args.eval_file).resolve():
            parser.error("--output must differ from --eval-file")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Persist each response so a long evaluation is inspectable and partial
        # results survive interruption. Base and adapter use identical sampling.
        with path.open('w') as handle:
            for i, row in enumerate(rows, 1):
                print(f"Evaluation {i}/{len(rows)}: {row['id']}", file=sys.stderr, flush=True)
                result = {**row, "response": answer(row["prompt"]),
                          "model": cfg["model"], "adapter": None if args.base else args.adapter,
                          "generation": {**options, "seed": cfg["seed"]}}
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                handle.flush()
        print(args.output)
    else:
        def display(prompt):
            response = answer(prompt, stream=not args.no_stream)
            if args.no_stream:
                print(response, flush=True)
        if args.prompt:
            display(args.prompt)
        else:
            while True:
                try:
                    prompt = input("You (Ctrl-D to exit): ").strip()
                except EOFError:
                    break
                if prompt:
                    display(prompt)
                    print()


if __name__ == '__main__':
    main()
