"""Use a trained adapter, or --base to compare the unmodified model."""
import argparse
import json
from pathlib import Path

from common import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="outputs/rationalist-lora/adapter")
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--prompt")
    parser.add_argument("--eval-file", help="JSONL prompts for before/after comparisons")
    parser.add_argument("--output", default="outputs/generations.jsonl")
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, set_seed
    cfg = config() if args.base else json.loads((Path(args.adapter).parent / "run_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg["model"], revision=cfg["revision"], dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
    )
    if not args.base:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def answer(prompt):
        set_seed(cfg["seed"])
        messages = [{"role": "system", "content": cfg["system_prompt"]}, {"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=700, do_sample=True,
                                    temperature=0.7, top_p=0.8, top_k=20,
                                    pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    if args.eval_file:
        from common import read_jsonl, write_jsonl
        write_jsonl(args.output, [{**row, "response": answer(row["prompt"]),
                                  "model": cfg["model"], "adapter": None if args.base else args.adapter}
                                 for row in read_jsonl(args.eval_file)])
        print(args.output)
    elif args.prompt:
        print(answer(args.prompt))
    else:
        while True:
            try:
                prompt = input("You (Ctrl-D to exit): ").strip()
            except EOFError:
                break
            if prompt:
                print(answer(prompt) + "\n")


if __name__ == "__main__":
    main()
