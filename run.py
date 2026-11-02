"""Small local Krillion post-training experiment (base vs LoRA SFT)."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER = ROOT / "results" / "lora"
SYSTEM = "You are playing Krillion. Give exactly one valid, specific answer to the prompt. Output only the answer, with no explanation."
SYSTEM_RULES = (
    "You are playing Krillion. Each question accepts one answer. An accepted answer "
    "earns more points when it is uncommon: 10, 15, 30, 60, 85, or 100 points. "
    "Choose a specific, real answer that fits every condition in the question, "
    "and aim for a high rarity tier. "
    "Output only the answer, with no explanation."
)
SYSTEM_RULES_CAUTIOUS = (
    "You are playing Krillion. Each question accepts one answer. An accepted answer "
    "earns more points when it is uncommon: 10, 15, 30, 60, 85, or 100 points. "
    "Choose a specific, real answer that fits every condition in the question, "
    "and aim for a high rarity tier without risking an invalid answer. "
    "Output only the answer, with no explanation."
)


def norm(s):
    return re.sub(r"[^\w]+", " ", s.casefold()).strip()


def prompt_text(tokenizer, prompt, system=SYSTEM, template_kwargs=None):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
        **(template_kwargs or {}),
    )


def target_answer(row):
    # Deterministic, diverse targets. Prefer a short 85-point answer over an
    # often idiosyncratic 100-point editor's pick.
    candidates = [a for a, score in row["answers"].items() if score >= 85 and len(a) <= 40]
    if not candidates:
        candidates = [a for a, score in row["answers"].items() if score >= 60]
    candidates.sort()
    index = int(hashlib.sha256(row["prompt"].encode()).hexdigest(), 16) % len(candidates)
    return candidates[index]


def load_model(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.float32)
    return tokenizer, model.to(device)


def make_examples(rows, tokenizer):
    examples = []
    for row in rows:
        prefix = tokenizer(prompt_text(tokenizer, row["prompt"]), add_special_tokens=False).input_ids
        answer = row["target"] if "target" in row else target_answer(row)
        completion = tokenizer(answer + tokenizer.eos_token,
                               add_special_tokens=False).input_ids
        ids = prefix + completion
        if len(ids) > 160:
            raise ValueError((row["prompt"], len(ids)))
        examples.append((ids, [-100] * len(prefix) + completion))
    return examples


def collator(batch, pad):
    width = max(len(ids) for ids, _ in batch)
    return {
        "input_ids": torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in batch]),
        "attention_mask": torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in batch]),
        "labels": torch.tensor([labels + [-100] * (width - len(labels)) for _, labels in batch]),
    }


@torch.no_grad()
def evaluate(rows, tokenizer, model, device, name):
    model.eval()
    records = []
    for i, row in enumerate(rows, 1):
        inputs = tokenizer(prompt_text(tokenizer, row["prompt"]), return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        output = model.generate(**inputs, max_new_tokens=24, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip()
        lookup = {norm(a): score for a, score in row["answers"].items()}
        score = lookup.get(norm(answer), 0)
        records.append({"date": row["date"], "prompt": row["prompt"], "answer": answer,
                        "score": score, "target_example": target_answer(row)})
        print(f"{name} {i}/{len(rows)} score={score:3d} {row['prompt']} -> {answer}", flush=True)
    summary = {"model": MODEL, "name": name, "count": len(records),
               "mean_score": sum(x["score"] for x in records) / len(records),
               "matched_rate": sum(x["score"] > 0 for x in records) / len(records),
               "records": records}
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"{name}: mean={summary['mean_score']:.2f}, matched={summary['matched_rate']:.3f}", flush=True)


def train(rows, tokenizer, model, device):
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                           target_modules=["q_proj", "v_proj", "o_proj"],
                                           task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    examples = make_examples(rows, tokenizer)
    loader = DataLoader(examples, batch_size=4, shuffle=True,
                        collate_fn=lambda batch: collator(batch, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    torch.manual_seed(42)
    random.seed(42)
    model.train()
    for epoch in range(2):
        total = 0.0
        for step, batch in enumerate(loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            if step % 10 == 0:
                print(f"epoch={epoch+1} step={step}/{len(loader)} loss={loss.item():.3f}", flush=True)
        print(f"epoch={epoch+1} mean_loss={total/len(loader):.3f}", flush=True)
    ADAPTER.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["baseline", "train", "eval", "eval-rl"])
    args = parser.parse_args()
    rows = json.loads((ROOT / "data" / "dives.json").read_text())
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}", flush=True)
    tokenizer, model = load_model(device)
    if args.mode == "baseline":
        evaluate([r for r in rows if r["split"] == "eval"], tokenizer, model, device, "baseline")
    elif args.mode == "train":
        train([r for r in rows if r["split"] == "train"], tokenizer, model, device)
    else:
        path = ADAPTER if args.mode == "eval" else ROOT / "results" / "grpo_lora"
        name = "sft" if args.mode == "eval" else "grpo"
        model = PeftModel.from_pretrained(model, path).to(device)
        evaluate([r for r in rows if r["split"] == "eval"], tokenizer, model, device, name)


if __name__ == "__main__":
    main()
