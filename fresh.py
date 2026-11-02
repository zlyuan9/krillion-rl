"""Matched-budget synthetic paraphrase test on untouched September dives."""

import argparse
import json
import random
import re
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader

from prepare import parse_page
from run import ROOT, collator, evaluate, load_model, make_examples, target_answer


FRESH = ROOT / "data" / "fresh_eval.json"
SYNTH = ROOT / "data" / "synthetic_train.json"


def tokens(text):
    stop = {"name", "a", "an", "the", "or", "of", "in", "with", "that", "is", "one"}
    return set(re.findall(r"[a-z0-9]+", text.lower())) - stop


def overlap(a, b):
    left, right = tokens(a), tokens(b)
    return len(left & right) / len(left | right) if left | right else 0


def prepare():
    old = json.loads((ROOT / "data" / "dives.json").read_text())
    train = [r for r in old if r["split"] == "train"]
    fresh = [r for p in sorted((ROOT / "data" / "raw").glob("2026-09-*.html"))
             for r in parse_page(p)]
    excluded = []
    kept = []
    for row in fresh:
        closest = max(train, key=lambda prior: overlap(row["prompt"], prior["prompt"]))
        similarity = overlap(row["prompt"], closest["prompt"])
        if similarity >= 0.55:
            excluded.append({"date": row["date"], "prompt": row["prompt"],
                             "training_prompt": closest["prompt"], "jaccard": similarity})
        else:
            kept.append(row)
    FRESH.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "data" / "fresh_excluded.json").write_text(json.dumps(excluded, ensure_ascii=False, indent=2) + "\n")
    examples = []
    for row in train:
        answer = target_answer(row)
        rest = row["prompt"][5:]
        for kind, prompt in [("real", row["prompt"]),
                             ("synthetic", "Give one example of " + rest),
                             ("synthetic", "Identify " + rest)]:
            examples.append({"source_date": row["date"], "kind": kind,
                             "prompt": prompt, "target": answer})
    SYNTH.write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n")
    print(f"Fresh September prompts: {len(kept)} kept, {len(excluded)} excluded for near overlap")
    print(f"Training examples: {len(examples)} ({len(train)} real + {2*len(train)} synthetic)")


def train(arm):
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the quick training run")
    torch.manual_seed(42)
    tokenizer, model = load_model("mps")
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                           target_modules=["q_proj", "v_proj", "o_proj"],
                                           task_type="CAUSAL_LM"))
    rows = json.loads(SYNTH.read_text())
    if arm == "control":
        # Same number of optimizer steps and target answers as augmented arm.
        rows = [row for row in rows if row["kind"] == "real"] * 3
    examples = make_examples(rows, tokenizer)
    loader = DataLoader(examples, batch_size=4, shuffle=True,
                        generator=torch.Generator().manual_seed(42),
                        collate_fn=lambda batch: collator(batch, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    model.train()
    for epoch in range(2):
        total = 0.0
        for step, batch in enumerate(loader, 1):
            batch = {key: value.to("mps") for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            if step % 25 == 0:
                print(f"{arm} epoch={epoch+1} step={step}/{len(loader)} loss={loss.item():.3f}", flush=True)
        print(f"{arm} epoch={epoch+1} mean_loss={total/len(loader):.3f}", flush=True)
    out = ROOT / "results" / f"{arm}_lora"
    model.save_pretrained(out)
    print(f"Saved {out}")


def eval_arm(arm):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer, model = load_model(device)
    if arm != "base":
        adapter = {"sft": "lora", "control": "control_lora", "synthetic": "synthetic_lora"}[arm]
        model = PeftModel.from_pretrained(model, ROOT / "results" / adapter).to(device)
    evaluate(json.loads(FRESH.read_text()), tokenizer, model, device, f"fresh_{arm}")


def summarize():
    names = ["base", "sft", "control", "synthetic"]
    results = {name: json.loads((ROOT / "results" / f"fresh_{name}.json").read_text())
               for name in names}
    paired = [(x["date"], x["score"] - y["score"])
              for x, y in zip(results["synthetic"]["records"], results["control"]["records"])]
    by_day = {}
    for day, delta in paired:
        by_day.setdefault(day, []).append(delta)
    rng = random.Random(42)
    days = list(by_day)
    bootstrap = []
    for _ in range(20_000):
        draw = [rng.choice(days) for _ in days]
        values = [delta for day in draw for delta in by_day[day]]
        bootstrap.append(sum(values) / len(values))
    bootstrap.sort()
    summary = {
        "eval_count": len(paired), "eval_dates": len(days),
        "models": {name: {"mean_score": data["mean_score"], "matched_rate": data["matched_rate"]}
                   for name, data in results.items()},
        "synthetic_minus_control_mean": sum(delta for _, delta in paired) / len(paired),
        "synthetic_minus_control_day_bootstrap_95pct": [bootstrap[500], bootstrap[19_500]],
        "bootstrap_seed": 42, "bootstrap_resamples": 20_000,
    }
    out = ROOT / "results" / "fresh_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "train-control", "train-synthetic",
                                         "eval-base", "eval-sft", "eval-control", "eval-synthetic",
                                         "summarize"])
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode.startswith("train-"):
        train(args.mode[6:])
    elif args.mode == "summarize":
        summarize()
    else:
        eval_arm(args.mode[5:])


if __name__ == "__main__":
    main()
