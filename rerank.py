"""Generate candidate answers, rank by model likelihood, filter via answer sheet."""

import argparse
import difflib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel

from run import ADAPTER, ROOT, SYSTEM, SYSTEM_RULES, SYSTEM_RULES_CAUTIOUS, load_model, norm, prompt_text


def answer_text(text):
    return text.strip().split("\n")[0].strip()


def matching_sheet(answers):
    normalized = {}
    for answer, score in answers.items():
        key = norm(answer)
        if key:
            normalized[key] = max(score, normalized.get(key, 0))
    return normalized


def match(answer, sheet, fuzzy=False):
    query = norm(answer)
    if query in sheet:
        return {"score": sheet[query], "method": "exact", "matched_answer": query}
    if not fuzzy or len(query) < 6:
        return None
    # Conservative typo/near-spelling recovery. This deliberately does not
    # attempt semantic equivalence or arbitrary first-name/surname aliases.
    near = difflib.get_close_matches(query, sheet.keys(), n=1, cutoff=0.92)
    if not near:
        return None
    key = near[0]
    if query[0] != key[0] or min(len(query), len(key)) / max(len(query), len(key)) < 0.8:
        return None
    return {"score": sheet[key], "method": "fuzzy", "matched_answer": key,
            "similarity": difflib.SequenceMatcher(None, query, key).ratio()}


@torch.no_grad()
def candidates(model, tokenizer, prompt, count, device, system_prompt=SYSTEM,
               template_kwargs=None):
    encoded = tokenizer(prompt_text(tokenizer, prompt, system=system_prompt,
                                    template_kwargs=template_kwargs),
                        return_tensors="pt", add_special_tokens=False)
    prefix = encoded["input_ids"].to(device)
    prefix_len = prefix.shape[1]
    greedy = model.generate(input_ids=prefix,
                            attention_mask=encoded["attention_mask"].to(device),
                            do_sample=False, max_new_tokens=24,
                            pad_token_id=tokenizer.eos_token_id)
    if count > 1:
        sampled = model.generate(input_ids=prefix,
                                 attention_mask=encoded["attention_mask"].to(device),
                                 do_sample=True, temperature=0.8, top_p=0.95,
                                 num_return_sequences=count - 1, max_new_tokens=24,
                                 pad_token_id=tokenizer.eos_token_id)
        width = max(greedy.shape[1], sampled.shape[1])
        generated = torch.cat([F.pad(greedy, (0, width - greedy.shape[1]), value=tokenizer.eos_token_id),
                               F.pad(sampled, (0, width - sampled.shape[1]), value=tokenizer.eos_token_id)])
    else:
        generated = greedy
    # Rank generated continuations under the actual model, rather than by the
    # order returned by the sampler. Score raw completions including EOS.
    attention = torch.ones_like(generated)
    lengths = []
    for index, sequence in enumerate(generated):
        completion = sequence[prefix_len:].tolist()
        try:
            length = completion.index(tokenizer.eos_token_id) + 1
        except ValueError:
            length = len(completion)
        lengths.append(length)
        attention[index, prefix_len + length:] = 0
    logits = model(input_ids=generated, attention_mask=attention).logits
    logprobs = torch.log_softmax(logits[:, prefix_len - 1:-1, :], dim=-1)
    ids = generated[:, prefix_len:]
    token_logprobs = logprobs.gather(2, ids.unsqueeze(-1)).squeeze(-1)
    results = []
    for i, length in enumerate(lengths):
        raw = tokenizer.decode(ids[i, :length], skip_special_tokens=True)
        results.append({"answer": answer_text(raw), "source": "greedy" if i == 0 else "sampled",
                        "mean_token_logprob": token_logprobs[i, :length].mean().item(),
                        "raw": raw.strip()})
    # Repeated samples count once; keep the highest likelihood instance.
    unique = {}
    for item in results:
        key = norm(item["answer"])
        if key and (key not in unique or item["mean_token_logprob"] > unique[key]["mean_token_logprob"]):
            unique[key] = item
    return sorted(unique.values(), key=lambda item: item["mean_token_logprob"], reverse=True)


def select(ranked, sheet, fuzzy):
    if fuzzy:
        exact = select(ranked, sheet, fuzzy=False)
        if exact["method"] == "exact":
            return exact
    for rank, item in enumerate(ranked, 1):
        found = match(item["answer"], sheet, fuzzy=fuzzy)
        if found:
            return {**item, **found, "rank": rank}
    if ranked:
        return {**ranked[0], "score": 0, "method": "no_match", "rank": 1}
    return {"answer": "", "score": 0, "method": "no_candidate", "rank": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test first N prompts")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--adapter", choices=["base", "sft", "grpo"], default="sft")
    parser.add_argument("--dataset", choices=["fresh", "demo-excluded"], default="fresh")
    parser.add_argument("--dataset-file", help="JSON rows relative to the project root, or absolute path")
    parser.add_argument("--prompt-style", choices=["legacy", "rules", "cautious"], default="legacy")
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    if args.dataset_file:
        data_path = Path(args.dataset_file)
        if not data_path.is_absolute():
            data_path = ROOT / data_path
    else:
        data_file = "fresh_eval.json" if args.dataset == "fresh" else "demo_excluded.json"
        data_path = ROOT / "data" / data_file
    rows = json.loads(data_path.read_text())
    if args.limit:
        rows = rows[:args.limit]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(42)
    tokenizer, base = load_model(device)
    adapter_path = {"sft": ADAPTER, "grpo": ROOT / "results" / "grpo_lora"}.get(args.adapter)
    model = (PeftModel.from_pretrained(base, adapter_path).to(device) if adapter_path else base).eval()
    system_prompt = {"legacy": SYSTEM, "rules": SYSTEM_RULES,
                     "cautious": SYSTEM_RULES_CAUTIOUS}[args.prompt_style]
    records = []
    for i, row in enumerate(rows, 1):
        ranked = candidates(model, tokenizer, row["prompt"], args.count, device,
                            system_prompt=system_prompt)
        sheet = matching_sheet(row["answers"])
        exact = select(ranked, sheet, fuzzy=False)
        fuzzy = select(ranked, sheet, fuzzy=True)
        records.append({"date": row["date"], "prompt": row["prompt"],
                        "ranked_candidates": ranked, "exact_selection": exact,
                        "fuzzy_selection": fuzzy})
        print(f"{i}/{len(rows)} exact={exact['score']:3d} fuzzy={fuzzy['score']:3d} "
              f"rank={exact['rank']} {row['prompt']} -> {exact['answer']}", flush=True)
    summary = {"model": args.adapter, "prompt_style": args.prompt_style,
               "dataset": str(data_path), "candidate_count": args.count,
               "candidate_pool": "1 greedy + sampled alternatives", "prompt_count": len(records),
               "exact_mean": sum(r["exact_selection"]["score"] for r in records) / len(records),
               "fuzzy_mean": sum(r["fuzzy_selection"]["score"] for r in records) / len(records),
               "exact_match_rate": sum(r["exact_selection"]["score"] > 0 for r in records) / len(records),
               "fuzzy_match_rate": sum(r["fuzzy_selection"]["score"] > 0 for r in records) / len(records),
               "records": records}
    suffix = f"_smoke{args.limit}" if args.limit else ""
    adapter_suffix = "" if args.adapter == "sft" else f"_{args.adapter}"
    dataset_suffix = (f"_{data_path.stem}" if args.dataset_file else
                      "" if args.dataset == "fresh" else "_demo_excluded")
    style_suffix = f"_{args.prompt_style}" if args.dataset_file else ""
    out = ROOT / "results" / f"rerank_{args.count}{adapter_suffix}{dataset_suffix}{style_suffix}{suffix}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"exact={summary['exact_mean']:.2f}, fuzzy={summary['fuzzy_mean']:.2f}; saved {out}")


if __name__ == "__main__":
    main()
