"""Frozen 4B zero-shot candidate-generation baseline on the September holdout."""

import argparse
import json
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rerank import candidates, match, matching_sheet, select
from run import ROOT, SYSTEM_RULES


MODEL = "Qwen/Qwen3.5-4B"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = json.loads((ROOT / "data" / "fresh_eval.json").read_text())
    if args.limit:
        rows = rows[:args.limit]
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required for local 4B inference")
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
                                                 dtype=torch.float16).to("mps").eval()
    records = []
    for i, row in enumerate(rows, 1):
        ranked = candidates(model, tokenizer, row["prompt"], 8, "mps",
                            system_prompt=SYSTEM_RULES,
                            template_kwargs={"enable_thinking": False})
        sheet = matching_sheet(row["answers"])
        chosen = select(ranked, sheet, fuzzy=False)
        pool_scores = [(match(item["answer"], sheet) or {"score": 0})["score"] for item in ranked]
        best = max(pool_scores, default=0)
        records.append({"date": row["date"], "prompt": row["prompt"],
                        "ranked_candidates": ranked, "selected": chosen,
                        "pool_best_points": best, "pool_scores": pool_scores})
        print(f"{i}/{len(rows)} selected={chosen['score']:3d} pool_best={best:3d} "
              f"{row['prompt']} -> {chosen['answer']}", flush=True)
    summary = {"model": MODEL, "checkpoint": "base post-trained model; no Krillion fine-tuning",
               "prompt_style": "rules", "thinking": False, "candidate_count": 8,
               "decoding": "1 greedy + 7 samples, temperature=0.8, top_p=0.95, 24 new tokens",
               "prompt_count": len(records),
               "selected_mean_points": sum(r["selected"]["score"] for r in records) / len(records),
               "pool_best_mean_points": sum(r["pool_best_points"] for r in records) / len(records),
               "pool_match_rate": sum(r["pool_best_points"] > 0 for r in records) / len(records),
               "mean_unique_candidates": statistics.mean(len(r["ranked_candidates"]) for r in records),
               "pool_beats_selected_count": sum(r["pool_best_points"] > r["selected"]["score"]
                                                for r in records),
               "records": records}
    suffix = f"_smoke{args.limit}" if args.limit else ""
    out = ROOT / "results" / f"qwen35_4b_fresh_rules{suffix}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    compact = {k: v for k, v in summary.items() if k != "records"}
    (ROOT / "results" / f"qwen35_4b_fresh_rules{suffix}_summary.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2) + "\n")
    print(f"selected={summary['selected_mean_points']:.2f}, "
          f"pool_best={summary['pool_best_mean_points']:.2f}; saved {out}")


if __name__ == "__main__":
    main()
