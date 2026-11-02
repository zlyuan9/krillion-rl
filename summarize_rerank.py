"""Summarize paired answer-sheet-filtered results across the September dives."""

import json
import random

from run import ROOT


def interval(left, right):
    by_day = {}
    for x, y in zip(left, right):
        assert x["date"] == y["date"] and x["prompt"] == y["prompt"]
        by_day.setdefault(x["date"], []).append(x["exact_selection"]["score"] - y["exact_selection"]["score"])
    rng = random.Random(42)
    days = list(by_day)
    values = []
    for _ in range(20_000):
        draw = [rng.choice(days) for _ in days]
        deltas = [v for day in draw for v in by_day[day]]
        values.append(sum(deltas) / len(deltas))
    values.sort()
    return [values[500], values[19_500]]


def main():
    files = {"base": "rerank_8_base.json", "sft": "rerank_8.json", "grpo": "rerank_8_grpo.json"}
    data = {name: json.loads((ROOT / "results" / file).read_text()) for name, file in files.items()}
    summary = {
        "prompt_count": 100, "candidate_pool": "1 greedy + 7 sampled",
        "ranking": "mean completion token log probability under the model",
        "selection": "highest-ranked exact answer-sheet match; fuzzy spelling only if no exact match",
        "display_score": "complete seven-round dive depth in metres",
        "models": {name: {"exact_mean": d["exact_mean"], "fuzzy_mean": d["fuzzy_mean"],
                          "exact_match_rate": d["exact_match_rate"]} for name, d in data.items()},
        "sft_minus_base_mean": data["sft"]["exact_mean"] - data["base"]["exact_mean"],
        "sft_minus_base_day_bootstrap_95pct": interval(data["sft"]["records"], data["base"]["records"]),
        "grpo_minus_sft_mean": data["grpo"]["exact_mean"] - data["sft"]["exact_mean"],
        "grpo_minus_sft_day_bootstrap_95pct": interval(data["grpo"]["records"], data["sft"]["records"]),
        "bootstrap_resamples": 20_000, "bootstrap_seed": 42,
    }
    demo_depth_path = ROOT / "results" / "demo_depth.json"
    if demo_depth_path.exists():
        depth = json.loads(demo_depth_path.read_text())
        for name in data:
            summary["models"][name]["demo_mean_depth_m"] = depth["models"][name]["mean_depth_m"]
        summary["demo_dive_count"] = depth["dive_count"]
    out = ROOT / "results" / "rerank_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
