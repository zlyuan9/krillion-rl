"""Express archived seven-question dive scores as Krillion-style depth metres."""

import argparse
import json
from collections import defaultdict

from prepare import parse_page
from run import ROOT


def prepare():
    included = {(row["date"], row["prompt"])
                for row in json.loads((ROOT / "data" / "fresh_eval.json").read_text())}
    all_rows = [row for path in sorted((ROOT / "data" / "raw").glob("2026-09-*.html"))
                for row in parse_page(path)]
    excluded = [row for row in all_rows if (row["date"], row["prompt"]) not in included]
    assert len(all_rows) == 105 and len(excluded) == 5
    out = ROOT / "data" / "demo_excluded.json"
    out.write_text(json.dumps(excluded, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(excluded)} omitted prompts to {out}")


def summarize():
    names = {"base": "_base", "sft": "", "grpo": "_grpo"}
    summary = {"depth_formula": "10 metres per point; sum all seven rounds",
               "maximum_depth_m": 7000, "dive_count": 15, "models": {}}
    for name, suffix in names.items():
        normal = ROOT / "results" / f"rerank_8{suffix}.json"
        omitted = ROOT / "results" / f"rerank_8{suffix}_demo_excluded.json"
        records = json.loads(normal.read_text())["records"] + json.loads(omitted.read_text())["records"]
        by_day = defaultdict(list)
        for record in records:
            chosen = record["exact_selection"]
            by_day[record["date"]].append({"prompt": record["prompt"],
                                          "answer": chosen["answer"],
                                          "points": chosen["score"],
                                          "depth_m": chosen["score"] * 10})
        assert len(by_day) == 15 and all(len(rounds) == 7 for rounds in by_day.values())
        dives = [{"date": day, "points": sum(r["points"] for r in rounds),
                  "depth_m": sum(r["depth_m"] for r in rounds), "rounds": rounds}
                 for day, rounds in sorted(by_day.items())]
        assert all(d["depth_m"] == 10 * d["points"] and 0 <= d["depth_m"] <= 7000
                   for d in dives)
        summary["models"][name] = {
            "mean_depth_m": sum(d["depth_m"] for d in dives) / len(dives),
            "mean_points": sum(d["points"] for d in dives) / len(dives),
            "dives": dives,
        }
    out = ROOT / "results" / "demo_depth.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    for name, result in summary["models"].items():
        print(f"{name}: mean depth {result['mean_depth_m']:.0f} m over 15 complete dives")
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "summarize"])
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else summarize()


if __name__ == "__main__":
    main()
