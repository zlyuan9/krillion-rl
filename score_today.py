"""Package a dated official-reveal run as a seven-round depth score."""

import argparse
import json

from run import ROOT, SYSTEM, SYSTEM_RULES, SYSTEM_RULES_CAUTIOUS


def result(date, style):
    path = ROOT / "results" / f"rerank_8_today_{date}_{style}.json"
    data = json.loads(path.read_text())
    if data["prompt_count"] != 7:
        raise ValueError(f"Expected seven rounds in {path}")
    rounds = [{"prompt": r["prompt"], "answer": r["exact_selection"]["answer"],
               "points": r["exact_selection"]["score"],
               "depth_m": 10 * r["exact_selection"]["score"],
               "selected_rank": r["exact_selection"]["rank"]}
              for r in data["records"]]
    points = sum(r["points"] for r in rounds)
    prompts = {"rules": SYSTEM_RULES, "cautious": SYSTEM_RULES_CAUTIOUS, "legacy": SYSTEM}
    return {"prompt_style": style, "system_prompt": prompts[style],
            "points": points, "depth_m": 10 * points, "rounds": rounds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("date", help="Krillion day in YYYY-MM-DD format")
    args = parser.parse_args()
    output = {"date": args.date, "source_today": "https://krillion.io/api/today",
              "source_reveal": f"https://krillion.io/api/reveal?date={args.date}",
              "scoring": "sum seven official reveal scores, multiply by 10 metres per point",
              "maximum_depth_m": 7000,
              "primary": result(args.date, "rules"),
              "cautious_comparison": result(args.date, "cautious"),
              "legacy_comparison": result(args.date, "legacy")}
    out = ROOT / "results" / f"today_{args.date}_score.json"
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"Rules-aware: {output['primary']['points']} points = {output['primary']['depth_m']} m")
    print(f"Cautious wording: {output['cautious_comparison']['points']} points = {output['cautious_comparison']['depth_m']} m")
    print(f"Legacy prompt: {output['legacy_comparison']['points']} points = {output['legacy_comparison']['depth_m']} m")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
