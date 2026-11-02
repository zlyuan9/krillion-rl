"""Convert a saved official Krillion daily API response and reveal to evaluator rows."""

import argparse
import json
from pathlib import Path

from run import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("date", help="Krillion day in YYYY-MM-DD format")
    args = parser.parse_args()
    raw = ROOT / "data" / "raw"
    today = json.loads((raw / f"{args.date}-official-today.json").read_text())
    reveal = json.loads((raw / f"{args.date}-official-reveal.json").read_text())
    if today["date"] != args.date or reveal["date"] != args.date:
        raise ValueError("API dates disagree")
    prompts = {item["id"]: item["text"] for item in today["prompts"]}
    rows = []
    for item in reveal["prompts"]:
        if item["text"] != prompts[item["id"]]:
            raise ValueError(f"Prompt wording mismatch: {item['id']}")
        answers = {}
        for answer in item["answers"]:
            if answer["score"] not in (10, 15, 30, 60, 85, 100):
                raise ValueError(f"Unexpected tier score: {answer}")
            key = answer["answer"]
            answers[key] = max(answer["score"], answers.get(key, 0))
        rows.append({"date": args.date, "prompt": item["text"],
                     "prompt_id": item["id"], "answers": answers})
    if len(rows) != 7:
        raise ValueError(f"Expected seven rounds, got {len(rows)}")
    out = ROOT / "data" / f"today_{args.date}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(rows)} official prompts, {sum(len(r['answers']) for r in rows)} accepted answer rows to {out}")


if __name__ == "__main__":
    main()
