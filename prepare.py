"""Parse saved, independently archived Krillion recap pages into a fixed dataset."""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "dives.json"


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def parse_page(path):
    soup = BeautifulSoup(path.read_text(), "html.parser")
    dives = []
    for section in soup.select("section.answer-block"):
        title = section.find("h2", recursive=False)
        details = section.select_one("details.answer-expand")
        if not title or not details:
            continue
        prompt = re.sub(r"^\d+\.\s*", "", clean(title.get_text(" ", strip=True)))
        answers = {}
        for item in details.select("ul.answer-list li"):
            score_span = item.select_one("span.score")
            if score_span is None:
                continue
            score = int(clean(score_span.get_text()))
            answer = clean(item.get_text(" ", strip=True)[: -len(score_span.get_text(" ", strip=True))])
            if answer:
                answers[answer] = score
        if not answers:
            raise ValueError(f"No answers for {path}: {prompt}")
        dives.append({"date": path.stem, "prompt": prompt, "answers": answers})
    if len(dives) != 7:
        raise ValueError(f"Expected seven prompts in {path}, got {len(dives)}")
    return dives


def main():
    rows = [row for path in sorted(RAW.glob("2026-08-*.html")) for row in parse_page(path)]
    seen = set()
    for row in rows:
        key = row["prompt"].casefold()
        if key in seen:
            raise ValueError(f"Duplicate prompt: {row['prompt']}")
        seen.add(key)
        row["split"] = "train" if row["date"] <= "2026-08-20" else "eval"
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(rows)} prompts: {sum(r['split']=='train' for r in rows)} train, "
          f"{sum(r['split']=='eval' for r in rows)} eval")
    print(f"Accepted answer rows: {sum(len(r['answers']) for r in rows)}")


if __name__ == "__main__":
    main()
