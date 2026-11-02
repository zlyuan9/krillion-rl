# krillion-rl

A small post-training experiment: teach an LLM to give uncommon, valid answers to [Krillion](https://krillion.io/) prompts.

## How it works

We parsed historical answer sheets from an [independent recap archive](https://krilliongame.com/krillion-answers/). August 1–20 trained a LoRA adapter on Llama 3.2 1B; August 21–25 and September 1–15 were held out. At evaluation, the model generates one greedy answer and seven samples. We rank them by mean token log probability and take the highest-ranked answer that exactly matches the archived sheet. Missing matches score zero. The optional fuzzy matcher added no points on the September set.

**This evaluation uses the held-out answer sheet to select an answer.** Live play would need Krillion's checker or another validity verifier. Archive matches and scores are estimates, not live submissions.

## Results

Mean archived points per prompt on the 100-prompt September holdout:

| Model | Greedy | Ranked selection | Best in generated pool |
| --- | ---: | ---: | ---: |
| Llama 3.2 1B base | 12.85 | 18.80 | 28.60 |
| Llama 3.2 1B SFT | 22.95 | **37.20** | **55.10** |
| SFT + 25 GRPO steps | — | 37.00 | 54.75 |
| Qwen3.5 4B, no Krillion training | — | 17.25 | 26.10 |

SFT gave the clear gain; short GRPO training, synthetic paraphrases, and this zero-shot 4B baseline did not improve it. The 4B run used a different, rules-aware prompt, so it is not a controlled model-size comparison. “Best in pool” is an answer-sheet-assisted upper bound, not a deployable score. See the [reranking](results/rerank_summary.json), [synthetic-data](results/fresh_summary.json), and [4B](results/qwen35_4b_fresh_rules_summary.json) summaries.

Krillion converts each point to 10 metres of depth. Across 15 complete September dives, the SFT model averaged **2,657 m** out of 7,000 m; see [per-dive results](results/demo_depth.json). On November 1, 2026, it scored **305 points / 3,050 m** against that day's public reveal sheet; see the [seven answers](results/today_2026-11-01_score.json).

## Run locally

Install `requirements.txt` and cache the base models named in `run.py` and `compare_4b.py`. Place the historical recap pages in `data/raw/`, then run:

```sh
python3 prepare.py
python3 run.py train
python3 rl.py
python3 fresh.py prepare
python3 rerank.py --count 8 --adapter base
python3 rerank.py --count 8 --adapter sft
python3 rerank.py --count 8 --adapter grpo
python3 summarize_rerank.py
python3 compare_4b.py
```

The scripts use an Apple GPU and cached model weights. Scraped pages, processed answer sheets, detailed candidate outputs, and trained adapters are local experiment artifacts excluded from this repo.
