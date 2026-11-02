# krillion-rl

Krillion post-training pilot.

A small, completed local experiment on rare-answer generation. This uses an
**independent historical recap** at [krilliongame.com](https://krilliongame.com/krillion-answers/),
not the live Krillion scorer or an official bulk dataset. The site describes its
pages as saved snapshots. The local downloaded pages are in `data/raw/` with dates
2026-08-01 through 2026-08-25.

The GitHub repository includes code and selected result files. Scraped pages,
processed answer sheets, most detailed candidate outputs, and trained adapters
are local experiment artifacts and are excluded from Git. Reproducing training or
evaluation requires collecting the public pages and running `prepare.py`, plus
downloading the base model weights specified in `run.py`.

## Data and scoring

- `prepare.py` extracts seven prompts and the full recorded answer/score list
  from each date: 175 prompts, 15,020 answer rows.
- Training uses August 1–20 (140 prompts); evaluation uses August 21–25
  (35 prompts). Splits are by whole dive. No duplicate prompt text was found.
- The model sees only a prompt at evaluation. Greedy decoding produces one
  answer with a 24-token cap.
- Scoring normalizes case and punctuation, then looks up the answer in that
  prompt's archived sheet. Missing matches score zero. This is a **recorded
  match rate**, not a definitive validity rate: aliases, autocorrect, or an
  omitted fair answer might score differently in the live game.

## Models

Base: cached `meta-llama/Llama-3.2-1B-Instruct`.

SFT: rank-8 LoRA on `q_proj`, `v_proj`, and `o_proj`; 2 epochs over 140 prompts,
batch size 4, learning rate 1e-4. Each prompt has one deterministically chosen
short answer from its recorded 85+ point tier (or 60+ if needed). This is a
simple way to teach rare-answer behavior, not an optimal target-selection
procedure.

GRPO: starts from the SFT adapter; 25 on-policy steps, four sampled completions
per prompt, exact score divided by 100 as reward, learning rate 1e-5, reference
KL coefficient 0.02. At 100 training prompt exposures, this is only 0.18 of an
epoch. `rl.py` contains the reward and full config.

## Held-out result

| Model | Mean recorded points / prompt | Recorded match rate | Dive totals, Aug 21–25 |
| --- | ---: | ---: | --- |
| Base | 12.00 | 82.9% | 80, 70, 60, 120, 90 |
| SFT | 30.86 | 68.6% | 245, 295, 230, 100, 210 |
| SFT + GRPO | 28.29 | 68.6% | 245, 205, 230, 100, 210 |

The SFT adapter raised score but caused more answers to miss the archived sheet.
GRPO changed only one of the 35 greedy evaluation answers: an apple variety
went from 100 to 10 points. Five held-out dives are too few to claim robust
generalization. Detailed individual answers and scores are in local `results/*.json`.

## Reproduce

Use the Python environment recorded in `requirements.txt` and an Apple GPU for
the quick run. The scripts use cached model weights (`local_files_only=True`).

```sh
python3 prepare.py
python3 run.py baseline
python3 run.py train
python3 run.py eval
python3 rl.py
python3 run.py eval-rl
```

The next useful experiment is a larger, varied training set with an untouched
future-date test set. Synthetic prompts can expand training, but their answer
sheets should be independently checked; evaluating on synthetic labels would
only measure how well the model learned the generator's preferences.

## Follow-up: synthetic paraphrases with a fresh holdout

`fresh.py` runs a controlled augmentation test. It parses the saved September
1–15 recaps and excludes five prompts with token-set Jaccard overlap ≥0.55 against
the original 140 training prompts. This leaves 100 new prompts across 15 dives.
No September answer sheet enters training.

For each original training prompt, the script creates two meaning-preserving
templates: `Name a lake` becomes `Give one example of a lake` and `Identify a
lake`. Each variant keeps the original recorded answer as its target. These are
**wording variants**, not 280 new Krillion categories or independently labeled
questions. The matched control repeats the 140 original examples three times.
Both new adapters start from the same base model and receive 420 examples per
epoch, two epochs, batch size 4, the same seed, optimizer, and learning rate.

| Model | Mean recorded points / prompt | Recorded match rate |
| --- | ---: | ---: |
| Base | 12.85 | 71% |
| Earlier short SFT (140 × 2 epochs) | **22.95** | **64%** |
| Repeated-original control (420 × 2 epochs) | 15.85 | 30% |
| Synthetic paraphrases (420 × 2 epochs) | 18.40 | 35% |

Synthetic paraphrases beat the step-matched repetition control by 2.55 points
per prompt. A paired bootstrap resampling whole days gives a 95% interval of
−1.99 to +7.32, so this small test does not establish a reliable gain. Both
longer runs did worse than the earlier short SFT adapter, and generated many
answers outside the archived sheet. The practical conclusion is to retain the
short SFT checkpoint as the best candidate so far and avoid simply increasing
training steps. See `results/fresh_summary.json`; the four detailed
`results/fresh_*.json` files remain in the local experiment directory.

```sh
python3 fresh.py prepare
python3 fresh.py train-control
python3 fresh.py train-synthetic
python3 fresh.py eval-base
python3 fresh.py eval-sft
python3 fresh.py eval-control
python3 fresh.py eval-synthetic
python3 fresh.py summarize
```

## Model-ranked candidate selection

`rerank.py` implements the intended answer-selection loop on the same 100
September prompts. It generates the model's greedy response plus seven sampled
alternatives, scores each complete response by its **mean token log probability
under that model**, and submits the highest-ranked response found in the
archived answer sheet. If none matches, it falls back to the top-ranked
response and scores zero. This is an approximate ranking over the generated
pool, not an exhaustive search over every possible answer.

| Model | One greedy answer | Ranked pool, exact match | Sheet match rate |
| --- | ---: | ---: | ---: |
| Base | 12.85 | 18.80 | 96% |
| Short SFT | 22.95 | **37.20** | 94% |
| SFT + short GRPO | — | 37.00 | 93% |

The SFT minus base difference under ranked selection is +18.4 points per
prompt; a paired bootstrap over the 15 dives gives a 95% interval of +14.1 to
+23.0. GRPO minus SFT is −0.2, with a 95% interval of −5.4 to +4.2.

The optional fuzzy matcher uses a ≥0.92 string similarity threshold plus
initial-letter and length checks. It runs **only when no candidate has an
exact match**, so it cannot displace a later exact match with a lower-scoring
near match. It added no points on this holdout. It is useful for minor spelling
or plural differences, but cannot safely resolve missing answers by meaning.

This selection method consults the held-out answer sheet at inference time.
For new daily dives, it requires access to the live checker or another
independent validity verifier. The archived score is still a snapshot score,
and answer omissions can undercount fair responses.

```sh
python3 rerank.py --count 8 --adapter base
python3 rerank.py --count 8 --adapter sft
python3 rerank.py --count 8 --adapter grpo
python3 summarize_rerank.py
```

## Stronger model baseline

`compare_4b.py` evaluates cached `Qwen/Qwen3.5-4B` without Krillion fine-tuning
on the same 100 September prompts. It disables thinking, uses the rules-aware
prompt, and generates one greedy plus seven sampled responses. The matching
and likelihood ranking are the same as above. The older Llama results used the
original prompt, so this is a practical baseline comparison rather than a
controlled test of model size alone.

| Model | Selected mean points | Best answer in generated pool | Mean distinct answers |
| --- | ---: | ---: | ---: |
| Llama 3.2 1B base | 18.80 | 28.60 | 3.97 |
| Llama 3.2 1B Krillion SFT | **37.20** | **55.10** | 6.60 |
| Qwen3.5 4B, no Krillion training | 17.25 | 26.10 | 3.32 |

For Qwen, the pool contained a higher-scoring answer than the likelihood
selection on 31 of 100 prompts. The SFT pool had a higher-scoring answer on
47 prompts. Task training helped both answer diversity and answer quality in
this pilot; simply moving to this larger zero-shot model did not. These
"best in pool" scores require the archived sheet and are diagnostic upper
bounds, not deployable scores. Qwen's aggregate output is in
[`results/qwen35_4b_fresh_rules_summary.json`](results/qwen35_4b_fresh_rules_summary.json).

```sh
python3 compare_4b.py
```

## November 1, 2026 live-day demo

`run.py` now also defines `SYSTEM_RULES`, which tells the model that accepted,
uncommon answers score more and asks it to choose a real, specific, high-tier
answer. It does not include the earlier phrase "without risking an invalid
answer"; that variant is preserved as `SYSTEM_RULES_CAUTIOUS`. The original
`SYSTEM` remains available for comparisons and for the training runs above.
The model sees only the prompt; `rerank.py` uses the answer sheet separately
to filter its eight ranked candidates.

The exact seven prompts and answer tiers came from Krillion's public
`/api/today` and `/api/reveal?date=2026-11-01` responses, saved in `data/raw/`.
With the revised rules-aware prompt, the short SFT adapter selected answers worth
30 + 60 + 85 + 30 + 30 + 10 + 60 = 305 points, or **3,050 m**. The cautious
wording gave 275 points (2,750 m), and the original prompt gave 190 points
(1,900 m), on this one day. The complete record is
`results/today_2026-11-01_score.json`. These are reveal-sheet-assisted scores;
the model did not submit a timed run to the website.

Full ranked candidates, likelihoods, choices, and scores are in local
`results/rerank_8*.json`; aggregate metrics are in
`results/rerank_summary.json`.

## Demo output: dive depth

[Krillion's public client](https://krillion.io/) states that every point sinks
the player 10 metres; seven 100-point answers reach 7,000 metres. For the demo,
`demo_depth.py` sums the seven archived tier scores for a date and multiplies
by 10. This changes **only the reported output unit**. `rl.py` still uses the
same tier-score reward divided by 100.

The 100-prompt benchmark had excluded five near-overlapping prompts. To make
complete seven-question dives, the depth demo scores those five separately and
combines them with the 100 existing outputs. It covers 15 complete September
dives. The benchmark metrics above retain the stricter 100-prompt split.

| Model | Mean complete-dive depth | Maximum possible |
| --- | ---: | ---: |
| Base | 1,313 m | 7,000 m |
| Short SFT | **2,657 m** | 7,000 m |
| SFT + short GRPO | 2,643 m | 7,000 m |

The [per-dive depths and seven scored answers](results/demo_depth.json)
are the demo output. These remain estimates based on the independent archive's
answer sheet rather than scores submitted to Krillion's server.

```sh
python3 demo_depth.py prepare
python3 rerank.py --count 8 --adapter base --dataset demo-excluded
python3 rerank.py --count 8 --adapter sft --dataset demo-excluded
python3 rerank.py --count 8 --adapter grpo --dataset demo-excluded
python3 demo_depth.py summarize
python3 summarize_rerank.py
```
