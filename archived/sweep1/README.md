# Sweep #1 archive — qwen generator / DeepSeek judge, unseeded blind prompt

Archived 2026-08-04. Superseded by sweep #2 in the live `results/` directory,
which uses a seeded blind prompt, an anti-overlap sighted prompt, and
Gemini 2.5 Flash as judge (see root [`RESULTS.md`](../../RESULTS.md) §9 and
`src/run_study.py`'s `--judge-model` default).

## What's here

| file | rows | grain |
|---|---:|---|
| `runs.csv` | 50 | one execution — the experimental cell, totals, provenance |
| `candidates.csv` | 1,010 | one generated concept |
| `rounds.csv` | 500 | one selection round |
| `run_full.log` | — | console log of the sweep that produced the above |

## Run configuration

| | |
|---|---|
| generator | `qwen/qwen3-235b-a22b-2507` (non-thinking release), temperature 0.7 |
| selector | `deepseek/deepseek-chat-v3.1`, temperature 0.0 |
| blind prompt | unseeded — "propose ONE concept," no anchor |
| design | 2 arms × pool size `n` ∈ {1, 2, 3, 5, 40} × 5 seeds × 10 rounds |
| scale | 50 runs · 1,010 candidates · 500 selection rounds |
| cost | 1,607 calls · 5.03M tokens · ≈ $1.12 · 3.6 core-hours (30 workers) |

## Why archived, not deleted

Two problems, both fully documented in the root `RESULTS.md`:

1. **The selector was mostly positional, not content-driven.** DeepSeek
   discarded the last-shown candidate at far above chance (e.g. 33% at
   n=41, where chance is 2.4%). A follow-up probe (`scripts/probe_judges.py`)
   found DeepSeek was not self-consistent across shuffles at any pool size
   tested — and neither were GPT-4o or Claude Sonnet 5, which additionally
   ignored the required output format outright. Gemini 2.5 Flash was the
   only judge showing content sensitivity above chance, hence the switch.
2. **Blind generation converged on one idea.** 314 of 1,010 candidates
   (31%) carried one of five near-duplicate titles, all describing
   piezoelectric sensors in the boot sole. Sweep #2's blind prompt seeds
   that exact idea deliberately instead of leaving it to chance.

The generated candidates are still good data and reusable as-is — blind-arm
generation never depended on the pool, so it isn't invalidated by the
selector bug. Full findings, tables, and reproduction commands live in the
root [`RESULTS.md`](../../RESULTS.md), which describes this exact sweep.
