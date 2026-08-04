# Results — sweep #1 (Mars haptic boot)

**Headline: the sweep ran cleanly and does not answer the research question.**
The selection step was not selecting — the judge was largely re-emitting the
order it was shown. A second, unrelated finding *is* valid and does not depend
on the selector: the generator produced the same concept over and over.

Nothing in this document is a quality claim. No human evaluation has been run.

---

## 1. What was run

| | |
|---|---|
| problem | `mars_boot` — Mars haptic boot (one brief) |
| generator | `qwen/qwen3-235b-a22b-2507` (non-thinking release), temperature 0.7 |
| selector | `deepseek/deepseek-chat-v3.1`, temperature 0.0 |
| design | 2 arms × pool size *n* ∈ {1, 2, 3, 5, 40} × 5 seeds × 10 rounds |
| scale | 50 runs · 1,010 candidates · 500 selection rounds |
| cost | 1,607 calls · 5.03M tokens · **≈ $1.12** · 3.6 core-hours (30 workers) |

Raw data (archived — `results/` now holds sweep #2, see §9):
`archived/sweep1/runs.csv`, `archived/sweep1/candidates.csv`,
`archived/sweep1/rounds.csv`.

## 2. Run health — all guards passed

| check | result |
|---|---|
| selection rounds cleanly parsed | **499 / 500** (one `discard_only` at n=41) |
| leaked reasoning tokens | **0** — the token axis is uncontaminated |
| empty/truncated generations | 99 caught and regenerated; **1 of 1,010** still short, and it was discarded anyway |
| concept length | median 215 words, p10 203, max 241 |
| blind arm saw pool context | never — `context_candidate_ids` empty on every blind row |

The infrastructure did its job. The problem is one level up.

---

## 3. Finding 1 (blocking) — the judge echoed presentation order

Every round shuffles the candidates before showing them. DeepSeek nonetheless
discarded the **last-shown** candidate 33% of the time at n=41, where chance is
2.4%. Measured across the sweep's own 500 rounds:

| candidates shown | returned the exact presented order | Spearman(rank, position) | last slot discarded | chance |
|---:|---:|---:|---:|---:|
| 2 | **91.0%** | 0.82 | 91% | 50% |
| 3 | 30.0% | 0.52 | 60% | 33% |
| 4 | 11.0% | 0.54 | 31% | 25% |
| 6 | 1.0% | 0.49 | 26% | 17% |
| 41 | **16.2%** | 0.46 | 33% | 2.4% |

Returning a 41-item list verbatim in 16% of rounds is not a ranking.

Because the presentation order is shuffled each round, this does **not** bias
toward any particular concept — it converts elimination into noise. The
arithmetic closes exactly in the n=1 cells:

| arm | last slot discarded | challenger discarded | challenger *was* last |
|---|---:|---:|---:|
| blind | 96.0% | **50.0%** | **46.0%** |
| sighted | 86.0% | 68.0% | 58.0% |

The blind challenger died at 50.0% and occupied the last slot 46.0% of the time.
Position accounts for the dynamics; no content signal is needed to explain them.

### Consequence

The dynamics table below is **recorded, not interpreted**. Every number in it is
consistent with a selection operator that is mostly positional, so none of it can
be read as evidence about selection pressure, about `blind` vs `sighted`, or
about BVSR.

| arm | n | challenger died | leader turnover | gens | tokens |
|---|---:|---:|---:|---:|---:|
| blind | 1 | 50.0% | 52.0% | 11 | 15,168 |
| blind | 2 | 34.0% | 50.0% | 12 | 18,739 |
| blind | 3 | 42.0% | 42.0% | 13 | 22,397 |
| blind | 5 | 30.0% | 22.0% | 15 | 29,792 |
| blind | 40 | 2.0% | 68.0% | 50 | 154,198 |
| sighted | 1 | 68.0% | 34.0% | 11 | 20,024 |
| sighted | 2 | 52.0% | 48.0% | 12 | 28,738 |
| sighted | 3 | 44.0% | 38.0% | 13 | 36,049 |
| sighted | 5 | 20.0% | 38.0% | 15 | 56,228 |
| sighted | 40 | 2.0% | 54.0% | 50 | 625,600 |

At n=40 the challenger is 1 of 41, so 2.4% is chance. Both arms sit at 2.0%.

---

## 4. Finding 2 — which judges can select at all

`scripts/probe_judges.py` applies the predecessor study's order-consistency
filter to the *selector*: take a real candidate set, present it under two
different shuffles, and ask whether the **same concept dies both times**. A
content-driven judge agrees with itself regardless of order.

| judge | n=6 (chance 17%) | n=41 (chance 2.4%) | notes |
|---|---:|---:|---|
| `google/gemini-2.5-flash` | **3/4 (75%)** | 0/4 | only judge above chance |
| `deepseek/deepseek-chat-v3.1` | 1/4 (25%) | 0/4 | the one used in the sweep |
| `openai/gpt-4o` | 1/4 (25%) | 5/8 unparsed | |
| `anthropic/claude-sonnet-5` | 8/8 unparsed | 8/8 unparsed | ignores the output format |

**Two conclusions, the second much firmer than the first:**

1. Gemini is the only judge showing content sensitivity, and only at small *n*.
   Consistent with it measuring near-neutral on position bias in the predecessor
   study. **Caveat: 4 sets per cell — directional, not established.**
2. **No judge is self-consistent at n=41.** One-shot ranking of 41 concepts is
   not a usable selection mechanism at any price. This kills the n=40 cell as
   designed — which is the divergent end of the dial, and the cell the study most
   wanted.

---

## 5. Finding 3 (valid, and independent of the selector) — variation is not blind

This does not depend on the judge at all: it is a property of the 1,010 generated
concepts, and generation is unaffected by the selection bug.

| | candidates | distinct texts | byte-identical repeats |
|---|---:|---:|---:|
| blind | 505 | 471 | 34 |
| sighted | 505 | 468 | 37 |
| **total** | **1,010** | **938** | **72** |

Exact repeats are the mild part. The concepts converge far harder than byte
equality suggests — the five most common titles across all 1,010 candidates:

```
"sensory sole"                        114
"sensory sole haptic feedback system"  75
"sensoryflex haptic boot"              53
"sensory conduit boot"                 46
"sensory sole haptic boot"             26
```

**314 of 1,010 concepts (31%) carry one of five titles.** The blind arm generates
each candidate in a fresh context with no knowledge of the pool, at temperature
0.7 — and still returns the same idea: piezoelectric sensors in the sole driving
actuators against the foot.

In Campbell's terms, variation that is supposed to be "uncorrelated with the
solution" is tightly correlated with a single solution. In Simonton's, initial
generation probability `p` and prior utility knowledge `v` are both high, so
sightedness `s = puv` is high and creativity `c = (1−p)u(1−v)` is small. The
theory's central prediction about LLMs holds here.

A pilot probe (8 blind generations each across four models) found the same
thing, and found that a stronger model does not fix it: Claude Sonnet 5 was the
most convergent of the four, with six of eight titles beginning "Pneumatic".

---

## 6. What cannot be concluded

- Nothing about whether iterating helps.
- Nothing about `blind` vs `sighted`.
- Nothing about selection pressure or the pool-size sweep.
- Nothing about quality — `final_eval.py` was deliberately **not** run. Grading
  survivors that were chosen largely by position would produce a number that
  looks like a result and is not one.

## 7. What survives

The expensive half. **1,010 generated candidates are good data** and are reusable
as-is. Blind-arm generation does not depend on the pool, so those prompts are
already in the disk cache: re-running the blind ladders under a different
selector costs *selection calls only* (~250 calls). The sighted arm must be
regenerated, because its prompts contain the pool.

## 8. Next steps, in order

1. **Widen the judge probe to ~20 sets** before spending again. Four sets is too
   thin to bet a re-run on. ~160 calls.
2. **Re-run n ∈ {1, 2, 3, 5} with Gemini** as selector, if the probe holds.
3. **Redesign n=40 rather than re-running it.** One-shot ranking is out. The
   Bradley–Terry alternative — challenger vs 3 random incumbents, both orders,
   cost constant in *n* — sidesteps the problem, because pairwise comparison is
   the one thing these judges do adequately.
4. **Add a diversity metric as a first-class outcome.** Given §5, a flat
   pool-size curve would be uninterpretable without it: "selection pressure does
   not matter" and "there was nothing to select between" produce the same graph.

## 9. Prompt change since this sweep — seeding the blind arm

`prompts/generate_blind.txt` and `prompts/generate_sighted.txt` were rewritten
after this sweep. **Any future run using them is not directly comparable to
the data in this document**, especially the blind arm.

**What changed.** The blind prompt used for sweep #1 was fully open: "propose
ONE concept that solves the problem below," no anchor. Every blind call now
opens with a fixed line instead:

> SEED IDEA: Piezo Boots — piezoelectric sensors in the boot sole feeding
> haptic actuators against the foot.

and asks the model to build its own variant from that seed rather than
restate it. The sighted prompt gained an explicit instruction to not overlap
with any idea already in the pool ("TABLE 1") and to prioritize a wild,
creative idea over feasibility or cost.

**Why.** §5 already showed the unseeded blind arm converges on this exact
mechanism without being asked to: 314 of 1,010 candidates (31%) carry one of
five near-duplicate titles, all piezoelectric-sensors-in-the-sole. That
convergence was treated as an accidental finding — noise the design had to
explain away. But it is not going away with a different phrasing or a
stronger model (§5, and the four-model pilot probe): it is the model's
dominant prior for this brief. Leaving the blind prompt unseeded meant real
API spend went toward measuring *which surface variant of the same idea* a
call would land on, not toward the generate-and-eliminate mechanism the study
exists to test. Seeding turns that convergence from an accidental confound
into a controlled starting condition: every replicate now begins from the
same fixed point, so a difference observed between arms or pool sizes going
forward is attributable to the ladder mechanism, not to which idea a given
replicate happened to free-associate to.

**Judge also changed, separately.** Seeding fixes the generator side only —
it has no bearing on Finding 1 (§3), the judge echoing presentation order.
That gets addressed by a second, independent change for sweep #2: the
selector moves from `deepseek/deepseek-chat-v3.1` to `google/gemini-2.5-flash`,
per §4's own probe — Gemini was the only judge of the four tested showing
above-chance content sensitivity at n=6 (75% vs. 17% chance), and unlike
GPT-4o and Claude Sonnet 5 it did not outright ignore the output format. This
is a probe-backed choice, not a validated fix: §4 also found Gemini's n=41
self-consistency was no better than chance, so `n=40` is still not trustworthy
without the Bradley–Terry redesign in §8 item 3.

Sweep #1's own selector data (archived, §1) is unaffected by this — it still
demonstrates the DeepSeek/positional-bias finding above regardless of what
sweep #2 does.

## 10. Reproduce

```bash
python src/run_study.py --dry-run                     # the plan, 0 calls
python src/analyze.py                                 # §2, §3 tables, on whatever is currently in results/
python scripts/probe_judges.py                         # §4 table, on whatever is currently in results/
```

`analyze.py` and `probe_judges.py` read `results/`, which now holds sweep #2
(seeded prompts, Gemini judge) — not the sweep #1 data this document
describes. To reproduce §2/§3/§4 exactly as reported here, point them at
`archived/sweep1/` instead (or copy those three CSVs back into `results/`).

## 11. Method note worth keeping

The order-consistency check in §4 should have been run on the *selector* before
the sweep, not after. It costs ~60 calls and would have caught this for about 5%
of what the sweep cost. Every guard that was built in advance — nonce collision,
empty-concept regeneration, blindness audit, reasoning-token accounting — held.
The one failure mode nobody instrumented in advance is the one that invalidated
the run.
