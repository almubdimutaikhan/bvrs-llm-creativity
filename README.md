# Does generating more ideas make them better?

**One problem. One model. A bare prompt. Then ten rounds of add-one, drop-one.**

The question is old and the answer is assumed. Donald Campbell argued in 1960
that creativity works like evolution: throw out variations *blindly*, let
selection keep what survives. Dean Keith Simonton spent sixty years formalising
it, and the current statement of the theory has actual arithmetic in it —
creativity `c = (1 − p)·u·(1 − v)`, sightedness `s = p·u·v`, and the claim that
**creativity is inversely related to sightedness**. The better you already know
which idea will work, the less creative it can be.

Large language models are the most sighted idea generators ever built. They are
trained on the answers. So the theory makes a sharp, uncomfortable prediction,
and nobody has tested it on them.

This repo tests it.

---

## The design in one picture

```
                    ┌──────────── pool of n ────────────┐
                    │  c000   c001   c002   ...   cN    │
                    └───────────────┬───────────────────┘
                                    │
             generate 1 more ───────┤
                                    │
      BLIND   fresh context,        │      SIGHTED  sees the whole pool,
      brief only, never sees  ──────┤──────  "here is what we have,
      the pool                      │        do better"
                                    ▼
                    ┌──────────── n + 1 ────────────────┐
                    │  shuffled, shown to a judge in ONE│
                    │  call, ranked best → worst        │
                    └───────────────┬───────────────────┘
                                    │  discard the worst
                                    ▼
                          back to a pool of n     × 10 rounds
```

Two knobs, and that is the entire experiment:

| knob | values | what it controls |
|---|---|---|
| **arm** | `blind` / `sighted` | where variation comes from — selection alone, or guided search |
| **pool size `n`** | 1, 2, 3, 5, 40 | **selection pressure** |

At `n=1`, half the population dies every round — pure exploitation. At `n=40`,
one of 41 dies — almost pure exploration. So `n` is a dial that runs from
convergent to divergent thinking, as a number you set rather than a prompt style
you hope the model honours. Crossed with the arm, it gives four named regimes:

|  | `n = 1` | `n = 40` |
|---|---|---|
| **blind** | random restart, hard selection | pure BVSR |
| **sighted** | iterative self-refinement | population-informed generation |

In evolutionary-computation terms this is a **(μ+1) evolution strategy** and the
sweep is a μ-sweep. That is deliberate: the psychology names the hypothesis, the
optimisation literature names the algorithm.

---

## What gets measured

**Quality against tokens — not against round.** A survivor at `n=40` cost five
times the generations of one at `n=1`. Plotting quality per round would hide
that; plotting it per token is the honest trade-off curve, and it is the chart
this study is for.

**Quality is ranked by a human.** The LLM judge inside the loop is a *component
of the method*, not the measurement — otherwise the study grades its own
homework. Final ranking is blind human 2AFC, and the natural reference is
candidate `c000`: the model's very first bare-prompt answer, already generated,
free.

> **The headline question:** does ten rounds of generate-and-eliminate beat the
> first thing the model said?

**Three diagnostics that come free, and any one of them can end the study early:**

- `discarded_was_challenger` — if the newcomer dies every round, generation is
  contributing nothing and the ladder is theatre.
- `discarded_position` vs `presentation_order` — the judge sees a shuffled list
  each round. Regress discard on position and you measure position bias directly
  instead of hoping it cancels.
- `rank_score` spread — if concepts are indistinguishable, selection has nothing
  to select on, and the curve will be flat no matter how many rounds you run.

---

## Output schema

Three tables, joined on `run_id` and `(run_id, candidate_id)`.

| file | grain | holds |
|---|---|---|
| `results/runs.csv` | one execution | the experimental cell, totals, provenance |
| `results/candidates.csv` | one concept | generation metadata, length, fate, score |
| `results/rounds.csv` | one selection | who was shown, in what order, who died, cost |

**CSV only — no side files.** The concept itself is in `candidates.text` and the
judge's verbatim reply is in `rounds.raw_response`. That is safe because
`append_row` whitespace-flattens *every* value on the way out, so a row can never
span more than one physical line no matter what the model returns. `text_sha256`
hashes the stored string, so it verifies from the CSV alone — and repeated hashes
are the fastest way to catch a generator that has started repeating itself.

What still stays out of the rows: the problem statement (identical in every row)
lives once in `problems/problems.json`, referenced by `problem_id`; prompt
templates live in `prompts/`, referenced by `gen_prompt_sha` / `sel_prompt_sha`.
Pool membership is recorded as ID lists, never as text.

Full column lists and the reasoning are in [`src/schema.py`](src/schema.py).

**Nothing is ever really deleted.** Discarded concepts keep their row with
`died_round` set, and every judge reply is retained. The elimination can
therefore be re-analysed later under a different rule without spending a cent.

---

## Two traps this code is built to avoid

**1. The cache nonce.** The blind arm sends a byte-identical prompt for every
candidate. With a content-addressed cache and no nonce, all candidates — and all
replicates — collapse onto one cached response. The pool fills with copies, the
study measures nothing, and it *looks like it worked*. Every call therefore
carries a nonce unique to the candidate, and the offline test asserts no nonce
repeats.

**2. Blindness has to be auditable.** `context_candidate_ids` records exactly
what each generator was shown. For the blind arm it must be empty in every row.
That is a one-line check, and without it "blind" is a claim rather than a fact.

---

## Run it

```bash
pip install -r requirements.txt
export AI_OPENROUTER_API_KEY=...

python src/run_study.py --dry-run                                # 0 API calls
python src/ladder.py --arm blind --pool-size 1 --rounds 10 --seed 0   # one ladder
python src/run_study.py --pool-sizes 1 2 3 5 40 --arms blind sighted --seeds 5
```

Every call is disk-cached, so re-runs are free and an interrupted sweep resumes.
The full sweep is ~1,500 calls / ~4.7M tokens by the dry-run estimate — single
digit dollars on Gemini Flash + DeepSeek, considerably more on a frontier model.
More than half of that budget is `sighted, n=40`, where each generation carries
40 concepts in its prompt.

---

## Status

Code is written and passes an offline end-to-end test. **No API calls have been
made and `results/` is empty.** There are no findings in this README — only a
design.

## Scope, stated plainly

One brief, one generator model. This is a **mechanism study**, not a
generalisation: the finding will be about this problem and this model.
Replicates (seeds) are the unit of analysis, which is what gives it real n.
Generalising across problems and models is a second study.

The result may well be flat. If LLM design concepts are homogeneous, selection
has little to select on and no amount of rounds will help — that is a clean,
reportable finding, and this design was chosen partly because both outcomes are
publishable.

## Credits

Blind-variation and selective-retention theory: Campbell, D. T. (1960),
*Psychological Review* 67(6), 380–400; Simonton, D. K. (2022), "The
Blind-Variation and Selective-Retention Theory of Creativity: Recent
Developments and Current Status of BVSR", *Creativity Research Journal*
([open access](https://doi.org/10.1080/10400419.2022.2059919)). BVSR is
contested — see Sternberg (1998), Gabora, Dasgupta and Weisberg — which is why
it is worth testing rather than assuming.

The Mars boot brief is case 029 of the sibling TRIZ-2AFC study.
