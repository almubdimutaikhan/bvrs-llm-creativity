#!/usr/bin/env python3
"""
Sweep the ladder over pool sizes, arms and replicates.

    pool sizes  n in {1, 2, 3, 5, 40}   selection pressure, high -> low
    arms        blind | sighted         where variation comes from
    seeds       replicates              THE UNIT OF ANALYSIS

One problem, one generator model. With a single brief the replicate is the
independent observation, which is what gives this design real n — unlike a
few-cases design where the case is the unit and n stays tiny.

Runs already present in results/runs.csv are skipped, so the sweep resumes.

  python src/run_study.py --dry-run
  python src/run_study.py --pool-sizes 1 --arms blind --seeds 1   # smoke
  python src/run_study.py --pool-sizes 1 2 3 5 40 --arms blind sighted --seeds 5
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ladder
import schema

# Rough per-call sizes, for --dry-run only. Measured values land in the CSVs.
EST_CONCEPT_TOKENS = 320
EST_GEN_OVERHEAD = 260
EST_SEL_OVERHEAD = 300
EST_GEN_OUT = 340
EST_SEL_OUT = 60


def existing_run_ids() -> set:
    if not schema.RUNS_CSV.exists():
        return set()
    with schema.RUNS_CSV.open(encoding="utf-8") as fh:
        return {r["run_id"] for r in csv.DictReader(fh)}


def estimate(arm: str, n: int, rounds: int, cap) -> tuple:
    """(calls, tokens) — an estimate, not a measurement."""
    calls = (n + rounds) + rounds
    gen = 0
    for k in range(n + rounds):
        pool_seen = 0
        if arm == "sighted":
            pool_seen = min(k, n)
            if cap:
                pool_seen = min(pool_seen, cap)
        gen += EST_GEN_OVERHEAD + pool_seen * EST_CONCEPT_TOKENS + EST_GEN_OUT
    sel = rounds * (EST_SEL_OVERHEAD + (n + 1) * EST_CONCEPT_TOKENS + EST_SEL_OUT)
    return calls, gen + sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="mars_boot")
    ap.add_argument("--pool-sizes", nargs="+", type=int, default=[1, 2, 3, 5, 40])
    ap.add_argument("--arms", nargs="+", default=["blind", "sighted"],
                    choices=["blind", "sighted"])
    ap.add_argument("--seeds", type=int, default=5, help="replicates per cell")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    ap.add_argument("--judge-model", default="deepseek/deepseek-chat-v3.1")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--sighted-cap", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.model.split("/")[0] == a.judge_model.split("/")[0]:
        print(f"[!] generator and judge are the same family "
              f"({a.model.split('/')[0]}) — self-preference bias is unmixable "
              f"with one generator model.\n")

    problem = ladder.load_problem(a.problem)
    cells = [(arm, n, s) for arm in a.arms for n in sorted(a.pool_sizes)
             for s in range(a.seeds)]
    done = existing_run_ids()

    total_calls = total_tokens = 0
    print(f"{problem['name']}  |  {a.model}  gen  /  {a.judge_model}  select")
    print(f"{len(a.arms)} arms x {len(a.pool_sizes)} pool sizes x {a.seeds} "
          f"seeds = {len(cells)} runs, {a.rounds} rounds each\n")
    print(f"{'arm':8s} {'n':>3s} {'calls':>7s} {'est tokens':>12s}")
    for arm in a.arms:
        for n in sorted(a.pool_sizes):
            c, t = estimate(arm, n, a.rounds, a.sighted_cap)
            total_calls += c * a.seeds
            total_tokens += t * a.seeds
            print(f"{arm:8s} {n:3d} {c:7d} {t:12,d}   x{a.seeds} seeds")
    print(f"\ntotal: {total_calls:,} calls, ~{total_tokens:,} tokens "
          f"(estimate; cached calls cost nothing)")
    if done:
        print(f"{len(done)} run(s) already in runs.csv will be skipped")
    if a.dry_run:
        print("\n--dry-run: no API calls made")
        return

    for i, (arm, n, seed) in enumerate(cells, 1):
        run_id = (f"{problem['id']}__{arm}__n{n}__"
                  f"{ladder._slug(a.model)}__s{seed}")
        if run_id in done:
            print(f"[{i}/{len(cells)}] skip {run_id}")
            continue
        row = ladder.run(problem, a.model, a.judge_model, arm, n, a.rounds,
                         seed, a.temperature, a.sighted_cap)
        flag = ""
        if row["n_parse_failures"]:
            flag += f"  [!] {row['n_parse_failures']} parse failures"
        if row["n_challenger_discarded"] == a.rounds:
            flag += "  [!] challenger died every round"
        print(f"[{i}/{len(cells)}] {run_id}  "
              f"{row['total_tokens']:,} tok  {row['total_wall_seconds']}s{flag}")

    print(f"\n-> {schema.RUNS_CSV}")
    print(f"-> {schema.CANDIDATES_CSV}")
    print(f"-> {schema.ROUNDS_CSV}")


if __name__ == "__main__":
    main()
