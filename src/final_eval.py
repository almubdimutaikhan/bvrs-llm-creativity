#!/usr/bin/env python3
"""
The headline contrast: does the ladder's survivor beat the model's FIRST answer?

For every run, pit final_candidate_id against c000 — the very first bare-prompt
concept of that same run, already generated and therefore free. Both texts come
from the same model, same brief, same temperature; the only difference is that
one went through n+1 rounds of generate-and-eliminate and the other is what the
model said immediately.

Controls carried from the predecessor study, where they mattered:
  - BOTH orders. Only order-consistent verdicts count; a judge that flips when
    the texts swap is expressing position bias, not preference.
  - The final judge is neither the generator NOR the in-loop selector. The
    selection judge chose the survivor; asking it to grade its own choice is
    not an evaluation.
  - Independent scoring before the pick, which lifted order-consistency
    50% -> 69% previously.

This is still an LLM proxy. The real measure is the human ranking; this exists
so the sweep can be read the day it finishes.

  python src/final_eval.py --dry-run
  python src/final_eval.py --judges google/gemini-2.5-flash
"""
import argparse
import csv
import itertools
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ladder
import schema
from llm import chat

ROOT = Path(__file__).resolve().parent.parent
OUT = schema.RESULTS / "final_eval.csv"

COLUMNS = [
    "run_id", "arm", "pool_size", "seed", "gen_model", "loop_judge",
    "judge_model", "survivor_id", "first_id", "order",
    "a_id", "b_id", "a_role", "b_role",
    "score_a", "score_b", "pick", "picked_id", "picked_role",
    "prompt_tokens", "completion_tokens", "latency_seconds", "cached",
    "raw_response",
]

_PICK = re.compile(r"answer\s*[:=]\s*\**\s*([AB])", re.I)
_SA = re.compile(r"score\s*a\s*[:=]\s*\**\s*([0-9.]+)", re.I)
_SB = re.compile(r"score\s*b\s*[:=]\s*\**\s*([0-9.]+)", re.I)


def family(model: str) -> str:
    return model.split("/")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=["google/gemini-2.5-flash"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    runs = list(csv.DictReader(schema.RUNS_CSV.open(encoding="utf-8")))
    cands = list(csv.DictReader(schema.CANDIDATES_CSV.open(encoding="utf-8")))
    text = {(c["run_id"], c["candidate_id"]): c["text"] for c in cands}
    tpl = (ROOT / "prompts" / "final_eval.txt").read_text(encoding="utf-8")

    pairs, degenerate = [], 0
    for r in runs:
        surv, first = r["final_candidate_id"], "c000"
        if surv == first:
            degenerate += 1          # the ladder never displaced the first answer
            continue
        ts, tf = text.get((r["run_id"], surv)), text.get((r["run_id"], first))
        if not ts or not tf:
            continue
        pairs.append((r, surv, first, ts, tf))

    jobs = [(p, j, o) for p in pairs for j in a.judges for o in ("sf", "fs")
            if family(j) not in (family(p[0]["model_name"]),
                                 family(p[0]["judge_model"]))]
    print(f"{len(runs)} runs -> {len(pairs)} comparable pairs "
          f"({degenerate} where the survivor IS the first answer) "
          f"-> {len(jobs)} judgements")
    if not jobs:
        sys.exit("no jobs — every judge shares a family with the generator "
                 "or the in-loop selector")
    if a.dry_run:
        print("  --dry-run: no API calls made")
        return

    problem = ladder.load_problem(runs[0]["problem_id"])

    def one(job):
        (r, surv, first, ts, tf), judge, order = job
        if order == "sf":
            a_id, b_id, a_txt, b_txt = surv, first, ts, tf
            a_role, b_role = "survivor", "first"
        else:
            a_id, b_id, a_txt, b_txt = first, surv, tf, ts
            a_role, b_role = "first", "survivor"
        res = chat(judge, "", tpl.format(problem=problem["statement"],
                                         a=a_txt, b=b_txt),
                   temperature=0.0, max_tokens=300,
                   nonce=f"final|{r['run_id']}|{judge}|{order}")
        m = _PICK.search(res.text or "")
        pick = m.group(1).upper() if m else ""
        return {
            "run_id": r["run_id"], "arm": r["arm"], "pool_size": r["pool_size"],
            "seed": r["seed"], "gen_model": r["model_name"],
            "loop_judge": r["judge_model"], "judge_model": judge,
            "survivor_id": surv, "first_id": first, "order": order,
            "a_id": a_id, "b_id": b_id, "a_role": a_role, "b_role": b_role,
            "score_a": (_SA.search(res.text or "") or [None, ""])[1]
                if _SA.search(res.text or "") else "",
            "score_b": (_SB.search(res.text or "") or [None, ""])[1]
                if _SB.search(res.text or "") else "",
            "pick": pick,
            "picked_id": {"A": a_id, "B": b_id}.get(pick, ""),
            "picked_role": {"A": a_role, "B": b_role}.get(pick, ""),
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "latency_seconds": res.latency_seconds,
            "cached": str(res.cached).lower(),
            "raw_response": res.text,
        }

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        records = list(ex.map(one, jobs))
    for rec in records:
        schema.append_row(OUT, COLUMNS, rec)
    print(f"\n{len(records)} judgements -> {OUT}")

    report(records)


def report(records):
    """Collapse the two orders; only agreeing verdicts count."""
    by = defaultdict(dict)
    meta = {}
    for r in records:
        key = (r["run_id"], r["judge_model"])
        by[key][r["order"]] = r["picked_role"]
        meta[key] = (r["arm"], int(r["pool_size"]))

    kept, flipped, unparsed = [], 0, 0
    for key, orders in by.items():
        if len(orders) < 2 or not all(orders.values()):
            unparsed += 1
            continue
        if len(set(orders.values())) == 1:
            kept.append((meta[key], orders["sf"]))
        else:
            flipped += 1
    total = len(by)
    print(f"\n{'='*66}\nSURVIVOR vs FIRST ANSWER")
    print(f"  {total} comparisons | order-consistent {len(kept)} "
          f"({len(kept)/max(total,1)*100:.0f}%) | flipped {flipped} | "
          f"unparsed {unparsed}")
    if not kept:
        print("  nothing order-consistent — no conclusion available")
        return
    print(f"\n{'arm':9s} {'n':>3s} {'n_runs':>7s} {'survivor wins':>14s}")
    for cell in sorted({c for c, _ in kept}):
        grp = [w for c, w in kept if c == cell]
        wins = sum(1 for w in grp if w == "survivor")
        print(f"{cell[0]:9s} {cell[1]:3d} {len(grp):7d} "
              f"{wins/len(grp)*100:13.1f}%")
    allw = sum(1 for _, w in kept if w == "survivor")
    print(f"\n  pooled: survivor wins {allw}/{len(kept)} "
          f"({allw/len(kept)*100:.1f}%)")
    print(f"\n  ONE problem, ONE generator. The replicate (seed) is the unit;\n"
          f"  these are descriptive percentages, not a significance test.")


if __name__ == "__main__":
    main()
