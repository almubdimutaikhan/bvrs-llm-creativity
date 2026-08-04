#!/usr/bin/env python3
"""
Does a judge rank CONTENT, or re-emit the order it was shown?

Takes real candidate sets from results/, presents each one to each judge under
TWO different shuffles, and asks the only question that matters:

    does the same concept get discarded both times?

A content-driven judge agrees with itself regardless of order. A position-driven
judge does not, because "the last one" is a different concept each time. This is
the order-consistency filter from the predecessor study, applied to the SELECTOR
rather than to the final evaluation -- and it should be run BEFORE a sweep, not
after one has already been paid for.

  python scripts/probe_judges.py
"""
import csv
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import ladder
import schema
from llm import chat

JUDGES = ["deepseek/deepseek-chat-v3.1",      # the one used in the sweep
          "google/gemini-2.5-flash",
          "anthropic/claude-sonnet-5",
          "openai/gpt-4o"]
POOL_SIZES = [6, 41]        # n_presented values to sample
ROUNDS_PER_SIZE = 4


def main():
    cands = list(csv.DictReader(schema.CANDIDATES_CSV.open(encoding="utf-8")))
    text = {(c["run_id"], c["candidate_id"]): (c["title"], c["text"])
            for c in cands}
    rounds = [r for r in csv.DictReader(schema.ROUNDS_CSV.open(encoding="utf-8"))
              if r["parse_status"] == "ok"]
    problem = ladder.load_problem("mars_boot")
    tpl = (ROOT / "prompts" / "select.txt").read_text(encoding="utf-8")

    picked = []
    for n in POOL_SIZES:
        grp = [r for r in rounds if int(r["n_presented"]) == n]
        picked += grp[:ROUNDS_PER_SIZE]

    jobs = []
    for r in picked:
        ids = r["presentation_order"].split(";")
        items = [(i, *text[(r["run_id"], i)]) for i in ids]
        for shuffle_no in (0, 1):
            order = list(items)
            random.Random(f"probe|{r['run_id']}|{r['round']}|{shuffle_no}"
                          .__hash__() & 0xffffffff).shuffle(order)
            for j in JUDGES:
                jobs.append((r, order, shuffle_no, j))

    print(f"{len(picked)} real candidate sets x {len(JUDGES)} judges "
          f"x 2 shuffles = {len(jobs)} calls\n")

    def one(job):
        r, order, shuffle_no, judge = job
        n = len(order)
        block = "\n\n".join(f"### Candidate {i+1}\nTITLE: {t}\n{b}"
                            for i, (_, t, b) in enumerate(order))
        res = chat(judge, "", tpl.format(n=n, problem=problem["statement"],
                                         candidates=block),
                   temperature=0.0, max_tokens=400,
                   nonce=f"judgeprobe|{r['run_id']}|{r['round']}|{shuffle_no}|{judge}")
        ranking, discard, status = ladder.parse_selection(res.text, n)
        ident = bool(ranking) and ranking == list(range(1, n + 1))
        return {"key": (r["run_id"], r["round"], judge), "n": n,
                "shuffle": shuffle_no, "status": status,
                "discard_id": order[discard - 1][0] if discard else None,
                "identity": ident}

    with ThreadPoolExecutor(max_workers=16) as ex:
        out = list(ex.map(one, jobs))

    by = defaultdict(dict)
    for o in out:
        by[o["key"]][o["shuffle"]] = o
    stats = defaultdict(lambda: {"agree": 0, "total": 0, "ident": 0,
                                 "calls": 0, "bad": 0})
    for (rid, rnd, judge), sh in by.items():
        n = list(sh.values())[0]["n"]
        s = stats[(judge, n)]
        for o in sh.values():
            s["calls"] += 1
            s["ident"] += int(o["identity"])
            s["bad"] += int(o["status"] != "ok")
        if len(sh) == 2 and all(o["discard_id"] for o in sh.values()):
            s["total"] += 1
            s["agree"] += int(sh[0]["discard_id"] == sh[1]["discard_id"])

    print(f"{'judge':30s} {'n':>3s} {'agree across shuffles':>22s} "
          f"{'identity echo':>14s} {'unparsed':>9s}")
    for (judge, n), s in sorted(stats.items()):
        agree = f"{s['agree']}/{s['total']}" if s["total"] else "-"
        pct = f" ({s['agree']/s['total']*100:.0f}%)" if s["total"] else ""
        print(f"{judge:30s} {n:3d} {agree + pct:>22s} "
              f"{s['ident']}/{s['calls']:<3d}{'':>7s} {s['bad']}/{s['calls']}")
    print(f"\nchance agreement at n is 1/n: {', '.join(f'n={n}: {100/n:.1f}%' for n in POOL_SIZES)}")
    print("A judge that ranks content should agree with itself well above chance.")


if __name__ == "__main__":
    main()
