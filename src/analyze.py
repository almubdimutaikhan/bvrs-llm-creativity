#!/usr/bin/env python3
"""
Read the three CSVs and report what the sweep actually did.

This reports DYNAMICS and HEALTH, not quality. Quality cannot come from inside
the loop: rank_score is relative to whichever pool a candidate happened to face,
so it is not comparable across runs, and the judge that produced it is part of
the method under test. The quality answer comes from src/final_eval.py (survivor
vs the model's first answer) and ultimately from human ranking.

What is decision-relevant here:

  challenger discard rate  if the newcomer nearly always dies, generation is
                           contributing nothing and the ladder is theatre
  leader turnover          how often the top concept actually changes
  position bias            discards should be spread across presented slots;
                           clustering means the judge is reading position
  parse health             a valid permutation of 1..41 is hard; if n=40 fails
                           to parse, its discards were random and it measures
                           nothing
  duplicate concepts       identical hashes = the generator repeating itself,
                           which drains the blind arm of variation

  python src/analyze.py
"""
import csv
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import schema


def load(path):
    if not path.exists():
        sys.exit(f"missing {path} — run src/run_study.py first")
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main():
    runs = load(schema.RUNS_CSV)
    cands = load(schema.CANDIDATES_CSV)
    rounds = load(schema.ROUNDS_CSV)

    by_run_c = defaultdict(list)
    for c in cands:
        by_run_c[c["run_id"]].append(c)
    by_run_r = defaultdict(list)
    for r in rounds:
        by_run_r[r["run_id"]].append(r)

    print(f"{len(runs)} runs | {len(cands)} candidates | {len(rounds)} rounds")
    models = {r["model_name"] for r in runs}
    judges = {r["judge_model"] for r in runs}
    temps = {r["temperature"] for r in runs}
    print(f"generator {'/'.join(models)}   judge {'/'.join(judges)}   "
          f"temp {'/'.join(temps)}")

    # ---- health first: if this is bad, nothing below means anything
    print(f"\n{'='*72}\nHEALTH")
    bad_parse = [r for r in rounds if r["parse_status"] != "ok"]
    print(f"  selection rounds not cleanly parsed : {len(bad_parse)}/{len(rounds)}")
    if bad_parse:
        for n_pres, grp in sorted(Counter(
                r["n_presented"] for r in bad_parse).items()):
            tot = sum(1 for r in rounds if r["n_presented"] == n_pres)
            print(f"      n_presented={n_pres:>3s}: {grp}/{tot} "
                  f"({Counter(r['parse_status'] for r in bad_parse if r['n_presented']==n_pres)})")
    short = sum(int(num(r["n_short_generations"])) for r in runs)
    reasoning = sum(int(num(r["total_reasoning_tokens"])) for r in runs)
    print(f"  regenerated (empty/short)           : {short}")
    print(f"  leaked reasoning tokens             : {reasoning}"
          + ("   [!] token axis is contaminated" if reasoning else ""))
    dupes = 0
    for rid, cs in by_run_c.items():
        h = Counter(c["text_sha256"] for c in cs)
        dupes += sum(v - 1 for v in h.values() if v > 1)
    print(f"  byte-identical concepts within a run: {dupes}")

    # ---- position bias
    print(f"\n{'='*72}\nPOSITION BIAS  (discards should spread evenly across slots)")
    for n_pres in sorted({int(r["n_presented"]) for r in rounds}):
        grp = [r for r in rounds if int(r["n_presented"]) == n_pres
               and r["parse_status"] == "ok"]
        if not grp:
            continue
        rel = [(num(r["discarded_position"]) - 1) / (n_pres - 1) for r in grp]
        first = sum(1 for r in grp if num(r["discarded_position"]) == 1)
        last = sum(1 for r in grp if num(r["discarded_position"]) == n_pres)
        print(f"  {n_pres:3d} shown, {len(grp):4d} rounds | mean rel. pos "
              f"{st.mean(rel):.3f} (0.5 = unbiased) | "
              f"first slot {first/len(grp)*100:4.1f}%  "
              f"last slot {last/len(grp)*100:4.1f}%  "
              f"(chance {1/n_pres*100:.1f}% each)")

    # ---- the dynamics table
    print(f"\n{'='*72}\nDYNAMICS")
    print(f"{'arm':8s} {'n':>3s} {'runs':>5s} {'chal.died':>10s} {'turnover':>9s} "
          f"{'gens':>5s} {'tokens':>9s} {'sec':>6s}")
    cells = sorted({(r["arm"], int(r["pool_size"])) for r in runs},
                   key=lambda t: (t[0], t[1]))
    for arm, n in cells:
        rs = [r for r in runs if r["arm"] == arm and int(r["pool_size"]) == n]
        rds = [x for r in rs for x in by_run_r[r["run_id"]]]
        if not rds:
            continue
        died = sum(1 for x in rds if x["discarded_was_challenger"] == "true")
        turn = sum(1 for x in rds if x["top_changed"] == "true")
        print(f"{arm:8s} {n:3d} {len(rs):5d} "
              f"{died/len(rds)*100:9.1f}% {turn/len(rds)*100:8.1f}% "
              f"{st.mean([num(r['n_generated']) for r in rs]):5.0f} "
              f"{st.mean([num(r['total_tokens']) for r in rs]):9,.0f} "
              f"{st.mean([num(r['total_wall_seconds']) for r in rs]):6.0f}")

    print("\n  chal.died = % of rounds where the NEW concept was the one "
          "discarded.\n  50% is the coin-flip line; near 100% means generating "
          "more adds nothing.")

    # ---- where the survivors came from
    print(f"\n{'='*72}\nSURVIVOR PROVENANCE  (is the final concept an early or late one?)")
    print(f"{'arm':8s} {'n':>3s} {'final gen_index, mean':>22s} {'  (0 = first ever generated)'}")
    for arm, n in cells:
        rs = [r for r in runs if r["arm"] == arm and int(r["pool_size"]) == n]
        idx = []
        for r in rs:
            fin = r["final_candidate_id"]
            for c in by_run_c[r["run_id"]]:
                if c["candidate_id"] == fin:
                    idx.append(num(c["gen_index"]))
        if idx:
            print(f"{arm:8s} {n:3d} {st.mean(idx):22.1f}   "
                  f"(max possible {n + 10 - 1})")

    # ---- length, the standing judge confound
    print(f"\n{'='*72}\nLENGTH  (a judge that prefers longer answers would show up here)")
    for arm, n in cells:
        rs = {r["run_id"] for r in runs
              if r["arm"] == arm and int(r["pool_size"]) == n}
        cs = [c for rid in rs for c in by_run_c[rid]]
        surv = [num(c["word_count"]) for c in cs if c["survived"] == "true"]
        dead = [num(c["word_count"]) for c in cs if c["survived"] == "false"]
        if surv and dead:
            print(f"{arm:8s} {n:3d}  survived {st.mean(surv):6.1f}w   "
                  f"discarded {st.mean(dead):6.1f}w   "
                  f"delta {st.mean(surv)-st.mean(dead):+6.1f}w")

    print(f"\n{'-'*72}")
    print("No quality claim is made above. Run src/final_eval.py for the "
          "survivor-vs-first-answer\ncontrast, and treat even that as a proxy "
          "until the human ranking is in.")


if __name__ == "__main__":
    main()
