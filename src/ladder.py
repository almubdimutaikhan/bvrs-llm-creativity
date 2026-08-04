#!/usr/bin/env python3
"""
One (mu+1) ladder: hold a pool of n concepts, add 1, discard 1, repeat.

    pool of n  --(generate 1)-->  n+1  --(one selection call)-->  n

The arm decides where variation comes from, and it is the whole experiment:

    blind    candidate k is generated in a FRESH context, brief only.
             It never sees the pool. Selection is the only intelligence.
             (Campbell 1960: variation "uncorrelated with the solution".)

    sighted  candidate k is shown the current pool and asked to do better.
             Variation is guided; this is hill-climbing.

The pool size n is the selection-pressure dial: at n=1 half the population dies
every round (pure exploitation); at n=40 one of 41 dies (almost pure
exploration). Sweeping n sweeps convergent -> divergent.

Nothing is ever really deleted. Discarded concepts stay on disk and in
candidates.csv with died_round set, so the elimination can be re-analysed
afterwards with a different rule, for free.

  python src/ladder.py --dry-run
  python src/ladder.py --arm blind --pool-size 1 --rounds 10 --seed 0
"""
import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import schema
from llm import chat

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"
PROBLEMS = ROOT / "problems" / "problems.json"

GEN_MAX_TOKENS = 700
SEL_MAX_TOKENS = 400

# Hybrid-thinking models (Qwen3, GPT-5, ...) emit hundreds of hidden reasoning
# tokens per call by default. This study's main chart plots quality against
# tokens; leaving thinking on would make that axis measure how much the model
# ruminated rather than what the ladder cost. Off by default, recorded either way.
NO_THINKING = {"enabled": False}

_TITLE = re.compile(r"^[\s*#]*TITLE\s*[:=]\s*(.+)$", re.I | re.M)
_CONCEPT = re.compile(r"^[\s*#]*CONCEPT\s*[:=]\s*(.*)\Z", re.I | re.M | re.S)
_RANKING = re.compile(r"RANKING\s*[:=]\s*\**\s*([0-9][0-9,\s]*)", re.I)
_DISCARD = re.compile(r"DISCARD\s*[:=]\s*\**\s*(\d+)", re.I)


# --------------------------------------------------------------------------
# helpers

def _read(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def _stable_seed(text: str) -> int:
    """Reproducible integer seed. Python's hash() is salted per process."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception:
        return ""


def _slug(model: str) -> str:
    return model.replace("/", "-").replace(":", "-")


def parse_concept(text: str) -> tuple[str, str]:
    """(title, body). Falls back gracefully if the model ignored the format."""
    tm = _TITLE.search(text or "")
    cm = _CONCEPT.search(text or "")
    title = tm.group(1).strip().strip("*").strip() if tm else ""
    body = cm.group(1).strip() if cm else (text or "").strip()
    if not title:
        first = body.splitlines()[0].strip() if body else ""
        title = first[:80]
    return title, body


def parse_selection(text: str, n: int) -> tuple[list, int, str]:
    """
    -> (ranking_1based, discard_1based, status)

    status is 'ok' (full valid ranking), 'discard_only' (ranking unusable but
    the discard is in range) or 'failed'. Recorded per round; a run with many
    non-'ok' rounds is not measuring what it claims to.
    """
    text = text or ""
    ranking, status = [], "failed"
    rm = _RANKING.search(text)
    if rm:
        nums = [int(x) for x in re.findall(r"\d+", rm.group(1))]
        if sorted(nums) == list(range(1, n + 1)):
            ranking, status = nums, "ok"

    dm = _DISCARD.search(text)
    discard = int(dm.group(1)) if dm else None
    if discard is not None and not (1 <= discard <= n):
        discard = None

    if status == "ok":
        # The ranking is authoritative; DISCARD is a checksum only.
        return ranking, (discard if discard == ranking[-1] else ranking[-1]), "ok"
    if discard is not None:
        return [], discard, "discard_only"
    return [], None, "failed"


def _pool_block(pool: list, cap) -> tuple[str, list]:
    """Render the pool for a sighted generator. Generation order, so context
    order does not vary with rank."""
    shown = pool
    if cap and len(pool) > cap:
        best = sorted(pool, key=lambda c: c["rank_score"], reverse=True)[:cap]
        shown = sorted(best, key=lambda c: c["gen_index"])
    block = "\n\n".join(f"Concept {i + 1} — {c['title']}\n{c['text']}"
                        for i, c in enumerate(shown))
    return block, [c["candidate_id"] for c in shown]


# --------------------------------------------------------------------------
# the ladder

def run(problem: dict, model: str, judge_model: str, arm: str, pool_size: int,
        rounds: int, seed: int, temperature: float = 0.9,
        sighted_cap=None, thinking: bool = False, notes: str = "") -> dict:

    assert arm in ("blind", "sighted")
    run_id = (f"{problem['id']}__{arm}__n{pool_size}__"
              f"{_slug(model)}__s{seed}")
    started = time.time()

    gen_tpl = _read("generate_blind.txt" if arm == "blind"
                    else "generate_sighted.txt")
    sel_tpl = _read("select.txt")
    text_dir = schema.TEXT_DIR / run_id
    text_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = schema.RESULTS / "raw" / run_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    pool, everyone = [], []
    tally = {"gen_p": 0, "gen_c": 0, "sel_p": 0, "sel_c": 0,
             "reasoning": 0, "calls": 0, "cached": 0}
    reasoning = None if thinking else NO_THINKING

    def generate(gen_index: int, born_round: int, origin: str) -> dict:
        if arm == "blind" or not pool:
            user = _read("generate_blind.txt").format(
                problem=problem["statement"])
            ctx_ids = []
        else:
            block, ctx_ids = _pool_block(pool, sighted_cap)
            user = gen_tpl.format(problem=problem["statement"], pool=block)
        cid = f"c{gen_index:03d}"
        # Nonce is mandatory: the blind prompt is byte-identical every time.
        res = chat(model, "", user, temperature=temperature,
                   max_tokens=GEN_MAX_TOKENS, nonce=f"{run_id}|{cid}",
                   reasoning=reasoning)
        tally["gen_p"] += res.prompt_tokens
        tally["gen_c"] += res.completion_tokens
        tally["reasoning"] += res.reasoning_tokens
        tally["calls"] += 1
        tally["cached"] += int(res.cached)

        title, body = parse_concept(res.text)
        (text_dir / f"{cid}.md").write_text(
            f"# {title}\n\n{body}\n", encoding="utf-8")
        cand = {
            "run_id": run_id, "candidate_id": cid, "gen_index": gen_index,
            "origin": origin, "born_round": born_round, "arm": arm,
            "context_candidate_ids": ";".join(ctx_ids), "context_n": len(ctx_ids),
            "title": title, "text": body,
            "word_count": len(body.split()), "char_count": len(body),
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "reasoning_tokens": res.reasoning_tokens,
            "latency_seconds": res.latency_seconds,
            "cached": str(res.cached).lower(),
            "died_round": "", "survived": "true",
            "rounds_present": 0, "rank_score": 0.0, "_points": [],
            "text_path": str((text_dir / f"{cid}.md").relative_to(schema.RESULTS)),
            "text_sha256": schema.sha256(body),
        }
        everyone.append(cand)
        return cand

    # ---- seed the pool
    for i in range(pool_size):
        pool.append(generate(i, 0, "seed"))

    # ---- rounds
    parse_failures = 0
    challenger_discarded = 0
    prev_top = None

    for rnd in range(1, rounds + 1):
        challenger = generate(pool_size + rnd - 1, rnd, "challenger")
        presented = pool + [challenger]
        n = len(presented)

        order = list(presented)
        random.Random(_stable_seed(f"{run_id}|round{rnd}")).shuffle(order)
        block = "\n\n".join(
            f"### Candidate {i + 1}\nTITLE: {c['title']}\n{c['text']}"
            for i, c in enumerate(order))
        user = sel_tpl.format(n=n, problem=problem["statement"],
                              candidates=block)

        ranking = []
        discard_idx = None
        status = "failed"
        retries = 0
        raw = ""
        sel_latency = 0.0
        sel_cached = True
        round_p = round_c = round_r = 0
        for attempt in range(3):
            suffix = "" if attempt == 0 else (
                f"\n\nYour previous answer did not parse. Output ONLY the two "
                f"lines. RANKING must contain each of the numbers 1..{n} "
                f"exactly once.")
            res = chat(judge_model, "", user + suffix, temperature=0.0,
                       max_tokens=SEL_MAX_TOKENS,
                       nonce=f"{run_id}|round{rnd}|try{attempt}",
                       reasoning=reasoning)
            tally["sel_p"] += res.prompt_tokens
            tally["sel_c"] += res.completion_tokens
            tally["reasoning"] += res.reasoning_tokens
            tally["calls"] += 1
            tally["cached"] += int(res.cached)
            round_p += res.prompt_tokens
            round_c += res.completion_tokens
            round_r += res.reasoning_tokens
            raw, sel_latency = res.text, res.latency_seconds
            sel_cached = sel_cached and res.cached
            ranking, discard_idx, status = parse_selection(res.text, n)
            retries = attempt
            if status != "failed":
                break

        (raw_dir / f"round{rnd:02d}.txt").write_text(raw, encoding="utf-8")

        if status == "failed":
            parse_failures += 1
            discard_idx = random.Random(
                _stable_seed(f"{run_id}|fallback{rnd}")).randrange(1, n + 1)

        # score everyone shown, best = 1.0
        if status == "ok" and n > 1:
            for pos, num in enumerate(ranking):
                cand = order[num - 1]
                cand["_points"].append((n - 1 - pos) / (n - 1))
                cand["rounds_present"] += 1
                cand["rank_score"] = round(
                    sum(cand["_points"]) / len(cand["_points"]), 4)

        dead = order[discard_idx - 1]
        dead["died_round"] = rnd
        dead["survived"] = "false"
        if dead["candidate_id"] == challenger["candidate_id"]:
            challenger_discarded += 1
        pool = [c for c in presented if c["candidate_id"] != dead["candidate_id"]]

        top = max(pool, key=lambda c: (c["rank_score"], -c["gen_index"]))
        schema.append_row(schema.ROUNDS_CSV, schema.ROUNDS_COLUMNS, {
            "run_id": run_id, "round": rnd, "n_presented": n,
            "challenger_id": challenger["candidate_id"],
            "pool_before": ";".join(c["candidate_id"] for c in presented
                                    if c["candidate_id"] != challenger["candidate_id"]),
            "pool_after": ";".join(c["candidate_id"] for c in pool),
            "presentation_order": ";".join(c["candidate_id"] for c in order),
            "ranking": ";".join(order[i - 1]["candidate_id"] for i in ranking),
            "discarded_id": dead["candidate_id"],
            "discarded_position": discard_idx,
            "discarded_was_challenger": str(
                dead["candidate_id"] == challenger["candidate_id"]).lower(),
            "top_id": top["candidate_id"],
            "top_changed": str(prev_top != top["candidate_id"]).lower(),
            "sel_prompt_tokens": round_p,
            "sel_completion_tokens": round_c,
            "sel_reasoning_tokens": round_r,
            "latency_seconds": sel_latency,
            "cached": str(sel_cached).lower(),
            "parse_status": status, "parse_retries": retries,
            "raw_response_path": str(
                (raw_dir / f"round{rnd:02d}.txt").relative_to(schema.RESULTS)),
        })
        prev_top = top["candidate_id"]

    # ---- write candidates
    for c in everyone:
        schema.append_row(schema.CANDIDATES_CSV, schema.CANDIDATES_COLUMNS,
                          {k: v for k, v in c.items()
                           if k not in ("text", "_points")})

    final = max(pool, key=lambda c: (c["rank_score"], -c["gen_index"]))
    row = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "problem_id": problem["id"],
        "method_name": f"{arm}_n{pool_size}",
        "arm": arm, "pool_size": pool_size, "n_rounds": rounds,
        "rounds_completed": rounds, "seed": seed,
        "model_name": model, "judge_model": judge_model,
        "temperature": temperature,
        "sighted_cap": sighted_cap if sighted_cap else "",
        "n_generated": len(everyone), "n_survivors": len(pool),
        "final_candidate_id": final["candidate_id"],
        "gen_prompt_tokens": tally["gen_p"],
        "gen_completion_tokens": tally["gen_c"],
        "sel_prompt_tokens": tally["sel_p"],
        "sel_completion_tokens": tally["sel_c"],
        "total_reasoning_tokens": tally["reasoning"],
        "total_tokens": sum(tally[k] for k in ("gen_p", "gen_c", "sel_p", "sel_c")),
        "total_wall_seconds": round(time.time() - started, 2),
        "n_api_calls": tally["calls"], "n_cached_calls": tally["cached"],
        "n_parse_failures": parse_failures,
        "n_challenger_discarded": challenger_discarded,
        "gen_prompt_sha": schema.sha256(gen_tpl)[:12],
        "sel_prompt_sha": schema.sha256(sel_tpl)[:12],
        "git_commit": _git_commit(), "notes": notes,
    }
    schema.append_row(schema.RUNS_CSV, schema.RUNS_COLUMNS, row)
    return row


# --------------------------------------------------------------------------

def load_problem(pid: str) -> dict:
    for p in json.loads(PROBLEMS.read_text())["problems"]:
        if p["id"] == pid:
            return p
    sys.exit(f"unknown problem id: {pid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="mars_boot")
    ap.add_argument("--arm", default="blind", choices=["blind", "sighted"])
    ap.add_argument("--pool-size", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="qwen/qwen3-32b")
    ap.add_argument("--judge-model", default="google/gemini-2.5-flash")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--sighted-cap", type=int, default=None,
                    help="max concepts shown to a sighted generator")
    ap.add_argument("--thinking", action="store_true",
                    help="allow hidden reasoning tokens (off by default: they "
                         "would dominate the token axis)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    problem = load_problem(a.problem)
    gens = a.pool_size + a.rounds
    print(f"{problem['name']}  arm={a.arm}  n={a.pool_size}  "
          f"rounds={a.rounds}  seed={a.seed}")
    print(f"  {gens} generation calls + {a.rounds} selection calls "
          f"= {gens + a.rounds} calls")
    if a.dry_run:
        print("  --dry-run: no API calls made")
        return

    row = run(problem, a.model, a.judge_model, a.arm, a.pool_size, a.rounds,
              a.seed, a.temperature, a.sighted_cap, a.thinking)
    print(f"\n{row['run_id']}")
    print(f"  final          : {row['final_candidate_id']}")
    print(f"  tokens         : {row['total_tokens']}  "
          f"(gen {row['gen_prompt_tokens'] + row['gen_completion_tokens']}, "
          f"sel {row['sel_prompt_tokens'] + row['sel_completion_tokens']})")
    print(f"  reasoning      : {row['total_reasoning_tokens']}  "
          f"(expect 0 unless --thinking)")
    print(f"  wall seconds   : {row['total_wall_seconds']}  "
          f"({row['n_cached_calls']}/{row['n_api_calls']} cached)")
    print(f"  parse failures : {row['n_parse_failures']}")
    print(f"  challenger died: {row['n_challenger_discarded']}/{a.rounds}")


if __name__ == "__main__":
    main()
