#!/usr/bin/env python3
"""
The output schema, in one place.

THREE tables, not one, joined on (run_id) and (run_id, candidate_id):

  runs.csv        one row per execution   — the experimental cell + its totals
  candidates.csv  one row per concept     — generation metadata + where the text is
  rounds.csv      one row per selection   — who was shown, who died, what it cost

Why not one flat file:
  - The problem statement is ~150 words and identical in every row. Flat CSV
    duplicates it thousands of times; here it lives in problems/problems.json
    and rows carry `problem_id`.
  - `left_over_solutions` as one cell holding 40 concepts is unreadable,
    unparseable and breaks on the first embedded comma.
  - CONCEPT TEXT NEVER ENTERS A CSV. It goes to
    results/candidates/<run_id>/<candidate_id>.md and the row carries a path and
    a sha256. Design concepts contain newlines, commas, quotes and markdown —
    embedding them is what forces "flatten the CSV" repairs later.

Conventions: snake_case; units in the name (_tokens, _seconds); ISO-8601 UTC
timestamps; booleans as true/false; ID lists joined with ";"; empty string for
not-applicable.
"""
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

RUNS_CSV = RESULTS / "runs.csv"
CANDIDATES_CSV = RESULTS / "candidates.csv"
ROUNDS_CSV = RESULTS / "rounds.csv"
TEXT_DIR = RESULTS / "candidates"

# --- one row per run -------------------------------------------------------
RUNS_COLUMNS = [
    "run_id",              # {problem}__{arm}__n{pool}__{model}__s{seed}
    "timestamp_utc",
    "problem_id",
    "method_name",         # human label for plots, e.g. "blind_n5"
    "arm",                 # blind | sighted
    "pool_size",           # n (mu). The selection-pressure dial.
    "n_rounds",            # rounds planned
    "rounds_completed",
    "seed",                # replicate index — the unit of analysis
    "model_name",          # generator
    "judge_model",         # selector (kept distinct from generator on purpose)
    "temperature",
    "sighted_cap",         # max concepts shown to a sighted generator ("" = all)
    "n_generated",         # pool_size + rounds_completed
    "n_survivors",
    "final_candidate_id",  # best survivor by cumulative rank score
    "gen_prompt_tokens",
    "gen_completion_tokens",
    "sel_prompt_tokens",
    "sel_completion_tokens",
    "total_tokens",
    "total_wall_seconds",
    "n_api_calls",
    "n_cached_calls",      # cached calls cost 0 and take ~0s — never mix them in
    "n_parse_failures",
    "n_challenger_discarded",   # how often the newcomer died immediately
    "gen_prompt_sha",      # exact template used, without duplicating it per row
    "sel_prompt_sha",
    "git_commit",
    "notes",
]

# --- one row per generated concept ----------------------------------------
CANDIDATES_COLUMNS = [
    "run_id",
    "candidate_id",        # c000, c001, ... unique within the run
    "gen_index",           # order of generation, 0-based
    "origin",              # seed | challenger
    "born_round",          # 0 for the initial pool, else the round that made it
    "arm",
    "context_candidate_ids",  # what the generator was shown. BLIND MUST BE EMPTY.
    "context_n",              # audit trail for the blind/sighted manipulation
    "title",
    "word_count",          # length is a known judge confound — report it
    "char_count",
    "prompt_tokens",
    "completion_tokens",
    "latency_seconds",
    "cached",
    "died_round",          # "" if it survived to the end
    "survived",
    "rounds_present",      # how many selections it faced
    "rank_score",          # mean normalised rank (1.0 best, 0.0 worst)
    "text_path",           # relative to results/
    "text_sha256",
]

# --- one row per selection event ------------------------------------------
ROUNDS_COLUMNS = [
    "run_id",
    "round",               # 1..n_rounds
    "n_presented",         # pool_size + 1
    "challenger_id",
    "pool_before",         # ";"-joined candidate_ids
    "pool_after",
    "presentation_order",  # ";"-joined ids IN THE ORDER SHOWN -> position bias
    "ranking",             # ";"-joined ids, best first, as returned
    "discarded_id",
    "discarded_position",  # 1-based slot in presentation_order
    "discarded_was_challenger",   # if ~always true, generation adds nothing
    "top_id",              # leader after this round
    "top_changed",
    "sel_prompt_tokens",       # summed over retries, not just the winning try
    "sel_completion_tokens",
    "latency_seconds",
    "cached",
    "parse_status",            # ok | discard_only | failed
    "parse_retries",
    "raw_response_path",   # judge's raw output, for audit
]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append_row(path: Path, columns: list, row: dict) -> None:
    """Append one row, writing the header if the file is new. Flushes each row
    so an interrupted run keeps everything it already paid for."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="raise")
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in columns})
