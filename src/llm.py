#!/usr/bin/env python3
"""
Cached OpenAI-compatible chat client that reports token usage and latency.

Two things here matter for this study and are easy to get wrong:

1. USAGE IS RECORDED. Every call returns prompt/completion token counts, because
   the experiment measures quality *per token*, not quality per round. A single
   summed "tokens" number cannot separate generation cost from selection cost,
   and the sighted arm's cost lives almost entirely in prompt tokens.

2. THE CACHE KEY INCLUDES A NONCE. The blind arm sends an IDENTICAL prompt for
   every candidate. Without a nonce, every candidate in a run — and every
   replicate of that run — collapses onto one cached response, the pool fills
   with copies of the same concept, and the experiment silently measures nothing
   while appearing to work. Callers MUST pass a nonce unique per call;
   ladder.py passes the candidate id.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "llm_cache"
BASE_URL = os.environ.get("GATEWAY_BASE_URL", "https://openrouter.ai/api/v1")

_KEY_NAMES = ("AI_OPENROUTER_API_KEY", "OPENROUTER_API_KEY",
              "AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_API_KEY")
_RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}


@dataclass
class Result:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    cached: bool
    model: str
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def api_key() -> str:
    """
    First key found in _KEY_NAMES ORDER — not in whichever order .env happens to
    list them. A .env holding both a Vercel gateway key and an OpenRouter key
    will otherwise hand the Vercel key to the OpenRouter base URL and 401.
    """
    found = {}
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            found[k.strip()] = v.strip().strip('"').strip("'")
    found.update({k: v for k, v in os.environ.items() if v})  # env wins

    for name in _KEY_NAMES:
        if found.get(name):
            return found[name]
    raise RuntimeError("No API key. Set AI_OPENROUTER_API_KEY in the "
                       "environment or in .env at the repo root.")


def _cache_key(model, system, user, params, nonce) -> str:
    blob = json.dumps({"model": model, "system": system, "user": user,
                       "params": params, "nonce": nonce}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def chat(model: str, system: str, user: str, temperature: float = 0.9,
         max_tokens: int = 1200, nonce: str = "", timeout: int = 180,
         reasoning=None) -> Result:
    """
    One chat completion. Disk-cached on (model, prompts, params, nonce).

    `reasoning` is passed through to the gateway. Hybrid-thinking models
    (Qwen3, GPT-5, ...) otherwise emit hundreds of hidden reasoning tokens per
    call, which would dominate the token axis this study plots quality against —
    the x-axis would become "how much did the model think", not "what did the
    ladder cost". Callers should pass {"enabled": False} unless testing that.
    Whatever is emitted is recorded in Result.reasoning_tokens either way.
    """
    params = {"temperature": temperature, "max_tokens": max_tokens}
    if reasoning is not None:
        params["reasoning"] = reasoning
    key = _cache_key(model, system, user, params, nonce)
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        d = json.loads(path.read_text())
        return Result(d["text"], d["prompt_tokens"], d["completion_tokens"],
                      d["latency_seconds"], True, model,
                      d.get("reasoning_tokens", 0))

    messages = ([{"role": "system", "content": system}] if system else []) \
        + [{"role": "user", "content": user}]
    payload = {"model": model, "messages": messages, **params}
    headers = {"Authorization": f"Bearer {api_key()}",
               "Content-Type": "application/json"}

    last = None
    for attempt in range(5):
        started = time.time()
        try:
            r = requests.post(f"{BASE_URL}/chat/completions", headers=headers,
                              json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last = str(exc)
            time.sleep(2 ** attempt)
            continue
        latency = round(time.time() - started, 3)
        if r.status_code == 200:
            body = r.json()
            text = body["choices"][0]["message"].get("content") or ""
            usage = body.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            res = Result(text, int(usage.get("prompt_tokens", 0)),
                         int(usage.get("completion_tokens", 0)),
                         latency, False, model,
                         int(details.get("reasoning_tokens", 0)))
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(
                {"text": res.text, "prompt_tokens": res.prompt_tokens,
                 "completion_tokens": res.completion_tokens,
                 "reasoning_tokens": res.reasoning_tokens,
                 "latency_seconds": res.latency_seconds,
                 "model": model, "nonce": nonce}, ensure_ascii=False))
            return res
        last = f"HTTP {r.status_code}: {r.text[:300]}"
        if r.status_code not in _RETRY_STATUS:
            break
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{model} failed after retries -> {last}")
