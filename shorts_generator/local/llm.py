"""LLM backend — Gemini or OpenAI over plain HTTPS (no vendor SDK to install), with an answer cache and
automatic model fallback.

Why not the SDKs: they pull in a dozen heavy packages for what is one POST request, and they hide the
difference between "busy right now" (503), "daily quota used up" (429) and "no such model" (404) — which
is exactly what decides whether to retry, wait, or try another model. Here:

* 503 / overloaded  -> retry the model twice, then move on to the next model
* 429 / quota       -> that model is parked for an hour (quota is per model), move on to the next
* 404 / not found   -> skip the model
* bad API key       -> stop at once with a clear message
Every successful, parseable answer is cached on disk, so a crash or a later failure never costs the same
call twice.
"""
import contextlib
import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from ..config import (
    LOCAL_OUTPUT_DIR,
    gemini_key,
    gemini_model,
    llm_cache_enabled,
    llm_provider,
    openai_key,
    openai_model,
    require_gemini_key,
    require_openai_key,
)
from . import control

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_NOT_TEXT_MODELS = re.compile(r"image|tts|audio|live|embed|omni|robotics|computer-use|vision|gemma|learnlm|aqa", re.I)


class LLMError(RuntimeError):
    """kind: auth | quota | overloaded | notfound | bad | network"""

    def __init__(self, message: str, kind: str = "bad", status: Optional[int] = None, retry_after: float = 0.0):
        super().__init__(message)
        self.kind, self.status, self.retry_after = kind, status, retry_after


def _sleep(seconds: float) -> None:
    """Sleep in short slices so Cancel is honoured."""
    if seconds >= _time_left():
        raise _out_of_time()
    end = time.time() + seconds
    while time.time() < end:
        control.check()
        time.sleep(min(0.5, max(0.0, end - time.time())))


_budget = threading.local()


@contextlib.contextmanager
def time_budget(seconds: float):
    """Inside the block, AI calls (retries, waits and all) give up after ``seconds``: for optional work, such as the
    popup plan of an edit, that must never hold up the render."""
    before = getattr(_budget, "until", None)
    until = time.time() + seconds
    _budget.until = until if before is None else min(before, until)
    try:
        yield
    finally:
        _budget.until = before


def _time_left() -> float:
    until = getattr(_budget, "until", None)
    return float("inf") if until is None else until - time.time()


def _out_of_time() -> LLMError:
    return LLMError("The AI service took too long to answer.", "overloaded")


def _read_timeout() -> float:
    left = _time_left()
    if left <= 1:
        raise _out_of_time()
    return min(180.0, left)


def _classify(resp: requests.Response) -> LLMError:
    try:
        body = resp.json()
        err = body.get("error", body) if isinstance(body, dict) else {}
        msg = str(err.get("message") or resp.text)[:300]
    except ValueError:
        msg = resp.text[:300]
    code = resp.status_code
    low = msg.lower()
    if code in (401, 403) or "api key not valid" in low or "api_key_invalid" in low or "invalid api key" in low:
        return LLMError(f"The API key was rejected ({code}). Check it in Settings. {msg}", "auth", code)
    if code == 404:
        return LLMError(f"Model not available: {msg}", "notfound", code)
    if code == 429:
        m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", low)
        wait = 0.0
        if m and any(m.groups()):
            wait = int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3) or 0)
        return LLMError(f"Quota used up or rate limit hit (429): {msg}", "quota", code, wait or 3600.0)
    if code >= 500:
        return LLMError(f"Service overloaded ({code}): {msg}", "overloaded", code)
    return LLMError(f"Request rejected ({code}): {msg}", "bad", code)


# --------------------------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------------------------- #
_models_lock = threading.Lock()
_models_cache: Dict = {"at": 0.0, "names": []}
_parked: Dict[str, float] = {}          # model -> unix time until which it is skipped (quota)


def _version(name: str) -> float:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def gemini_models(key: Optional[str] = None, refresh: bool = False) -> List[str]:
    """Text-capable Gemini Flash models this key can use, best first (newest, full-size before 'lite')."""
    key = key or gemini_key()
    with _models_lock:
        if not refresh and _models_cache["names"] and time.time() - _models_cache["at"] < 3600:
            return list(_models_cache["names"])
    names: List[str] = []
    token = None
    try:
        for _ in range(5):
            params = {"pageSize": 100, **({"pageToken": token} if token else {})}
            r = requests.get(f"{GEMINI_BASE}/models", params=params, headers={"x-goog-api-key": key}, timeout=20)
            if r.status_code != 200:
                raise _classify(r)
            data = r.json()
            for m in data.get("models", []):
                name = str(m.get("name", "")).replace("models/", "")
                if ("generateContent" in (m.get("supportedGenerationMethods") or []) and "flash" in name
                        and not _NOT_TEXT_MODELS.search(name)):
                    names.append(name)
            token = data.get("nextPageToken")
            if not token:
                break
    except requests.RequestException:
        return list(_models_cache["names"])
    names = sorted(set(names), key=lambda n: ("lite" in n, -_version(n), n))
    with _models_lock:
        _models_cache.update(at=time.time(), names=names)
    return names


def _gemini_generate(model: str, prompt: str, key: str) -> str:
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json", "maxOutputTokens": 8192},
    }
    try:
        r = requests.post(f"{GEMINI_BASE}/models/{model}:generateContent",
                          headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                          json=body, timeout=(15, _read_timeout()))
    except requests.RequestException as e:
        raise LLMError(f"Network problem talking to Gemini: {e}", "network") from e
    if r.status_code != 200:
        raise _classify(r)
    data = r.json()
    cands = data.get("candidates") or []
    if not cands:
        raise LLMError(f"Gemini returned no answer ({data.get('promptFeedback') or 'blocked?'}).", "bad")
    parts = (cands[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        raise LLMError(f"Gemini returned an empty answer (finish reason: {cands[0].get('finishReason')}).", "bad")
    return text


def _model_chain(primary: str, key: str) -> List[str]:
    found = gemini_models(key)
    if primary == "auto":
        return found or ["gemini-flash-latest"]
    return [primary] + [m for m in found if m != primary]


_last_model: Dict[str, str] = {}
_parked_kind: Dict[str, str] = {}
ROUNDS = 3                      # whole-chain attempts when every model is busy (waits 20 s, then 40 s, between rounds)


def call_gemini_llm(prompt: str) -> str:
    key = require_gemini_key()
    primary = gemini_model()
    problems: List[str] = []
    for rnd in range(ROUNDS):
        chain = _model_chain(primary, key)
        # In "auto" mode start with the model that answered last time instead of re-trying one that is down.
        pref = _last_model.get("gemini")
        if primary == "auto" and pref in chain:
            chain = [pref] + [m for m in chain if m != pref]
        problems, saw_busy = [], False
        for idx, model in enumerate(chain):
            if _parked.get(model, 0) > time.time():
                saw_busy = saw_busy or _parked_kind.get(model) == "overloaded"
                problems.append(f"{model}: unavailable for now")
                continue
            for attempt in range(3):
                control.check()
                try:
                    text = _gemini_generate(model, prompt, key)
                    _last_model["gemini"] = model
                    return text
                except LLMError as e:
                    if e.kind == "auth":
                        raise
                    if e.kind == "overloaded" and attempt < 2:
                        print(f"[llm/gemini] {model} is busy (503), retrying in {5 * (attempt + 1)}s", flush=True)
                        _sleep(5 * (attempt + 1))
                        continue
                    saw_busy = saw_busy or e.kind == "overloaded"
                    _parked_kind[model] = e.kind
                    if e.kind == "quota":
                        _parked[model] = time.time() + min(e.retry_after or 3600.0, 12 * 3600)
                    elif e.kind in ("overloaded", "notfound"):
                        _parked[model] = time.time() + (20 if e.kind == "overloaded" else 3600)   # skip it for a while
                    problems.append(f"{model}: {e}")
                    nxt = next((m for m in chain[idx + 1:] if _parked.get(m, 0) <= time.time()), None)
                    print(f"[llm/gemini] {model} unavailable ({e.kind})" + (f" -> trying {nxt}" if nxt else ""), flush=True)
                    break
        if not saw_busy or rnd == ROUNDS - 1:          # nothing worth waiting for (e.g. daily quota gone), or out of rounds
            break
        wait = 20 * (rnd + 1)
        print(f"[llm/gemini] every model is busy right now; waiting {wait}s and trying again ({rnd + 2}/{ROUNDS})", flush=True)
        control.report("rank", None, "The AI service is busy: waiting a moment and trying again")
        _sleep(wait)
    quota_only = all("quota" in p.lower() or "not available" in p.lower() or "unavailable for now" in p.lower() for p in problems)
    hint = (" The free daily quota looks used up: try again later, or add another key / switch provider in Settings."
            if quota_only else " The service is overloaded: your progress is saved, so trying again continues from here.")
    raise LLMError("Every Gemini model failed." + hint + " Details: " + " | ".join(problems[-3:]), "overloaded")


# --------------------------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------------------------- #
def call_openai_llm(prompt: str) -> str:
    key, model = require_openai_key(), openai_model()
    last: Optional[LLMError] = None
    for attempt in range(4):
        control.check()
        try:
            r = requests.post(
                OPENAI_URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "temperature": 0.7, "response_format": {"type": "json_object"},
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=(15, _read_timeout()),
            )
        except requests.RequestException as e:
            last = LLMError(f"Network problem talking to OpenAI: {e}", "network")
        else:
            if r.status_code == 200:
                _last_model["openai"] = model
                return r.json()["choices"][0]["message"]["content"] or ""
            last = _classify(r)
            if last.kind in ("auth", "notfound", "bad"):
                raise last
        if attempt < 3:
            print(f"[llm/openai] {last.kind}, retrying in {2 ** (attempt + 1)}s", flush=True)
            _sleep(2 ** (attempt + 1))
    raise last or LLMError("OpenAI failed.")


# --------------------------------------------------------------------------------------------- #
# Dispatch + cache
# --------------------------------------------------------------------------------------------- #
def _cache_path(provider: str, model: str, prompt: str) -> Path:
    key = hashlib.sha256(f"{provider}\0{model}\0{prompt}".encode("utf-8")).hexdigest()
    return Path(LOCAL_OUTPUT_DIR) / ".llm_cache" / f"{key}.json"


def _is_valid_json_answer(text: str) -> bool:
    """Only cache answers that actually parse: an invalid one must be retried, not replayed."""
    from ..highlights import _parse_json_loose

    try:
        return bool(_parse_json_loose(text))
    except Exception:
        return False


def call_local_llm(prompt: str, use_cache: bool = True) -> str:
    """Ask the configured provider. Cached on disk by (provider, model setting, prompt)."""
    provider = llm_provider()
    if provider == "openai":
        model, fn = openai_model(), call_openai_llm
    else:
        provider, model, fn = "gemini", gemini_model(), call_gemini_llm

    path = _cache_path(provider, model, prompt) if (use_cache and llm_cache_enabled()) else None
    if path is not None and path.exists():
        try:
            text = json.loads(path.read_text(encoding="utf-8"))["text"]
            print(f"[llm/{provider}] cache hit ({len(text)} chars): no API call", flush=True)
            return text
        except (OSError, ValueError, KeyError):
            pass

    text = fn(prompt)
    if path is not None and _is_valid_json_answer(text):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": _last_model.get(provider, model), "text": text}, ensure_ascii=False),
                            encoding="utf-8")
        except OSError:
            pass
    return text


def test_connection() -> Dict:
    """One tiny real request: does the configured provider/key/model work, and which model answered?"""
    provider = llm_provider()
    started = time.time()
    try:
        if provider == "gemini" and not gemini_key():
            return {"ok": False, "provider": provider, "error": "No Gemini API key yet. Paste one and press Save."}
        if provider == "openai" and not openai_key():
            return {"ok": False, "provider": provider, "error": "No OpenAI API key yet. Paste one and press Save."}
        call_local_llm('Reply with exactly this JSON and nothing else: {"ok": true}', use_cache=False)
        return {"ok": True, "provider": provider, "model": _last_model.get(provider, ""),
                "seconds": round(time.time() - started, 1)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "provider": provider, "error": str(e)[:400]}
