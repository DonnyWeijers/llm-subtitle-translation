#!/usr/bin/env python3
"""subtrans: an LLM subtitle translation service with tiered model routing, free quality checks and exact
cost per call. Python standard library only.

It speaks the Lingarr API that Bazarr uses for its "lingarr" translator: Bazarr POSTs /api/translate/content with
{"lines": [{"position": n, "line": "..."}], "title", "sourceLanguage", "targetLanguage", "translatedSubtitlePath", ...}
and expects [{"position": n, "line": "..."}] back.
  - step 1 = TIER1 (cheap model), step 2 = TIER2 (stronger model, other family), both via OpenRouter. The step for a
    file can be set with POST /api/tier; default step 1. The models were picked with the benchmark in ../bench.
  - 100-line batches with the benchmark's prompt; a batch with missing or unparsable lines is retried once, then sent
    to the other step's model, so one bad batch never leaves English on screen.
  - free quality checks on every translation (missing lines, English left in, odd line lengths, text shifted onto
    neighbouring lines; calibrated on the benchmark). Step 1 fails a check -> only the failing batches are redone
    (step 1 at a higher reasoning effort, then step 2), then the whole file with step 2 if it still fails; still
    failing = weak. JUDGE scores a 100-line sample of a random JUDGE_RATE share of translations, for the quality
    figure only.
  - every call's exact cost (OpenRouter usage.cost, USD) is logged to $DATA/costs.jsonl; per file: $DATA/state.json.
GET /api/state (per file), GET /api/stats (totals, this month, per step, OpenRouter key usage), POST /api/tier.
"""
import concurrent.futures as cf
import datetime as dt
import http.server
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request

DATA = os.environ.get("DATA", "/data")
PORT = int(os.environ.get("PORT", 9876))
KEY_FILE = os.environ.get("OPENROUTER_KEY_FILE", "/run/secrets/openrouter/subtitles.key")


def key():
    """Read on every call, so a rotated key works without a restart. Mount the key's folder, not the file: a
    replaced file is a new inode that a single-file bind mount never sees."""
    return open(KEY_FILE).read().strip()
TIERS = {1: os.environ.get("TIER1", "openai/gpt-6-luna"), 2: os.environ.get("TIER2", "google/gemini-3.7-flash")}
JUDGE = os.environ.get("JUDGE", "anthropic/claude-sonnet-5.5")
THRESHOLD = int(os.environ.get("THRESHOLD", 72))
JUDGE_RATE = float(os.environ.get("JUDGE_RATE", 0.2))  # spot check: 1 in 5 translations
# free checks: share of lines (shift: of 8-line windows) above which a translation fails. Calibrated on the benchmark:
# good models 0-0.5 % shifted windows, models that shift lines 2-30 %
LIMITS = {"missing": 0.02, "untranslated": 0.03, "odd_length": 0.03, "shifted": 0.015}
BATCH = 100
REASONING = {"openai/gpt-6-luna": {"effort": "low"}}  # as benchmarked
# retries of failing batches before step 2 (measured per 100 lines: Luna low $0.00085, medium $0.0012, high $0.0017,
# xhigh $0.0027, Gemini 3.7 Flash $0.011). Luna's shifts are random (high shifted a batch that low/medium/xhigh did
# not), so another Luna pass usually comes out clean; more effort is no guarantee
RETRY_EFFORTS = os.environ.get("RETRY_EFFORTS", "medium,high,xhigh").split(",")
EUR_PER_USD = float(os.environ.get("EUR_PER_USD", 0.86))
LANG = {"nl": "Dutch", "en": "English"}
LOCK = threading.Lock()

INSTRUCTION = """You are an assistant that translates subtitles to {lang}.
    You will receive the following JSON type:

    class SubtitleObject(typing.TypedDict):
        index: str
        content: str

    Request: list[SubtitleObject]

    The 'index' key is the index of the subtitle dialog.
    The 'content' key is the dialog to be translated.

    The indices must remain the same in the response as in the request.
    Dialogs must be translated as they are without any changes.
    If a line has a comma or multiple sentences, try to keep one line to about 40-50 characters.
    """
RUBRIC = """You are a senior Dutch subtitle editor reviewing a Dutch subtitle for a TV series, against the English
original. Lines are numbered on both sides and must correspond one to one: Dutch text that belongs to another English
line is a serious meaning error (it appears at the wrong moment on screen).

Score 0-100 each:
- accuracy: meaning preserved, including context, idioms, tone, who is addressed; omissions a good subtitler would
  make for brevity are fine, wrong meaning or text on the wrong line is not
- fluency: natural spoken Dutch as on Dutch television, no anglicisms or literal word order
- style: usable as subtitles: concise, readable, consistent names and forms of address (je/u), nothing left in English
- overall: 90+ professional broadcast quality; 80 good, a few minor issues; 70 understandable but clearly
  machine-like or with a couple of meaning errors; 60 several meaning errors; below 50 misleading or broken

List at most 5 of the most serious problems (line number, short note in English).
Answer ONLY with JSON: {"accuracy": n, "fluency": n, "style": n, "overall": n,
 "problems": [{"line": n, "type": "meaning|fluency|style|untranslated|misaligned", "note": "..."}]}"""


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def load(name, default):
    try:
        return json.load(open(os.path.join(DATA, name)))
    except (OSError, ValueError):
        return default


def save(name, obj):
    tmp = os.path.join(DATA, f".{name}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(DATA, name))


def log_cost(kind, model, usd, title, extra=None):
    with LOCK, open(os.path.join(DATA, "costs.jsonl"), "a") as f:
        f.write(json.dumps({"t": now(), "kind": kind, "model": model, "usd": usd, "title": title, **(extra or {})},
                           ensure_ascii=False) + "\n")


def openrouter(model, system, user, extra=None, timeout=600):
    body = {"model": model, "usage": {"include": True},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], **(extra or {})}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 method="POST", headers={"Content-Type": "application/json",
                                                         "Authorization": f"Bearer {key()}",
                                                         "X-Title": "subtrans"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    if "error" in r:
        raise RuntimeError(str(r["error"])[:200])
    return (r["choices"][0]["message"].get("content") or ""), float((r.get("usage") or {}).get("cost") or 0)


def parse(text, wanted):
    m = re.search(r"\[.*\]", text or "", re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except ValueError:
        items = []
    got = {str(i.get("index")): (i.get("content") or "").strip() for i in items if isinstance(i, dict)}
    return {k: got[k] for k in wanted if got.get(k)}


def translate_batch(batch, tier, lang, title, effort=None):
    """batch: [(position, line)] -> ({position: dutch}, usd, model used). effort: Luna's reasoning effort override."""
    wanted = [str(p) for p, _ in batch]
    payload = json.dumps([{"index": str(p), "content": line} for p, line in batch], ensure_ascii=False)
    usd, out = 0.0, {}
    for model in (TIERS[tier], TIERS[tier], TIERS[2 if tier == 1 else 1]):  # retry once, then the other model
        try:
            reasoning = {"effort": effort} if effort and model == TIERS[1] else REASONING.get(model)
            text, cost = openrouter(model, INSTRUCTION.format(lang=lang), payload,
                                    {"reasoning": reasoning} if reasoning else None)
        except Exception as e:  # noqa: BLE001
            print(f"{now()} {title}: {model} batch error {str(e)[:120]}", flush=True)
            time.sleep(5)
            continue
        usd += cost
        log_cost("translate", model, cost, title)
        got = parse(text, wanted)
        out.update({int(k): v for k, v in got.items()})  # back to Bazarr's integer positions
        if len(out) >= len(wanted) * 0.98:
            return out, usd, model
    if not out:
        raise RuntimeError("batch failed with both models")
    return out, usd, model


def judge(lines, translated, title):
    """Score a 100-line sample from the middle; returns (score dict, usd)."""
    pos = [p for p, _ in lines]
    mid = max(0, len(pos) // 2 - BATCH // 2)
    sample = lines[mid:mid + BATCH]
    en = "\n".join(f"{i + 1}. {l.replace(chr(10), ' / ')}" for i, (_, l) in enumerate(sample))
    nl = "\n".join(f"{i + 1}. {translated.get(p, '').replace(chr(10), ' / ')}" for i, (p, _) in enumerate(sample))
    try:
        text, usd = openrouter(JUDGE, RUBRIC, f"ENGLISH ({title}):\n{en}\n\nDUTCH SUBTITLE:\n{nl}",
                               {"temperature": 0}, timeout=300)
        m = re.search(r"\{.*\}", text, re.S)
        res = json.loads(m.group(0))
    except Exception as e:  # noqa: BLE001
        return {"overall": None, "error": str(e)[:150]}, 0.0
    log_cost("judge", JUDGE, usd, title)
    return res, usd


def clean(x):
    return re.sub(r"<[^>]+>|\{[^}]+\}", "", x or "").replace("\n", " ").strip()


def corr(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) ** 0.5 if va and vb else 0.0


def checks(lines, translated):
    """Free quality checks -> ({name: share}, [failed names]). No LLM involved."""
    n = max(1, len(lines))
    missing = sum(1 for p, _ in lines if not clean(translated.get(p)))
    untr = sum(1 for p, en in lines if len(clean(en).split()) >= 4 and clean(translated.get(p)).lower() == clean(en).lower())
    longs = [(len(clean(en)), len(clean(translated.get(p)))) for p, en in lines if len(clean(en)) >= 25]
    odd = sum(1 for e, d in longs if d and not 0.4 <= d / e <= 2.2)
    # text shifted to a neighbouring line: in an 8-line window the Dutch lengths follow the English lengths of the
    # line before/after better than their own
    E = [len(clean(en)) for _, en in lines]
    D = [len(clean(translated.get(p))) for p, _ in lines]
    wins = bad = 0
    for s in range(1, len(lines) - 9, 4):
        own = corr(E[s:s + 8], D[s:s + 8])
        near = max(corr(E[s - 1:s + 7], D[s:s + 8]), corr(E[s + 1:s + 9], D[s:s + 8]))
        wins += 1
        bad += near > own + 0.25 and own < 0.6
    res = {"missing": missing / n, "untranslated": untr / n, "odd_length": odd / max(1, len(longs)),
           "shifted": bad / max(1, wins)}
    return {k: round(v, 4) for k, v in res.items()}, [k for k, v in res.items() if v > LIMITS[k]]


def translate_all(lines, tier, lang, title, effort=None):
    batches = [lines[i:i + BATCH] for i in range(0, len(lines), BATCH)]
    translated, usd, models = {}, 0.0, set()
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        for out, cost, model in pool.map(lambda b: translate_batch(b, tier, lang, title, effort), batches):
            translated.update(out)
            usd += cost
            models.add(model)
    return translated, usd, models


def handle_translate(req):
    lines = [(int(x["position"]), x["line"]) for x in req.get("lines", []) if str(x.get("line", "")).strip()]
    title = req.get("title") or "?"
    path = req.get("translatedSubtitlePath") or title
    lang = LANG.get((req.get("targetLanguage") or "nl")[:2], req.get("targetLanguage") or "Dutch")
    state = load("state.json", {})
    tier = int((state.get(path) or {}).get("next_tier") or 1)
    t0 = time.time()
    translated, usd, models = translate_all(lines, tier, lang, title)
    check, failed = checks(lines, translated)
    first = None
    if failed and tier == 1:  # fix it straight away: a broken translation should not wait for anything
        first = {"model": TIERS[1], "checks": check, "failed": failed, "redo": []}
        # a shift or a skipped stretch usually sits in one or two batches, so only the batches that fail the checks on
        # their own are redone: first Luna at each RETRY_EFFORTS (its shifts are random), then Gemini (step 2) for what still
        # fails; the whole file is checked after each pass, and only if it still fails is the whole file redone with
        # Gemini
        def bad_batches():
            return [b for b in (lines[i:i + BATCH] for i in range(0, len(lines), BATCH)) if checks(b, translated)[1]]
        for step, effort in [(1, e) for e in RETRY_EFFORTS if e] + [(2, None)]:
            bad = bad_batches()
            if not failed or not bad:
                break
            print(f"{now()} {title}: {failed} {check}, redo {len(bad)} of {-(-len(lines) // BATCH)} batches with "
                  f"{TIERS[step]}{' ' + effort if effort else ''}", flush=True)
            fixed, usd2, models2 = translate_all([x for b in bad for x in b], step, lang, title, effort)
            translated.update(fixed)
            usd += usd2
            models |= models2
            tier = step
            check, failed = checks(lines, translated)
            first["redo"].append({"model": TIERS[step], "effort": effort, "batches": len(bad), "failed": failed})
        if failed:
            print(f"{now()} {title}: still {failed} {check}, step 2 whole file", flush=True)
            tier = 2
            translated, usd2, models2 = translate_all(lines, tier, lang, title)
            usd += usd2
            models |= models2
            check, failed = checks(lines, translated)
            first["redo"].append({"model": TIERS[2], "effort": None, "batches": "all", "failed": failed})
    missing = len(lines) - len(translated)
    score, judge_usd = judge(lines, translated, title) if random.random() < JUDGE_RATE else ({}, 0.0)
    overall = score.get("overall")
    weak = bool(failed)
    entry = {"title": title, "tier": tier, "model": TIERS[tier], "models_used": sorted(models), "lines": len(lines),
             "missing": missing, "checks": check, "failed_checks": failed, "step1": first,
             "score": overall, "accuracy": score.get("accuracy"), "fluency": score.get("fluency"),
             "problems": (score.get("problems") or [])[:5], "weak": weak, "usd": round(usd, 5),
             "judge_usd": round(judge_usd, 5), "seconds": round(time.time() - t0), "at": now(),
             "history": ((state.get(path) or {}).get("history") or [])[-4:]
             + [{"tier": tier, "score": overall, "failed": failed, "at": now()}]}
    with LOCK:
        state = load("state.json", {})
        state[path] = entry
        save("state.json", state)
    print(f"{now()} {title}: step {tier} {TIERS[tier]} {len(lines)} lines, {missing} missing, checks "
          f"{'OK' if not failed else failed}, score {overall}, "
          f"${usd + judge_usd:.4f}, {entry['seconds']} s", flush=True)
    # untranslated lines keep their English text (Bazarr only replaces positions it gets back)
    return [{"position": p, "line": translated[p]} for p, _ in lines if p in translated]


_balance = {"at": 0, "data": {}}


def balance():
    if time.time() - _balance["at"] > 600:
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key()}"})
            d = json.load(urllib.request.urlopen(req, timeout=20)).get("data", {})
            _balance.update(at=time.time(), data={k: d.get(k) for k in ("usage", "usage_monthly", "usage_daily",
                                                                         "limit", "limit_remaining")})
        except Exception as e:  # noqa: BLE001
            _balance["data"]["error"] = str(e)[:100]
    return _balance["data"]


def luna_checks(cur, days=30):
    """How often step 1 fails a free check, over translations of the last `days` days. Only
    entries made with the checks (they have "checks") that started at step 1; /test paths are left out by the caller."""
    since = (dt.datetime.now() - dt.timedelta(days=days)).isoformat(timespec="seconds")
    runs = [e for e in cur if "checks" in e and e.get("at", "") >= since and (e.get("tier") == 1 or e.get("step1"))]
    failed = [e for e in runs if e.get("step1")]
    by = {}
    for e in failed:
        for k in e["step1"].get("failed") or []:
            by[k] = by.get(k, 0) + 1
    return {"days": days, "checked": len(runs), "failed": len(failed),
            "fail_pct": round(100 * len(failed) / len(runs), 1) if runs else None, "by_check": by,
            "still_failing": sum(1 for e in runs if e.get("weak")),
            "recent": [{"title": e["title"].split(" · ")[0], "at": e["at"][:16], "failed": e["step1"].get("failed") or [],
                        "checks": e["step1"].get("checks") or {}, "fixed": not e.get("weak")}
                       for e in sorted(failed, key=lambda e: e["at"], reverse=True)[:10]]}


def stats():
    costs = []
    try:
        costs = [json.loads(l) for l in open(os.path.join(DATA, "costs.jsonl"))]
    except OSError:
        pass
    month = dt.date.today().strftime("%Y-%m")
    state = load("state.json", {})
    cur = [e for p, e in state.items() if not p.startswith("/test/")]
    usd_month = sum(c["usd"] for c in costs if c["t"].startswith(month))
    scored = [e["score"] for e in cur if e.get("score") is not None]
    by_tier = {f"step{t}": sum(1 for e in cur if e["tier"] == t) for t in TIERS}
    return {"eur_total": round(sum(c["usd"] for c in costs) * EUR_PER_USD, 3),
            "eur_month": round(usd_month * EUR_PER_USD, 3),
            "eur_month_judge": round(sum(c["usd"] for c in costs if c["t"].startswith(month) and c["kind"] == "judge")
                                     * EUR_PER_USD, 3),
            "translations": len(cur), "translations_month": sum(1 for e in cur if e["at"].startswith(month)),
            **by_tier, "avg_score": round(sum(scored) / len(scored), 1) if scored else None,
            "weak": sum(1 for e in cur if e.get("weak")),
            "weak_after_step2": [e["title"].split(" · ")[0] for e in cur if e.get("weak") and e["tier"] == 2][:10],
            "luna_checks": luna_checks(cur), "judged": len(scored), "judge_rate": JUDGE_RATE, "limits": LIMITS,
            "tiers": {f"step{t}": m for t, m in TIERS.items()}, "judge": JUDGE, "threshold": THRESHOLD,
            "openrouter": balance(), "updated": now()}


class Handler(http.server.BaseHTTPRequestHandler):
    def reply(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")

    def do_GET(self):
        if self.path == "/api/stats":
            return self.reply(200, stats())
        if self.path == "/api/state":
            return self.reply(200, load("state.json", {}))
        if self.path in ("/", "/health"):
            return self.reply(200, {"ok": True, "tiers": TIERS})
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        try:
            if self.path == "/api/translate/content":
                return self.reply(200, handle_translate(self.body()))
            if self.path == "/api/tier":  # {"path": translatedSubtitlePath, "tier": 2}
                b = self.body()
                with LOCK:
                    state = load("state.json", {})
                    state.setdefault(b["path"], {"title": b.get("title", "?"), "tier": 1, "at": now()})["next_tier"] = int(b["tier"])
                    save("state.json", state)
                return self.reply(200, {"ok": True})
        except Exception as e:  # noqa: BLE001
            print(f"{now()} error: {e}", flush=True)
            return self.reply(500, {"error": str(e)[:200]})
        self.reply(404, {"error": "not found"})

    def log_message(self, fmt, *args):  # quiet; our own lines go to stdout
        pass


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    print(f"{now()} subtrans on :{PORT}, steps {TIERS}, free checks {LIMITS}, judge {JUDGE} on {JUDGE_RATE:.0%}",
          flush=True)
    http.server.ThreadingHTTPServer(("", PORT), Handler).serve_forever()
