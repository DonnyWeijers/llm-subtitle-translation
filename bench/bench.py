#!/usr/bin/env python3
"""Subtitle translation benchmark, phase 1: translate the same passages with every model and measure them.

  OPENROUTER_KEY=... PAIRS=./pairs RUN=bench1 python3 bench.py

Test set: PAIRS holds episodes that have an English subtitle AND a human Dutch one: <name>.en.srt + <name>.nl.srt.
From each of N_EP episodes, two LINES-line English passages (at 30 % and 70 %) are translated the way Bazarr does it
(its system instruction + a JSON list of lines, one call). Per model and passage it stores the lines and:
  chrF   character n-gram F-score vs the human Dutch lines in the same time window (the usual MT metric; "copy" =
         the English itself = the floor)
  cost   exact, from OpenRouter (usage.cost, USD), scaled to EUR per average episode (EP_CHARS Dutch characters)
  speed  seconds per call; errors: missing lines, unparsable JSON, lines left in English, lines > 50 characters
Output: out/<RUN>/ (one JSON per model and passage, testset.json, summary.json). Phase 2 (judge.py) reads it.
Model names are OpenRouter ids; "<model>#nothink" turns thinking off, "<model>@<effort>" sets the reasoning effort.
Use a key with its own monthly cap for benchmarks, so a run can't eat the production budget.
"""
import concurrent.futures as cf
import glob
import json
import os
import re
import time
import urllib.request
from collections import Counter

N_EP, LINES, STARTS = int(os.environ.get("N_EP", 4)), 100, (0.3, 0.7)  # two 100-line passages per episode
EUR_PER_USD, EP_CHARS = 0.86, 26000
RUN = os.environ.get("RUN", time.strftime("%Y%m%d-%H%M"))
OUT = os.path.join(os.environ.get("OUT", "out"), RUN)
PAIRS = os.environ.get("PAIRS", "pairs")
OR_KEY = os.environ.get("OPENROUTER_KEY", "")
MODELS = os.environ.get("MODELS", ",".join([
    "google/gemini-3.8-flash", "google/gemini-3.7-flash", "google/gemini-3.5-flash-lite",
    "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4.1-flash", "openai/gpt-6-luna",
    "mistralai/mistral-small-2603", "google/gemma-4-31b-it", "qwen/qwen3.6-flash",
    "meta-llama/llama-3.3-70b-instruct", "z-ai/glm-5.3-flash", "openai/gpt-5-mini",
    "anthropic/claude-haiku-4.5", "deepseek/deepseek-v4-flash#nothink",
    "deepseek/deepseek-v4.1-flash#nothink"])).split(",")
REASONING = {"openai/gpt-6-luna", "openai/gpt-5-mini", "z-ai/glm-5.3-flash"}  # run with the lowest reasoning effort
WORKERS = int(os.environ.get("WORKERS", 6))  # 1 for :free models (rate limits)
INSTRUCTION = """You are an assistant that translates subtitles to Dutch.
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
EN_WORDS = set("the you and is are to of what that this it i'm don't we he she they have not with for was your".split())


def parse_srt(path):
    out = []
    for b in re.split(r"\n\s*\n", open(path, encoding="utf-8", errors="replace").read().strip()):
        parts = b.splitlines()
        idx = next((i for i, p in enumerate(parts[:3]) if "-->" in p), None)
        if idx is None:
            continue
        m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+) --> (\d+):(\d+):(\d+)[,.](\d+)", parts[idx].strip())
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start, end = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000, g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        text = re.sub(r"<[^>]+>|\{[^}]+\}", "", "\n".join(parts[idx + 1:])).strip()
        if text:
            out.append((start, end, text))
    return out


def chrf(hyp, ref, n=6, beta=2.0):
    hyp, ref = re.sub(r"\s+", "", hyp.lower()), re.sub(r"\s+", "", ref.lower())
    ps, rs = [], []
    for k in range(1, n + 1):
        h = Counter(hyp[i:i + k] for i in range(len(hyp) - k + 1))
        r = Counter(ref[i:i + k] for i in range(len(ref) - k + 1))
        m = sum((h & r).values())
        ps.append(m / max(sum(h.values()), 1))
        rs.append(m / max(sum(r.values()), 1))
    p, r = sum(ps) / n, sum(rs) / n
    return 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if p + r else 0.0


def test_set():
    picked, episodes = [], 0
    for en in sorted(glob.glob(os.path.join(PAIRS, "*.en.srt"))):
        nl = en[:-len(".en.srt")] + ".nl.srt"
        if not os.path.exists(nl):
            continue
        e, d = parse_srt(en), parse_srt(nl)
        if len(e) < 300 or len(d) < 250:
            continue
        segs = []
        for start in STARTS:
            i = int(len(e) * start)
            src = e[i:i + LINES]
            t0, t1 = src[0][0], src[-1][1]
            ref = [x[2] for x in d if x[1] > t0 and x[0] < t1]
            if len(ref) >= LINES * 0.6:  # else the Dutch file doesn't cover this part (other cut / out of sync)
                segs.append({"title": os.path.basename(en)[:-len(".en.srt")], "src": [x[2] for x in src],
                             "ref": "\n".join(ref)})
        if len(segs) < len(STARTS):
            continue
        for s in segs:
            picked.append({"id": len(picked), **s})
        episodes += 1
        if episodes == N_EP:
            break
    if not picked:
        raise SystemExit(f"no usable <name>.en.srt + <name>.nl.srt pairs in {PAIRS}")
    return picked


def http_json(url, body, headers, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def parse_lines(text, n):
    m = re.search(r"\[.*\]", text or "", re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except ValueError:
        return [""] * n, False
    by = {str(i.get("index")): i.get("content", "") for i in items if isinstance(i, dict)}
    return [by.get(str(k), "") or "" for k in range(n)], True


def translate(model, lines):
    batch = json.dumps([{"index": str(i), "content": t} for i, t in enumerate(lines)], ensure_ascii=False)
    nothink = model.endswith("#nothink")
    name, _, effort = model.replace("#nothink", "").partition("@")
    body = {"model": name, "usage": {"include": True},
            "messages": [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": batch}]}
    if nothink:
        body["reasoning"] = {"enabled": False}
    elif effort:
        body["reasoning"] = {"effort": effort}
    elif name in REASONING:
        body["reasoning"] = {"effort": "low"}
    r = http_json("https://openrouter.ai/api/v1/chat/completions", body, {"Authorization": f"Bearer {OR_KEY}"})
    if "error" in r:
        raise RuntimeError(str(r["error"])[:200])
    u = r.get("usage", {})
    text = r["choices"][0]["message"].get("content") or ""
    think = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    out, ok = parse_lines(text, len(lines))
    return out, ok, {"tokens_in": u.get("prompt_tokens", 0), "tokens_out": u.get("completion_tokens", 0),
                     "tokens_thinking": think, "usd": float(u.get("cost") or 0)}


def english_left(src, out):
    """Lines that still read as English: 3+ common English words, or 2+ and identical to the source."""
    def words(x):
        return set(re.findall(r"[a-z']+", x.lower())) & EN_WORDS
    return sum(1 for s, x in zip(src, out)
               if x and (len(words(x)) >= 3 or (len(words(x)) >= 2 and x.strip().lower() == s.strip().lower())))


def run_one(model, ep):
    best = None
    tries = 4 if ":free" in model else 2  # free models: rate limits (429) -> wait and retry
    for attempt in range(tries):  # retry on an error or unparsable answer, like Bazarr's retries
        if attempt and best and best.get("error"):
            time.sleep(30 * attempt if ":free" in model else 5)
        t0 = time.time()
        try:
            out, ok, use = translate(model, ep["src"])
            err = None
        except Exception as e:  # noqa: BLE001
            out, ok, use, err = [""] * len(ep["src"]), False, {"usd": 0.0}, str(e)[:200]
        res = {"model": model, "episode": ep["id"], "title": ep["title"], "seconds": round(time.time() - t0, 1),
               "parsed": ok, "error": err, "attempt": attempt + 1, **use, "lines": out}
        if best:
            res["usd"] += best["usd"]  # a retry costs too
        best = res
        if ok and not err:
            break
    hyp = "\n".join(best["lines"])
    best.update({
        "chrF": round(chrf(hyp, ep["ref"]), 2),
        "missing": sum(1 for x in best["lines"] if not x.strip()),
        "english_left": english_left(ep["src"], best["lines"]),
        "too_long": sum(1 for x in best["lines"] for ln in x.splitlines() if len(ln) > 50),
        "out_chars": len(hyp.replace("\n", "")),
    })
    with open(os.path.join(OUT, f"{model.replace(':', '_').replace('/', '_')}__{ep['id']}.json"), "w") as f:
        json.dump(best, f, ensure_ascii=False)
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    # TESTSET=<earlier run>: the same passages as that run, so new models compare with its summary
    eps = json.load(open(os.path.join(os.path.dirname(OUT), os.environ["TESTSET"], "testset.json"))) \
        if os.environ.get("TESTSET") else test_set()
    json.dump(eps, open(os.path.join(OUT, "testset.json"), "w"), ensure_ascii=False)
    for e in eps:  # floor: the English itself
        with open(os.path.join(OUT, f"copy__{e['id']}.json"), "w") as f:
            json.dump({"model": "copy", "episode": e["id"], "chrF": round(chrf("\n".join(e["src"]), e["ref"]), 2),
                       "lines": e["src"], "usd": 0}, f, ensure_ascii=False)
    jobs = [(m, e) for m in MODELS for e in eps]
    results = []
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for res in pool.map(lambda me: run_one(*me), jobs):
            results.append(res)
            print(json.dumps({k: res[k] for k in ("model", "episode", "chrF", "usd", "seconds", "missing",
                                                     "english_left", "parsed", "error")}), flush=True)
    summary = {}
    for m in MODELS:
        rs = [r for r in results if r["model"] == m]
        chars = sum(r["out_chars"] for r in rs) or 1
        usd = sum(r["usd"] for r in rs)
        summary[m] = {"chrF": round(sum(r["chrF"] for r in rs) / len(rs), 2),
                      "eur_per_episode": round(usd * EUR_PER_USD / chars * EP_CHARS, 4),
                      "sec_per_100_lines": round(sum(r["seconds"] for r in rs) / len(rs), 1),
                      "missing": sum(r["missing"] for r in rs), "english_left": sum(r["english_left"] for r in rs),
                      "too_long": sum(r["too_long"] for r in rs), "failed_calls": sum(1 for r in rs if not r["parsed"]),
                      "thinking_tokens": sum(r.get("tokens_thinking", 0) for r in rs),
                      "usd_spent": round(usd, 4)}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
