#!/usr/bin/env python3
"""Subtitle translation benchmark, phase 2: blind LLM judges.

  OPENROUTER_KEY=... RUN=bench1 MODELS="openai/gpt-6-luna,google/gemini-3.7-flash,human" python3 judge.py

Each judge sees the English passage (numbered lines) and ONE Dutch version (a model's, or the human subtitle for the
same time window: "human" = the anchor), never which one it is. Rubric scores 0-100 (accuracy, fluency, subtitle
style, overall) + up to 5 worst errors. Two judges from different families (Anthropic, OpenAI), so no judge rates
"its own" candidates alone; jobs run in a shuffled order. Results: out/<RUN>/judge.json and judge-summary.json.
"""
import concurrent.futures as cf
import json
import os
import random
import re
import time
import urllib.request

RUN = os.environ.get("RUN", "bench1")
DIR = os.path.join(os.environ.get("OUT", "out"), RUN)
OR_KEY = os.environ.get("OPENROUTER_KEY", "")
JUDGES = os.environ.get("JUDGES", "anthropic/claude-sonnet-5.5,openai/gpt-6-sol").split(",")
MODELS = os.environ.get("MODELS", "human").split(",")
RUBRIC = """You are a senior Dutch subtitle editor reviewing a Dutch subtitle for a TV series, against the English
original. The Dutch may be made by a human subtitler or by software; judge only the result.

Score 0-100 each:
- accuracy: meaning preserved, including context ("I do." answering "know what you're doing?"), idioms, tone, who is
  addressed; omissions that a good subtitler would make for brevity are fine, wrong meaning is not
- fluency: natural spoken Dutch as on Dutch television, no anglicisms or literal word order
- style: usable as subtitles: concise, readable, consistent names and forms of address (je/u), nothing left in English
  that should be Dutch
- overall: 90+ professional broadcast quality; 80 good, a few minor issues; 70 understandable but clearly
  machine-like or with a couple of meaning errors; 60 several meaning errors; below 50 misleading or broken

List at most 5 of the most serious problems (line number of the English line, short note in English).
Answer ONLY with JSON: {"accuracy": n, "fluency": n, "style": n, "overall": n,
 "problems": [{"line": n, "type": "meaning|fluency|style|untranslated", "note": "..."}]}"""


def call(judge, prompt):
    body = {"model": judge, "usage": {"include": True}, "temperature": 0,
            "messages": [{"role": "system", "content": RUBRIC}, {"role": "user", "content": prompt}]}
    if judge.startswith("openai/"):
        body["reasoning"] = {"effort": "low"}  # keep the judge's own thinking (cost) small
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 method="POST", headers={"Content-Type": "application/json",
                                                         "Authorization": f"Bearer {OR_KEY}"})
    err = None
    for attempt in range(3):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=300))
            text = r["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0)), float((r.get("usage") or {}).get("cost") or 0)
        except Exception as e:  # noqa: BLE001
            err = str(e)[:150]
            time.sleep(10 * (attempt + 1))
    return {"error": err}, 0.0


def main():
    segs = json.load(open(os.path.join(DIR, "testset.json")))
    jobs = []
    for s in segs:
        english = "\n".join(f"{i + 1}. {t.replace(chr(10), ' / ')}" for i, t in enumerate(s["src"]))
        for m in MODELS:
            if m == "human":
                dutch = s["ref"].replace("\n\n", "\n")
            else:
                f = os.path.join(DIR, f"{m.replace(':', '_').replace('/', '_')}__{s['id']}.json")
                if not os.path.exists(f):
                    continue
                dutch = "\n".join(x.replace("\n", " / ") for x in json.load(open(f))["lines"])
            prompt = f"ENGLISH:\n{english}\n\nDUTCH SUBTITLE (same passage):\n{dutch}"
            for j in JUDGES:
                jobs.append((j, m, s["id"], prompt))
    random.Random(48).shuffle(jobs)  # no fixed order per model

    def run(job):
        j, m, sid, prompt = job
        res, usd = call(j, prompt)
        return {"judge": j, "model": m, "segment": sid, "usd": usd, **res}

    out = []
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        for r in pool.map(run, jobs):
            out.append(r)
            print(json.dumps({k: r.get(k) for k in ("judge", "model", "segment", "overall", "usd", "error")}), flush=True)
    json.dump(out, open(os.path.join(DIR, "judge.json"), "w"), ensure_ascii=False, indent=1)
    summ = {}
    for m in MODELS:
        rs = [r for r in out if r["model"] == m and isinstance(r.get("overall"), (int, float))]
        if not rs:
            continue
        per_judge = {j: round(sum(r["overall"] for r in rs if r["judge"] == j)
                              / max(1, sum(1 for r in rs if r["judge"] == j)), 1) for j in JUDGES}
        summ[m] = {"overall": round(sum(r["overall"] for r in rs) / len(rs), 1), "per_judge": per_judge,
                   "accuracy": round(sum(r.get("accuracy", 0) for r in rs) / len(rs), 1),
                   "fluency": round(sum(r.get("fluency", 0) for r in rs) / len(rs), 1),
                   "style": round(sum(r.get("style", 0) for r in rs) / len(rs), 1),
                   "min_segment": min(r["overall"] for r in rs), "n": len(rs)}
    summ["_cost_usd"] = round(sum(r["usd"] for r in out), 3)
    json.dump(summ, open(os.path.join(DIR, "judge-summary.json"), "w"), indent=1)
    print("SUMMARY " + json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
