#!/usr/bin/env python3
"""OpenRouter spend and budget per app, as one JSON status for a dashboard (run from cron, e.g. every 15 minutes).

Every AI call goes through OpenRouter, with one API key per app, each with its own monthly cap: when a key hits its
cap its calls stop, whatever the account balance, so a runaway benchmark can't stop the production app. This script
writes OUT (no secrets in it):
  balance   account credit left: credits bought - used (OpenRouter /credits)
  keys      per key/app: monthly cap, remaining, spend today / week / month, projected month-end
  key       the key that must not stop (MAIN_APP), for a single "is production safe" tile
  month     spend this calendar month (UTC, as OpenRouter counts it), projected to month end, by app (from the keys)
            and by model (from subtrans' own per-call log, exact usage.cost)
Benchmarks are one-offs: their spend is counted once, not extrapolated to the month end.

Settings (environment):
  KEY_DIR    folder with one key file per app (default ~/.config/openrouter); keys are never printed or written
  KEYS       app=file pairs, e.g. "Subtitles=subtitles.key,Benchmarks=bench.key"
  MAIN_APP   the app whose key must not stop (default: the first in KEYS)
  COSTS      subtrans' costs.jsonl (optional; titles starting with TEST count as Benchmarks)
  OUT        output file (default ./openrouter_status.json)
  EUR_PER_USD
"""
import calendar
import datetime as dt
import json
import os
import tempfile
import urllib.error
import urllib.request
from collections import defaultdict

KEY_DIR = os.path.expanduser(os.environ.get("KEY_DIR", "~/.config/openrouter"))
KEYS = [tuple(p.split("=", 1)) for p in
        os.environ.get("KEYS", "Subtitles=subtitles.key,Benchmarks=bench.key").split(",") if "=" in p]
MAIN_APP = os.environ.get("MAIN_APP", KEYS[0][0] if KEYS else "")
COSTS = os.environ.get("COSTS", "")
OUT = os.environ.get("OUT", "openrouter_status.json")
EUR_PER_USD = float(os.environ.get("EUR_PER_USD", 0.86))
ONE_OFF_APPS = {"Benchmarks"}


def api(path, keyfile):
    req = urllib.request.Request(f"https://openrouter.ai/api/v1/{path}",
                                 headers={"Authorization": f"Bearer {open(keyfile).read().strip()}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["data"]


def logged(month):
    """(app, model) -> USD this month from the per-call log."""
    out = defaultdict(float)
    if not COSTS:
        return out
    try:
        with open(COSTS) as f:
            for line in f:
                try:
                    c = json.loads(line)
                except ValueError:
                    continue
                if c.get("t", "").startswith(month):
                    app = "Benchmarks" if (c.get("title") or "").startswith("TEST") else MAIN_APP
                    out[(app, c["model"])] += float(c.get("usd") or 0)
    except OSError:
        pass
    return out


def main():
    now = dt.datetime.now(dt.timezone.utc)
    month = now.strftime("%Y-%m")
    days = calendar.monthrange(now.year, now.month)[1]
    elapsed = (now.day - 1) + now.hour / 24 + now.minute / 1440
    keys, credits = [], None
    for app, name in KEYS:
        f = os.path.join(KEY_DIR, name)
        if not os.path.exists(f):
            continue
        try:
            k = api("key", f)
            credits = credits or api("credits", f)
        except urllib.error.HTTPError as e:  # a revoked key: report it instead of failing the whole status
            keys.append({"app": app, "error": f"HTTP {e.code}", "month": 0.0, "projected": 0.0})
            continue
        m = float(k.get("usage_monthly") or 0)
        keys.append({"app": app, "limit": k.get("limit"), "reset": k.get("limit_reset"),
                     "remaining": round(float(k["limit_remaining"]), 2) if k.get("limit_remaining") is not None else None,
                     "month": round(m, 3), "today": round(float(k.get("usage_daily") or 0), 3),
                     "week": round(float(k.get("usage_weekly") or 0), 3),
                     "projected": round(m / elapsed * days, 2) if elapsed >= 1 and app not in ONE_OFF_APPS else round(m, 2)})
    if credits is None:
        raise SystemExit(f"no usable key in {KEY_DIR} for {[n for _, n in KEYS]}")
    spent_keys = sum(k["month"] for k in keys)
    by_model, by_app_log = defaultdict(float), defaultdict(float)
    for (app, model), usd in logged(month).items():
        by_model[model] += usd
        by_app_log[app] += usd
    logged_total = sum(by_model.values())
    # per key is exact, but only covers the keys that exist now: a key deleted mid-month takes its spend with it,
    # so then the per-call log gives the fuller picture
    if logged_total > spent_keys + 0.01:
        spent, by_app, extra = logged_total, dict(by_app_log), logged_total - spent_keys
    else:
        spent, by_app, extra = spent_keys, {k["app"]: k["month"] for k in keys}, 0.0

    def eur(usd):
        return round(usd * EUR_PER_USD, 2)

    status = {
        "balance": round(float(credits["total_credits"]) - float(credits["total_usage"]), 2),
        "credits_total": round(float(credits["total_credits"]), 2),
        "used_total": round(float(credits["total_usage"]), 2),
        "keys": keys,
        "key": next((k for k in keys if k["app"] == MAIN_APP), keys[0] if keys else {}),
        "month": {"usd": round(spent, 2), "eur": eur(spent),
                  "projected_usd": round(sum(k["projected"] for k in keys) + extra, 2) if elapsed >= 1 else None,
                  "source": "logs" if extra else "keys",
                  "by_app": [{"app": a, "usd": round(u, 3), "eur": eur(u)} for a, u in sorted(by_app.items(), key=lambda kv: -kv[1])],
                  "by_model": [{"model": m, "usd": round(u, 3), "eur": eur(u)} for m, u in sorted(by_model.items(), key=lambda kv: -kv[1])][:8]},
        "eur_per_usd": EUR_PER_USD,
        "updated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(OUT)), prefix=".openrouter_status.")
    with os.fdopen(fd, "w") as f:
        json.dump(status, f)
    os.chmod(tmp, 0o644)
    os.replace(tmp, OUT)
    print(json.dumps(status))


if __name__ == "__main__":
    main()
