"""E14 — predictive (incremental) validity: does IGS predict construction/verification outcomes
BEYOND forward solving (AIME)? Runs Y1 (constrained construction) and Y2 (error detection) on the
E10 cohort, then for each outcome Y computes rho(IGS,Y), rho(AIME,Y) and the partial rho(IGS,Y|AIME)
with a spread guard and the pre-registered rule from docs/specs/2026-06-08-e14-predictive-validity.md.

Usage: python scripts/run_predictive.py   (reads data/results/e10_results.json for IGS + AIME)
"""
from __future__ import annotations

import argparse
import json
import os
import re

from dotenv import load_dotenv

from inversa.analysis import partial_spearman, spearman, spearman_ci
from inversa.bench_core import map_concurrent
from inversa.cli_experiment import OPENROUTER_BASE_URL, _make_adapter
from inversa.predictive import check_construction
from inversa.tasks.posing import extract_equation


def _yesno(text: str):
    if not text:
        return None
    tail = text.rsplit("####", 1)[-1].lower()
    m = re.search(r"\b(yes|no)\b", tail)
    if m:
        return m.group(1)
    hits = re.findall(r"\b(yes|no)\b", text.lower())
    return hits[-1] if hits else None


def _score_model(model, y1, y2, provider, base_url, key_env):
    ad = _make_adapter(model, provider, base_url, key_env, max_tokens=3000,
                       reasoning_max_tokens=2000, temperature=0.0)

    def run(items, grade):
        ok = answered = 0
        for it in items:
            try:
                raw = ad.generate(it["prompt"])
            except Exception:
                continue
            answered += 1
            if grade(it, raw):
                ok += 1
        return (ok / answered) if answered else None

    y1s = run(y1, lambda it, raw: check_construction(it["checks"], extract_equation(raw))[0])
    y2s = run(y2, lambda it, raw: _yesno(raw) == it["answer"])
    return {"y1": y1s, "y2": y2s}


def _spread_ok(vals):
    return len(set(round(v, 3) for v in vals)) >= 4 and (max(vals) - min(vals)) >= 0.15


def _analyze(name, igs, aime, y):
    r_iy, lo_iy, hi_iy = spearman_ci(igs, y)
    r_ay = spearman(aime, y)
    pr = partial_spearman(igs, y, aime)
    saturated = not _spread_ok(y)
    gap = (r_iy - r_ay) if (r_iy is not None and r_ay is not None) else None
    if saturated:
        verdict = "INCONCLUSIVE (outcome saturated — no spread; cannot test prediction)"
        supported = False
    elif pr is not None and pr >= 0.30 and gap is not None and gap >= 0.10:
        verdict = "SUPPORTED: IGS predicts this outcome incrementally over forward solving"
        supported = True
    elif pr is not None and pr < 0.10:  # not meaningfully positive (near-zero OR negative)
        verdict = ("REJECTED: no incremental prediction over AIME — IGS adds nothing beyond "
                   "forward solving for this outcome (partial <= 0)")
        supported = False
    else:
        verdict = "INCONCLUSIVE: partial signal but below the pre-registered bar"
        supported = False
    return {"outcome": name, "n": len(y), "rho_igs_y": r_iy, "ci_igs_y": [lo_iy, hi_iy],
            "rho_aime_y": r_ay, "partial_rho_igs_y_given_aime": pr, "gap_igs_minus_aime": gap,
            "outcome_saturated": saturated, "supported": supported, "verdict": verdict}


def main(argv=None) -> None:
    load_dotenv()
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="E14 predictive (incremental) validity")
    ap.add_argument("--e10", default="data/results/e10_results.json")
    ap.add_argument("--y1", default="data/banks/construct_constraints.json")
    ap.add_argument("--y2", default="data/banks/verify_pairs.json")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--base-url", default=OPENROUTER_BASE_URL)
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--out", default="data/results/predictive_results.json")
    args = ap.parse_args(argv)

    e10 = json.load(open(args.e10, encoding="utf-8"))["models"]
    base = {r["model"]: {"igs": r["igs"], "aime": r["aime"]}
            for r in e10 if r.get("igs") is not None}
    y1 = json.load(open(args.y1, encoding="utf-8"))["items"]
    y2 = json.load(open(args.y2, encoding="utf-8"))["items"]
    models = list(base)

    scored = map_concurrent(
        lambda m: _score_model(m, y1, y2, args.provider, args.base_url, args.key_env),
        models, args.max_workers)
    for m, s in zip(models, scored):
        if s is None:
            continue
        base[m].update(s)
        print(f"[done] {m}: Y1(construct)={_fmt(s['y1'])} Y2(verify)={_fmt(s['y2'])} "
              f"(IGS={base[m]['igs']:.2f} AIME={base[m]['aime']:.0%})", flush=True)

    analyses = []
    for name in ("y1", "y2"):
        rows = [(v["igs"], v["aime"], v[name]) for v in base.values()
                if v.get(name) is not None]
        if len(rows) >= 5:
            igs = [r[0] for r in rows]; aime = [r[1] for r in rows]; yv = [r[2] for r in rows]
            analyses.append(_analyze(name, igs, aime, yv))

    out = {"benchmark": "E14 predictive validity", "models": base, "analyses": analyses}
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n=== E14 predictive (incremental) validity ===")
    for a in analyses:
        nm = "Y1 constrained-construction" if a["outcome"] == "y1" else "Y2 error-detection"
        print(f"\n{nm} (N={a['n']}):")
        print(f"  rho(IGS,Y)   = {_f(a['rho_igs_y'])}   rho(AIME,Y) = {_f(a['rho_aime_y'])}")
        print(f"  partial rho(IGS,Y | AIME) = {_f(a['partial_rho_igs_y_given_aime'])}"
              f"   gap = {_f(a['gap_igs_minus_aime'])}")
        print(f"  VERDICT: {a['verdict']}")
    print(f"\nwrote {args.out}")
    os._exit(0)


def _fmt(v):
    return "n/a" if v is None else f"{v:.0%}"


def _f(v):
    return "None" if v is None else f"{v:+.2f}"


if __name__ == "__main__":
    main()
