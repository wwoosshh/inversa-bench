"""E13 — discriminant validity: is IGS *math-specific*, or does it just track general capability (g)?

Convergent validity (E10) showed IGS correlates with AIME (rho=+0.93). But every benchmark
correlates with overall model strength, so a high IGS-AIME correlation alone cannot tell
"IGS is a math test" from "IGS is a g-meter." This runs a NON-math benchmark (history,
literature, philosophy, law, linguistics, ... ; exact-letter machine grading, no LLM judge) on
the same E10 models and asks:

  - does IGS correlate with AIME (math) MORE than with the non-math axis?
  - does IGS<->AIME survive partialling out the non-math axis (math signal above g)?

PRE-REGISTERED decision rule (fixed before seeing results):
  STRONG (IGS is math-specific) supported iff  rho(IGS,AIME) - rho(IGS,nonmath) >= 0.15
  AND partial rho(IGS,AIME | nonmath) >= 0.30 ; otherwise fall back to the WEAK claim
  (IGS is a contamination-proof capability measure that agrees with hard math benchmarks).

Usage: python scripts/run_discriminant.py   (reads data/results/e10_results.json for IGS+AIME)
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

PROMPT = ("Answer this multiple-choice question. Think briefly if needed, but you MUST end with "
          "ONLY the single letter of the correct choice on a final line prefixed by '#### '.\n"
          "Example final line: '#### C'\n\n{q}\n"
          "A) {A}\nB) {B}\nC) {C}\nD) {D}")


def extract_choice(text: str):
    """Pull the chosen letter A-D: prefer the text after the last '####', else the last bare letter."""
    if not text:
        return None
    tail = text.rsplit("####", 1)[-1]
    m = re.search(r"[A-D]", tail.upper())
    if m:
        return m.group(0)
    hits = re.findall(r"\b([A-D])\b", text.upper())
    return hits[-1] if hits else None


def score_model(model, questions, provider, base_url, key_env):
    """Return non-math accuracy for one model (None if it cannot be measured at all)."""
    ad = _make_adapter(model, provider, base_url, key_env, max_tokens=3000,
                       reasoning_max_tokens=2000, temperature=0.0)
    correct = answered = 0
    for q in questions:
        try:
            raw = ad.generate(PROMPT.format(q=q["question"], **q["choices"]))
        except Exception:
            continue
        answered += 1
        if extract_choice(raw) == q["answer"]:
            correct += 1
    return (correct / answered) if answered else None


def main(argv=None) -> None:
    load_dotenv()
    try:  # Windows cp949 console crashes on non-ASCII at print time
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="E13 discriminant validity (IGS vs non-math)")
    ap.add_argument("--e10", default="data/results/e10_results.json")
    ap.add_argument("--bank", default="data/banks/nonmath_mc_set.json")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--base-url", default=OPENROUTER_BASE_URL)
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--out", default="data/results/discriminant_results.json")
    args = ap.parse_args(argv)

    e10 = json.load(open(args.e10, encoding="utf-8"))["models"]
    base = {r["model"]: {"igs": r["igs"], "aime": r["aime"]}
            for r in e10 if r.get("igs") is not None}
    questions = json.load(open(args.bank, encoding="utf-8"))["questions"]
    models = list(base)

    accs = map_concurrent(
        lambda m: score_model(m, questions, args.provider, args.base_url, args.key_env),
        models, args.max_workers)
    for m, a in zip(models, accs):
        if a is not None:
            base[m]["nonmath"] = a
            print(f"[done] {m}: nonmath={a:.0%} (IGS={base[m]['igs']:.2f} AIME={base[m]['aime']:.0%})",
                  flush=True)
        else:
            print(f"[skip] {m}: non-math unmeasurable", flush=True)

    rows = [{"model": m, **v} for m, v in base.items() if "nonmath" in v]
    igs = [r["igs"] for r in rows]
    aime = [r["aime"] for r in rows]
    nonmath = [r["nonmath"] for r in rows]

    r_ia, lo_ia, hi_ia = spearman_ci(igs, aime)
    r_in, lo_in, hi_in = spearman_ci(igs, nonmath)
    r_an = spearman(aime, nonmath)
    pr = partial_spearman(igs, aime, nonmath)

    # A discriminant test is only valid if the non-math control actually SPREADS the models
    # (acts as a real capability axis). If it saturates (near-constant), a large gap is an
    # artifact of zero variance, not evidence of math-specificity — guard against that.
    nm_distinct = len(set(round(v, 3) for v in nonmath))
    nm_range = (max(nonmath) - min(nonmath)) if nonmath else 0.0
    nonmath_saturated = nm_distinct < 4 or nm_range < 0.15

    strong = (not nonmath_saturated and r_ia is not None and r_in is not None
              and (r_ia - r_in) >= 0.15 and pr is not None and pr >= 0.30)
    if nonmath_saturated:
        verdict = (f"INCONCLUSIVE: the non-math control SATURATED (distinct values={nm_distinct}, "
                   f"range={nm_range:.2f}) — it has no spread, so it cannot act as a discriminant / "
                   f"general-capability axis and math-specificity is UNTESTABLE here. Build a harder "
                   f"non-math control (hard reasoning, not factual recall) or adopt the WEAK claim.")
    elif strong:
        verdict = ("STRONG supported: IGS is math-specific (correlates with AIME well above the "
                   "non-math axis, and survives controlling for it)")
    else:
        verdict = ("WEAK (fall back): IGS reads as a contamination-proof capability measure that "
                   "agrees with hard math benchmarks; math-specificity is not established by this test")

    analysis = {
        "n": len(rows),
        "rho_igs_aime": r_ia, "ci_igs_aime": [lo_ia, hi_ia],
        "rho_igs_nonmath": r_in, "ci_igs_nonmath": [lo_in, hi_in],
        "rho_aime_nonmath": r_an,
        "partial_rho_igs_aime_given_nonmath": pr,
        "rule": "STRONG iff (rho_igs_aime - rho_igs_nonmath) >= 0.15 AND partial >= 0.30",
        "strong_supported": strong,
        "verdict": verdict,
    }
    json.dump({"analysis": analysis, "models": sorted(rows, key=lambda x: -x["igs"])},
              open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n=== E13 discriminant validity ===")
    print(f"N={len(rows)}")
    print(f"  rho(IGS, AIME)    = {r_ia:+.2f}  [{lo_ia:+.2f},{hi_ia:+.2f}]   (convergent, math)")
    print(f"  rho(IGS, nonmath) = {r_in:+.2f}  [{lo_in:+.2f},{hi_in:+.2f}]   (discriminant, non-math)")
    print(f"  rho(AIME, nonmath)= {r_an:+.2f}   (the g baseline)")
    print(f"  partial rho(IGS, AIME | nonmath) = {pr:+.2f}" if pr is not None else "  partial = None")
    print(f"  gap rho(IGS,AIME) - rho(IGS,nonmath) = {(r_ia - r_in):+.2f}")
    print(f"VERDICT: {verdict}")
    print(f"wrote {args.out}")
    os._exit(0)


if __name__ == "__main__":
    main()
