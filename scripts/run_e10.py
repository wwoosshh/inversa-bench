"""E10 — discrimination-power competition: hard forward (AIME) vs IGS, same models.
Computes Spearman(IGS, AIME) rank-agreement + a bootstrap CI, and the discrimination power of
each axis (std, distinct levels, spread, fraction tied at max). Then classifies per §5.6:
  (A) rankings differ -> IGS construct must be proven separately;
  (B1) agree & forward sharper -> IGS dies first (no durable value);
  (B2) agree & IGS sharper -> validated AND future-proof (the win).
Usage: python scripts/run_e10.py [--models ...]
"""
import argparse
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from inversa.analysis import spearman_ci
from inversa.cli_experiment import _make_adapter
from inversa.scoring import igs as igs_fn
from inversa.tasks.adversarial import parse_numeric_answer

PROMPT = ("Solve this AIME competition problem. The answer is an integer from 0 to 999. "
          "Reason step by step, then end with a line in EXACTLY this form:\n#### <integer>\n\n{q}")


def disc(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return {}
    mx = max(vals)
    return {"std": statistics.pstdev(vals), "spread": mx - min(vals),
            "distinct": len(set(round(v, 3) for v in vals)),
            "frac_tied_max": sum(1 for v in vals if abs(v - mx) < 1e-9) / len(vals)}


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--bank", default="data/banks/aime_set.json")
    ap.add_argument("--igs", default="data/results/paper_level3_all.json")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--deadline", type=float, default=2400.0)
    ap.add_argument("--json-out", default="data/results/e10_results.json")
    args = ap.parse_args()

    problems = json.load(open(args.bank, encoding="utf-8"))["problems"][:args.n]
    igs_rows = json.load(open(args.igs, encoding="utf-8"))
    igs_by_model = {r["model"]: igs_fn(r["pose_validity"], r["transform_validity"]) for r in igs_rows}
    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              or sorted(igs_by_model))  # default: all models that have an IGS

    def run_one(m):
        try:
            ad = _make_adapter(m, "openrouter", "https://openrouter.ai/api/v1",
                               "OPENROUTER_API_KEY", args.max_tokens)
            ok = 0
            for p in problems:
                ans = parse_numeric_answer(ad.generate(PROMPT.format(q=p["problem"])))
                if ans is not None and abs(ans - float(p["answer"])) < 0.5:
                    ok += 1
            acc = ok / len(problems)
        except Exception as e:
            print(f"[skip] {m}: {type(e).__name__}: {e}", flush=True)
            return None
        print(f"[done] {m}: AIME={acc:.0%} (IGS={igs_by_model.get(m)})", flush=True)
        return {"model": m, "aime": acc, "igs": igs_by_model.get(m)}

    def analyze(res):
        pairs = [(r["igs"], r["aime"]) for r in res if r["igs"] is not None]
        rho, lo, hi = spearman_ci([p[0] for p in pairs], [p[1] for p in pairs])
        d_igs = disc([p[0] for p in pairs])
        d_aime = disc([p[1] for p in pairs])
        verdict = "indeterminate"
        if rho is not None:
            agree = rho >= 0.5
            sharper = (d_igs.get("std", 0) > d_aime.get("std", 0)
                       and d_igs.get("distinct", 0) >= d_aime.get("distinct", 0))
            if not agree:
                verdict = "A: rankings differ -> must prove IGS construct (E3/E5)"
            elif sharper:
                verdict = "B2: agree AND IGS sharper -> validated + future-proof (WIN)"
            else:
                verdict = "B1: agree but forward sharper -> IGS dies first (no durable value)"
        return {"n": len(pairs), "spearman_igs_aime": rho, "ci": [lo, hi],
                "disc_igs": d_igs, "disc_aime": d_aime, "verdict": verdict}

    def writeout(res):
        json.dump({"analysis": analyze(res), "models": sorted(res, key=lambda x: -(x["aime"] or 0))},
                  open(args.json_out, "w", encoding="utf-8"), indent=2)

    res = []
    ex = ThreadPoolExecutor(max_workers=min(len(models), args.max_workers))
    futs = {ex.submit(run_one, m): m for m in models}
    try:
        for fut in as_completed(futs, timeout=args.deadline):
            r = fut.result()
            if r is not None:
                res.append(r)
                writeout(res)
    except TimeoutError:
        print(f"[deadline] abandoned: {[m for f,m in futs.items() if not f.done()]}", flush=True)
    writeout(res)
    a = analyze(res)
    print(f"\nN={a['n']}  Spearman(IGS,AIME)={a['spearman_igs_aime']} CI={a['ci']}")
    print(f"disc IGS ={a['disc_igs']}")
    print(f"disc AIME={a['disc_aime']}")
    print(f"VERDICT: {a['verdict']}")
    ex.shutdown(wait=False)
    os._exit(0)


if __name__ == "__main__":
    main()
