"""E12b: inverse contamination gap = transform_validity(familiar/textbook sources) −
transform_validity(random sources). Predicted ≈ 0 (the transform answer is computed from the
input regardless of familiarity → no leakable fixed item). Pairs with run_forward_gap (E12) to
complete the head-to-head: forward gap > 0 vs inverse gap ≈ 0.
Usage: python scripts/run_inverse_gap.py [--models ...]
"""
import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from inversa.cli_experiment import _make_adapter
from inversa.tasks.structural import run_transform_item, validity_rate

DEFAULT = ("meta-llama/llama-3.1-8b-instruct,qwen/qwen-2.5-7b-instruct,openai/gpt-4o-mini,"
           "anthropic/claude-3.5-haiku,google/gemini-3.5-flash,meta-llama/llama-3.3-70b-instruct,"
           "deepseek/deepseek-v3.2,anthropic/claude-opus-4.8")


def _validity(adapter, items):
    out = [run_transform_item(adapter, it["source"], it["g_desc"], it["r_value"], it["target_value"])
           for it in items]
    return validity_rate(out)


def _mean_ci(vals, n=2000, seed=0):
    if len(vals) < 3:
        return (sum(vals) / len(vals) if vals else None, None, None)
    rng = random.Random(seed)
    b = sorted(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals) for _ in range(n))
    return (sum(vals) / len(vals), b[int(0.025 * n)], b[int(0.975 * n) - 1])


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT)
    ap.add_argument("--familiar", default="data/banks/transform_bank_familiar.json")
    ap.add_argument("--random", default="data/banks/transform_bank.json")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--deadline", type=float, default=1500.0)
    ap.add_argument("--json-out", default="data/results/inverse_gap_results.json")
    args = ap.parse_args()

    fam = json.load(open(args.familiar, encoding="utf-8"))["items"]
    rnd = json.load(open(args.random, encoding="utf-8"))["items"]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    def run_one(m):
        try:
            ad = _make_adapter(m, "openrouter", "https://openrouter.ai/api/v1",
                               "OPENROUTER_API_KEY", args.max_tokens)
            vf, vr = _validity(ad, fam), _validity(ad, rnd)
        except Exception as e:
            print(f"[skip] {m}: {type(e).__name__}: {e}", flush=True)
            return None
        gap = vf - vr
        print(f"[done] {m}: familiar={vf:.0%} random={vr:.0%} GAP={gap:+.0%}", flush=True)
        return {"model": m, "familiar": vf, "random": vr, "gap": gap}

    def writeout(res):
        gaps = [r["gap"] for r in res]
        mean, lo, hi = _mean_ci(gaps)
        json.dump({"summary": {"n_models": len(res), "mean_gap": mean,
                               "gap_ci_lo": lo, "gap_ci_hi": hi},
                   "models": sorted(res, key=lambda x: -x["gap"])},
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
    m, lo, hi = _mean_ci([r["gap"] for r in res])
    ci = "n/a" if lo is None else f"[{lo:+.1%}, {hi:+.1%}]"
    print(f"\ninverse mean gap = {m:+.1%} 95% CI {ci} (N={len(res)})")
    ex.shutdown(wait=False)
    os._exit(0)


if __name__ == "__main__":
    main()
