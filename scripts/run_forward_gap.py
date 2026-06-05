"""E12: measure forward score inflation = acc(contaminated GSM8K) - acc(fresh GSM-Symbolic),
matched by template (same difficulty). A positive gap = forward scores are inflated by having
seen the exact items (contamination) + surface-fragility. Per-model gaps + bootstrap CI.
Usage: python scripts/run_forward_gap.py [--models ...]
"""
import argparse
import html
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from inversa.cli_experiment import _make_adapter
from inversa.tasks.adversarial import parse_numeric_answer

DEFAULT = ("meta-llama/llama-3.1-8b-instruct,qwen/qwen-2.5-7b-instruct,cohere/command-r-08-2024,"
           "mistralai/mistral-small-2603,anthropic/claude-3.5-haiku,openai/gpt-4o-mini,"
           "meta-llama/llama-3.3-70b-instruct,mistralai/mistral-large,google/gemini-3.5-flash,"
           "deepseek/deepseek-v3.2,qwen/qwen3.7-plus,anthropic/claude-opus-4.8")

PROMPT = ("Solve this grade-school math word problem. Reason briefly, then end with a line "
          "in EXACTLY this form:\n#### <final integer answer>\n\nProblem:\n{q}")
_TOL = 1e-6


def _acc(adapter, items, qkey, akey):
    ok = 0
    for it in items:
        ans = parse_numeric_answer(adapter.generate(PROMPT.format(q=it[qkey])))
        if ans is not None and abs(ans - float(it[akey])) < _TOL:
            ok += 1
    return ok / len(items)


def _mean_ci(vals, n=2000, seed=0):
    if len(vals) < 3:
        return (sum(vals) / len(vals) if vals else None, None, None)
    rng = random.Random(seed)
    boots = []
    for _ in range(n):
        samp = [vals[rng.randrange(len(vals))] for _ in vals]
        boots.append(sum(samp) / len(samp))
    boots.sort()
    return (sum(vals) / len(vals), boots[int(0.025 * n)], boots[int(0.975 * n) - 1])


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT)
    ap.add_argument("--bank", default="data/banks/forward_gap_set.json")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--deadline", type=float, default=1800.0)
    ap.add_argument("--out", default="data/results/forward_gap_report.html")
    ap.add_argument("--json-out", default="data/results/forward_gap_results.json")
    args = ap.parse_args()

    pairs = json.load(open(args.bank, encoding="utf-8"))["pairs"]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    def run_one(m):
        try:
            ad = _make_adapter(m, "openrouter", "https://openrouter.ai/api/v1",
                               "OPENROUTER_API_KEY", args.max_tokens)
            acc_c = _acc(ad, pairs, "contaminated_q", "contaminated_ans")
            acc_f = _acc(ad, pairs, "fresh_q", "fresh_ans")
        except Exception as e:
            print(f"[skip] {m}: {type(e).__name__}: {e}", flush=True)
            return None
        gap = acc_c - acc_f
        print(f"[done] {m}: GSM8K(contam)={acc_c:.0%} symbolic(fresh)={acc_f:.0%} GAP={gap:+.0%}", flush=True)
        return {"model": m, "acc_contaminated": acc_c, "acc_fresh": acc_f, "gap": gap}

    def writeout(res):
        gaps = [r["gap"] for r in res]
        mean, lo, hi = _mean_ci(gaps)
        n_pos = sum(1 for g in gaps if g > 0)
        summary = {"n_models": len(res), "mean_gap": mean, "gap_ci_lo": lo, "gap_ci_hi": hi,
                   "n_positive_gap": n_pos}
        json.dump({"summary": summary, "models": sorted(res, key=lambda x: -x["gap"])},
                  open(args.json_out, "w", encoding="utf-8"), indent=2)
        esc = html.escape
        rows = "".join(
            f"<tr><td>{esc(r['model'])}</td><td style='text-align:center'>{r['acc_contaminated']:.0%}</td>"
            f"<td style='text-align:center'>{r['acc_fresh']:.0%}</td>"
            f"<td style='text-align:center;color:{'#b91c1c' if r['gap']>0 else '#16a34a'}'><b>{r['gap']:+.0%}</b></td></tr>"
            for r in sorted(res, key=lambda x: -x["gap"]))
        ci = "n/a" if lo is None else f"[{lo:+.1%}, {hi:+.1%}]"
        out = (f"<!doctype html><meta charset='utf-8'><title>Forward contamination gap (E12)</title>"
               f"<style>body{{font-family:Segoe UI,sans-serif;max-width:860px;margin:2rem auto}}"
               f"table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #ddd;padding:6px 9px;font-size:14px}}"
               f".box{{background:#fef2f2;border:1px solid #fecaca;border-radius:6px;padding:12px 16px;margin:1rem 0}}</style>"
               f"<h1>Forward score inflation — GSM8K vs GSM-Symbolic (E12)</h1>"
               f"<p>Matched by template (same difficulty). <b>gap = acc(GSM8K original, in training) "
               f"− acc(GSM-Symbolic fresh)</b>. Positive gap = forward score inflated by contamination "
               f"+ surface-fragility. {len(pairs)} matched problems/model.</p>"
               f"<div class='box'><b>Mean gap = {mean:+.1%}</b>, 95% CI {ci}; "
               f"{n_pos}/{len(res)} models show a positive gap.</div>"
               f"<table><tr><th>model</th><th>GSM8K (contaminated)</th><th>GSM-Symbolic (fresh)</th>"
               f"<th>gap</th></tr>{rows}</table>")
        open(args.out, "w", encoding="utf-8").write(out)

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
    print(f"wrote {args.out} ({len(res)} models)")
    ex.shutdown(wait=False)
    os._exit(0)


if __name__ == "__main__":
    main()
