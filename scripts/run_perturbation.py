"""Run the isomorphic-perturbation fragility test (Experiment F, cheap signal).
Outputs a models x templates accuracy matrix. Flat near-1.0 within a template = robust solving;
fragmented accuracy across equi-difficult fresh variants = non-robust (shortcut/memorization).
Usage: python scripts/run_perturbation.py [--models ...]
"""
import argparse
import html
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from inversa.cli_experiment import _make_adapter
from inversa.tasks.solving import solve_item

DEFAULT = ("meta-llama/llama-3.1-8b-instruct,qwen/qwen-2.5-7b-instruct,openai/gpt-4o-mini,"
           "anthropic/claude-haiku-4.5,google/gemini-3.5-flash,mistralai/mistral-large,"
           "deepseek/deepseek-v3.2,anthropic/claude-opus-4.8")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT)
    ap.add_argument("--bank", default="data/banks/perturbation_set.json")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--deadline", type=float, default=900.0)
    ap.add_argument("--out", default="data/results/perturbation_report.html")
    ap.add_argument("--json-out", default="data/results/perturbation_results.json")
    args = ap.parse_args()

    problems = json.load(open(args.bank, encoding="utf-8"))["problems"]
    templates = sorted({p["template"] for p in problems})
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    def run_one(m):
        try:
            ad = _make_adapter(m, "openrouter", "https://openrouter.ai/api/v1",
                               "OPENROUTER_API_KEY", args.max_tokens)
            by_t = defaultdict(list)
            for p in problems:
                it = solve_item(ad, p["equation"], p["answer"])
                by_t[p["template"]].append(bool(it.correct))
        except Exception as e:
            print(f"[skip] {m}: {type(e).__name__}: {e}", flush=True)
            return None
        accs = {t: sum(by_t[t]) / len(by_t[t]) for t in by_t}
        overall = sum(sum(by_t[t]) for t in by_t) / sum(len(by_t[t]) for t in by_t)
        print(f"[done] {m}: overall={overall:.0%} " +
              " ".join(f"{t.split('_')[0]}={accs.get(t,0):.0%}" for t in templates), flush=True)
        return {"model": m, "by_template": accs, "overall": overall}

    def writeout(res):
        json.dump(res, open(args.json_out, "w", encoding="utf-8"), indent=2)
        esc = html.escape
        hdr = "".join(f"<th>{esc(t)}</th>" for t in templates)
        rows = ""
        for r in sorted(res, key=lambda x: x["overall"]):
            cells = ""
            for t in templates:
                a = r["by_template"].get(t)
                col = "#16a34a" if a == 1.0 else ("#b91c1c" if (a or 0) < 0.5 else "#ca8a04")
                cells += f"<td style='text-align:center;color:{col}'>{'-' if a is None else format(a,'.0%')}</td>"
            rows += f"<tr><td>{esc(r['model'])}</td>{cells}<td style='text-align:center'><b>{r['overall']:.0%}</b></td></tr>"
        # column means
        means = ""
        for t in templates:
            vals = [r["by_template"][t] for r in res if t in r["by_template"]]
            means += f"<td style='text-align:center'><i>{(sum(vals)/len(vals)):.0%}</i></td>" if vals else "<td>-</td>"
        out = (f"<!doctype html><meta charset='utf-8'><title>Perturbation fragility</title>"
               f"<style>body{{font-family:Segoe UI,sans-serif;max-width:900px;margin:2rem auto}}"
               f"table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #ddd;padding:6px 9px;font-size:14px}}</style>"
               f"<h1>Isomorphic-perturbation fragility (Experiment F)</h1>"
               f"<p>Accuracy per model x template over 6 structurally-identical, constant-perturbed, "
               f"sympy-verified instances. Flat 100% within a template = robust reasoning; "
               f"&lt;100% / scattered on equi-difficult fresh variants = non-robust (shortcut/memorization).</p>"
               f"<table><tr><th>model</th>{hdr}<th>overall</th></tr>{rows}"
               f"<tr><td><i>mean</i></td>{means}<td></td></tr></table>")
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
