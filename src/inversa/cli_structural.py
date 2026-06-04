"""CLI: level-3 generative probes across models -> glass-box HTML + JSON.

Form A (struct-pose): validity of posing a UNIQUE-real-root equation for targets along a
novelty ladder (integer -> irrational). A recall-driven model collapses on un-memorable rungs.
Form B (transform): validity of structurally transforming a random ugly-root equation so its
unique real root is g(r). Recall is impossible here (output is pinned to a fresh input).
"""
from __future__ import annotations

import argparse
import html
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from inversa.cli_experiment import OPENROUTER_BASE_URL, _make_adapter
from inversa.tasks.structural import (
    run_struct_pose_item,
    run_transform_item,
    validity_rate,
)


def _esc(v) -> str:
    return html.escape(str(v))


def evaluate(model, adapter, pose_targets, transform_items):
    pose = [run_struct_pose_item(adapter, t["target_desc"], t["target_value"], t["novelty"])
            for t in pose_targets]
    trans = [run_transform_item(adapter, it["source"], it["g_desc"], it["r_value"], it["target_value"])
             for it in transform_items]
    by_nov = OrderedDict((p.novelty, p.valid) for p in pose)  # one item per rung
    return {
        "model": model,
        "pose_validity": validity_rate(pose),
        "pose_by_novelty": by_nov,
        "transform_validity": validity_rate(trans),
        "pose_items": pose,
        "transform_items": trans,
    }


def _render(results, novelty_order):
    def cell(ok):
        return ('<td style="text-align:center;color:#16a34a">OK</td>' if ok
                else '<td style="text-align:center;color:#b91c1c">X</td>')

    head = "".join(f"<th>{_esc(n)}</th>" for n in novelty_order)
    curve_rows = []
    for r in results:
        cells = "".join(cell(r["pose_by_novelty"].get(n, False)) for n in novelty_order)
        curve_rows.append(
            f"<tr><td><b>{_esc(r['model'])}</b></td>{cells}"
            f"<td style='text-align:center'>{r['pose_validity']:.0%}</td>"
            f"<td style='text-align:center'>{r['transform_validity']:.0%}</td></tr>")

    evidence = []
    for r in results:
        pose_rows = "".join(
            f"<tr><td>{_esc(it.novelty)}</td><td>{_esc(it.target_desc)}</td>"
            f"<td><code>{_esc(it.equation)}</code></td><td>{'OK' if it.valid else 'X'}</td></tr>"
            for it in r["pose_items"])
        trans_rows = "".join(
            f"<tr><td><code>{_esc(it.source)}</code></td><td>{_esc(it.g_desc)}</td>"
            f"<td>{it.target_value:.4f}</td><td><code>{_esc(it.equation)}</code></td>"
            f"<td>{'OK' if it.valid else 'X'}</td></tr>"
            for it in r["transform_items"])
        evidence.append(
            f"<details><summary><b>{_esc(r['model'])}</b> - per-item evidence</summary>"
            f"<h4>Form A - structural-target posing (unique real root?)</h4>"
            f"<table><tr><th>novelty</th><th>target</th><th>posed equation</th><th>valid</th></tr>{pose_rows}</table>"
            f"<h4>Form B - structural transform (unique real root = g(r)?)</h4>"
            f"<table><tr><th>source</th><th>g(r)</th><th>target</th><th>constructed equation</th><th>valid</th></tr>{trans_rows}</table>"
            f"</details>")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Inversa - level-3 (construction vs recall)</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2rem auto;color:#1a1a1a;line-height:1.5;padding:0 1rem}}
table{{border-collapse:collapse;width:100%;margin:.6rem 0}}
th,td{{border-bottom:1px solid #ddd;padding:6px 9px;text-align:left;font-size:14px}}
th{{font-size:12px;color:#555;text-transform:uppercase;letter-spacing:.03em}}
code{{background:#eef;padding:1px 5px;border-radius:3px;font-size:12px}}
details{{margin:.5rem 0;border:1px solid #e5e5e5;border-radius:6px;padding:.4rem .8rem}}
summary{{cursor:pointer}} h4{{margin:.8rem 0 .2rem;font-size:13px;color:#444}}
.box{{background:#f7f7f8;border:1px solid #e5e5e5;border-radius:6px;padding:12px 16px;margin:1rem 0;font-size:14px}}
.caveat{{color:#666;font-size:13px}}</style></head><body>
<h1>Inversa level-3: does it construct structure, or recall templates?</h1>
<p>Every task pins the answer to test-time information so a memorized problem cannot satisfy it.
<b>Form A</b>: pose an equation whose <b>unique</b> real solution is the given target, along a
novelty ladder. The naive template (a minimal polynomial) carries conjugate roots and fails
uniqueness, so the right side of the ladder demands real construction. <b>Form B</b>: given a
random ugly-root cubic, transform its structure so the unique real root becomes g(r) - recall
is impossible because the output depends entirely on a freshly-generated input.</p>
<div class="box"><b>Validity along the novelty ladder</b> (OK = unique-real-root constraint met). A model
that <i>recalls</i> stays OK on the left and turns X on the right; a model that <i>constructs</i> holds across.</div>
<table><tr><th>Model</th>{head}<th>Pose&nbsp;valid</th><th>Transform&nbsp;valid</th></tr>
{''.join(curve_rows)}</table>
<h2>Per-item evidence (glass-box)</h2>{''.join(evidence)}
<p class="caveat">All judgments are sympy forward-verified. Form B items are sanity-checked at
bank-build time to have a unique-real-root canonical answer, so a failure is the model's, not the task's.</p>
</body></html>"""


def main(argv=None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Inversa level-3 construction-vs-recall probe")
    ap.add_argument("--models", required=True, help="comma-separated models")
    ap.add_argument("--provider", default="openrouter",
                    choices=["anthropic", "openrouter", "openai_compatible"])
    ap.add_argument("--base-url", default=OPENROUTER_BASE_URL)
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--pose-bank", default="data/banks/struct_pose_targets.json")
    ap.add_argument("--transform-bank", default="data/banks/transform_bank.json")
    ap.add_argument("--out", default="data/results/level3_report.html")
    ap.add_argument("--json-out", default="data/results/level3_results.json")
    args = ap.parse_args(argv)

    pose_targets = json.load(open(args.pose_bank, encoding="utf-8"))["targets"]
    transform_items = json.load(open(args.transform_bank, encoding="utf-8"))["items"]
    novelty_order = [t["novelty"] for t in pose_targets]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    def run_one(m):
        try:
            ad = _make_adapter(m, args.provider, args.base_url, args.key_env, args.max_tokens)
            res = evaluate(m, ad, pose_targets, transform_items)
        except Exception as e:
            print(f"[skip] {m}: {type(e).__name__}: {e}", flush=True)
            return None
        print(f"[done] {m}: pose_valid={res['pose_validity']:.0%} "
              f"transform_valid={res['transform_validity']:.0%}", flush=True)
        return res

    with ThreadPoolExecutor(max_workers=min(len(models), args.max_workers)) as ex:
        results = [r for r in ex.map(run_one, models) if r is not None]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(_render(results, novelty_order))
    slim = [{k: v for k, v in r.items() if k not in ("pose_items", "transform_items")}
            for r in results]
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2)
    print(f"wrote {args.out} and {args.json_out}")


if __name__ == "__main__":
    main()
