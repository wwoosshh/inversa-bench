"""Inversa benchmark engine — the canonical entry point that turns a list of models into a
clear, ranked **Inversa Generative Score (IGS)**.

IGS = mean(pose_validity, transform_validity): a machine-verified, recall-resistant score of a
model's ability to *construct* mathematical structure (answer→problem), designed to discriminate
where forward solving saturates and to be immune to benchmark contamination (transform inputs are
randomized at test time → no fixed item can leak; cf. forward inflation measured in run_forward_gap).

Usage:
  python -m inversa.cli_bench --models "modelA,modelB,..." [--solve-bank ... for a saturation ref]
Outputs: data/results/igs_benchmark.{json,html} and a ranked console table.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from inversa.cli_experiment import OPENROUTER_BASE_URL, _make_adapter
from inversa.cli_structural import evaluate
from inversa.scoring import generative_score
from inversa.tasks.solving import accuracy, solve_batch

BENCHMARK_NAME = "Inversa Generative Score (IGS) v1"


def _bar(p, color="#0d9488"):
    w = max(0.0, min(100.0, p * 100.0))
    return (f'<div style="background:#eee;border-radius:3px;width:120px;display:inline-block;'
            f'vertical-align:middle"><div style="background:{color};width:{w:.0f}%;height:13px;'
            f'border-radius:3px"></div></div> <b>{p:.2f}</b>')


def _render(results, solve_ref):
    esc = html.escape
    rows = ""
    for i, r in enumerate(sorted(results, key=lambda x: -x["igs"]), 1):
        sv = solve_ref.get(r["model"])
        svc = "—" if sv is None else f"{sv:.0%}"
        rows += (f"<tr><td>{i}</td><td><b>{esc(r['model'])}</b></td><td>{_bar(r['igs'])}</td>"
                 f"<td style='text-align:center'>{r['pose_validity']:.2f}</td>"
                 f"<td style='text-align:center'>{r['transform_validity']:.2f}</td>"
                 f"<td style='text-align:center;color:#888'>{svc}</td></tr>")
    return (f"<!doctype html><meta charset='utf-8'><title>{esc(BENCHMARK_NAME)}</title>"
            f"<style>body{{font-family:Segoe UI,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}}"
            f"table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #ddd;padding:7px 10px;font-size:14px}}"
            f"th{{font-size:12px;color:#555;text-transform:uppercase}}</style>"
            f"<h1>{esc(BENCHMARK_NAME)}</h1>"
            f"<p>Ranked by <b>IGS = mean(pose, transform)</b> — recall-resistant construction ability. "
            f"Solve is a saturation reference only (not part of IGS).</p>"
            f"<table><tr><th>#</th><th>model</th><th>IGS</th><th>pose</th><th>transform</th>"
            f"<th>solve (ref)</th></tr>{rows}</table>")


def main(argv=None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Inversa benchmark engine (IGS)")
    ap.add_argument("--models", required=True)
    ap.add_argument("--provider", default="openrouter",
                    choices=["anthropic", "openrouter", "openai_compatible"])
    ap.add_argument("--base-url", default=OPENROUTER_BASE_URL)
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--deadline", type=float, default=1800.0)
    ap.add_argument("--pose-bank", default="data/banks/struct_pose_targets.json")
    ap.add_argument("--transform-bank", default="data/banks/transform_bank.json")
    ap.add_argument("--solve-bank", default="", help="optional solve bank for a saturation-reference column")
    ap.add_argument("--out", default="data/results/igs_benchmark.html")
    ap.add_argument("--json-out", default="data/results/igs_benchmark.json")
    args = ap.parse_args(argv)

    pose_targets = json.load(open(args.pose_bank, encoding="utf-8"))["targets"]
    transform_items = json.load(open(args.transform_bank, encoding="utf-8"))["items"]
    solve_problems = (json.load(open(args.solve_bank, encoding="utf-8"))["problems"]
                      if args.solve_bank else [])
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    solve_ref: dict = {}

    def run_one(m):
        try:
            ad = _make_adapter(m, args.provider, args.base_url, args.key_env, args.max_tokens)
            res = evaluate(m, ad, pose_targets, transform_items)
            gs = generative_score(m, res["pose_items"], res["transform_items"])
            if solve_problems:
                solve_ref[m] = accuracy(solve_batch(ad, solve_problems))
        except Exception as e:
            print(f"[skip] {m}: {type(e).__name__}: {e}", flush=True)
            return None
        print(f"[done] {m}: IGS={gs.igs:.2f} (pose={gs.pose_validity:.2f} transform={gs.transform_validity:.2f})",
              flush=True)
        return {"model": m, "igs": gs.igs, "pose_validity": gs.pose_validity,
                "transform_validity": gs.transform_validity}

    def writeout(results):
        ranked = sorted(results, key=lambda x: -x["igs"])
        for i, r in enumerate(ranked, 1):
            r["rank"] = i
            r["solve_ref"] = solve_ref.get(r["model"])
        meta = {"benchmark": BENCHMARK_NAME, "n_models": len(ranked),
                "pose_bank": args.pose_bank, "transform_bank": args.transform_bank}
        json.dump({**meta, "results": ranked}, open(args.json_out, "w", encoding="utf-8"), indent=2)
        open(args.out, "w", encoding="utf-8").write(_render(results, solve_ref))

    results = []
    ex = ThreadPoolExecutor(max_workers=min(len(models), args.max_workers))
    futs = {ex.submit(run_one, m): m for m in models}
    try:
        for fut in as_completed(futs, timeout=args.deadline):
            r = fut.result()
            if r is not None:
                results.append(r)
                writeout(results)
    except TimeoutError:
        print(f"[deadline] abandoned: {[m for f,m in futs.items() if not f.done()]}", flush=True)
    writeout(results)
    print(f"\n=== {BENCHMARK_NAME} ===")
    for r in sorted(results, key=lambda x: -x["igs"]):
        print(f"  {r['igs']:.2f}  {r['model']}")
    print(f"wrote {args.out} and {args.json_out} ({len(results)} models)")
    ex.shutdown(wait=False)
    os._exit(0)


if __name__ == "__main__":
    main()
