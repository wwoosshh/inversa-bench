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
import random
import sys
import threading

from dotenv import load_dotenv

from inversa.bench_core import ResultCache, evaluate_leaderboard, map_concurrent
from inversa.cli_experiment import OPENROUTER_BASE_URL, _make_adapter
from inversa.pose_random import random_pose_targets
from inversa.tasks.solving import accuracy, solve_batch
from inversa.tasks.structural import run_struct_pose_item, run_transform_item

BENCHMARK_NAME = "Inversa Generative Score (IGS) v1"


def _usage_dict(usage):
    """Serialize an adapter's UsageTotals into a JSON-friendly token-spend summary, so the
    engine can report real cost + truncations per model. None for adapters that don't track
    usage (e.g. Anthropic/fake). Mirrors the two diagnosed reasoning-model failure modes:
    `reasoning_tokens` exposes uncapped spend; `truncations` exposes score-corrupting cutoffs."""
    if usage is None:
        return None
    return {"calls": usage.calls, "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "reasoning_tokens": usage.reasoning_tokens, "truncations": usage.truncations}


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
    try:  # Windows consoles default to cp949/cp1252 and crash on non-ASCII (warning glyphs,
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # em dashes) at print time
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Inversa benchmark engine (IGS)")
    ap.add_argument("--models", required=True)
    ap.add_argument("--provider", default="openrouter",
                    choices=["anthropic", "openrouter", "openai_compatible"])
    ap.add_argument("--base-url", default=OPENROUTER_BASE_URL)
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=8000,
                    help="output token cap. Reasoning models need headroom: at 2048 many "
                         "providers truncate before the '#### eq' line (empty content -> false "
                         "zeros). Raised so honoring-providers finish; ignoring-providers are "
                         "bounded by --reasoning-max-tokens instead.")
    ap.add_argument("--reasoning-max-tokens", type=int, default=4000,
                    help="reasoning-token budget forwarded to OpenRouter. Bounds spend on "
                         "providers that ignore --max-tokens (e.g. deepseek-r1 burned 3-4x the "
                         "cap). Default 4000 caps cost; pass 0 to disable (uncapped).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature. Default 0 makes scoring near-deterministic so "
                         "fewer repeats are needed; pass <0 to send no temperature at all.")
    ap.add_argument("--max-workers", type=int, default=16,
                    help="global cap on in-flight (model,item) calls. Every call across all "
                         "models shares one pool, so a slow model holds only its own slot.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="MAX repeats per model. With adaptive repeats (default) only models "
                         "whose rank is still statistically ambiguous use the extra repeats.")
    ap.add_argument("--no-adaptive", action="store_true",
                    help="repeat every model --repeats times instead of only ambiguous ranks")
    ap.add_argument("--cache", default="data/results/igs_cache.json",
                    help="item-level result cache for resume; a re-run skips calls already made "
                         "(keyed by model/bank/repeat/item). Pass '' to disable.")
    ap.add_argument("--deadline", type=float, default=0.0,
                    help="optional wall-clock seconds before a forced exit; progress is in the "
                         "cache, so re-running with the same --cache resumes. 0 = no deadline.")
    ap.add_argument("--pose-bank", default="data/banks/struct_pose_targets.json")
    ap.add_argument("--pose-random", type=int, default=0,
                    help="generate N pose targets FRESH at test time instead of loading --pose-bank "
                         "(contamination-immune: no fixed item to leak). The seed + generated "
                         "targets are recorded in the output JSON so the run stays reproducible.")
    ap.add_argument("--pose-seed", type=int, default=None,
                    help="seed for --pose-random (default: a fresh OS-random seed each run).")
    ap.add_argument("--pose-no-trivial", action="store_true",
                    help="reject trivial pose answers: a linear restatement (x = target) does not "
                         "count, forcing genuine construction (degree>=2 or radicals/exp/log). "
                         "Anti-gaming; off by default for comparability with the fixed-bank runs.")
    ap.add_argument("--truncation-missing", action="store_true",
                    help="treat a truncated item (model ran out of token budget before emitting an "
                         "equation) as MISSING — excluded from the validity rate — instead of a "
                         "wrong answer. Avoids deflating reasoning models; off by default.")
    ap.add_argument("--transform-bank", default="data/banks/transform_bank.json")
    ap.add_argument("--solve-bank", default="", help="optional solve bank for a saturation-reference column")
    ap.add_argument("--out", default="data/results/igs_benchmark.html")
    ap.add_argument("--json-out", default="data/results/igs_benchmark.json")
    args = ap.parse_args(argv)

    if args.pose_random > 0:
        pose_seed = (args.pose_seed if args.pose_seed is not None
                     else random.SystemRandom().randrange(2 ** 31))
        pose_targets = random_pose_targets(args.pose_random, random.Random(pose_seed))
        pose_source = f"random(n={args.pose_random}, seed={pose_seed})"
        print(f"[pose] generated {args.pose_random} fresh targets (seed={pose_seed}, "
              f"contamination-immune)", flush=True)
    else:
        pose_seed = None
        pose_targets = json.load(open(args.pose_bank, encoding="utf-8"))["targets"]
        pose_source = args.pose_bank
    transform_items = json.load(open(args.transform_bank, encoding="utf-8"))["items"]
    solve_problems = (json.load(open(args.solve_bank, encoding="utf-8"))["problems"]
                      if args.solve_bank else [])
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    reasoning_cap = args.reasoning_max_tokens or None        # 0 -> uncapped
    temperature = args.temperature if args.temperature >= 0 else None  # <0 -> send nothing
    cache = ResultCache.load(args.cache) if args.cache else None

    # One adapter per model, lazily built and shared across that model's concurrently-scheduled
    # items (its usage tally is lock-guarded). A lock guards the dict so two item-threads racing
    # to first-touch the same model don't build two adapters.
    adapters: dict = {}
    adapters_lock = threading.Lock()

    def get_adapter(m):
        with adapters_lock:
            ad = adapters.get(m)
            if ad is None:
                ad = _make_adapter(m, args.provider, args.base_url, args.key_env, args.max_tokens,
                                   reasoning_max_tokens=reasoning_cap, temperature=temperature)
                adapters[m] = ad
            return ad

    # rep 0 is one big batch with no natural checkpoints, so count completed CALLS (cache misses
    # only — hits never reach score_item) to show progress and flush the cache mid-run for
    # durability on a long unattended run.
    progress = {"n": 0}
    prog_lock = threading.Lock()

    def score_item(model, bank, item, rep):
        ad = get_adapter(model)
        if bank == "pose":
            it = run_struct_pose_item(ad, item["target_desc"], item["target_value"],
                                      item.get("novelty", ""),
                                      require_nontrivial=args.pose_no_trivial)
        else:
            it = run_transform_item(ad, item["source"], item["g_desc"], item["r_value"],
                                    item["target_value"])
        with prog_lock:
            progress["n"] += 1
            n = progress["n"]
        if n % 50 == 0:
            print(f"[progress] {n} live calls completed", flush=True)
        if cache is not None and n % 250 == 0:
            cache.save()
        if args.truncation_missing and not it.answered:
            return None  # truncated, no equation -> missing (engine excludes it from the rate)
        return it.valid

    def on_rep_done(rep):
        if cache is not None:
            cache.save()
        print(f"[rep {rep}] complete (cache saved)" if cache else f"[rep {rep}] complete",
              flush=True)

    if args.deadline and args.deadline > 0:
        def _bail():
            print(f"[deadline] {args.deadline:.0f}s elapsed — forcing exit; re-run with the same "
                  f"--cache to resume from where this stopped.", flush=True)
            os._exit(2)
        t = threading.Timer(args.deadline, _bail)
        t.daemon = True
        t.start()

    results = evaluate_leaderboard(
        models, pose_targets, transform_items, score_item,
        repeats=max(1, args.repeats), max_workers=args.max_workers,
        cache=cache, adaptive=not args.no_adaptive, on_rep_done=on_rep_done,
        log=lambda m: print(m, flush=True))
    if cache is not None:
        cache.save()

    # Optional solve-reference column, one accuracy per model, concurrent across models.
    solve_ref: dict = {}
    if solve_problems:
        present = [r["model"] for r in results]
        accs = map_concurrent(lambda m: accuracy(solve_batch(get_adapter(m), solve_problems)),
                              present, args.max_workers)
        solve_ref = {m: a for m, a in zip(present, accs) if a is not None}

    # Attach real token spend per model (from the shared adapters) for the cost report.
    for r in results:
        r["usage"] = _usage_dict(getattr(adapters.get(r["model"]), "usage", None))

    ranked = sorted(results, key=lambda x: -x["igs"])
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
        r["solve_ref"] = solve_ref.get(r["model"])
    meta = {"benchmark": BENCHMARK_NAME, "n_models": len(ranked),
            "pose_source": pose_source, "pose_seed": pose_seed,
            "transform_bank": args.transform_bank, "pose_no_trivial": args.pose_no_trivial,
            "truncation_missing": args.truncation_missing,
            "repeats_max": max(1, args.repeats), "adaptive": not args.no_adaptive}
    if args.pose_random > 0:  # record the fresh targets so a contamination-immune run stays auditable
        meta["pose_targets"] = pose_targets
    json.dump({**meta, "results": ranked}, open(args.json_out, "w", encoding="utf-8"), indent=2)
    open(args.out, "w", encoding="utf-8").write(_render(ranked, solve_ref))

    print(f"\n=== {BENCHMARK_NAME} ===")
    for r in ranked:
        print(f"  {r['igs']:.2f}  {r['model']}  (reps={r['reps_done']})")
    usages = [r["usage"] for r in ranked if r.get("usage")]
    if usages:
        out_tok = sum(u["completion_tokens"] for u in usages)
        reason_tok = sum(u["reasoning_tokens"] for u in usages)
        trunc = [r["model"] for r in ranked if (r.get("usage") or {}).get("truncations")]
        print(f"token spend: {out_tok:,} completion ({reason_tok:,} reasoning) across "
              f"{len(usages)} models")
        if trunc:
            print(f"[WARN] truncated (scores unreliable - raise --max-tokens): {trunc}")
    print(f"wrote {args.out} and {args.json_out} ({len(ranked)} models)")
    sys.stdout.flush()  # os._exit skips buffer flushing — force the summary out first
    os._exit(0)


if __name__ == "__main__":
    main()
