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
import os
import sys
import threading

from dotenv import load_dotenv

from inversa.bench_run import BENCHMARK_NAME, _usage_dict, run_leaderboard_job  # noqa: F401
from inversa.cli_experiment import OPENROUTER_BASE_URL


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

    # Map argparse flags -> the params dict the shared job understands (cli_bench is now a thin
    # wrapper around bench_run.run_leaderboard_job, which the GUI also drives).
    params = {
        "models": args.models, "provider": args.provider, "base_url": args.base_url,
        "key_env": args.key_env, "max_tokens": args.max_tokens,
        "reasoning_max_tokens": args.reasoning_max_tokens, "temperature": args.temperature,
        "max_workers": args.max_workers, "repeats": args.repeats, "adaptive": not args.no_adaptive,
        "pose_bank": args.pose_bank, "pose_random": args.pose_random, "pose_seed": args.pose_seed,
        "pose_no_trivial": args.pose_no_trivial, "transform_bank": args.transform_bank,
        "solve_bank": args.solve_bank, "truncation_missing": args.truncation_missing,
        "cache": args.cache, "out": args.out, "json_out": args.json_out,
    }

    # Optional wall-clock watchdog: progress is in the cache, so a forced exit is resumable.
    if args.deadline and args.deadline > 0:
        def _bail():
            print(f"[deadline] {args.deadline:.0f}s elapsed — forcing exit; re-run with the same "
                  f"--cache to resume.", flush=True)
            os._exit(2)
        t = threading.Timer(args.deadline, _bail)
        t.daemon = True
        t.start()

    run_leaderboard_job(params, on_log=lambda m: print(m, flush=True))
    sys.stdout.flush()  # os._exit skips buffer flushing — force the summary out first
    os._exit(0)


if __name__ == "__main__":
    main()
