"""The Inversa benchmark run, factored out of the CLI so the CLI and the GUI drive the SAME engine
path. `run_leaderboard_job(params, on_log, abort, adapter_factory)` builds adapters, scores every
(model, item) via `evaluate_leaderboard`, writes the JSON/HTML/leaderboard, and returns the ranked
results. Progress goes to `on_log` (the CLI prints it; the GUI streams it); `abort` (a threading
Event) lets a caller stop a run; `adapter_factory` is injectable so tests run with no network.
"""
from __future__ import annotations

import html
import json
import random
import threading
from typing import Any, Callable, Dict, List, Optional

from inversa.bench_core import ResultCache, evaluate_leaderboard, map_concurrent
from inversa.cli_experiment import OPENROUTER_BASE_URL, _make_adapter
from inversa.pose_random import random_pose_targets
from inversa.tasks.solving import accuracy, solve_batch
from inversa.tasks.structural import run_struct_pose_item, run_transform_item

BENCHMARK_NAME = "Inversa Generative Score (IGS) v1"

DEFAULTS: Dict[str, Any] = {
    "provider": "openrouter", "base_url": OPENROUTER_BASE_URL, "key_env": "OPENROUTER_API_KEY",
    "api_key": None, "max_tokens": 8000, "reasoning_max_tokens": 4000, "temperature": 0.0,
    "max_workers": 16, "repeats": 1, "adaptive": True,
    "pose_bank": "data/banks/struct_pose_targets.json", "pose_random": 0, "pose_seed": None,
    "pose_no_trivial": False, "transform_bank": "data/banks/transform_bank.json", "solve_bank": "",
    "truncation_missing": False, "cache": "data/results/igs_cache.json",
    "out": "data/results/igs_benchmark.html", "json_out": "data/results/igs_benchmark.json",
}


def _usage_dict(usage):
    """Serialize an adapter's UsageTotals to a JSON-friendly token-spend summary (None if the adapter
    doesn't track usage). `reasoning_tokens` exposes runaway reasoning; `truncations` exposes cutoffs."""
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


def render_html(results, solve_ref) -> str:
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
            f"<p>Ranked by <b>IGS = mean(pose, transform)</b>. Solve is a saturation reference only.</p>"
            f"<table><tr><th>#</th><th>model</th><th>IGS</th><th>pose</th><th>transform</th>"
            f"<th>solve (ref)</th></tr>{rows}</table>")


def run_leaderboard_job(params: Dict[str, Any], on_log: Optional[Callable[[str], None]] = None,
                        abort: Optional[threading.Event] = None,
                        adapter_factory: Optional[Callable[[str], Any]] = None) -> Dict[str, Any]:
    """Run the IGS benchmark for `params` (a dict mirroring the CLI flags). Returns
    {results, meta, out, json_out, token_total}. Pure of argparse/stdout so CLI + GUI share it."""
    p = {**DEFAULTS, **params}
    log = on_log or (lambda _m: None)
    models = p["models"]
    if isinstance(models, str):
        models = [m.strip() for m in models.split(",") if m.strip()]

    # --- banks -------------------------------------------------------------------------------
    if p["pose_random"] and p["pose_random"] > 0:
        seed = p["pose_seed"] if p["pose_seed"] is not None else random.SystemRandom().randrange(2 ** 31)
        pose_targets = random_pose_targets(p["pose_random"], random.Random(seed))
        pose_source, pose_seed = f"random(n={p['pose_random']}, seed={seed})", seed
        log(f"[pose] generated {p['pose_random']} fresh targets (seed={seed}, contamination-immune)")
    else:
        pose_seed = None
        pose_targets = json.load(open(p["pose_bank"], encoding="utf-8"))["targets"]
        pose_source = p["pose_bank"]
    transform_items = json.load(open(p["transform_bank"], encoding="utf-8"))["items"]
    solve_problems = (json.load(open(p["solve_bank"], encoding="utf-8"))["problems"]
                      if p["solve_bank"] else [])

    reasoning_cap = p["reasoning_max_tokens"] or None
    temperature = p["temperature"] if p["temperature"] >= 0 else None
    cache = ResultCache.load(p["cache"]) if p["cache"] else None

    # --- adapters (one per model, shared; factory injectable for tests/no-network) -----------
    if adapter_factory is None:
        def adapter_factory(m):  # noqa: E306
            return _make_adapter(m, p["provider"], p["base_url"], p["key_env"], p["max_tokens"],
                                 reasoning_max_tokens=reasoning_cap, temperature=temperature,
                                 api_key=p["api_key"])
    adapters: Dict[str, Any] = {}
    adapters_lock = threading.Lock()

    def get_adapter(m):
        with adapters_lock:
            ad = adapters.get(m)
            if ad is None:
                ad = adapter_factory(m)
                adapters[m] = ad
            return ad

    progress = {"n": 0}
    prog_lock = threading.Lock()

    def score_item(model, bank, item, rep):
        ad = get_adapter(model)
        if bank == "pose":
            it = run_struct_pose_item(ad, item["target_desc"], item["target_value"],
                                      item.get("novelty", ""), require_nontrivial=p["pose_no_trivial"])
        else:
            it = run_transform_item(ad, item["source"], item["g_desc"], item["r_value"],
                                    item["target_value"])
        with prog_lock:
            progress["n"] += 1
            n = progress["n"]
        if n % 50 == 0:
            log(f"[progress] {n} live calls completed")
        if cache is not None and n % 250 == 0:
            cache.save()
        if p["truncation_missing"] and not it.answered:
            return None
        return it.valid

    def on_rep_done(rep):
        if cache is not None:
            cache.save()
        log(f"[rep {rep}] complete")

    results = evaluate_leaderboard(
        models, pose_targets, transform_items, score_item,
        repeats=max(1, p["repeats"]), max_workers=p["max_workers"], cache=cache,
        adaptive=p["adaptive"], on_rep_done=on_rep_done, log=log, abort=abort)
    if cache is not None:
        cache.save()

    solve_ref: Dict[str, float] = {}
    if solve_problems and results:
        present = [r["model"] for r in results]
        accs = map_concurrent(lambda m: accuracy(solve_batch(get_adapter(m), solve_problems)),
                              present, p["max_workers"])
        solve_ref = {m: a for m, a in zip(present, accs) if a is not None}

    for r in results:
        r["usage"] = _usage_dict(getattr(adapters.get(r["model"]), "usage", None))
    ranked = sorted(results, key=lambda x: -x["igs"])
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
        r["solve_ref"] = solve_ref.get(r["model"])

    meta = {"benchmark": BENCHMARK_NAME, "n_models": len(ranked), "pose_source": pose_source,
            "pose_seed": pose_seed, "transform_bank": p["transform_bank"],
            "pose_no_trivial": p["pose_no_trivial"], "truncation_missing": p["truncation_missing"],
            "repeats_max": max(1, p["repeats"]), "adaptive": p["adaptive"]}
    if p["pose_random"] and p["pose_random"] > 0:
        meta["pose_targets"] = pose_targets
    if p["json_out"]:
        json.dump({**meta, "results": ranked}, open(p["json_out"], "w", encoding="utf-8"), indent=2)
    if p["out"]:
        open(p["out"], "w", encoding="utf-8").write(render_html(ranked, solve_ref))

    usages = [r["usage"] for r in ranked if r.get("usage")]
    token_total = {"completion": sum(u["completion_tokens"] for u in usages),
                   "reasoning": sum(u["reasoning_tokens"] for u in usages),
                   "truncated_models": [r["model"] for r in ranked
                                        if (r.get("usage") or {}).get("truncations")]}
    log(f"=== {BENCHMARK_NAME} ===")
    for r in ranked:
        log(f"  {r['igs']:.2f}  {r['model']}  (reps={r['reps_done']})")
    if usages:
        log(f"token spend: {token_total['completion']:,} completion "
            f"({token_total['reasoning']:,} reasoning) across {len(usages)} models")
        if token_total["truncated_models"]:
            log(f"[WARN] truncated (raise max-tokens or use truncation-missing): "
                f"{token_total['truncated_models']}")
    log(f"wrote {p['out']} and {p['json_out']} ({len(ranked)} models)")
    return {"results": ranked, "meta": meta, "out": p["out"], "json_out": p["json_out"],
            "token_total": token_total}
