"""CLI: run the dissociation experiment across models; write report.html + results.json."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from inversa.adapters.anthropic_adapter import AnthropicAdapter
from inversa.adapters.openai_compatible import OpenAICompatibleAdapter
from inversa.experiment import correlations, evaluate_model_detailed
from inversa.report_html import render_html_detailed

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _make_adapter(model: str, provider: str, base_url: str, key_env: str, max_tokens: int = 512,
                  reasoning_max_tokens: int | None = None, temperature: float | None = None):
    """Build an adapter for `model`. provider=anthropic -> native Anthropic;
    anything else -> OpenAI-compatible endpoint (OpenRouter, Ollama, ...).
    `reasoning_max_tokens` bounds reasoning-model spend on OpenRouter (ignored by Anthropic);
    `temperature` (e.g. 0.0) cuts run-to-run noise so fewer repeats are needed."""
    if provider == "anthropic":
        return AnthropicAdapter(model=model)
    return OpenAICompatibleAdapter(model, base_url=base_url, api_key_env=key_env,
                                   max_tokens=max_tokens,
                                   reasoning_max_tokens=reasoning_max_tokens,
                                   temperature=temperature)


def main(argv=None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Inversa dissociation experiment")
    parser.add_argument("--models",
                        default="claude-opus-4-8,claude-sonnet-4-6,claude-haiku-4-5-20251001",
                        help="comma-separated poser models")
    parser.add_argument("--weak-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "openrouter", "openai_compatible"],
                        help="adapter provider for all models")
    parser.add_argument("--base-url", default=OPENROUTER_BASE_URL,
                        help="base URL for openrouter/openai_compatible providers")
    parser.add_argument("--key-env", default="OPENROUTER_API_KEY",
                        help="env var holding the API key for openrouter/openai_compatible")
    parser.add_argument("--solve-bank", default="data/banks/solve_set_v1.json")
    parser.add_argument("--targets", default="3,7,12,42")
    parser.add_argument("--levels", default="1,2,3")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="output token cap; raise so reasoning models reach the '#### <ans>' line")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="models evaluated concurrently (each is network-bound; wall-clock ~ slowest model)")
    parser.add_argument("--out", default="data/results/report.html")
    parser.add_argument("--json-out", default="data/results/results.json")
    parser.add_argument("--deadline", type=float, default=1200.0,
                        help="overall seconds before abandoning stuck/slow models (keeps partial results)")
    args = parser.parse_args(argv)

    with open(args.solve_bank, encoding="utf-8") as f:
        solve_problems = json.load(f)["problems"]
    targets = [int(x) for x in args.targets.split(",")]
    levels = [int(x) for x in args.levels.split(",")]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    def run_one(m):
        """Evaluate one poser end-to-end. Each model gets its OWN poser+weak adapters
        (independent OpenAI clients) so the calls are safe to run concurrently."""
        try:
            poser = _make_adapter(m, args.provider, args.base_url, args.key_env, args.max_tokens)
            weak = _make_adapter(args.weak_model, args.provider, args.base_url, args.key_env, args.max_tokens)
            run = evaluate_model_detailed(m, poser, weak, solve_problems, targets, levels, targets)
        except Exception as e:  # one unreachable/mis-slugged model must not abort the whole run
            print(f"[skip] {m}: {type(e).__name__}: {e}", flush=True)
            return None
        s = run.scores
        print(f"[done] {m}: solve={s.solve_accuracy:.0%} "
              f"mace={s.calibration_mace} adv={s.adversarial_success_rate:.0%}", flush=True)
        return run

    def writeout(runs):
        scores = [r.scores for r in runs]
        cors = correlations(scores)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(render_html_detailed(runs, cors))
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"scores": [s.__dict__ for s in scores], "correlations": cors}, f, indent=2)
        return cors

    # Incremental write per completed model + overall deadline + force-exit: a network drop or a
    # GIL-holding verify hang can no longer lose the models that already finished (this exact
    # failure cost a full multi-model run earlier).
    runs = []
    ex = ThreadPoolExecutor(max_workers=min(len(models), args.max_workers))
    futs = {ex.submit(run_one, m): m for m in models}
    try:
        for fut in as_completed(futs, timeout=args.deadline):
            r = fut.result()
            if r is not None:
                runs.append(r)
                writeout(runs)
    except TimeoutError:
        stuck = [m for f, m in futs.items() if not f.done()]
        print(f"[deadline] {len(stuck)} model(s) abandoned: {stuck}", flush=True)

    cors = writeout(runs)
    print(f"correlations: {cors}")
    print(f"wrote {args.out} and {args.json_out} ({len(runs)} models)")
    ex.shutdown(wait=False)
    os._exit(0)


if __name__ == "__main__":
    main()
