"""E8 runner: transform-validity on the HARD polynomial-transform bank, per model, ranked.
Goal: does higher generative difficulty resolve the IGS top (gemini-flash-lite / opus / qwen3.7
were all 1.0 on the easy bank)? Usage: python scripts/run_transform_hard.py [--models ...]
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from inversa.cli_experiment import _make_adapter
from inversa.tasks.structural import run_transform_item, validity_rate

DEFAULT = ("anthropic/claude-opus-4.8,google/gemini-3.1-flash-lite,qwen/qwen3.7-plus,"
           "anthropic/claude-haiku-4.5,deepseek/deepseek-v3.2,meta-llama/llama-4-scout,"
           "openai/gpt-4o-mini,google/gemini-3.5-flash,mistralai/mistral-large,"
           "anthropic/claude-sonnet-4.6,meta-llama/llama-3.3-70b-instruct,"
           "mistralai/mistral-small-2603,anthropic/claude-3.5-haiku,qwen/qwen-2.5-7b-instruct")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT)
    ap.add_argument("--bank", default="data/banks/transform_hard.json")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--deadline", type=float, default=1800.0)
    ap.add_argument("--json-out", default="data/results/transform_hard_results.json")
    args = ap.parse_args()

    items = json.load(open(args.bank, encoding="utf-8"))["items"]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    def run_one(m):
        try:
            ad = _make_adapter(m, "openrouter", "https://openrouter.ai/api/v1",
                               "OPENROUTER_API_KEY", args.max_tokens)
            out = [run_transform_item(ad, it["source"], it["g_desc"], it["r_value"], it["target_value"])
                   for it in items]
            v = validity_rate(out)
        except Exception as e:
            print(f"[skip] {m}: {type(e).__name__}: {e}", flush=True)
            return None
        print(f"[done] {m}: hard-transform validity={v:.0%}", flush=True)
        return {"model": m, "hard_transform_validity": v}

    def writeout(res):
        json.dump(sorted(res, key=lambda x: -x["hard_transform_validity"]),
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
    print("\n=== HARD transform validity (ranked) ===")
    for r in sorted(res, key=lambda x: -x["hard_transform_validity"]):
        print(f"  {r['hard_transform_validity']:.0%}  {r['model']}")
    ex.shutdown(wait=False)
    os._exit(0)


if __name__ == "__main__":
    main()
