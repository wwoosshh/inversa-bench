"""Run a posing batch from a JSON target bank and print a report."""
from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from inversa.adapters.anthropic_adapter import AnthropicAdapter
from inversa.report import format_summary, summarize
from inversa.tasks.posing import run_batch


def main(argv=None) -> None:
    # Load ANTHROPIC_API_KEY from a .env in the cwd / project root if present.
    load_dotenv()
    parser = argparse.ArgumentParser(description="Inversa slice-1 math runner")
    parser.add_argument("--bank", required=True, help="path to target bank JSON")
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--verbose", action="store_true",
                        help="also print each item's raw model output")
    args = parser.parse_args(argv)

    with open(args.bank, encoding="utf-8") as f:
        data = json.load(f)
    targets = data.get("targets")
    if not isinstance(targets, list) or not targets:
        parser.error(f"bank {args.bank!r} must contain a non-empty 'targets' list")

    adapter = AnthropicAdapter(model=args.model)
    results = run_batch(adapter, targets)

    # Glass-box output (design §3.5): surface the full evidence chain per item.
    print(format_summary(summarize(results)))
    for r in results:
        v = r.verification
        print(f"  target={r.target} eq={r.equation!r} "
              f"solutions={v.solutions} valid={v.valid} unique={v.unique}")
        if args.verbose:
            print(f"    raw_output={r.raw_output!r}")


if __name__ == "__main__":
    main()
