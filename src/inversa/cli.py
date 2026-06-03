"""Run a posing batch from a JSON target bank and print a report."""
from __future__ import annotations

import argparse
import json

from inversa.adapters.anthropic_adapter import AnthropicAdapter
from inversa.report import format_summary, summarize
from inversa.tasks.posing import run_batch


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Inversa slice-1 math runner")
    parser.add_argument("--bank", required=True, help="path to target bank JSON")
    parser.add_argument("--model", default="claude-opus-4-8")
    args = parser.parse_args(argv)

    with open(args.bank, encoding="utf-8") as f:
        targets = json.load(f)["targets"]

    adapter = AnthropicAdapter(model=args.model)
    results = run_batch(adapter, targets)

    print(format_summary(summarize(results)))
    for r in results:
        v = r.verification
        print(f"  target={r.target} eq={r.equation!r} "
              f"valid={v.valid} unique={v.unique}")


if __name__ == "__main__":
    main()
