"""CLI: run a difficulty-calibration batch from a target bank and print a report."""
from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from inversa.adapters.anthropic_adapter import AnthropicAdapter
from inversa.report import calibration_summary, format_calibration_summary
from inversa.tasks.calibration import run_calibration_batch


def main(argv=None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Inversa difficulty-calibration runner")
    parser.add_argument("--bank", required=True, help="path to target bank JSON")
    parser.add_argument("--levels", default="1,2,3,4",
                        help="comma-separated requested polynomial degrees")
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--verbose", action="store_true",
                        help="also print each item's raw model output")
    args = parser.parse_args(argv)

    with open(args.bank, encoding="utf-8") as f:
        data = json.load(f)
    targets = data.get("targets")
    if not isinstance(targets, list) or not targets:
        parser.error(f"bank {args.bank!r} must contain a non-empty 'targets' list")
    try:
        levels = [int(x) for x in args.levels.split(",")]
    except ValueError:
        parser.error("--levels must be comma-separated integers, e.g. 1,2,3,4")

    adapter = AnthropicAdapter(model=args.model)
    items = run_calibration_batch(adapter, targets, levels)

    # Glass-box output (design §3.5): per-item requested vs actual degree + verdict.
    print(format_calibration_summary(calibration_summary(items)))
    for it in items:
        d = it.difficulty
        print(f"  target={it.target} req_deg={it.requested_level} eq={it.equation!r} "
              f"actual_deg={d.degree} band={d.band} valid={it.valid} "
              f"cal_err={it.calibration_error}")
        if args.verbose:
            print(f"    raw_output={it.raw_output!r}")


if __name__ == "__main__":
    main()
