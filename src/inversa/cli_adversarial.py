"""CLI: run a model-vs-model adversarial posing batch and print a report."""
from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from inversa.adapters.anthropic_adapter import AnthropicAdapter
from inversa.report import adversarial_summary, format_adversarial_summary
from inversa.tasks.adversarial import run_adversarial_batch


def main(argv=None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Inversa adversarial (model-vs-model) runner")
    parser.add_argument("--bank", required=True, help="path to target bank JSON")
    parser.add_argument("--poser-model", default="claude-opus-4-8")
    parser.add_argument("--weak-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--verbose", action="store_true",
                        help="also print each item's raw poser/weak output")
    args = parser.parse_args(argv)

    with open(args.bank, encoding="utf-8") as f:
        data = json.load(f)
    targets = data.get("targets")
    if not isinstance(targets, list) or not targets:
        parser.error(f"bank {args.bank!r} must contain a non-empty 'targets' list")

    poser = AnthropicAdapter(model=args.poser_model)
    weak = AnthropicAdapter(model=args.weak_model)
    items = run_adversarial_batch(poser, weak, targets)

    # Glass-box output (design §3.5): per-item equation, validity, weak answer, verdict.
    print(format_adversarial_summary(adversarial_summary(items)))
    for it in items:
        print(f"  target={it.target} eq={it.equation!r} valid={it.valid} "
              f"weak_answer={it.weak_answer} weak_correct={it.weak_correct} "
              f"adversarial_success={it.adversarial_success}")
        if args.verbose:
            print(f"    poser_raw={it.raw_poser_output!r}")
            print(f"    weak_raw={it.weak_raw_output!r}")


if __name__ == "__main__":
    main()
