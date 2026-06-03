"""CLI: run the dissociation experiment across models; write report.html + results.json."""
from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from inversa.adapters.anthropic_adapter import AnthropicAdapter
from inversa.experiment import correlations, evaluate_model
from inversa.report_html import render_html


def main(argv=None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Inversa dissociation experiment")
    parser.add_argument("--models",
                        default="claude-opus-4-8,claude-sonnet-4-6,claude-haiku-4-5-20251001",
                        help="comma-separated poser models")
    parser.add_argument("--weak-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--solve-bank", default="data/banks/solve_set_v1.json")
    parser.add_argument("--targets", default="3,7,12,42")
    parser.add_argument("--levels", default="1,2,3")
    parser.add_argument("--out", default="data/results/report.html")
    parser.add_argument("--json-out", default="data/results/results.json")
    args = parser.parse_args(argv)

    with open(args.solve_bank, encoding="utf-8") as f:
        solve_problems = json.load(f)["problems"]
    targets = [int(x) for x in args.targets.split(",")]
    levels = [int(x) for x in args.levels.split(",")]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    weak = AnthropicAdapter(model=args.weak_model)

    scores = []
    for m in models:
        poser = AnthropicAdapter(model=m)
        s = evaluate_model(m, poser, weak, solve_problems, targets, levels, targets)
        scores.append(s)
        print(f"[done] {m}: solve={s.solve_accuracy:.0%} "
              f"mace={s.calibration_mace} adv={s.adversarial_success_rate:.0%}")

    cors = correlations(scores)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(render_html(scores, cors))
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump({"scores": [s.__dict__ for s in scores], "correlations": cors}, f, indent=2)
    print(f"correlations: {cors}")
    print(f"wrote {args.out} and {args.json_out}")


if __name__ == "__main__":
    main()
