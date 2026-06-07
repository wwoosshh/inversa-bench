"""Build the IGS leaderboard dashboard (HTML + summary JSON) from a (possibly partial) run.

Reads the engine's leaderboard JSON and the attempted roster, so coverage is reported honestly
(measured vs attempted, by family, IGS spread, token spend, unmeasured list). Safe to run on a
credit-aborted partial run — it synthesizes whatever was measured.

Usage:
  python scripts/build_dashboard.py \
      --results data/results/igs_leaderboard_30.json \
      --roster  data/banks/leaderboard_roster.json \
      --out     data/results/igs_dashboard.html \
      --summary-out data/results/igs_dashboard_summary.json
"""
from __future__ import annotations

import argparse
import json

from inversa.dashboard import render_html, summarize_leaderboard


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build the IGS leaderboard dashboard")
    ap.add_argument("--results", default="data/results/igs_leaderboard_30.json")
    ap.add_argument("--roster", default="data/banks/leaderboard_roster.json")
    ap.add_argument("--out", default="data/results/igs_dashboard.html")
    ap.add_argument("--summary-out", default="data/results/igs_dashboard_summary.json")
    args = ap.parse_args(argv)

    data = json.load(open(args.results, encoding="utf-8"))
    results = data["results"] if isinstance(data, dict) and "results" in data else data
    roster = json.load(open(args.roster, encoding="utf-8"))
    if isinstance(roster, dict):  # tolerate {"models": [...]} or a bare list
        roster = roster.get("models") or roster.get("roster") or []

    summary = summarize_leaderboard(results, roster)
    meta = data if isinstance(data, dict) else {}
    htmlpage = render_html(results, summary,
                           pose_bank=meta.get("pose_bank", ""),
                           transform_bank=meta.get("transform_bank", ""))
    open(args.out, "w", encoding="utf-8").write(htmlpage)
    json.dump(summary, open(args.summary_out, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print(f"measured {summary['n_measured']}/{summary['n_attempted']} models "
          f"({summary['n_families']} families), IGS {summary['igs_min']} to {summary['igs_max']}")
    print(f"wrote {args.out} and {args.summary_out}")


if __name__ == "__main__":
    main()
