"""Prune IGS banks down to their DISCRIMINATING items, using a finished run's resume cache.

An item that every model passed (or every model failed) costs a call but carries zero ranking
information — dropping it shrinks every subsequent leaderboard pass with no loss of separation.
This reads the per-item verdicts cli_bench already wrote to its --cache, so no new API calls.

Usage:
  python -m scripts.prune_bank --models "m1,m2,..." \
      --cache data/results/igs_cache.json \
      --pose-bank data/banks/struct_pose_targets.json \
      --transform-bank data/banks/transform_bank.json
Writes <bank>.pruned.json next to each input bank (override with --pose-out / --transform-out).
"""
from __future__ import annotations

import argparse
import json

from inversa.bench_core import (
    ResultCache,
    discriminating_item_indices,
    discrimination_matrix,
)


def _prune(bank_path: str, items_key: str, bank_name: str, models, cache: ResultCache,
           out_path: str) -> None:
    data = json.load(open(bank_path, encoding="utf-8"))
    items = data[items_key]
    matrix = discrimination_matrix(cache, models, bank_name, items)
    keep = discriminating_item_indices(matrix)
    data[items_key] = [items[i] for i in keep]
    data["pruned_from"] = len(items)
    data["pruned_using_models"] = len(matrix)
    json.dump(data, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    dead = len(items) - len(keep)
    print(f"{bank_name}: {len(items)} -> {len(keep)} items "
          f"({dead} non-discriminating dropped, over {len(matrix)} complete models) -> {out_path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Prune IGS banks to discriminating items")
    ap.add_argument("--models", required=True, help="comma-separated roster used in the run")
    ap.add_argument("--cache", default="data/results/igs_cache.json")
    ap.add_argument("--pose-bank", default="data/banks/struct_pose_targets.json")
    ap.add_argument("--transform-bank", default="data/banks/transform_bank.json")
    ap.add_argument("--pose-out", default="")
    ap.add_argument("--transform-out", default="")
    args = ap.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cache = ResultCache.load(args.cache)
    pose_out = args.pose_out or args.pose_bank.replace(".json", ".pruned.json")
    trans_out = args.transform_out or args.transform_bank.replace(".json", ".pruned.json")

    _prune(args.pose_bank, "targets", "pose", models, cache, pose_out)
    _prune(args.transform_bank, "items", "transform", models, cache, trans_out)


if __name__ == "__main__":
    main()
