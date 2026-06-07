"""IRT-scale an existing IGS run (rebuttal #6): turn the ordinal raw-mean IGS into an interval
(logit) ability scale via a Rasch fit over the model x item correctness matrix recovered from the
resume cache. Pools pose + transform items into ONE matrix (dropping the arbitrary 50/50 weight) and
reports per-item difficulty (b) and discrimination (point-biserial). No API calls.

Usage:
  python scripts/build_irt.py --results data/results/igs_leaderboard_30.json \
      --cache data/results/igs_cache.json --out data/results/igs_irt.json
"""
from __future__ import annotations

import argparse
import json

from inversa.analysis import spearman
from inversa.bench_core import ResultCache, cache_key
from inversa.irt import point_biserial, rasch_fit


def _load_items(bank_path, items_key):
    data = json.load(open(bank_path, encoding="utf-8"))
    return data[items_key]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="IRT (Rasch) scaling of an IGS run")
    ap.add_argument("--results", default="data/results/igs_leaderboard_30.json")
    ap.add_argument("--cache", default="data/results/igs_cache.json")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--out", default="data/results/igs_irt.json")
    args = ap.parse_args(argv)

    rj = json.load(open(args.results, encoding="utf-8"))
    meta = rj if isinstance(rj, dict) else {}
    results = rj["results"] if isinstance(rj, dict) else rj
    models = [r["model"] for r in results]
    raw_igs = {r["model"]: r["igs"] for r in results}

    pose_bank = meta.get("pose_bank") or "data/banks/struct_pose_targets30.json"
    transform_bank = meta.get("transform_bank") or "data/banks/transform_bank.json"
    if meta.get("pose_targets"):          # a --pose-random run records its fresh targets inline
        pose_items = meta["pose_targets"]
    else:
        pose_items = _load_items(pose_bank, "targets")
    transform_items = _load_items(transform_bank, "items")

    cache = ResultCache.load(args.cache)
    banks = [("pose", pose_items), ("transform", transform_items)]

    # Build the complete model x item matrix from the cache (drop models missing any item).
    labels, matrix = [], []
    for m in models:
        row, complete = [], True
        for bank, items in banks:
            for it in items:
                hit = cache.get(cache_key(m, bank, args.rep, it))
                if hit is None:
                    complete = False
                    break
                row.append(1 if hit["valid"] else 0)
            if not complete:
                break
        if complete:
            labels.append(m)
            matrix.append(row)

    theta, diff = rasch_fit(matrix)
    pb = point_biserial(matrix)

    ranked = sorted(
        [{"model": labels[j], "theta": round(theta[j], 3), "igs_raw": raw_igs.get(labels[j])}
         for j in range(len(labels))], key=lambda r: -r["theta"])
    for i, r in enumerate(ranked, 1):
        r["rank"] = i

    n_pose = len(pose_items)
    items_out = []
    for i in range(len(matrix[0]) if matrix else 0):
        bank = "pose" if i < n_pose else "transform"
        idx = i if i < n_pose else i - n_pose
        items_out.append({"bank": bank, "index": idx, "difficulty_b": round(diff[i], 3),
                          "discrimination_pbis": (round(pb[i], 3) if pb[i] is not None else None)})

    # sanity: theta is a *monotone rescaling* of raw IGS (interval, not a new ranking)
    rho = spearman([r["theta"] for r in ranked], [r["igs_raw"] for r in ranked])
    out = {"benchmark": "IGS (Rasch-scaled)", "n_models": len(labels),
           "n_items": len(matrix[0]) if matrix else 0, "n_pose": n_pose,
           "spearman_theta_vs_raw_igs": rho, "models": ranked, "items": items_out}
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"Rasch fit: {len(labels)} models x {out['n_items']} items "
          f"(theta vs raw-IGS Spearman = {rho:+.3f})")
    print("top 5 by theta:")
    for r in ranked[:5]:
        print(f"  theta={r['theta']:+.2f}  igs={r['igs_raw']:.2f}  {r['model']}")
    finite = [r for r in ranked if abs(r["theta"]) < 6]
    print(f"theta range (finite): {min(r['theta'] for r in finite):+.2f} .. "
          f"{max(r['theta'] for r in finite):+.2f}; {len(ranked)-len(finite)} at clamp boundary")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
