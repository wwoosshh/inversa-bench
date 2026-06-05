"""E10-redux: recompute IGS using the HARD polynomial transform (E8) and re-run the discrimination
comparison vs AIME. Question: now that E8 resolved the IGS top, does hard-IGS match/exceed AIME's
top resolution while keeping rank-agreement? Pure analysis over existing results (no API).
"""
import json
import statistics

from inversa.analysis import spearman_ci
from inversa.scoring import igs as igs_fn

pose = {r["model"]: r["pose_validity"] for r in json.load(open("data/results/paper_level3_all.json"))}
easy_tx = {r["model"]: r["transform_validity"] for r in json.load(open("data/results/paper_level3_all.json"))}
hard_tx = {r["model"]: r["hard_transform_validity"]
           for r in json.load(open("data/results/transform_hard_results.json"))}
aime = {r["model"]: r["aime"] for r in json.load(open("data/results/e10_results.json"))["models"]}

models = sorted(set(pose) & set(hard_tx) & set(aime))
rows = []
for m in models:
    rows.append({"model": m, "aime": aime[m],
                 "easy_igs": igs_fn(pose[m], easy_tx[m]),
                 "hard_igs": igs_fn(pose[m], hard_tx[m])})


def disc(vals):
    mx = max(vals)
    return {"std": round(statistics.pstdev(vals), 3), "distinct": len(set(round(v, 3) for v in vals)),
            "tied_at_max": sum(1 for v in vals if abs(v - mx) < 1e-9)}


def corr(key):
    xs = [r[key] for r in rows]
    ys = [r["aime"] for r in rows]
    return spearman_ci(xs, ys)


print(f"N = {len(rows)} (pose ∩ hard-transform ∩ AIME)\n")
print(f"{'model':34s} {'AIME':>6} {'easyIGS':>8} {'hardIGS':>8}")
for r in sorted(rows, key=lambda x: -x["aime"]):
    print(f"{r['model']:34s} {r['aime']*100:5.0f}% {r['easy_igs']:8.2f} {r['hard_igs']:8.2f}")

for label, key in [("easy-IGS", "easy_igs"), ("hard-IGS", "hard_igs")]:
    rho, lo, hi = corr(key)
    print(f"\n{label}: Spearman vs AIME = {rho:+.2f} [{lo:+.2f},{hi:+.2f}]  disc={disc([r[key] for r in rows])}")
print(f"AIME    : disc={disc([r['aime'] for r in rows])}")
