"""Generate paper figures from the REAL result JSONs. Output: docs/paper/figures/*.png
Run: python scripts/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/paper/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 130, "savefig.bbox": "tight"})


def short(m):
    return m.split("/")[-1]


def load(f):
    return json.load(open(f"data/results/{f}.json", encoding="utf-8"))


# ---- Fig 1: forward contamination gap (E12) ----------------------------------
d = load("forward_gap_results")
ms = sorted(d["models"], key=lambda r: -r["gap"])
names = [short(r["model"]) for r in ms]
y = range(len(ms))
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.barh(y, [r["acc_contaminated"] * 100 for r in ms], height=0.4, label="GSM8K (contaminated)", color="#b91c1c")
ax.barh([i + 0.42 for i in y], [r["acc_fresh"] * 100 for r in ms], height=0.4, label="GSM-Symbolic (fresh)", color="#0d9488")
ax.set_yticks([i + 0.21 for i in y]); ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("accuracy (%)"); ax.invert_yaxis()
ax.set_title(f"Forward contamination gap (E12): mean +{d['summary']['mean_gap']*100:.1f}%  "
             f"95% CI [{d['summary']['gap_ci_lo']*100:+.1f}, {d['summary']['gap_ci_hi']*100:+.1f}]")
ax.legend(fontsize=8, loc="lower right")
fig.savefig(f"{OUT}/fig_forward_gap.png"); plt.close(fig)

# ---- Fig 2: IGS vs AIME scatter (E10) ----------------------------------------
e = load("e10_results")
pts = [(r["igs"], r["aime"], short(r["model"])) for r in e["models"] if r["igs"] is not None]
rho = e["analysis"]["spearman_igs_aime"]
fig, ax = plt.subplots(figsize=(6.4, 5.4))
ax.scatter([p[0] for p in pts], [p[1] * 100 for p in pts], color="#2563eb", zorder=3)
for x, yy, n in pts:
    ax.annotate(n, (x, yy * 100), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("IGS (Inversa Generative Score)"); ax.set_ylabel("AIME accuracy (%)")
ax.set_title(f"E10: IGS vs a hard forward benchmark (AIME)\nSpearman = +{rho:.2f} [{e['analysis']['ci'][0]:+.2f}, {e['analysis']['ci'][1]:+.2f}]  (N={e['analysis']['n']})")
ax.grid(alpha=0.3)
fig.savefig(f"{OUT}/fig_igs_vs_aime.png"); plt.close(fig)

# ---- Fig 3: difficulty escalation (E8): easy -> hard -> brutal ----------------
easy = {r["model"]: r.get("transform_validity") for r in load("paper_level3_all")}
hard = {r["model"]: r["hard_transform_validity"] for r in load("transform_hard_results")}
brutal = {r["model"]: r["hard_transform_validity"] for r in load("transform_brutal_results")}
common = [m for m in hard if m in easy and m in brutal]
common.sort(key=lambda m: -brutal[m])
fig, ax = plt.subplots(figsize=(7.6, 4.6))
xs = [0, 1, 2]
for m in common:
    ys = [easy[m] * 100, hard[m] * 100, brutal[m] * 100]
    ax.plot(xs, ys, marker="o", label=short(m), linewidth=1.6)
ax.set_xticks(xs); ax.set_xticklabels(["easy\n(Mobius)", "hard\n(r^2, r^3)", "brutal\n(r^4, composite)"])
ax.set_ylabel("transform validity (%)")
ax.set_title("E8: raising construction difficulty resolves the IGS top\n(all 100% on easy -> spread 12-88% on brutal; no model at 100%)")
ax.legend(fontsize=6.5, ncol=2, loc="lower left"); ax.grid(alpha=0.3); ax.set_ylim(-3, 105)
fig.savefig(f"{OUT}/fig_escalation.png"); plt.close(fig)

# ---- Fig 4: IGS leaderboard (engine-aggregated, finer 30-item banks) ---------
lv = load("igs_leaderboard_30")["results"]
trunc = set(load("igs_dashboard_summary")["truncated_models"])
rows = sorted(([short(r["model"]), r["igs"], r["model"] in trunc] for r in lv), key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(7.6, max(6.0, 0.16 * len(rows))))
# truncated models (low-confidence) drawn in amber so the figure self-flags them
ax.barh(range(len(rows)), [r[1] for r in rows],
        color=["#d97706" if r[2] else "#0d9488" for r in rows])
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] + (" *" if r[2] else "") for r in rows], fontsize=6)
ax.set_xlabel("IGS = mean(pose, transform)")
ax.set_title(f"IGS leaderboard (N={len(rows)}, 30-item banks, temp 0; * = truncated/low-confidence)")
ax.set_xlim(0, 1.0); ax.margins(y=0.005)
fig.savefig(f"{OUT}/fig_leaderboard.png"); plt.close(fig)

# ---- Fig 5: pipeline schematic ----------------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 2.4)); ax.axis("off")
boxes = [("TARGET\n(an answer,\ne.g. 'unique\nroot = 3')", "#dbeafe"),
         ("PROMPT\nthe model to\nCONSTRUCT a\nproblem", "#dbeafe"),
         ("MODEL\noutputs an\nequation\n'#### x**3-27=0'", "#fef9c3"),
         ("EXTRACT\nparse the\nequation\n(regex+clean)", "#e5e7eb"),
         ("VERIFY (sympy)\nsolve & check\nUNIQUE real\nroot == target", "#dcfce7"),
         ("SCORE\nvalid? -> IGS\n(machine,\nno LLM judge)", "#dcfce7")]
n = len(boxes); w = 1.0 / n
for i, (txt, c) in enumerate(boxes):
    ax.add_patch(plt.Rectangle((i * w + 0.005, 0.15), w - 0.03, 0.7, facecolor=c, edgecolor="#555"))
    ax.text(i * w + w / 2, 0.5, txt, ha="center", va="center", fontsize=7.5)
    if i < n - 1:
        ax.annotate("", xy=(i * w + w - 0.01, 0.5), xytext=(i * w + w - 0.03, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="#333"))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Inversa scoring pipeline: answer -> constructed problem -> machine verification", fontsize=10)
fig.savefig(f"{OUT}/fig_pipeline.png"); plt.close(fig)

print("wrote figures to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
