"""Build the Inversa Generative Score (IGS) leaderboard from existing results — the formal,
standalone generative indicator. Demonstrates the value proposition: in the regime where the
SOLVE axis has saturated (many models at 100%, top unmeasurable), IGS still ranks models.

Usage: python scripts/build_leaderboard.py <level3.json> <experiment.json>
"""
import html
import json
import sys

from inversa.scoring import score_from_rates

lvl_path = sys.argv[1] if len(sys.argv) > 1 else "data/results/macro3_level3.json"
exp_path = sys.argv[2] if len(sys.argv) > 2 else "data/results/macro3_experiment.json"

lvl = {r["model"]: r for r in json.load(open(lvl_path, encoding="utf-8"))}
solve = {s["model"]: s["solve_accuracy"]
         for s in json.load(open(exp_path, encoding="utf-8"))["scores"]}

rows = []
for m, r in lvl.items():
    gs = score_from_rates(m, r["pose_validity"], r["transform_validity"])
    rows.append({"model": m, "igs": gs.igs, "pose": gs.pose_validity,
                 "transform": gs.transform_validity, "solve": solve.get(m)})
rows.sort(key=lambda x: x["igs"], reverse=True)

# The headline statistic: among models that MAX the solve axis, how widely does IGS spread?
solved = [x for x in rows if x["solve"] is not None]
maxsolve = max((x["solve"] for x in solved), default=None)
ceil = [x for x in solved if x["solve"] == maxsolve]
ceil_igs = [x["igs"] for x in ceil]
callout = ""
if len(ceil) >= 2:
    callout = (f"{len(ceil)} models share the TOP solve score ({maxsolve:.0%}) - solve cannot "
               f"rank them. Their IGS still spreads {min(ceil_igs):.2f}-{max(ceil_igs):.2f}.")

print(f"\n{'#':>2}  {'model':40s} {'IGS':>5}  {'pose':>5} {'xform':>6}  {'solve(ref)':>10}")
print("-" * 74)
for i, x in enumerate(rows, 1):
    sv = "n/a" if x["solve"] is None else f"{x['solve']:.0%}"
    print(f"{i:>2}  {x['model']:40s} {x['igs']:5.2f}  {x['pose']:5.2f} {x['transform']:6.2f}  {sv:>10}")
print("\n" + callout + "\n")


def esc(v):
    return html.escape(str(v))


def bar(p, color="#0d9488"):
    w = max(0, min(100, p * 100))
    return (f'<div style="background:#eee;border-radius:3px;width:120px;display:inline-block;vertical-align:middle">'
            f'<div style="background:{color};width:{w:.0f}%;height:13px;border-radius:3px"></div></div> {p:.2f}')


trows = "".join(
    f"<tr><td>{i}</td><td><b>{esc(x['model'])}</b></td><td>{bar(x['igs'])}</td>"
    f"<td style='text-align:center'>{x['pose']:.2f}</td><td style='text-align:center'>{x['transform']:.2f}</td>"
    f"<td style='text-align:center;color:#888'>{'n/a' if x['solve'] is None else format(x['solve'],'.0%')}</td></tr>"
    for i, x in enumerate(rows, 1))

out = f"""<!doctype html><meta charset="utf-8"><title>Inversa Generative Score</title>
<style>body{{font-family:Segoe UI,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1a1a1a;line-height:1.5}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border-bottom:1px solid #ddd;padding:7px 10px;font-size:14px;text-align:left}}
th{{font-size:12px;color:#555;text-transform:uppercase}}.box{{background:#f0fdfa;border:1px solid #99f6e4;border-radius:6px;padding:12px 16px;margin:1rem 0}}
.cap{{color:#666;font-size:13px}}</style>
<h1>Inversa Generative Score (IGS)</h1>
<p>A standalone, machine-verified indicator of a model's ability to <b>construct</b> mathematical
structure — recall-proof by design. <b>IGS = mean(pose validity, transform validity)</b>. Solve
accuracy is shown only as a <i>saturation reference</i>; it is not part of IGS.</p>
<div class="box"><b>Why it exists:</b> {esc(callout)}</div>
<table><tr><th>#</th><th>model</th><th>IGS</th><th>pose</th><th>transform</th><th>solve (ref)</th></tr>{trows}</table>
<p class="cap">pose = construct an equation whose unique real solution is a given irrational/structural target
(naive templates break uniqueness). transform = structurally re-root a freshly-randomized equation to g(r)
(output pinned to a test-time input → memorization cannot help). All judgments sympy-verified.</p>
"""
open("data/results/igs_leaderboard.html", "w", encoding="utf-8").write(out)
json.dump(rows, open("data/results/igs_leaderboard.json", "w", encoding="utf-8"), indent=2)
print("wrote data/results/igs_leaderboard.html and .json")
