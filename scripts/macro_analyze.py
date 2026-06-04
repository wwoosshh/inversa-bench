"""Macro check: does generative ability work as a performance indicator across a large,
diverse, capability-graded model set? Merges the forward/solve + adversarial experiment
(cli_experiment) with the level-3 construction probe (cli_structural) by model, then reports
cross-model Spearman correlations of SOLVE vs each generative axis, and among the generative
axes themselves (do they cohere into one 'generative' factor?).

Run after both experiments:
  python scripts/macro_analyze.py data/results/macro_experiment.json data/results/macro_level3.json
"""
import html
import json
import sys

from inversa.analysis import spearman

exp_path = sys.argv[1] if len(sys.argv) > 1 else "data/results/macro_experiment.json"
lvl_path = sys.argv[2] if len(sys.argv) > 2 else "data/results/macro_level3.json"

exp = {s["model"]: s for s in json.load(open(exp_path, encoding="utf-8"))["scores"]}
lvl = {r["model"]: r for r in json.load(open(lvl_path, encoding="utf-8"))}

rows = []
for m in exp:
    if m not in lvl:
        continue
    e, l = exp[m], lvl[m]
    rows.append({
        "model": m,
        "solve": e["solve_accuracy"],
        "calib_mace": e["calibration_mace"],
        "adv_validity": e["pose_validity_rate"],
        "adv_fool": e["pose_fool_rate"],
        "l3_pose": l["pose_validity"],
        "l3_transform": l["transform_validity"],
    })
rows.sort(key=lambda r: (r["solve"], r["l3_transform"]))


def _corr(ka, kb):
    pairs = [(r[ka], r[kb]) for r in rows if r[ka] is not None and r[kb] is not None]
    if len(pairs) < 3:
        return None
    return spearman([p[0] for p in pairs], [p[1] for p in pairs])


GEN = [("calib_mace", "calibration MACE (lower=better)"),
       ("adv_validity", "adversarial pose-validity"),
       ("l3_pose", "level-3 pose"),
       ("l3_transform", "level-3 transform")]

solve_corr = {k: _corr("solve", k) for k, _ in GEN}
# coherence: do generative axes correlate with each other?
coh = {
    "adv_validity ~ l3_pose": _corr("adv_validity", "l3_pose"),
    "adv_validity ~ l3_transform": _corr("adv_validity", "l3_transform"),
    "l3_pose ~ l3_transform": _corr("l3_pose", "l3_transform"),
}


def f(v):
    return "n/a" if v is None else f"{v:+.2f}"


def fp(v):
    return "n/a" if v is None else f"{v:.2f}"


print(f"\nN = {len(rows)} models (intersection of both experiments)\n")
hdr = f"{'model':40s} {'solve':>6} {'calMACE':>8} {'advVal':>7} {'advFool':>8} {'L3pose':>7} {'L3xform':>8}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['model']:40s} {r['solve']:6.2f} {fp(r['calib_mace']):>8} {r['adv_validity']:7.2f} "
          f"{fp(r['adv_fool']):>8} {r['l3_pose']:7.2f} {r['l3_transform']:8.2f}")

print("\n=== SOLVE vs generative (Spearman; near 0 / opposite sign => dissociates) ===")
for k, label in GEN:
    print(f"  solve vs {label:38s}: {f(solve_corr[k])}")
print("\n=== generative coherence (do the gen axes track each other?) ===")
for k, v in coh.items():
    print(f"  {k:32s}: {f(v)}")

# write a compact glass-box HTML
def esc(v):
    return html.escape(str(v))


trows = "".join(
    f"<tr><td>{esc(r['model'])}</td><td>{r['solve']:.2f}</td><td>{fp(r['calib_mace'])}</td>"
    f"<td>{r['adv_validity']:.2f}</td><td>{fp(r['adv_fool'])}</td>"
    f"<td>{r['l3_pose']:.2f}</td><td>{r['l3_transform']:.2f}</td></tr>"
    for r in rows)
crows = "".join(f"<tr><td>solve vs {esc(lab)}</td><td>{f(solve_corr[k])}</td></tr>" for k, lab in GEN)
crows += "".join(f"<tr><td>{esc(k)}</td><td>{f(v)}</td></tr>" for k, v in coh.items())
htmlout = f"""<!doctype html><meta charset="utf-8"><title>Inversa macro</title>
<style>body{{font-family:Segoe UI,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border-bottom:1px solid #ddd;padding:6px 9px;font-size:14px;text-align:left}}</style>
<h1>Inversa macro check (N={len(rows)})</h1>
<p>Forward solving vs generative proxies across a diverse, capability-graded model set.</p>
<table><tr><th>model</th><th>solve</th><th>calib MACE</th><th>adv validity</th><th>adv fool</th><th>L3 pose</th><th>L3 transform</th></tr>{trows}</table>
<h2>Cross-model correlations (Spearman)</h2><table><tr><th>pair</th><th>rho</th></tr>{crows}</table>
"""
open("data/results/macro_report.html", "w", encoding="utf-8").write(htmlout)
print("\nwrote data/results/macro_report.html")
