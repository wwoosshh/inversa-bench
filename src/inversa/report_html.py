"""Self-contained HTML report (inline CSS, percentage bars — no external deps).
Glass-box: shows every per-model number plus the dissociation verdict and caveats."""
from __future__ import annotations

from typing import List

from inversa.experiment import ModelScores


def _bar(pct: float, color: str) -> str:
    w = max(0.0, min(100.0, pct * 100.0))
    return (f'<div style="background:#eee;border-radius:3px;width:140px;display:inline-block;'
            f'vertical-align:middle">'
            f'<div style="background:{color};width:{w:.0f}%;height:14px;border-radius:3px"></div></div>'
            f' <span>{pct:.0%}</span>')


def _fmt_corr(v) -> str:
    return "n/a" if v is None else f"{v:+.2f}"


def render_html(scores: List[ModelScores], corr: dict) -> str:
    rows = []
    for s in scores:
        mace = "n/a" if s.calibration_mace is None else f"{s.calibration_mace:.2f}"
        rows.append(
            "<tr>"
            f"<td><b>{s.model}</b></td>"
            f"<td>{_bar(s.solve_accuracy, '#2563eb')}</td>"
            f"<td style='text-align:center'>{mace}</td>"
            f"<td>{_bar(s.adversarial_success_rate, '#16a34a')}</td>"
            "</tr>"
        )
    table = "\n".join(rows)
    c_mace = _fmt_corr(corr.get("solve_vs_calibration_mace"))
    c_adv = _fmt_corr(corr.get("solve_vs_adversarial"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Inversa — dissociation report</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:820px;margin:2rem auto;color:#1a1a1a;line-height:1.5}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border-bottom:1px solid #ddd;padding:8px 10px;text-align:left}}
th{{font-size:13px;color:#555;text-transform:uppercase;letter-spacing:.03em}}
.box{{background:#f7f7f8;border:1px solid #e5e5e5;border-radius:6px;padding:12px 16px;margin:1rem 0}}
.caveat{{color:#666;font-size:14px}}
code{{background:#eee;padding:1px 4px;border-radius:3px}}
</style></head><body>
<h1>Inversa — generative vs solving (dissociation)</h1>
<p>Each model is scored on <b>solving</b> (accuracy on a fixed equation set) and on two
<b>generative</b> proxies: <b>calibration</b> (mean abs error hitting a requested difficulty degree — <i>lower is better</i>)
and <b>adversarial success</b> (rate of posing valid equations a weaker model fails). The question:
do the generative scores <b>diverge</b> from solving?</p>
<table>
<tr><th>Model</th><th>Solve accuracy</th><th>Calibration MACE &darr;</th><th>Adversarial success</th></tr>
{table}
</table>
<div class="box">
<b>Dissociation (Spearman across models):</b><br>
solve vs calibration-MACE: <code>{c_mace}</code> (near 0 / positive &rArr; generative diverges from solving)<br>
solve vs adversarial: <code>{c_adv}</code> (near 0 / negative &rArr; diverges; +1 &rArr; tracks solving)
</div>
<p class="caveat"><b>Honest caveats:</b> few models &rArr; correlations are <i>suggestive, not conclusive</i>.
Difficulty = polynomial degree (a v1 proxy). All scores are machine-verified (sympy), but whether
they measure "understanding" beyond solving is exactly the hypothesis under test. Domain: math equations only.</p>
</body></html>"""
