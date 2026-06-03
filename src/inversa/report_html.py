"""Self-contained HTML report (inline CSS — no external deps). Glass-box: per-model
scores + dissociation verdict + honest caveats, and (detailed mode) the full per-item
evidence so a human can inspect every judgment. All model-generated text is HTML-escaped."""
from __future__ import annotations

import html
from typing import List

from inversa.experiment import ModelRun, ModelScores


def _esc(v) -> str:
    return html.escape(str(v))


def _bar(pct: float, color: str) -> str:
    w = max(0.0, min(100.0, pct * 100.0))
    return (f'<div style="background:#eee;border-radius:3px;width:140px;display:inline-block;'
            f'vertical-align:middle">'
            f'<div style="background:{color};width:{w:.0f}%;height:14px;border-radius:3px"></div></div>'
            f' <span>{pct:.0%}</span>')


def _fmt_corr(v) -> str:
    return "n/a" if v is None else f"{v:+.2f}"


def _summary_table(scores: List[ModelScores]) -> str:
    rows = []
    for s in scores:
        mace = "n/a" if s.calibration_mace is None else f"{s.calibration_mace:.2f}"
        rows.append(
            "<tr>"
            f"<td><b>{_esc(s.model)}</b></td>"
            f"<td>{_bar(s.solve_accuracy, '#2563eb')}</td>"
            f"<td style='text-align:center'>{mace}</td>"
            f"<td>{_bar(s.adversarial_success_rate, '#16a34a')}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _page(scores: List[ModelScores], corr: dict, extra_body: str = "") -> str:
    table = _summary_table(scores)
    c_mace = _fmt_corr(corr.get("solve_vs_calibration_mace"))
    c_adv = _fmt_corr(corr.get("solve_vs_adversarial"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Inversa - dissociation report</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:2rem auto;color:#1a1a1a;line-height:1.5;padding:0 1rem}}
table{{border-collapse:collapse;width:100%;margin:.6rem 0}}
th,td{{border-bottom:1px solid #ddd;padding:6px 9px;text-align:left;font-size:14px}}
th{{font-size:12px;color:#555;text-transform:uppercase;letter-spacing:.03em}}
.box{{background:#f7f7f8;border:1px solid #e5e5e5;border-radius:6px;padding:12px 16px;margin:1rem 0}}
.caveat{{color:#666;font-size:14px}}
code{{background:#eef;padding:1px 5px;border-radius:3px;font-size:13px}}
details{{margin:.5rem 0;border:1px solid #e5e5e5;border-radius:6px;padding:.4rem .8rem}}
summary{{cursor:pointer;font-size:15px}}
h4{{margin:.8rem 0 .2rem;font-size:13px;color:#444}}
</style></head><body>
<h1>Inversa - generative vs solving (dissociation)</h1>
<p>Each model is scored on <b>solving</b> (accuracy on a fixed equation set) and two
<b>generative</b> proxies: <b>calibration</b> (mean abs error hitting a requested difficulty degree - <i>lower is better</i>)
and <b>adversarial success</b> (rate of posing valid equations a weaker model fails). Question:
do the generative scores <b>diverge</b> from solving?</p>
<table>
<tr><th>Model</th><th>Solve accuracy</th><th>Calibration MACE &darr;</th><th>Adversarial success</th></tr>
{table}
</table>
<div class="box">
<b>Dissociation (Spearman across models):</b><br>
solve vs calibration-MACE: <code>{c_mace}</code> (near 0 / positive &rArr; diverges; negative &rArr; tracks solving)<br>
solve vs adversarial: <code>{c_adv}</code> (near 0 / negative &rArr; diverges; +1 &rArr; tracks solving)
</div>
{extra_body}
<p class="caveat"><b>Honest caveats:</b> few models &rArr; correlations are <i>suggestive, not conclusive</i> (and undefined when a column has no variance).
Difficulty = polynomial degree (a v1 proxy). When the weak solver is also one of the tested poser models (typically the weakest),
that model's adversarial score is measured against <i>itself</i>, not an external baseline - interpret with care.
All scores are machine-verified (sympy), but whether they measure "understanding" beyond solving is the hypothesis under test. Domain: math equations only.</p>
</body></html>"""


def render_html(scores: List[ModelScores], corr: dict) -> str:
    return _page(scores, corr)


def _evidence_sections(runs: List[ModelRun]) -> str:
    out = []
    for r in runs:
        solve_rows = "".join(
            f"<tr><td><code>{_esc(it.equation)}</code></td><td>{_esc(it.expected)}</td>"
            f"<td>{_esc(it.answer)}</td><td>{'OK' if it.correct else 'X'}</td></tr>"
            for it in r.solve_items)
        cal_rows = "".join(
            f"<tr><td>{_esc(it.requested_level)}</td><td><code>{_esc(it.equation)}</code></td>"
            f"<td>{_esc(it.difficulty.degree)} (band {_esc(it.difficulty.band)})</td>"
            f"<td>{_esc(it.calibration_error)}</td><td>{'OK' if it.valid else 'X'}</td></tr>"
            for it in r.calibration_items)
        adv_rows = "".join(
            f"<tr><td><code>{_esc(it.equation)}</code></td><td>{'OK' if it.valid else 'X'}</td>"
            f"<td>{_esc(it.weak_answer)}</td><td>{'OK' if it.weak_correct else 'X'}</td>"
            f"<td>{'FOOLED' if it.adversarial_success else '-'}</td></tr>"
            for it in r.adversarial_items)
        out.append(
            f"<details><summary><b>{_esc(r.model)}</b> - per-item evidence</summary>"
            f"<h4>Solve</h4><table><tr><th>equation</th><th>expected</th><th>model answer</th><th>correct</th></tr>{solve_rows}</table>"
            f"<h4>Calibration (requested degree &rarr; actual)</h4><table><tr><th>req deg</th><th>posed equation</th><th>actual deg (band)</th><th>cal err</th><th>valid</th></tr>{cal_rows}</table>"
            f"<h4>Adversarial (did it fool the weak model?)</h4><table><tr><th>posed equation</th><th>valid</th><th>weak answer</th><th>weak correct</th><th>fooled?</th></tr>{adv_rows}</table>"
            f"</details>"
        )
    return "<h2>Per-item evidence (glass-box)</h2>" + "\n".join(out)


def render_html_detailed(runs: List[ModelRun], corr: dict) -> str:
    return _page([r.scores for r in runs], corr, extra_body=_evidence_sections(runs))
