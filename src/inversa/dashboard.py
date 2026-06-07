"""Synthesize the IGS leaderboard into a single self-contained dashboard (HTML + summary dict),
honest about coverage on a PARTIAL run. `summarize_leaderboard` is pure (tested); `render_html`
turns the summary + rows into a standalone page suitable as paper/README evidence.

Why a dedicated dashboard (vs the engine's raw table): a paper-grade artifact has to show not
just the ranking but the *accounting* — measured vs attempted, by family, IGS spread, token
spend, and which models went unmeasured (and roughly why) — so the leaderboard cannot be read as
hiding the gaps.
"""
from __future__ import annotations

import html
from typing import Any, Dict, List, Optional, Sequence


def _family(model: str) -> str:
    return model.split("/")[0] if "/" in model else model


def summarize_leaderboard(results: Sequence[Dict[str, Any]],
                          roster: Sequence[str]) -> Dict[str, Any]:
    """Aggregate a (possibly partial) leaderboard: coverage vs the attempted roster, per-family
    bests, IGS spread, token spend, truncations, and the unmeasured list. `results` are the
    measured rows (each has model/igs/pose_validity/transform_validity/reps_done/usage)."""
    measured = {r["model"] for r in results}
    unmeasured = [m for m in roster if m not in measured]

    fam_rows: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        fam_rows.setdefault(_family(r["model"]), []).append(r)
    families = []
    for fam, rows in fam_rows.items():
        best = max(rows, key=lambda r: r["igs"])
        families.append({
            "family": fam, "n": len(rows),
            "best_igs": best["igs"], "best_model": best["model"],
            "mean_igs": round(sum(r["igs"] for r in rows) / len(rows), 4),
        })
    families.sort(key=lambda f: -f["best_igs"])

    igss = [r["igs"] for r in results]
    usages = [r.get("usage") for r in results if r.get("usage")]
    truncated = [r["model"] for r in results if (r.get("usage") or {}).get("truncations")]

    return {
        "n_attempted": len(roster),
        "n_measured": len(results),
        "n_unmeasured": len(unmeasured),
        "unmeasured": unmeasured,
        "families": families,
        "n_families": len(families),
        "igs_min": min(igss) if igss else None,
        "igs_max": max(igss) if igss else None,
        "total_completion_tokens": sum(u.get("completion_tokens", 0) for u in usages),
        "total_reasoning_tokens": sum(u.get("reasoning_tokens", 0) for u in usages),
        "total_prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usages),
        "truncated_models": truncated,
    }


def _bar(p: float, color: str = "#0d9488") -> str:
    w = max(0.0, min(100.0, p * 100.0))
    return (f'<div class="bar"><div style="background:{color};width:{w:.0f}%"></div></div>'
            f'<b>{p:.2f}</b>')


def render_html(results: Sequence[Dict[str, Any]], summary: Dict[str, Any], *,
                title: str = "Inversa Generative Score — Leaderboard Dashboard",
                pose_bank: str = "", transform_bank: str = "") -> str:
    """A standalone, dependency-free HTML dashboard: headline coverage, full ranked table with
    pose/transform breakdown, per-family bests, and an explicit unmeasured-models accounting."""
    esc = html.escape
    ranked = sorted(results, key=lambda r: -r["igs"])

    rows = ""
    for i, r in enumerate(ranked, 1):
        u = r.get("usage") or {}
        trunc = " ⚠" if u.get("truncations") else ""
        rows += (f"<tr><td>{i}</td><td><b>{esc(r['model'])}</b>{trunc}</td>"
                 f"<td>{_bar(r['igs'])}</td>"
                 f"<td class=c>{r['pose_validity']:.2f}</td>"
                 f"<td class=c>{r['transform_validity']:.2f}</td>"
                 f"<td class=c style='color:#888'>{r.get('reps_done', 1)}</td></tr>")

    fam_rows = "".join(
        f"<tr><td><b>{esc(f['family'])}</b></td><td class=c>{f['n']}</td>"
        f"<td class=c>{f['best_igs']:.2f}</td><td>{esc(f['best_model'])}</td>"
        f"<td class=c>{f['mean_igs']:.2f}</td></tr>"
        for f in summary["families"])

    unmeasured = ", ".join(esc(m) for m in summary["unmeasured"]) or "—"
    rng = ("n/a" if summary["igs_min"] is None
           else f"{summary['igs_min']:.2f} – {summary['igs_max']:.2f}")
    reason_tok = summary["total_reasoning_tokens"]
    tok = f"{summary['total_completion_tokens']:,} completion ({reason_tok:,} reasoning)"
    banks = (f"pose: <code>{esc(pose_bank)}</code> · transform: <code>{esc(transform_bank)}</code>"
             if pose_bank or transform_bank else "")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:980px;margin:2rem auto;
padding:0 1rem;color:#1a1a1a;line-height:1.5}}
h1{{margin:.2rem 0}} h2{{margin:1.6rem 0 .4rem;font-size:16px}}
.sub{{color:#666;font-size:13px;margin:.2rem 0 1rem}}
.cards{{display:flex;flex-wrap:wrap;gap:12px;margin:1rem 0}}
.card{{background:#f0fdfa;border:1px solid #99f6e4;border-radius:8px;padding:12px 16px;min-width:130px}}
.card .v{{font-size:24px;font-weight:700;color:#0f766e}} .card .l{{font-size:12px;color:#555}}
table{{border-collapse:collapse;width:100%;margin:.4rem 0}}
th,td{{border-bottom:1px solid #e5e5e5;padding:6px 9px;font-size:14px;text-align:left}}
th{{font-size:11px;color:#555;text-transform:uppercase;letter-spacing:.03em}}
td.c{{text-align:center}} code{{background:#eef;padding:1px 5px;border-radius:3px;font-size:12px}}
.bar{{background:#eee;border-radius:3px;width:120px;height:13px;display:inline-block;
vertical-align:middle;margin-right:6px}} .bar>div{{height:13px;border-radius:3px}}
.unmeasured{{color:#666;font-size:12px;background:#fafafa;border:1px solid #eee;border-radius:6px;
padding:8px 12px;margin:.4rem 0}}</style></head><body>
<h1>{esc(title)}</h1>
<p class="sub"><b>IGS = mean(pose validity, transform validity)</b> — machine-verified (sympy),
recall-proof by construction. {banks}</p>
<div class="cards">
<div class="card"><div class="v">{summary['n_measured']}</div><div class="l">models measured</div></div>
<div class="card"><div class="v">{summary['n_attempted']}</div><div class="l">attempted</div></div>
<div class="card"><div class="v">{summary['n_families']}</div><div class="l">families</div></div>
<div class="card"><div class="v">{rng}</div><div class="l">IGS range</div></div>
<div class="card"><div class="v">{summary['n_unmeasured']}</div><div class="l">unmeasured</div></div>
</div>
<p class="sub">Token spend: {tok}.</p>
<h2>Ranking</h2>
<table><tr><th>#</th><th>model</th><th>IGS</th><th>pose</th><th>transform</th><th>reps</th></tr>
{rows}</table>
<h2>By family (best model)</h2>
<table><tr><th>family</th><th>n</th><th>best IGS</th><th>best model</th><th>mean IGS</th></tr>
{fam_rows}</table>
<h2>Coverage</h2>
<div class="unmeasured"><b>Unmeasured ({summary['n_unmeasured']}/{summary['n_attempted']}):</b>
{unmeasured}<br><span style="color:#999">Unmeasured = provider-incompatible (404 / non-serverless /
unsupported endpoint), reasoning-model timeout, or account/credit limit reached mid-run.</span></div>
</body></html>"""
