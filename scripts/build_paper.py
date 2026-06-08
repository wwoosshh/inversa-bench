"""Build the publishable single-PDF papers from the canonical markdown.

Pipeline (pip-only, no LaTeX): regenerate figures -> pandoc (markdown -> Typst, with an arXiv-style
title block from metadata) -> typst compile -> docs/paper/inversa-en.pdf and inversa-ko.pdf. The
canonical .md files are left untouched (GitHub-rendered); we only strip the duplicate H1 (the title
block carries the title) into a throwaway build file.

Setup once:  python -m pip install -e ".[paper]"
Build:       python scripts/build_paper.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pypandoc
import typst

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "paper"
AUTHOR = "Woosunghyeun (우성현) — Chungwoon University · nunconnect1@gmail.com"
DATE = "June 8, 2026"
LICENSE = "CC-BY-4.0"

DOCS = [("inversa-paper.md", "inversa-en.pdf"), ("inversa-paper.ko.md", "inversa-ko.pdf")]

# Raw-Typst styling injected at the top of the body (pandoc passes ```{=typst} through verbatim):
# tighter margins + justified body + smaller table text so the wide tables (Appendix V, the
# hypothesis ledger) fit a single column cleanly.
PREAMBLE = """```{=typst}
#set page(margin: (x: 1.6cm, y: 1.9cm))
#set text(size: 10pt)
#set par(justify: true)
#show table: set text(size: 8.2pt)
#show table.cell.where(y: 0): strong
#show raw: set text(size: 8.5pt)
// pandoc wraps tables in a (non-breakable) figure; allow them to split across pages so long
// tables (e.g. the 59-model leaderboard) flow instead of overflowing the bottom margin.
#show figure: set block(breakable: true)
```

"""


_ZWSP = "​"  # zero-width space: an invisible break opportunity


def _breakable_table_row(line: str) -> str:
    """In table rows only, add zero-width break points after '/' and '_' so long file paths and
    reproduce-commands (no spaces) can wrap inside a cell instead of overflowing into the next."""
    s = line.lstrip()
    if not s.startswith("|"):
        return line
    if set(s) <= set("|-: "):  # the |---|:--| separator row
        return line
    # after / _ (paths, identifiers) and ASCII - (hyphenated model names); NOT '.' (would split
    # decimals like 0.93). The unicode minus '−' in CIs is untouched.
    return (line.replace("/", "/" + _ZWSP).replace("_", "_" + _ZWSP).replace("-", "-" + _ZWSP))


def _load(name: str):
    return json.load(open(ROOT / "data" / "results" / f"{name}.json", encoding="utf-8"))


def _short(m: str) -> str:
    return m.split("/")[-1]


def _fmt(v) -> str:
    """2-decimal display for a numeric cell; em-dash for missing."""
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.2f}"
    return str(v)


def data_appendix(lang: str) -> str:
    """Build 'Appendix D' from the real result JSONs so the published PDF carries the actual
    per-model evidence inline (not just file paths) and a reader can verify every claim from the
    document alone. Returns markdown (portrait); each table is guarded so a missing file is skipped."""
    ko = lang == "ko"
    H = {"title": ("## 부록 D — 전체 모델별 원자료 (PDF만으로 검증 가능하도록 직접 수록)"
                   if ko else
                   "## Appendix D — Full per-model data (embedded so the PDF verifies on its own)"),
         "intro": ("아래 표는 본문 주장에 쓰인 `data/results/*.json`의 실제 값을 그대로 옮긴 것입니다. "
                   "E12 오염 격차·E12b 역격차의 모델별 값은 §3.1·§3.2 본문 표에 이미 있습니다." if ko else
                   "These tables reproduce the actual values from `data/results/*.json` used in the "
                   "text, so the paper verifies standalone. Per-model E12/E12b gaps are already in the "
                   "§3.1/§3.2 body tables.")}
    out = ["```{=typst}\n#pagebreak()\n```\n", H["title"], "", H["intro"], ""]

    def tbl(title, header, rows):
        out.append(f"**{title}**")
        out.append("")
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join(["---"] * len(header)) + "|")
        out.extend(rows)
        out.append("")

    # D.1 — full leaderboard (the big one that was only in the HTML dashboard)
    try:
        lv = _load("igs_leaderboard_30")["results"]
        try:
            trunc = set(_load("igs_dashboard_summary")["truncated_models"])
        except Exception:
            trunc = set()
        rows = []
        for r in sorted(lv, key=lambda x: x.get("rank", 999)):
            star = " *" if r["model"] in trunc else ""
            rows.append(f"| {r.get('rank','')} | {_short(r['model'])}{star} | "
                        f"{r['pose_validity']:.2f} | {r['transform_validity']:.2f} | {r['igs']:.2f} |")
        cap = ("D.1 — IGS 리더보드 전체 (N={n}, 30문항, temp 0; * = truncation/저신뢰)" if ko
               else "D.1 — Full IGS leaderboard (N={n}, 30-item banks, temp 0; * = truncated/low-confidence)")
        tbl(cap.format(n=len(rows)),
            (["#", "모델", "pose", "transform", "IGS"] if ko else ["#", "model", "pose", "transform", "IGS"]),
            rows)
    except Exception as e:  # noqa: BLE001
        out.append(f"*(leaderboard table unavailable: {e})*\n")

    # D.2 — discriminant E13 (IGS, AIME, non-math logic control) — superset of E10
    try:
        dm = _load("discriminant_results")["models"]
        rows = [f"| {_short(m['model'])} | {_fmt(m.get('igs'))} | {_fmt(m.get('aime'))} | "
                f"{_fmt(m.get('nonmath', m.get('logic')))} |" for m in dm]
        tbl(("D.2 — 판별 타당도 E13: IGS · AIME · 비수학 논리 통제 (N={n})" if ko
             else "D.2 — Discriminant validity E13: IGS · AIME · non-math logic control (N={n})").format(n=len(rows)),
            (["모델", "IGS", "AIME", "논리(통제)"] if ko else ["model", "IGS", "AIME", "logic (control)"]),
            rows)
    except Exception as e:  # noqa: BLE001
        out.append(f"*(E13 table unavailable: {e})*\n")

    # D.3 — predictive E14 (IGS, AIME, Y1 construction, Y2 self-verify)
    try:
        pm = _load("predictive_results")["models"]
        rows = [f"| {_short(k)} | {_fmt(v.get('igs'))} | {_fmt(v.get('aime'))} | "
                f"{_fmt(v.get('y1'))} | {_fmt(v.get('y2'))} |" for k, v in pm.items()]
        tbl(("D.3 — 예측 타당도 E14: IGS · AIME · Y1(구성) · Y2(자가검증) (N={n})" if ko
             else "D.3 — Predictive validity E14: IGS · AIME · Y1(construct) · Y2(self-verify) (N={n})").format(n=len(rows)),
            (["모델", "IGS", "AIME", "Y1", "Y2"] if ko else ["model", "IGS", "AIME", "Y1", "Y2"]),
            rows)
    except Exception as e:  # noqa: BLE001
        out.append(f"*(E14 table unavailable: {e})*\n")

    # D.4 — Rasch theta (interval scaling)
    try:
        im = _load("igs_irt")["models"]
        rows = [f"| {m.get('rank','')} | {_short(m['model'])} | {m['theta']:.2f} | {m['igs_raw']:.2f} |"
                for m in im]
        tbl(("D.4 — Rasch θ (구간 척도) (N={n})" if ko else "D.4 — Rasch θ (interval scaling) (N={n})").format(n=len(rows)),
            (["#", "모델", "θ", "IGS"] if ko else ["#", "model", "θ", "IGS (raw)"]), rows)
    except Exception as e:  # noqa: BLE001
        out.append(f"*(IRT table unavailable: {e})*\n")

    # D.5 — difficulty escalation E8 (easy -> hard -> brutal)
    try:
        easy = {r["model"]: r.get("transform_validity") for r in _load("paper_level3_all")}
        hard = {r["model"]: r["hard_transform_validity"] for r in _load("transform_hard_results")}
        brutal = {r["model"]: r["hard_transform_validity"] for r in _load("transform_brutal_results")}
        common = sorted([m for m in hard if m in easy and m in brutal], key=lambda m: -brutal[m])
        rows = [f"| {_short(m)} | {easy[m]:.2f} | {hard[m]:.2f} | {brutal[m]:.2f} |" for m in common]
        tbl(("D.5 — 난이도 확장 E8: easy(뫼비우스) → hard(r²,r³) → brutal(r⁴,합성) (N={n})" if ko
             else "D.5 — Difficulty escalation E8: easy(Möbius) → hard(r²,r³) → brutal(r⁴,composite) (N={n})").format(n=len(rows)),
            (["모델", "easy", "hard", "brutal"] if ko else ["model", "easy", "hard", "brutal"]), rows)
    except Exception as e:  # noqa: BLE001
        out.append(f"*(escalation table unavailable: {e})*\n")

    return "\n".join(out) + "\n"


def regen_figures() -> None:
    print("[build] regenerating figures…", flush=True)
    subprocess.check_call([sys.executable, "scripts/make_figures.py"], cwd=str(ROOT))


def build_one(md_name: str, pdf_name: str) -> None:
    lang = "ko" if md_name.endswith(".ko.md") else "en"
    md = (PAPER / md_name).read_text(encoding="utf-8")
    title, body = "Inversa", []
    took_title = False
    for ln in md.splitlines():
        if not took_title and ln.startswith("# "):
            title = ln[2:].replace("*", "").strip()  # title block carries it; drop md emphasis
            took_title = True
            continue
        if ln.startswith("## Appendix V") or ln.startswith("## 부록 V"):
            # embed the full per-model data tables (portrait) so the PDF verifies standalone, THEN
            # switch to landscape for Appendix V's very wide reproduce-command table.
            body.extend(data_appendix(lang).splitlines())
            body.append("```{=typst}\n#pagebreak()\n#set page(flipped: true, margin: 1.4cm)\n```\n")
        body.append(ln)
    body = [_breakable_table_row(ln) for ln in body]
    tmp_md = PAPER / "_build.md"
    tmp_typ = PAPER / "_build.typ"
    tmp_md.write_text(PREAMBLE + "\n".join(body), encoding="utf-8")
    try:
        pypandoc.convert_file(
            str(tmp_md), "typst", outputfile=str(tmp_typ),
            extra_args=["--standalone", "--wrap=preserve",
                        "-M", f"title={title}", "-M", f"author={AUTHOR}", "-M", f"date={DATE}"])
        typst.compile(str(tmp_typ), output=str(PAPER / pdf_name))
    finally:
        tmp_md.unlink(missing_ok=True)
        tmp_typ.unlink(missing_ok=True)
    kb = (PAPER / pdf_name).stat().st_size // 1024
    print(f"[build] wrote docs/paper/{pdf_name}  ({kb} KB)", flush=True)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    regen_figures()
    for md_name, pdf_name in DOCS:
        build_one(md_name, pdf_name)
    print("[build] done.", flush=True)


if __name__ == "__main__":
    main()
