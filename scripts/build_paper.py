"""Build the publishable single-PDF papers from the canonical markdown.

Pipeline (pip-only, no LaTeX): regenerate figures -> pandoc (markdown -> Typst, with an arXiv-style
title block from metadata) -> typst compile -> docs/paper/inversa-en.pdf and inversa-ko.pdf. The
canonical .md files are left untouched (GitHub-rendered); we only strip the duplicate H1 (the title
block carries the title) into a throwaway build file.

Setup once:  python -m pip install -e ".[paper]"
Build:       python scripts/build_paper.py
"""
from __future__ import annotations

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


def regen_figures() -> None:
    print("[build] regenerating figures…", flush=True)
    subprocess.check_call([sys.executable, "scripts/make_figures.py"], cwd=str(ROOT))


def build_one(md_name: str, pdf_name: str) -> None:
    md = (PAPER / md_name).read_text(encoding="utf-8")
    title, body = "Inversa", []
    took_title = False
    for ln in md.splitlines():
        if not took_title and ln.startswith("# "):
            title = ln[2:].replace("*", "").strip()  # title block carries it; drop md emphasis
            took_title = True
            continue
        # Appendix V holds a very wide reference table; switch to landscape from here so its long
        # reproduce-commands/paths fit instead of overlapping a narrow column.
        if ln.startswith("## Appendix V") or ln.startswith("## 부록 V"):
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
