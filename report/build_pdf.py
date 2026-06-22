"""Render report/REPORT.md to report/REPORT.pdf using fpdf2 (pure-Python, no
system dependencies — works on Windows, Colab and Linux alike).

Supports the Markdown subset used in REPORT.md: headings (#..####), paragraphs,
bold, inline code, links, bullet/numbered lists, blockquotes, fenced code blocks,
pipe tables and horizontal rules. Non-latin-1 glyphs are transliterated to ASCII
so the built-in core fonts can render them without bundling a TTF.

Usage:  python report/build_pdf.py [input.md] [output.pdf]
"""
from __future__ import annotations

import os
import re
import sys

from fpdf import FPDF
from fpdf.fonts import FontFace

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(HERE, "REPORT.md")
DEFAULT_OUT = os.path.join(HERE, "REPORT.pdf")

# Map common Unicode to ASCII so core fonts (latin-1) can render the text.
_UNI = {
    "≤": "<=", "≥": ">=", "→": "->", "←": "<-", "↔": "<->", "≈": "~",
    "—": "-", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"',
    "…": "...", "×": "x", "•": "-", " ": " ", "🤗": "HF", "✅": "",
}


def sanitize(text: str) -> str:
    for k, v in _UNI.items():
        text = text.replace(k, v)
    return "".join(c if ord(c) < 256 else "" for c in text)


def inline(text: str) -> str:
    """Flatten inline markdown that we don't render structurally.

    ``**bold**`` is left intact for fpdf's markdown renderer; single-marker
    emphasis is stripped (fpdf doesn't render it) using boundary-aware patterns
    so identifiers like ``max_seq_length`` are never mangled.
    """
    text = re.sub(r"`([^`]*)`", r"\1", text)                 # `code` -> code
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)  # [a](b) -> a (b)
    text = re.sub(r"(?<!\*)\*(?!\*)([^*]+?)\*(?!\*)", r"\1", text)   # *italic* -> italic
    text = re.sub(r"(?<![\w*])_([^_]+?)_(?![\w*])", r"\1", text)     # _italic_ -> italic
    return sanitize(text)


class PDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140)
        self.cell(0, 8, str(self.page_no()), align="C")
        self.set_text_color(0)


def _heading(pdf: PDF, level: int, text: str):
    sizes = {1: 20, 2: 15, 3: 12.5, 4: 11}
    pdf.ln(3 if level > 1 else 1)
    pdf.set_font("Helvetica", "B", sizes.get(level, 11))
    if level == 1:
        pdf.set_text_color(20, 40, 90)
    elif level == 2:
        pdf.set_text_color(30, 60, 120)
    else:
        pdf.set_text_color(0)
    pdf.multi_cell(0, sizes.get(level, 11) * 0.55, inline(text.strip()))
    pdf.set_text_color(0)
    if level <= 2:
        y = pdf.get_y() + 1
        pdf.set_draw_color(200)
        pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
        pdf.ln(3)
    else:
        pdf.ln(1)


def _paragraph(pdf: PDF, text: str):
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, inline(text), markdown=True)
    pdf.ln(1.5)


def _list_item(pdf: PDF, text: str, ordered: bool, idx: int):
    pdf.set_font("Helvetica", "", 10)
    bullet = f"{idx}." if ordered else "-"
    x0 = pdf.get_x()
    pdf.cell(7, 5, bullet)
    pdf.set_x(x0 + 7)
    pdf.multi_cell(0, 5, inline(text), markdown=True)
    pdf.set_x(x0)
    pdf.ln(0.5)


def _blockquote(pdf: PDF, text: str):
    pdf.set_fill_color(245, 247, 250)
    pdf.set_font("Helvetica", "I", 9.5)
    pdf.set_x(pdf.l_margin + 3)
    pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 3, 5,
                   inline(text), fill=True)
    pdf.ln(2)


def _code_block(pdf: PDF, lines):
    pdf.set_font("Courier", "", 8)
    pdf.set_fill_color(244, 244, 244)
    avail = pdf.w - pdf.l_margin - pdf.r_margin
    for ln in lines:
        pdf.multi_cell(avail, 4.2, sanitize(ln) if ln else " ", fill=True)
    pdf.ln(2)


def _table(pdf: PDF, rows):
    # rows: list of list-of-cell-strings; first row is the header.
    pdf.set_font("Helvetica", "", 8.5)
    head_style = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=(40, 70, 120))
    with pdf.table(
        first_row_as_headings=True,
        headings_style=head_style,
        line_height=5,
        text_align="LEFT",
        cell_fill_color=(244, 246, 249),
        cell_fill_mode="ROWS",
    ) as table:
        for r, cells in enumerate(rows):
            row = table.row()
            for c in cells:
                row.cell(inline(c).replace("**", ""))
    pdf.ln(2)


def _image(pdf: PDF, path: str):
    if not os.path.exists(path):
        pdf.set_font("Courier", "", 8)
        pdf.set_text_color(150)
        pdf.multi_cell(0, 5, f"[missing image: {path}]")
        pdf.set_text_color(0)
        return
    avail = pdf.w - pdf.l_margin - pdf.r_margin
    w = avail * 0.82
    pdf.image(path, x=(pdf.w - w) / 2, w=w)
    pdf.ln(3)


def _hr(pdf: PDF):
    pdf.ln(1)
    pdf.set_draw_color(210)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)


def render(md_path: str, pdf_path: str) -> None:
    base_dir = os.path.dirname(os.path.abspath(md_path))
    with open(md_path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")

    pdf = PDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.set_margins(18, 16, 18)
    pdf.add_page()

    i, n = 0, len(lines)
    para: list = []

    def flush_para():
        nonlocal para
        if para:
            _paragraph(pdf, " ".join(p.strip() for p in para))
            para = []

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # fenced code block
        if stripped.startswith("```"):
            flush_para()
            block = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            _code_block(pdf, block)
            i += 1
            continue

        # pipe table
        if stripped.startswith("|") and "|" in stripped[1:]:
            flush_para()
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.match(r"^[\s:\-]+$", "".join(cells)):  # skip --- separator
                    tbl.append(cells)
                i += 1
            if tbl:
                _table(pdf, tbl)
            continue

        # blank line -> paragraph break
        if stripped == "":
            flush_para()
            i += 1
            continue

        # image: ![alt](path)
        m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)\s*$", stripped)
        if m:
            flush_para()
            _image(pdf, os.path.join(base_dir, m.group(1)))
            i += 1
            continue

        # horizontal rule
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            flush_para()
            _hr(pdf)
            i += 1
            continue

        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush_para()
            _heading(pdf, len(m.group(1)), m.group(2))
            i += 1
            continue

        # blockquote
        if stripped.startswith(">"):
            flush_para()
            quote = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip())
                i += 1
            _blockquote(pdf, " ".join(quote))
            continue

        # list item
        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m:
            flush_para()
            _list_item(pdf, m.group(2), ordered=True, idx=int(m.group(1)))
            i += 1
            continue
        m = re.match(r"^[-*]\s+(.*)$", stripped)
        if m:
            flush_para()
            _list_item(pdf, m.group(1), ordered=False, idx=0)
            i += 1
            continue

        # default: paragraph text
        para.append(line)
        i += 1

    flush_para()
    pdf.output(pdf_path)
    print(f"[build_pdf] wrote {pdf_path} ({pdf.page_no()} pages)")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
    dst = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    render(src, dst)
