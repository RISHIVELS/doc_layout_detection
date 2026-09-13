# Converts memo/MEMO.md into a clean 2-page PDF for submission. Plain
# reportlab, black text on white, no metadata beyond a bare title.
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent.parent


def inline_markdown_to_xml(text: str) -> str:
    """**bold** -> <b>bold</b>, `code` -> monospace font tag. Order
    matters - code spans first, so bold markers inside a code span
    (there aren't any here, but just in case) don't get touched."""
    text = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def parse_blocks(source: str) -> list[str]:
    """Splits on blank lines so a markdown paragraph that's hand-wrapped
    across several source lines becomes one flowing block, not one
    Paragraph per line (which was inflating this to 3 pages with extra
    gaps between what should be the same paragraph)."""
    blocks, current = [], []
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append(" ".join(current))
                current = []
            continue
        if line.startswith("> "):
            line = line[2:]  # blockquote marker - stray without this on wrapped lines
        current.append(line)
    if current:
        blocks.append(" ".join(current))
    return blocks


def main() -> None:
    source = (ROOT / "memo" / "MEMO.md").read_text(encoding="utf-8")
    out_path = ROOT / "memo" / "MEMO.pdf"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MemoTitle", parent=styles["Title"], fontSize=15, spaceAfter=10, textColor="black",
    )
    h2_style = ParagraphStyle(
        "MemoH2", parent=styles["Heading2"], fontSize=11, spaceBefore=8, spaceAfter=3,
        textColor="black",
    )
    body_style = ParagraphStyle(
        "MemoBody", parent=styles["Normal"], fontSize=9, leading=11.5, spaceAfter=5,
        textColor="black",
    )

    story = []
    for block in parse_blocks(source):
        if block.startswith("# "):
            story.append(Paragraph(inline_markdown_to_xml(block[2:]), title_style))
        elif block.startswith("## "):
            story.append(Paragraph(inline_markdown_to_xml(block[3:]), h2_style))
        else:
            story.append(Paragraph(inline_markdown_to_xml(block), body_style))

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        title="",
        author="",
        subject="",
        creator="",
    )
    doc.build(story)

    # reportlab still stamps its own Producer string and a CreationDate/
    # ModDate regardless of the kwargs above - strip those too, same
    # discipline as the evaluation report PDF.
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(out_path)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata({"/Title": "", "/Author": "", "/Subject": "", "/Creator": "", "/Producer": ""})
    with open(out_path, "wb") as f:
        writer.write(f)

    print(f"wrote {out_path} ({len(reader.pages)} pages)")


if __name__ == "__main__":
    main()
