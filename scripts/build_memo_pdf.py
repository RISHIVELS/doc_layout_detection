# Converts memo/MEMO.md into a designed 2-page PDF for submission. Plain
# reportlab, black text on white, a real results table pulled from
# reports/metrics.json, no metadata beyond a bare structure.
from __future__ import annotations

import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, NextPageTemplate, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent


def inline_markdown_to_xml(text: str) -> str:
    """`code` -> monospace font tag, [text](url) -> a real clickable
    reportlab <link>, **bold** -> <b>, *italic* -> <i>.

    Links first, before bold/italic - markdown link syntax uses brackets
    and parens, not asterisks, but I want the URL itself protected from
    the code-span regex (a URL can contain backticks in theory) and
    resolved before anything else touches the surrounding text. Without
    this, [text](url) was silently passing through as plain literal
    text with the brackets still in it - no link ever got into the PDF."""
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#1a56db">\1</link>', text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def parse_blocks(source: str) -> list[str]:
    """Splits on blank lines so a markdown paragraph hand-wrapped across
    several source lines becomes one flowing block, not one Paragraph per
    line - that was inflating the page count with extra gaps between
    what should be the same paragraph."""
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


def results_table(metrics: dict) -> Table:
    """Real per-class numbers from reports/metrics.json, not retyped by
    hand into the memo text - the same values the README and evaluation
    report cite."""
    header = ["Class", "Precision", "Recall", "mAP50", "mAP50-95"]
    rows = [header]
    for name, m in sorted(metrics["per_class"].items(), key=lambda kv: kv[1]["mAP50"], reverse=True):
        rows.append([name, f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['mAP50']:.3f}", f"{m['mAP50_95']:.3f}"])

    o = metrics["overall"]
    rows.append(["Overall", f"{o['precision']:.3f}", f"{o['recall']:.3f}", f"{o['mAP50']:.3f}", f"{o['mAP50_95']:.3f}"])

    table = Table(rows, colWidths=[1.5 * inch, 0.85 * inch, 0.75 * inch, 0.75 * inch, 0.85 * inch])
    table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
        ("FONT", (0, 1), (-1, -2), "Helvetica", 8.5),
        ("FONT", (0, -1), (-1, -1), "Helvetica-Bold", 8.5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.black),
        ("LINEABOVE", (0, -1), (-1, -1), 0.75, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
    ]))
    return table


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(LETTER[0] / 2, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    source = (ROOT / "memo" / "MEMO.md").read_text(encoding="utf-8")
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    out_path = ROOT / "memo" / "MEMO.pdf"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MemoTitle", parent=styles["Title"], fontSize=15, spaceAfter=3, textColor="black",
    )
    subtitle_style = ParagraphStyle(
        "MemoSubtitle", parent=styles["Normal"], fontSize=9, textColor=colors.grey,
        spaceAfter=10, alignment=1,
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
    blocks = parse_blocks(source)
    title_done = False
    for block in blocks:
        if block.startswith("# "):
            story.append(Paragraph(inline_markdown_to_xml(block[2:]), title_style))
            story.append(Paragraph("RT-DETR-L fine-tuned on DocLayNet, evaluated on a held-out test split", subtitle_style))
            title_done = True
        elif block.startswith("## "):
            heading_text = block[3:]
            story.append(Paragraph(inline_markdown_to_xml(heading_text), h2_style))
            # drop the real results table in right after the evaluation
            # section's heading, before its body text
            if heading_text.startswith("3. Evaluation"):
                pass  # table inserted after this section's paragraphs below instead
        else:
            story.append(Paragraph(inline_markdown_to_xml(block), body_style))
            # insert the table right after the first paragraph of section 3
            if "mAP50-95 = 0.412" in block:
                story.append(Spacer(1, 4))
                story.append(results_table(metrics))
                story.append(Spacer(1, 6))

    doc = BaseDocTemplate(
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
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="memo", frames=[frame], onPage=_footer)])
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
