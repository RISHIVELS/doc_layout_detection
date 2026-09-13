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
    what should be the same paragraph.

    A line starting with '- ' also opens a new block, so consecutive
    bullets (which have no blank line between them) don't get merged into
    one run-on paragraph. Their own wrapped continuation lines still fold
    into the bullet they belong to."""
    blocks, current = [], []

    def flush():
        if current:
            blocks.append(" ".join(current))
            current.clear()

    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.startswith("> "):
            line = line[2:]  # blockquote marker - stray without this on wrapped lines
        if line.startswith("- ") or line.startswith("#"):
            flush()  # bullets and headings each start their own block
        current.append(line)
    flush()
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


BOARD_INSET = 0.38 * inch
RULE_GREY = colors.HexColor("#c9c9c9")


def _board_and_footer(canvas, doc):
    """Ruled border plus the page number, drawn together because both live
    outside the flowable frame. Two rules rather than one - a 1.1pt outer
    and a hairline inside it; a single heavy box reads like a certificate.
    Same frame as the submission cover sheet, so the two documents look
    like one set."""
    canvas.saveState()
    width, height = LETTER
    canvas.setStrokeColor(colors.black)
    canvas.setLineWidth(1.1)
    canvas.rect(BOARD_INSET, BOARD_INSET, width - 2 * BOARD_INSET, height - 2 * BOARD_INSET)
    canvas.setLineWidth(0.4)
    canvas.setStrokeColor(RULE_GREY)
    inner = BOARD_INSET + 4
    canvas.rect(inner, inner, width - 2 * inner, height - 2 * inner)

    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(colors.HexColor("#3d3d3d"))
    canvas.drawCentredString(width / 2, BOARD_INSET + 14, f"{doc.page} / 2")
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
        spaceAfter=8, alignment=1,
    )
    h2_style = ParagraphStyle(
        "MemoH2", parent=styles["Heading2"], fontSize=10, spaceBefore=6, spaceAfter=2.5,
        textColor="black",
    )
    # Leading and paragraph gaps trimmed to buy back the vertical the board
    # margins cost. 10.2 on 8.5pt is tight but still comfortably readable;
    # anything below this started to look cramped on screen.
    body_style = ParagraphStyle(
        "MemoBody", parent=styles["Normal"], fontSize=8.5, leading=10.2, spaceAfter=3.6,
        textColor="black",
    )
    bullet_style = ParagraphStyle(
        "MemoBullet", parent=body_style, leftIndent=10, bulletIndent=0, spaceAfter=2.4,
        bulletFontSize=8.5,
    )
    # The repo/demo/report row sits directly under the title, so it should
    # sit on the same centre line as the title and subtitle - left-aligned
    # it read as the first line of body text rather than as a header.
    links_style = ParagraphStyle(
        "MemoLinks", parent=body_style, alignment=1, spaceBefore=1, spaceAfter=9,
    )

    story = []
    links_row_next = False
    for block in parse_blocks(source):
        if links_row_next:
            # only the one block immediately after the title
            links_row_next = False
            if not block.startswith(("#", "- ")):
                story.append(Paragraph(inline_markdown_to_xml(block), links_style))
                continue
        # Explicit placeholder rather than sniffing for a prose string.
        # The old version triggered on "mAP50-95 = 0.412" appearing in the
        # text, which silently dropped the whole table the moment I
        # reworded that sentence - exactly the kind of failure that ships
        # unnoticed because nothing errors.
        if block.strip() == "{{RESULTS_TABLE}}":
            story.append(Spacer(1, 4))
            story.append(results_table(metrics))
            story.append(Spacer(1, 8))
        elif block.startswith("# "):
            story.append(Paragraph(inline_markdown_to_xml(block[2:]), title_style))
            story.append(Paragraph("RT-DETR-L fine-tuned on DocLayNet, evaluated on a held-out test split", subtitle_style))
            links_row_next = True
        elif block.startswith("## "):
            story.append(Paragraph(inline_markdown_to_xml(block[3:]), h2_style))
        elif block.startswith("- "):
            # bulletText is what actually draws the marker - the style's
            # indent alone would just give silently un-bulleted text
            story.append(Paragraph(inline_markdown_to_xml(block[2:]), bullet_style, bulletText="•"))
        else:
            story.append(Paragraph(inline_markdown_to_xml(block), body_style))

    doc = BaseDocTemplate(
        str(out_path),
        pagesize=LETTER,
        # Pulled in from 0.5/0.65 to clear the board rules. That costs
        # roughly five lines of vertical run over two pages, paid back by
        # the tighter leading below - it still lands on exactly 2 pages.
        topMargin=0.62 * inch,
        bottomMargin=0.72 * inch,
        leftMargin=0.78 * inch,
        rightMargin=0.78 * inch,
        title="",
        author="",
        subject="",
        creator="",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="memo", frames=[frame], onPage=_board_and_footer)])
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
