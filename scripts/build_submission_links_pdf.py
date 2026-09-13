# Renders memo/SUBMISSION_LINKS.txt as a designed one-page PDF - grouped
# sections with accent bars and light backgrounds, not just stacked
# label/value text. Real clickable hyperlinks, not plain URL text.
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent

ACCENT = colors.HexColor("#1a56db")
ACCENT_BG = colors.HexColor("#eef3fd")
INK = colors.HexColor("#111111")
MUTED = colors.HexColor("#5b5b5b")
RULE = colors.HexColor("#dcdcdc")

# Grouped into sections rather than one flat list - each tuple is
# (section title, [(label, value, is_url), ...])
SECTIONS = [
    ("Project", [
        ("Name", "Document Layout Detection & Reasoning API (RT-DETR on DocLayNet)", False),
    ]),
    ("Source & Weights", [
        ("GitHub Repository", "https://github.com/RISHIVELS/doc_layout_detection", True),
        ("Trained Model Weights", "https://github.com/RISHIVELS/doc_layout_detection/releases/download/weights-v1/best.pt", True),
        ("Model filename", "best.pt", False),
        ("Model placement after download", "weights/best.pt", False),
    ]),
    ("Live Demo & Reports", [
        ("Live Demo (no setup - upload a page, get real detection + Q&amp;A)", "https://huggingface.co/spaces/RISHIVEL/RAP_DocLayout_DetectionD", True),
        ("Visual Evaluation Report", "https://github.com/RISHIVELS/doc_layout_detection/blob/main/reports/evaluation_report.pdf", True),
    ]),
    ("API & Documentation", [
        ("Endpoints", "GET /health   ·   POST /detect   ·   POST /ask", False),
        ("Setup and API usage", "See README.md in the GitHub repository.", False),
        ("Written Memo", "MEMO.pdf", False),
    ]),
]


def section_table(title: str, rows: list[tuple[str, str, bool]], styles: dict) -> Table:
    """One section as a table with a colored left accent bar, a tinted
    header row, and label/value pairs underneath - not just plain text
    stacked with no visual grouping."""
    body_rows = []
    for label, value, is_url in rows:
        value_para = Paragraph(
            f'<link href="{value}">{value}</link>' if is_url else value,
            styles["link"] if is_url else styles["value"],
        )
        body_rows.append([Paragraph(label, styles["label"]), value_para])

    table_data = [[Paragraph(title, styles["section_title"]), ""]] + body_rows
    table = Table(table_data, colWidths=[1.9 * inch, 4.5 * inch])

    style_cmds = [
        ("SPAN", (0, 0), (1, 0)),
        ("BACKGROUND", (0, 0), (1, 0), ACCENT_BG),
        ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT),  # left accent bar down the whole block
        ("TOPPADDING", (0, 0), (-1, 0), 5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 1), (-1, -2), 0.4, RULE),
    ]
    table.setStyle(TableStyle(style_cmds))
    return table


def main() -> None:
    out_path = ROOT / "memo" / "SUBMISSION_LINKS.pdf"
    base = getSampleStyleSheet()

    styles = {
        "section_title": ParagraphStyle(
            "SectionTitle", parent=base["Normal"], fontSize=10, fontName="Helvetica-Bold",
            textColor=ACCENT,
        ),
        "label": ParagraphStyle(
            "Label", parent=base["Normal"], fontSize=8.5, textColor=MUTED,
        ),
        "value": ParagraphStyle(
            "Value", parent=base["Normal"], fontSize=9.5, textColor=INK, leading=12,
        ),
        "link": ParagraphStyle(
            "Link", parent=base["Normal"], fontSize=9.5, textColor=ACCENT, leading=12,
        ),
    }

    title_style = ParagraphStyle(
        "Title", parent=base["Title"], fontSize=17, textColor=INK, spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=base["Normal"], fontSize=10, textColor=MUTED, spaceAfter=4,
    )

    story = [
        Paragraph("RAP Pre-Hackathon Screening Submission", title_style),
        Paragraph("Constrained Object Detection &amp; Reasoning API", subtitle_style),
    ]

    # a thin full-width rule under the header instead of relying on
    # whitespace alone to separate header from content
    rule = Table([[""]], colWidths=[6.4 * inch], rowHeights=[1])
    rule.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1.2, INK)]))
    story.append(rule)
    story.append(Spacer(1, 14))

    for title, rows in SECTIONS:
        story.append(KeepTogether(section_table(title, rows, styles)))
        story.append(Spacer(1, 12))

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        title="", author="", subject="", creator="",
    )
    doc.build(story)

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
