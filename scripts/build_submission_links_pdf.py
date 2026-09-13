# Renders memo/SUBMISSION_LINKS.txt as a clean one-page PDF. Kept
# deliberately plain: black text, one accent color used only for links,
# thin rules for structure instead of colored panels - the boxed/tinted
# version read as templated rather than simple.
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent

ACCENT = colors.HexColor("#1a56db")
INK = colors.HexColor("#111111")
MUTED = colors.HexColor("#767676")
RULE = colors.HexColor("#e3e3e3")

SECTIONS = [
    ("Project", [
        ("Name", "Document Layout Detection & Reasoning API (RT-DETR on DocLayNet)", False),
    ]),
    ("Source & weights", [
        ("GitHub repository", "https://github.com/RISHIVELS/doc_layout_detection", True),
        ("Trained model weights", "https://github.com/RISHIVELS/doc_layout_detection/releases/download/weights-v1/best.pt", True),
        ("Model filename", "best.pt", False),
        ("Model placement after download", "weights/best.pt", False),
    ]),
    ("Live demo & reports", [
        ("Live demo (no setup - upload a page, get real detection + Q&amp;A)", "https://huggingface.co/spaces/RISHIVEL/RAP_DocLayout_DetectionD", True),
        ("Visual evaluation report", "https://github.com/RISHIVELS/doc_layout_detection/blob/main/reports/evaluation_report.pdf", True),
    ]),
    ("API & documentation", [
        ("Endpoints", "GET /health   ·   POST /detect   ·   POST /ask", False),
        ("Setup and API usage", "See README.md in the GitHub repository.", False),
        ("Written memo", "MEMO.pdf", False),
    ]),
]


def main() -> None:
    out_path = ROOT / "memo" / "SUBMISSION_LINKS.pdf"
    base = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title", parent=base["Title"], fontSize=16, textColor=INK, spaceAfter=2, alignment=0,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=base["Normal"], fontSize=9.5, textColor=MUTED,
    )
    # small-caps-style section label via letter-spacing and size rather
    # than a colored band - a common quiet way to divide sections without
    # decoration
    section_style = ParagraphStyle(
        "Section", parent=base["Normal"], fontSize=8.5, textColor=MUTED,
        spaceBefore=18, spaceAfter=6, leading=10,
    )
    label_style = ParagraphStyle(
        "Label", parent=base["Normal"], fontSize=9, textColor=MUTED, leading=12,
    )
    value_style = ParagraphStyle(
        "Value", parent=base["Normal"], fontSize=10, textColor=INK, leading=13,
    )
    link_style = ParagraphStyle(
        "Link", parent=value_style, textColor=ACCENT,
    )

    story = [
        Paragraph("RAP Pre-Hackathon Screening Submission", title_style),
        Paragraph("Constrained Object Detection &amp; Reasoning API", subtitle_style),
        Spacer(1, 16),
    ]

    for title, rows in SECTIONS:
        story.append(Paragraph(title.upper(), section_style))

        table_data = []
        for label, value, is_url in rows:
            value_para = Paragraph(
                f'<link href="{value}">{value}</link>' if is_url else value,
                link_style if is_url else value_style,
            )
            table_data.append([Paragraph(label, label_style), value_para])

        table = Table(table_data, colWidths=[1.9 * inch, 4.5 * inch])
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, RULE),
        ]))
        story.append(table)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
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
