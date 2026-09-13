# Renders memo/SUBMISSION_LINKS.txt as a clean one-page PDF - same
# design language as build_memo_pdf.py, with real clickable hyperlinks
# instead of plain URL text.
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent.parent

# (label, value, is_url) - built directly rather than parsed from the txt
# file, since the links need real <link> tags reportlab can render as
# clickable, not just plain text that happens to look like a URL.
ENTRIES = [
    ("Project", "Document Layout Detection & Reasoning API (RT-DETR on DocLayNet)", False),
    ("GitHub Repository", "https://github.com/RISHIVELS/doc_layout_detection", True),
    ("Trained Model Weights", "https://github.com/RISHIVELS/doc_layout_detection/releases/download/weights-v1/best.pt", True),
    ("Model filename", "best.pt", False),
    ("Model placement after download", "weights/best.pt", False),
    ("Live Demo", "https://huggingface.co/spaces/RISHIVEL/RAP_DocLayout_DetectionD", True),
    ("Visual Evaluation Report", "https://github.com/RISHIVELS/doc_layout_detection/blob/main/reports/evaluation_report.pdf", True),
    ("API", "GET /health   ·   POST /detect   ·   POST /ask", False),
    ("Setup and API usage", "See README.md in the GitHub repository.", False),
    ("Written Memo", "MEMO.pdf", False),
]


def main() -> None:
    out_path = ROOT / "memo" / "SUBMISSION_LINKS.pdf"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"], fontSize=15, spaceAfter=3, textColor="black",
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=9, textColor=colors.grey,
        spaceAfter=18,
    )
    label_style = ParagraphStyle(
        "Label", parent=styles["Normal"], fontSize=9, textColor=colors.grey, spaceAfter=1,
    )
    value_style = ParagraphStyle(
        "Value", parent=styles["Normal"], fontSize=10.5, textColor="black", spaceAfter=13,
        leading=13,
    )
    link_style = ParagraphStyle(
        "Link", parent=value_style, textColor=colors.HexColor("#1a56db"),
    )

    story = [
        Paragraph("RAP Pre-Hackathon Screening Submission", title_style),
        Paragraph("Constrained Object Detection &amp; Reasoning API", subtitle_style),
    ]

    for label, value, is_url in ENTRIES:
        story.append(Paragraph(label, label_style))
        if is_url:
            story.append(Paragraph(f'<link href="{value}">{value}</link>', link_style))
        else:
            story.append(Paragraph(value, value_style))

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
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
