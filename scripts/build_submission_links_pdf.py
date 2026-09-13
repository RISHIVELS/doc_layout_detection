# Renders the submission cover sheet as a one-page PDF. Design brief I set
# myself: whoever opens this should get every link, the headline result,
# and how to run it without scrolling or opening anything else.
#
# So - a ruled board around the page, the two links they will actually
# click (demo + repo) boxed at the top, everything else in scannable rows,
# and the result strip pinned to the foot. Black on white, one blue used
# for links only. No tinted panels; that version read as templated.
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

ACCENT = colors.HexColor("#0b4fd1")
INK = colors.HexColor("#000000")
# previous grey was #767676 - too light once printed, labels washed out
# against the URLs next to them. Darkened until label and value read at
# the same distance.
LABEL = colors.HexColor("#3d3d3d")
RULE = colors.HexColor("#c9c9c9")

REPO = "https://github.com/RISHIVELS/doc_layout_detection"
DEMO = "https://huggingface.co/spaces/RISHIVEL/RAP_DocLayout_DetectionD"

BOARD_INSET = 0.42 * inch

# (label, value, href) - href None means plain text. Where a link exists,
# the visible text is the path inside the repo rather than the whole URL:
# the full one wrapped mid-word ("...weights-v1/b est.pt") and looked
# broken. The repo root is printed in the hero box directly above, so a
# relative path is still resolvable on paper.
SECTIONS = [
    ("What was built", [
        ("Part A - detection", "RT-DETR-L fine-tuned on DocLayNet. All 11 classes are non-COCO, "
                               "so no pretrained class carries any of the score.", None),
        ("Part B - reasoning", "POST /ask routes a question to the detector or refuses it. "
                               "&quot;What is the invoice total?&quot; is refused by design - a layout "
                               "detector reads structure, never text.", None),
        ("The refusal rule", "insufficient_information is set in deterministic Python before the "
                             "LLM writes anything. The model is told the verdict; it is never asked "
                             "for one.", None),
        ("Constraints met", "No agentic frameworks. Own training code. Seeded and reproducible "
                            "(seed 42, full config in reports/run_metadata.json).", None),
    ]),
    ("Trained weights", [
        ("Download", "/releases/download/weights-v1/best.pt",
         REPO + "/releases/download/weights-v1/best.pt"),
        ("Place at", "weights/best.pt &nbsp;&#183;&nbsp; 66 MB Ultralytics checkpoint", None),
    ]),
    ("Reports and evidence", [
        ("Visual evaluation report", "/reports/evaluation_report.pdf",
         REPO + "/blob/main/reports/evaluation_report.pdf"),
        ("Raw metrics", "/reports/metrics.json",
         REPO + "/blob/main/reports/metrics.json"),
        ("Failure-case images", "/reports/failure_examples/",
         REPO + "/tree/main/reports/failure_examples"),
        ("Build log", "/memo/BUILD_LOG.md", REPO + "/blob/main/memo/BUILD_LOG.md"),
        ("Written memo", "MEMO.pdf - in this submission folder", None),
    ]),
    ("Running it", [
        ("API endpoints", "GET /health &nbsp; &#183; &nbsp; POST /detect &nbsp; &#183; &nbsp; POST /ask", None),
        ("Local setup", "README.md in the repository - venv, Docker, and curl examples", None),
        ("Training script", "/scripts/train.py", REPO + "/blob/main/scripts/train.py"),
    ]),
]

FOOTER_FACTS = (
    "RT-DETR-L fine-tuned on DocLayNet  ·  11 non-COCO classes  ·  "
    "mAP50 0.622, mAP50-95 0.412  ·  15 epochs on a Tesla T4  ·  41.3 ms/image"
)


def draw_board(canvas, doc):
    """The border. Two rules rather than one - a 1.1pt outer and a hairline
    just inside it. A single heavy box looks like a certificate; the pair
    reads like a printed form, which is what this actually is."""
    canvas.saveState()
    width, height = LETTER
    canvas.setStrokeColor(INK)
    canvas.setLineWidth(1.1)
    canvas.rect(BOARD_INSET, BOARD_INSET, width - 2 * BOARD_INSET, height - 2 * BOARD_INSET)
    canvas.setLineWidth(0.4)
    canvas.setStrokeColor(RULE)
    inner = BOARD_INSET + 4
    canvas.rect(inner, inner, width - 2 * inner, height - 2 * inner)

    # Facts strip sits on the board's bottom edge, drawn outside the
    # flowable frame, so it stays pinned no matter how content above
    # reflows.
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(LABEL)
    canvas.drawCentredString(width / 2, BOARD_INSET + 15, FOOTER_FACTS)
    canvas.restoreState()


def main() -> None:
    out_path = ROOT / "memo" / "SUBMISSION_LINKS.pdf"
    base = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "T", parent=base["Title"], fontSize=17, textColor=INK, alignment=0,
        spaceAfter=1, leading=20,
    )
    subtitle_style = ParagraphStyle(
        "S", parent=base["Normal"], fontSize=9.5, textColor=LABEL, leading=12,
    )
    section_style = ParagraphStyle(
        "Sec", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8.2,
        textColor=INK, spaceBefore=14, spaceAfter=5, leading=10,
    )
    label_style = ParagraphStyle(
        "L", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8.6,
        textColor=LABEL, leading=12,
    )
    value_style = ParagraphStyle(
        "V", parent=base["Normal"], fontSize=9.2, textColor=INK, leading=12,
    )
    link_style = ParagraphStyle("Lk", parent=value_style, textColor=ACCENT)

    hero_tag_style = ParagraphStyle(
        "HT", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8,
        textColor=INK, leading=11,
    )
    hero_note_style = ParagraphStyle(
        "HN", parent=base["Normal"], fontSize=8.4, textColor=LABEL, leading=11,
    )
    hero_url_style = ParagraphStyle(
        "HU", parent=base["Normal"], fontSize=10, textColor=ACCENT, leading=13,
    )

    story = [
        Paragraph("RAP Pre-Hackathon Screening &#8212; Round 1", title_style),
        Paragraph(
            "Constrained Object Detection &amp; Reasoning API "
            "&nbsp;&#183;&nbsp; Document layout detection",
            subtitle_style,
        ),
        Spacer(1, 13),
    ]

    # Hero: the two things an evaluator opens first, given their own box so
    # they get found before any reading happens.
    def hero_row(tag: str, note: str, url: str) -> list:
        return [
            Paragraph(tag, hero_tag_style),
            [
                Paragraph(note, hero_note_style),
                Paragraph('<link href="%s">%s</link>' % (url, url), hero_url_style),
            ],
        ]

    hero = Table(
        [
            hero_row(
                "LIVE DEMO",
                "Runs in the browser, no setup &#8212; upload a page, get real "
                "detections and ask a question",
                DEMO,
            ),
            hero_row(
                "SOURCE CODE",
                "Training code, API, reasoning layer, Dockerfile, build log",
                REPO,
            ),
        ],
        colWidths=[1.15 * inch, 5.3 * inch],
    )
    hero.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.9, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(hero)

    for heading, rows in SECTIONS:
        block = [Paragraph(heading.upper(), section_style)]
        data = []
        for label, value, href in rows:
            para = Paragraph(
                '<link href="%s">%s</link>' % (href, value) if href else value,
                link_style if href else value_style,
            )
            data.append([Paragraph(label, label_style), para])

        table = Table(data, colWidths=[1.55 * inch, 4.9 * inch])
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("LINEABOVE", (0, 0), (-1, 0), 0.7, INK),
            ("LINEBELOW", (0, 0), (-1, -2), 0.35, RULE),
        ]))
        block.append(table)
        # a heading must not end up alone at the foot, separated from its rows
        story.append(KeepTogether(block))

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        topMargin=0.82 * inch,
        bottomMargin=0.95 * inch,
        leftMargin=0.78 * inch,
        rightMargin=0.78 * inch,
        title="", author="", subject="", creator="",
    )
    doc.build(story, onFirstPage=draw_board, onLaterPages=draw_board)

    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(out_path)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata({"/Title": "", "/Author": "", "/Subject": "", "/Creator": "", "/Producer": ""})
    with open(out_path, "wb") as f:
        writer.write(f)

    print("wrote %s (%d pages)" % (out_path, len(reader.pages)))


if __name__ == "__main__":
    main()
