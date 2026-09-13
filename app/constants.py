# Single source of truth for the class list. Had it duplicated across the
# dataset script, training yaml and API code early on and they drifted -
# no error, just silently wrong labels. Everything imports from here now.
from __future__ import annotations

# DocLayNet's 0-indexed alphabetical order. Verified against rendered pages
# (see prepare_dataset.py --verify), don't trust the dataset card blindly.
# Don't reorder - these ids are baked into the label files and the weights.
CLASS_NAMES: list[str] = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]

CLASS_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS: dict[int, str] = {idx: name for name, idx in CLASS_TO_ID.items()}

NUM_CLASSES: int = len(CLASS_NAMES)

MODEL_VERSION: str = "rtdetr-l-doclaynet-v1"

# The 6 doc types DocLayNet covers. Used to break mAP down per category at
# eval time - one aggregate number can hide a model that's just good at
# financial reports (the biggest slice) and bad at everything else.
DOC_CATEGORIES: list[str] = [
    "financial_reports",
    "scientific_articles",
    "laws_and_regulations",
    "government_tenders",
    "manuals",
    "patents",
]

# RT-DETR has a fixed number of object queries, so it can only ever emit
# this many boxes per image. Dense pages can exceed it - measured in eval
# so that shows up as "query saturation", not unexplained recall loss.
DEFAULT_QUERY_BUDGET: int = 300
