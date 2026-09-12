"""
I keep the class vocabulary in one file because I got bitten by this early on.

My first version had the class list written out separately in the dataset prep
script, in the training YAML and again in the API response builder. The moment
those three drifted apart, the labels would still be valid integers, the model
would still train, and every metric would still look reasonable - it would just
be quietly learning the wrong thing. There is no error message for that. So the
list lives here and everything else imports it.

A note on the ordering, because it matters: this is DocLayNet's own 0-indexed
alphabetical ordering. I verified it against rendered pages rather than trusting
the dataset card (see scripts/prepare_dataset.py --verify). Do not reorder it.
The integer ids are written into the YOLO label files on disk and baked into the
trained weights, so changing the order here silently invalidates both.

The other reason I like this class set: none of these eleven exist in COCO.
COCO has person, car, dog, chair. It has no concept of a "Section-header" or a
"Formula". That is what makes the fine-tuning real - there is no pretrained
checkpoint anywhere that can emit one of these labels, so any mAP I report had
to be earned.
"""

from __future__ import annotations

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

"""
DocLayNet pages come from six quite different kinds of document. I carry this
through dataset prep so that at evaluation time I can break mAP down by
category instead of only reporting one aggregate number.

I wanted this because a single mAP figure cannot tell me whether the model
actually learned document structure or just learned what annual reports look
like - financial reports are the largest slice of the data, so a model that is
good at those and useless at patents would still post a respectable headline
score. Splitting the metric out is the only way to catch that.
"""
DOC_CATEGORIES: list[str] = [
    "financial_reports",
    "scientific_articles",
    "laws_and_regulations",
    "government_tenders",
    "manuals",
    "patents",
]

"""
RT-DETR predicts a fixed number of objects per image - it does not scale with
how busy the page is. That is a consequence of the DETR design: there are N
learned object queries and each one produces at most one box.

This is a genuine ceiling for my use case and I only realised it while reading
the architecture properly. A dense patent page or a financial statement can
carry a large number of annotated regions, and if a page has more regions than
the model has queries, the model *cannot* output them all no matter how well it
was trained. I measure how often this happens on the test set and report it,
because otherwise it shows up as unexplained recall loss.
"""
DEFAULT_QUERY_BUDGET: int = 300
