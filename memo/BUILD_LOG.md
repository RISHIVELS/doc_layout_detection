# Build log

Working notes from building this, kept as I went rather than written after
the fact.

## Reading the brief

Weighting is not what the title suggests: 25% is Part A hidden-set
performance, the rest is dataset sourcing, evaluation honesty, failure
analysis, Part B correctness, code quality, and reproducibility. Roughly
40% is written reasoning about the work, not the work itself. That shaped
everything below — I optimized for a system I could explain precisely,
not for the highest number.

Hard gates: no agentic frameworks anywhere, RT-DETR with my own training
code, at least one non-COCO class or Part A scores zero, and training has
to be reproducible or Part A caps at 50%.

## Picking the domain

First idea was construction PPE (helmet/vest detection) — dropped it
because `person` is a COCO class, so part of the score would be inherited
from the pretrained backbone rather than earned. Also already taken by
another candidate. Traffic was worse on the same axis — car, truck, person,
traffic light are all COCO classes.

Landed on document layout detection: all 11 DocLayNet classes (Table,
Text, Title, etc.) are non-COCO, so every point of mAP is earned. It also
gives Part B a real architectural refusal case instead of a tuned
threshold — "what's the invoice total" is a question about content a
layout detector structurally cannot answer, not a low-confidence guess.
Adjacent to what RAP builds (RAPFlow does document extraction), which
made the domain choice defensible rather than arbitrary.

Risk I accepted: DocLayNet is rendered/scanned pages, not camera photos.
If the hidden eval set is phone photos of paper, this model will do
worse. Mitigated with rotation/perspective augmentation, not eliminated.

## Dataset

`pierreguillou/DocLayNet-base` — a 10% subset of IBM's DocLayNet, same
annotation quality, 8k images instead of 80k. Full dataset is ~30GB and
takes 45 min just to download; not worth it against a one-day budget.

Used the author's train/val/test split rather than re-splitting myself —
reproducible, comparable to published work. Measured the one risk that
comes with that (pages from the same source PDF leaking across the split)
rather than assuming it away: 0% overlap.

The real gotcha in this dataset: `bboxes_block` repeats each region's box
once per text line, so a 40-region page reads as ~1,100 duplicate
annotations if you don't dedupe. Wrote `dedupe_annotations()` to collapse
on `(box, category)` before writing YOLO labels.

## Reasoning layer — why Groq, and the architecture

Switched from an original OpenAI plan to Groq: faster, generous free
tier, and it supports `json_schema` with `strict: true` — the router's
output is structurally guaranteed to match my schema (constrained
decoding), not just prompted-and-hoped-for. `llama-3.3-70b-versatile` was
deprecated for free tier in June 2026, so the router runs `gpt-oss-20b`
and synthesis runs `gpt-oss-120b`.

Design point I care about most: the guardrail that decides
`insufficient_information` is plain deterministic Python, evaluated
*before* the LLM writes anything. Asking a model "are you confident?"
just gets more text — it's equally fluent at a confident-sounding guess
as an honest refusal. The verdict is handed to the LLM as a fact, not
asked as a question.

## Training config choices

`fliplr`, `flipud`, `mosaic` all forced to 0 — Ultralytics defaults these
on for photographs, but a mirrored or four-way-composited document page
is not a document. Kept small-angle rotation and mild perspective, since
those model real scanner/camera skew and are the only hedge against the
domain-shift risk above.

`lr0=1e-4` not the usual `1e-2` — DETR-family transformer components are
unstable at high LR, and I'm fine-tuning from COCO weights, not training
from scratch.

Cut epochs from 30 to 15 partway through after losing the first training
run (see below) and needing to fit a much tighter remaining window —
noting the pivot rather than hiding it.

## Kaggle infrastructure issues

Several real environment problems hit in the first hour of actually
running this — none were modelling bugs, all were dependency/platform
collisions:

- **`datasets>=4.0.0`** dropped support for legacy "loading script"
  datasets entirely; this dataset repo is one. Pinned `datasets<4.0.0`.
- Loading a script-based dataset prompts for confirmation
  (`trust_remote_code`), which hangs forever under Kaggle's non-interactive
  commit mode. Passed `trust_remote_code=True` explicitly, after reading
  what the script actually does.
- The dataset repo's own schema declares box coordinates as `int64`; the
  real values are floats. Older `pyarrow` silently truncated them, newer
  `pyarrow` correctly refuses. Fixed by overriding the load schema to
  `float64` rather than editing someone else's script.
- Ultralytics auto-registers a Ray Tune callback whenever `ray` is
  importable, with no check for whether Tune is running. Kaggle ships ray
  pre-installed; the callback crashed at the end of epoch 1 calling a
  renamed internal Ray API. Fixed with `SETTINGS["raytune"] = False`.
- A notebook cell did `%cd` into a directory and then `rm -rf`'d it on
  the next run, corrupting the shell. Fixed by stepping out first.
- Label-check renders used PIL's default ~10px font — unreadable once
  resized. Switched to a real font (matplotlib's bundled DejaVu, since
  that's a dependency I already have).

**One of these actually cost a full training run**: the session was
running interactively rather than via Save & Run All (Commit), and Kaggle
recycled the container after a period of inactivity, wiping the dataset
and every checkpoint through epoch 10 (which had reached mAP50 0.69).
Rebuilt from scratch with epochs cut to 15 to fit the remaining time, and
switched to committed runs for good.

## API, Docker, Streamlit

FastAPI wraps the same `Detector`/reasoning pipeline used everywhere else
— no separate logic path. Caught one real bug via the tests before it
shipped: the detector dependency was written as a bare default argument
(`= get_detector()`) instead of `Depends(get_detector)`, which would have
silently defeated test overrides while looking completely normal in
production.

Docker: CPU-only torch wheel (no CUDA toolkit bloat for a container that
only serves inference), non-root user. Built and actually ran it —
confirmed the app starts cleanly with no weights present and `/health`
reports that instead of crashing.

Streamlit app reuses the exact same `Detector` and pipeline as the API,
for Hugging Face Spaces — no duplicated logic, just a UI on top.

## Real results

Final run: 15 epochs, Tesla T4, 2.27h. mAP50 0.622, mAP50-95 0.412.

The most useful single finding: `Footnote` (47 instances) scored 0.182
mAP50, `Page-footer` — a similarly thin, similarly-positioned class —
scored 0.843. Difference is `Page-footer` has 9x more training instances
and a very consistent position. Revises my pre-training hypothesis from
"thin objects fail" to "thin *and rare* objects fail; thin-but-frequent
ones are fine."

`Picture` had decent recall (0.622) but weak precision (0.416) —
mined-failure images show composite regions (a diagram plus its embedded
formula text) getting split into multiple overlapping boxes instead of
one region.

Live-tested the full pipeline against the trained weights on a real,
previously-unseen page: asking a content question ("what law is this
about") correctly routed to `out_of_scope` before detection ever ran —
the real example used in the memo.
