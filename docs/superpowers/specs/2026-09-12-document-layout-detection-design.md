# Constrained Object Detection & Reasoning API — Design

**Date:** 2026-09-12
**Submission:** RAP Pre-Hackathon Screening, Round 1 (CV + Applied ML track)
**Deadline:** ~1 working day
**Compute:** Kaggle free tier (T4 16 GB, 12 h session, 30 h/week)

---

## 1. Problem & domain

**Domain:** Document layout analysis framed as object detection.

Given a document page image, predict the location and type of every structural
region on the page: tables, figures, headings, body text, captions, formulas.

This is the perception stage that sits beneath any document-understanding
pipeline. Text extraction alone yields a flat stream of characters; layout
detection yields *structure* — without it a table collapses into meaningless
linear text and a page footer is indistinguishable from a contract clause.

**Why this domain:** Rapid Acceleration Partners builds RAPFlow, a content
intelligence platform whose stated purpose is automated data extraction from
unstructured documents and forms. Document layout detection is the upstream
perception stage for exactly that class of product.

We claim domain adjacency to RAP's product line, not knowledge of their stack.

### Classes (11, all non-COCO)

`Caption`, `Footnote`, `Formula`, `List-item`, `Page-footer`, `Page-header`,
`Picture`, `Section-header`, `Table`, `Text`, `Title`

None of these exist in COCO's 80-class vocabulary. Hard Constraint 3 is
satisfied unarguably: no off-the-shelf pretrained checkpoint can produce a
single one of these labels. Every point of mAP must be earned by fine-tuning.

**Decision: train all 11 classes, do not trim.** Dropping the rare, small
classes (`Footnote`, `Page-footer`) would inflate headline mAP. Their poor
per-class AP is failure-case evidence, which the rubric weights at 15%.
Trimming deletes evidence and reads as metric-gaming.

---

## 2. Dataset

**Source:** `pierreguillou/DocLayNet-base` on Hugging Face — a pre-processed
10% subset of IBM Research's DocLayNet.

| Property | Value |
|---|---|
| Images | 8,057 (train 6,910 / val 648 / test 499) |
| Download size | 3.8 GB |
| Image size | 1025 x 1025 PNG, uniform |
| Annotation format | COCO-style `[x, y, w, h]` in 1025-px space |
| License | CDLA-Permissive-1.0 |
| Provenance | DocLayNet (IBM Deep Search), 80,863 expert-annotated pages |

Upstream DocLayNet spans 6 document categories: `financial_reports`,
`scientific_articles`, `laws_and_regulations`, `government_tenders`,
`manuals`, `patents`. Each row carries `doc_category` and `collection`.

**Why the subset and not full DocLayNet:** the full release is ~30 GB and
takes ~45 min to download on a hosted notebook. With a one-day budget, 3.8 GB
of the same expert annotations is the correct trade. This is a deliberate
constraint-driven choice and is documented as such in the memo.

### Known data hazards to handle in preparation

1. **Repeated bounding boxes.** `bboxes_block` is stored per text-line, with
   the parent block's box repeated once per line. Naive ingestion produces
   thousands of duplicate annotations per page. **Must deduplicate on
   `(bbox, category)` before writing labels.**
2. **Category index base is unverified.** Evidence suggests 0-indexed
   alphabetical (`0=Caption ... 10=Title`). The prep script must *verify this
   empirically* by cross-referencing a sample against rendered images, not
   assume it.
3. **Degenerate boxes.** Zero-width/height and out-of-bounds boxes must be
   dropped and counted; the count is reported.

### Split strategy

Use the dataset author's published train/val/test splits as the primary
result: reproducible, and comparable to published DocLayNet work.

**Additionally report per-`doc_category` mAP on the test set.** This is the
substantive evaluation contribution — it answers whether the model generalised
across document types or merely memorised financial reports. A single aggregate
mAP cannot answer that.

Leakage note for the memo: DocLayNet pages are drawn from multi-page source
PDFs. If a random split placed pages from the same PDF on both sides, aggregate
mAP is optimistic. We verify overlap using `original_filename` and report the
measured page-level leakage rather than assuming it is zero.

---

## 3. Model & training

**RT-DETR-L** via Ultralytics, fine-tuned from COCO-pretrained weights.
Permitted explicitly by Hard Constraint 2.

### Why RT-DETR suits this task

- **Global self-attention captures long-range layout context.** Whether a block
  is a `Caption` depends on a `Picture` above it; a `Section-header` is defined
  by what follows. A CNN's local receptive field models this poorly.
- **No NMS.** DETR-family models do direct set prediction with Hungarian
  bipartite matching, so duplicates are suppressed during training, not by a
  hand-tuned post-processing threshold. On dense two-column pages, adjacent text
  blocks overlap enough that NMS can wrongly suppress a legitimate neighbour.

### Known limitations (carried into the memo, not hidden)

- Transformer detectors underperform CNNs on very small objects. `Footnote` and
  `Page-footer` are small and thin; expect them to be our worst classes.
- **Fixed query budget.** RT-DETR emits a fixed number of predictions per image
  (default 300). Dense patent and financial pages can exceed this. When they do,
  the model *structurally cannot* output every region. We measure how many test
  pages exceed the budget and report it.
- DETR-family models converge more slowly than YOLO, which constrains us on a
  one-day budget.

### Hyperparameters (baseline)

| Parameter | Value | Rationale |
|---|---|---|
| `imgsz` | 640 | Time budget. Known limitation for thin classes; documented. |
| `epochs` | 30-40 | Fits the Kaggle session; checkpoint every epoch. |
| `batch` | 8 | T4 16 GB with AMP. |
| `optimizer` | AdamW | Standard for DETR-family. |
| `lr0` | 1e-4 | DETR-family needs lower LR than YOLO defaults. |
| `cos_lr` | True | Smooth decay within a short schedule. |
| **`fliplr`** | **0.0** | **Critical.** Default is 0.5. Documents have directional reading order; mirroring a page produces layouts that cannot occur. |
| `flipud` | 0.0 | Same reasoning. |
| `degrees` | 3.0 | Small rotation only — scan skew is real, page rotation is not. |
| `perspective` | 0.0005 | Mild, for scan/camera robustness. |
| `hsv_s` / `hsv_v` | low | Pages are predominantly white; heavy colour jitter is unrealistic. |
| `mosaic` | 0.0 | Mosaic composites four images into one, producing impossible multi-page collages. Harmful here. |

Every value, the seed, wall-clock training time, and exact GPU model are logged
to the repo. Hard Constraint 4 caps Part A at 50% if training cannot be
reproduced from our instructions.

---

## 4. Evaluation

- mAP@50 and mAP@50-95, overall and **per class**
- Precision / recall per class
- **Confusion matrix** — the substantive result. Expected confusions:
  `Text` vs `List-item`, `Title` vs `Section-header`, `Caption` vs `Text`.
  These are cases where pixels genuinely underdetermine the label and human
  annotators disagree; they are annotation-protocol ambiguity, not model error.
- **Per-`doc_category` mAP** — generalisation across document types.
- **Query-budget saturation rate** — % of test pages with more ground-truth
  regions than the model can emit.

**What these metrics do not tell us**, stated plainly in the memo: mAP measures
box localisation and classification. It says nothing about whether the inferred
reading order is correct, whether a detected `Table`'s internal structure is
recoverable, or whether the page's semantic hierarchy is right. A model can
score well on mAP and still be useless for downstream extraction.

### Failure mining

`scripts/mine_failures.py` scores every test image, ranks by error, and dumps
the worst N with side-by-side prediction/ground-truth visualisations. The five
memo failure cases are selected from evidence, not invented.

---

## 5. API

FastAPI, two endpoints.

### `POST /detect`
Image upload -> detections.

```json
{
  "detections": [
    {"class_name": "Table", "class_id": 8,
     "bbox": {"x1": 91.0, "y1": 604.2, "x2": 934.5, "y2": 889.1},
     "confidence": 0.94}
  ],
  "image_size": {"width": 1025, "height": 1025},
  "inference_ms": 41.2,
  "model_version": "rtdetr-l-doclaynet-v1"
}
```

### `POST /ask`
Image + natural-language question -> reasoned answer.

---

## 6. Part B — reasoning layer

Hand-written. No LangChain, LangGraph, CrewAI, AutoGen, or equivalent anywhere
in the codebase (Hard Constraint 1). Plain `groq` SDK calls only.

```
/ask(image, question)
 |
 1. INTENT ROUTER  (app/reasoning/router.py)
    - Deterministic regex prefilter short-circuits obviously non-visual
      questions before spending a token.
    - One Groq call with strict json_schema (constrained decoding), given the model's class
      vocabulary:
        {needs_detection, target_classes, task_type, reason}
      task_type in {count, presence, compare, describe, out_of_scope}
    - If the question targets information outside the class vocabulary
      ("what is the invoice total?", "who signed this?"),
      needs_detection=false, task_type=out_of_scope.
 |
 2. DETECT (only if routed there) -> RT-DETR boxes/classes/scores
 |
 3. EVIDENCE BUILDER  (app/reasoning/evidence.py) — pure Python, deterministic
    per-class counts, confidence stats, areas, reading-order sort,
    spatial relations (above/below/contains).
 |
 4. CONFIDENCE GUARDRAIL  (app/reasoning/guardrail.py) — deterministic,
    evaluated BEFORE the LLM:
      - zero detections of a target class                -> insufficient
      - all target detections below tau = 0.50           -> insufficient
      - counting question with >30% of target detections
        in the ambiguous band [0.25, 0.50)               -> insufficient
      - query-budget saturation reached                  -> insufficient
    The verdict is a hard constraint the LLM cannot override.
 |
 5. ANSWER SYNTHESIS  (app/reasoning/synthesize.py)
    Second LLM call, given ONLY the evidence JSON plus the guardrail verdict.
    The LLM never receives the image.
```

**The load-bearing design decision:** the LLM never sees pixels. It is
structurally incapable of hallucinating visual content, because it receives only
a JSON summary of detector output. The guardrail is deterministic Python, not a
politely-worded prompt asking the model to be careful.

### The two honesty cases

1. **Out-of-scope (architectural).** *"What is the invoice total?"* The detector
   knows exactly **where** the table is and nothing about **what it says**. The
   refusal is not a tuned threshold — it is the system recognising the boundary
   of its own perception vocabulary. This is the primary memo example.
2. **Low confidence (statistical).** *"How many tables are on this page?"* on a
   dense financial report returning three table regions at 0.31 / 0.38 / 0.44.
   The guardrail fires; the response states that possible tables were detected
   at low confidence and a reliable count cannot be given.

### Response shape

```json
{
  "answer": "I detected 3 possible table regions, but all are below the
             confidence threshold. I cannot give you a reliable count.",
  "used_detection": true,
  "insufficient_information": true,
  "confidence": "low",
  "evidence": {"class_counts": {"Table": 3},
               "confidence_stats": {"Table": {"max": 0.44, "mean": 0.38}}},
  "detections": [...],
  "reasoning_trace": {
    "router": {"needs_detection": true, "target_classes": ["Table"],
               "task_type": "count", "reason": "..."},
    "guardrail": {"triggered": true, "rule": "all_below_tau",
                  "tau": 0.5, "observed_max": 0.44}
  }
}
```

`reasoning_trace` is included deliberately: it makes the decision path auditable
by a reviewer rather than asking them to trust the output.

---

## 7. Operating domain and its boundary

Trained on rendered/scanned document pages at uniform 1025 x 1025. **Camera-
captured document photographs — perspective skew, shadows, page curl, glare —
are out of distribution.** Mild perspective and rotation augmentation partially
mitigates this; it does not eliminate it.

This limitation is declared in the README and the memo rather than discovered by
the reviewers on the hidden evaluation set.

---

## 8. Packaging (bonus, 10%)

- `Dockerfile` + `docker-compose.yml`, CPU inference by default
- Structured JSON request/response logging with latency per request
- Error handling: malformed image, oversized upload, unsupported MIME type,
  missing API key, model not loaded, LLM timeout
- `GET /health` reporting model load state and version
- README with runnable `curl` examples and full sample payloads for both
  endpoints

---

## 9. Deliverables checklist (mapped to the brief)

| Required | Artifact |
|---|---|
| Source: training script | `scripts/train.py` |
| Source: evaluation script | `scripts/evaluate.py` |
| Source: FastAPI app, both endpoints | `app/main.py` |
| Model weights + load path | Release asset / Kaggle output, path in README |
| Memo: domain & sourcing rationale | `memo/MEMO.md` §1 |
| Memo: split strategy + justification | `memo/MEMO.md` §2 |
| Memo: metrics and their limits | `memo/MEMO.md` §3 |
| Memo: 5 failure cases + root cause | `memo/MEMO.md` §4 (from `mine_failures.py`) |
| Memo: Part B routing + insufficient-info example | `memo/MEMO.md` §5 |
| API usage instructions + sample payloads | `README.md` |
| Bonus: Docker, logging, error handling | `Dockerfile`, `app/logging_config.py` |

## 10. Hard-constraint compliance

| # | Constraint | How satisfied |
|---|---|---|
| 1 | No agentic frameworks | Plain `groq` SDK. No LangChain/LangGraph/CrewAI/AutoGen. Verified by dependency audit of `requirements.txt`. |
| 2 | No AutoML; own code; RT-DETR | Ultralytics RT-DETR, own training/eval scripts. |
| 3 | >= 1 non-COCO class | All 11 classes are non-COCO. |
| 4 | Reproducibility | Seeds, hyperparameters, GPU model, wall-clock time, exact commands in README. |
| 5 | Tool use disclosed | AI assistance used and disclosed; every decision documented with rationale here for verbal defense. |

---

## 11. Out of scope (YAGNI)

- Multi-agent orchestration — explicitly excluded by the brief
- OCR / text content extraction — layout only; this boundary *is* the
  out-of-scope honesty case
- Reading-order reconstruction beyond a simple top-to-bottom sort
- Model ensembling, TTA, hyperparameter search — no time, and marginal under
  this rubric
- Training on full 30 GB DocLayNet
