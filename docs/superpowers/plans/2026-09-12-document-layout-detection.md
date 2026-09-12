# Document Layout Detection & Reasoning API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune RT-DETR on DocLayNet document-layout data and expose it through a FastAPI service with a hand-written natural-language reasoning layer that refuses to guess.

**Architecture:** Ultralytics RT-DETR-L fine-tuned on an 8k-image DocLayNet subset (11 non-COCO classes), trained on Kaggle T4. A FastAPI app wraps the weights with two endpoints: `/detect` returns raw detections; `/ask` runs a four-stage hand-written pipeline — regex prefilter, LLM intent router, deterministic evidence builder, deterministic confidence guardrail — then synthesises an answer from structured evidence only. The LLM never receives pixels.

**Tech Stack:** Python 3.11, Ultralytics (RT-DETR), PyTorch, HuggingFace `datasets`, FastAPI, Pydantic v2, `openai` SDK, pytest, Docker.

**Spec:** `docs/superpowers/specs/2026-09-12-document-layout-detection-design.md`

## Global Constraints

- **No agentic frameworks.** LangChain, LangGraph, CrewAI, AutoGen or equivalent must not appear in `requirements.txt`, imports, or vendored code. Plain `openai` SDK only.
- **Detector must be RT-DETR.** Ultralytics `RTDETR('rtdetr-l.pt')`.
- **All 11 DocLayNet classes trained.** No class trimming.
- **Class list, fixed order (0-indexed):** `Caption`, `Footnote`, `Formula`, `List-item`, `Page-footer`, `Page-header`, `Picture`, `Section-header`, `Table`, `Text`, `Title`
- **`fliplr=0.0` and `flipud=0.0` and `mosaic=0.0`** in training. Non-negotiable: mirrored/composited pages are physically impossible documents.
- **Reproducibility:** every run sets `seed=42`; hyperparameters, GPU model, and wall-clock time are written to `runs/<name>/run_metadata.json` and surfaced in the README.
- **The LLM never receives the image.** Only the evidence JSON.
- **Guardrail is deterministic Python**, evaluated before any synthesis call, and its verdict cannot be overridden by the LLM.

---

## Execution order note

Tasks 1-5 are **time-critical**: they end with training launched on Kaggle, which runs ~3 h unattended. Tasks 6-13 are written and tested locally **while training runs**. Do not serialise them after training.

---

## File structure

| File | Responsibility |
|---|---|
| `app/constants.py` | Class names, id maps, model version string. Single source of truth. |
| `app/schemas.py` | Pydantic request/response models for both endpoints. |
| `app/detector.py` | RT-DETR load + inference wrapper. Knows nothing about reasoning. |
| `app/reasoning/router.py` | Regex prefilter + LLM intent routing. Returns a `RouteDecision`. |
| `app/reasoning/evidence.py` | Pure functions: detections -> structured evidence. No I/O. |
| `app/reasoning/guardrail.py` | Pure functions: evidence + route -> `GuardrailVerdict`. No I/O. |
| `app/reasoning/synthesize.py` | LLM answer synthesis from evidence JSON only. |
| `app/reasoning/pipeline.py` | Orchestrates the four stages. The only file that knows the order. |
| `app/logging_config.py` | Structured JSON logging with per-request latency. |
| `app/main.py` | FastAPI wiring, error handlers, `/health`, `/detect`, `/ask`. |
| `scripts/prepare_dataset.py` | HF DocLayNet-base -> YOLO format on disk. Dedupe + validation. |
| `scripts/train.py` | RT-DETR fine-tune, writes `run_metadata.json`. |
| `scripts/evaluate.py` | mAP, per-class, per-doc-category, confusion matrix, query saturation. |
| `scripts/mine_failures.py` | Rank test images by error, dump worst-N visualisations. |
| `notebooks/kaggle_train.ipynb` | Thin Kaggle wrapper that clones the repo and calls the scripts. |

---

### Task 1: Scaffolding and class constants

**Files:**
- Create: `requirements.txt`, `.gitignore`, `.env.example`, `app/constants.py`, `app/__init__.py`, `app/reasoning/__init__.py`
- Test: `tests/test_constants.py`

**Interfaces:**
- Produces: `CLASS_NAMES: list[str]` (11, fixed order), `CLASS_TO_ID: dict[str, int]`, `ID_TO_CLASS: dict[int, str]`, `MODEL_VERSION: str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_constants.py
from app.constants import CLASS_NAMES, CLASS_TO_ID, ID_TO_CLASS

def test_eleven_classes_in_fixed_alphabetical_order():
    assert CLASS_NAMES == [
        "Caption", "Footnote", "Formula", "List-item", "Page-footer",
        "Page-header", "Picture", "Section-header", "Table", "Text", "Title",
    ]

def test_no_class_is_a_coco_class():
    coco_sample = {"person", "car", "dog", "chair", "bottle", "tv", "book"}
    assert not {c.lower() for c in CLASS_NAMES} & coco_sample

def test_id_maps_are_inverses():
    assert all(ID_TO_CLASS[CLASS_TO_ID[c]] == c for c in CLASS_NAMES)
    assert CLASS_TO_ID["Caption"] == 0 and CLASS_TO_ID["Title"] == 10
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_constants.py -v`. Expected: `ModuleNotFoundError: app.constants`.
- [ ] **Step 3: Implement `app/constants.py`** — the list above, plus derived dicts and `MODEL_VERSION = "rtdetr-l-doclaynet-v1"`.
- [ ] **Step 4: Run test to verify it passes** — `pytest tests/test_constants.py -v`. Expected: 3 passed.
- [ ] **Step 5: Write `requirements.txt`** pinning: `ultralytics`, `torch`, `datasets`, `fastapi`, `uvicorn[standard]`, `pydantic>=2`, `openai`, `python-multipart`, `pillow`, `numpy`, `pytest`. **No LangChain/LangGraph/CrewAI/AutoGen.**
- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: project scaffolding and DocLayNet class constants"`

---

### Task 2: Dataset preparation — HF to YOLO

**Files:**
- Create: `scripts/prepare_dataset.py`
- Test: `tests/test_prepare_dataset.py`

**Interfaces:**
- Produces:
  - `coco_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]` — takes `[x, y, w, h]` in pixels, returns normalised `(xc, yc, w, h)`
  - `dedupe_annotations(bboxes: list[list[float]], categories: list[int]) -> list[tuple[tuple[float,...], int]]`
  - `is_valid_bbox(bbox: list[float], img_w: int, img_h: int) -> bool`

**Why this task matters:** `bboxes_block` repeats each block's box once per text line. Naive ingestion produces thousands of duplicate labels per page and will silently wreck training. This is the single highest-risk step in the build.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prepare_dataset.py
import pytest
from scripts.prepare_dataset import coco_to_yolo, dedupe_annotations, is_valid_bbox

def test_coco_to_yolo_centers_and_normalises():
    # box at x=0,y=0,w=512,h=512 on a 1024x1024 page -> center (0.25,0.25), size (0.5,0.5)
    assert coco_to_yolo([0, 0, 512, 512], 1024, 1024) == (0.25, 0.25, 0.5, 0.5)

def test_coco_to_yolo_full_page_box():
    assert coco_to_yolo([0, 0, 1025, 1025], 1025, 1025) == (0.5, 0.5, 1.0, 1.0)

def test_dedupe_collapses_line_level_repetition():
    # one text block repeated over 3 lines, then a distinct block
    bboxes = [[10, 10, 100, 50]] * 3 + [[10, 200, 100, 50]]
    cats = [9, 9, 9, 8]
    out = dedupe_annotations(bboxes, cats)
    assert len(out) == 2
    assert ((10.0, 10.0, 100.0, 50.0), 9) in out

def test_dedupe_keeps_same_box_with_different_category():
    out = dedupe_annotations([[1, 1, 5, 5], [1, 1, 5, 5]], [9, 10])
    assert len(out) == 2

def test_dedupe_preserves_first_seen_order():
    out = dedupe_annotations([[5, 5, 1, 1], [1, 1, 1, 1], [5, 5, 1, 1]], [1, 2, 1])
    assert [c for _, c in out] == [1, 2]

@pytest.mark.parametrize("bbox,expected", [
    ([0, 0, 100, 100], True),
    ([0, 0, 0, 100], False),      # zero width
    ([0, 0, 100, 0], False),      # zero height
    ([-5, 0, 100, 100], False),   # negative origin
    ([1000, 0, 100, 100], False), # exceeds right edge on a 1025 page
])
def test_is_valid_bbox(bbox, expected):
    assert is_valid_bbox(bbox, 1025, 1025) is expected
```

- [ ] **Step 2: Run tests to verify they fail** — `pytest tests/test_prepare_dataset.py -v`. Expected: import error.
- [ ] **Step 3: Implement the three pure functions** in `scripts/prepare_dataset.py`.
- [ ] **Step 4: Run tests to verify they pass** — `pytest tests/test_prepare_dataset.py -v`. Expected: 9 passed.
- [ ] **Step 5: Add the CLI driver** to the same file: `--split {train,validation,test} --out DIR --limit N`. It streams `load_dataset("pierreguillou/DocLayNet-base")`, writes `images/<split>/<id>.png` and `labels/<split>/<id>.txt`, and records per-split counts of images, kept annotations, deduped-away annotations, and dropped-invalid annotations into `<out>/prep_report.json`.
- [ ] **Step 6: Verify the category index base empirically** — add `--verify N` which renders N random pages with their decoded class names drawn on the boxes into `assets/label_check/`. **Open these images and confirm the labels are right before training.** The spec assumes 0-indexed alphabetical; this step proves or disproves it. Do not skip.
- [ ] **Step 7: Commit** — `git commit -m "feat: DocLayNet to YOLO conversion with line-level dedupe"`

---

### Task 3: Dataset config and integrity check

**Files:**
- Create: `configs/doclaynet.yaml`
- Modify: `scripts/prepare_dataset.py` (emit the yaml)

- [ ] **Step 1: Write `configs/doclaynet.yaml`** with `path`, `train: images/train`, `val: images/validation`, `test: images/test`, and `names:` mapping 0-10 to `CLASS_NAMES` in the fixed order.
- [ ] **Step 2: Sanity check** — confirm every `images/<split>/*.png` has a matching `labels/<split>/*.txt`, and that no label file is empty without being recorded in `prep_report.json`. Fail loudly on mismatch.
- [ ] **Step 3: Inspect `assets/label_check/` renders** — confirm a `Table` box actually surrounds a table. **Gate: do not proceed to Task 4 until this is visually confirmed.**
- [ ] **Step 4: Commit** — `git commit -m "feat: dataset config and integrity verification"`

---

### Task 4: Training script

**Files:**
- Create: `scripts/train.py`

**Interfaces:**
- Produces: weights at `runs/detect/<name>/weights/best.pt`, metadata at `runs/detect/<name>/run_metadata.json`

- [ ] **Step 1: Implement the training entrypoint** — `RTDETR("rtdetr-l.pt")`, `.train()` with: `data=configs/doclaynet.yaml`, `imgsz=640`, `epochs=35`, `batch=8`, `optimizer="AdamW"`, `lr0=1e-4`, `cos_lr=True`, `seed=42`, `amp=True`, `save_period=1`, `patience=10`, and the augmentation overrides: **`fliplr=0.0`, `flipud=0.0`, `mosaic=0.0`**, `degrees=3.0`, `perspective=0.0005`, `hsv_s=0.2`, `hsv_v=0.2`.
- [ ] **Step 2: Write `run_metadata.json`** capturing: all hyperparameters, `torch.cuda.get_device_name(0)`, total VRAM, wall-clock seconds, `ultralytics.__version__`, `torch.__version__`, git commit SHA, and the `prep_report.json` contents. This is what protects Part A from the 50% reproducibility cap.
- [ ] **Step 3: Smoke test locally** — run with `--epochs 1 --limit 20` on CPU to prove the script runs end to end before spending GPU quota. Expected: completes, writes weights and metadata.
- [ ] **Step 4: Commit** — `git commit -m "feat: RT-DETR training script with document-appropriate augmentation"`

---

### Task 5: Kaggle notebook and LAUNCH TRAINING

**Files:**
- Create: `notebooks/kaggle_train.ipynb`

- [ ] **Step 1: Write the notebook** — cells that: enable internet, `pip install -q ultralytics datasets`, clone the repo, run `prepare_dataset.py` for all three splits, run the `--verify` render, then run `train.py`.
- [ ] **Step 2: Confirm GPU** — `!nvidia-smi` must show a Tesla T4. Record the exact model for `run_metadata.json`.
- [ ] **Step 3: Run the label-check cell and eyeball the renders.** Gate.
- [ ] **Step 4: LAUNCH TRAINING.** Use "Save Version -> Save & Run All (Commit)" so the session survives browser disconnect.
- [ ] **Step 5: Immediately proceed to Task 6 while it trains.** Do not wait.
- [ ] **Step 6: Commit** — `git commit -m "feat: Kaggle training notebook"`

---

### Task 6: Detector wrapper and schemas

**Files:**
- Create: `app/detector.py`, `app/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces:
  - `Detection` (pydantic): `class_name: str`, `class_id: int`, `bbox: BBox`, `confidence: float`
  - `BBox` (pydantic): `x1, y1, x2, y2: float`
  - `Detector.predict(image: PIL.Image, conf: float = 0.25) -> tuple[list[Detection], float]` returning detections and inference milliseconds
  - `Detector.is_loaded: bool`, `Detector.query_budget: int`

- [ ] **Step 1: Write the failing test** — assert `BBox` rejects `x2 < x1`, that `Detection.confidence` is constrained to `[0, 1]`, and that `Detection` serialises to the exact JSON shape in spec §5.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `app/schemas.py` then `app/detector.py`.** The detector lazy-loads weights from `MODEL_PATH` env var, exposes `is_loaded` without raising, and converts Ultralytics results into `Detection` objects using `ID_TO_CLASS`.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `git commit -m "feat: detector wrapper and API schemas"`

---

### Task 7: Evidence builder (pure, deterministic)

**Files:**
- Create: `app/reasoning/evidence.py`
- Test: `tests/test_evidence.py`

**Interfaces:**
- Produces: `build_evidence(detections: list[Detection], image_w: int, image_h: int) -> Evidence` where `Evidence` carries `class_counts: dict[str, int]`, `confidence_stats: dict[str, ConfStats]` (`max`, `mean`, `min`), `total_detections: int`, `reading_order: list[str]`, `relations: list[Relation]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evidence.py
from app.reasoning.evidence import build_evidence
from app.schemas import Detection, BBox

def d(name, cid, conf, x1=0, y1=0, x2=10, y2=10):
    return Detection(class_name=name, class_id=cid, confidence=conf,
                     bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2))

def test_counts_by_class():
    ev = build_evidence([d("Table", 8, .9), d("Table", 8, .8), d("Text", 9, .7)], 1025, 1025)
    assert ev.class_counts == {"Table": 2, "Text": 1}

def test_confidence_stats_per_class():
    ev = build_evidence([d("Table", 8, .9), d("Table", 8, .5)], 1025, 1025)
    assert ev.confidence_stats["Table"].max == 0.9
    assert ev.confidence_stats["Table"].mean == 0.7
    assert ev.confidence_stats["Table"].min == 0.5

def test_empty_detections_yield_empty_evidence_not_crash():
    ev = build_evidence([], 1025, 1025)
    assert ev.class_counts == {} and ev.total_detections == 0

def test_reading_order_is_top_to_bottom_then_left_to_right():
    lower = d("Text", 9, .9, y1=500, y2=600)
    upper_right = d("Table", 8, .9, x1=600, x2=700, y1=100, y2=200)
    upper_left = d("Title", 10, .9, x1=0, x2=100, y1=100, y2=200)
    ev = build_evidence([lower, upper_right, upper_left], 1025, 1025)
    assert ev.reading_order == ["Title", "Table", "Text"]

def test_containment_relation_detected():
    outer = d("Table", 8, .9, x1=0, y1=0, x2=500, y2=500)
    inner = d("Caption", 0, .9, x1=10, y1=10, x2=100, y2=50)
    ev = build_evidence([outer, inner], 1025, 1025)
    assert any(r.kind == "contains" and r.subject == "Table" and r.object == "Caption"
               for r in ev.relations)
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `evidence.py`.** Pure functions only — no I/O, no network, no model access.
- [ ] **Step 4: Run to verify they pass** — 5 passed.
- [ ] **Step 5: Commit** — `git commit -m "feat: deterministic evidence builder"`

---

### Task 8: Confidence guardrail (pure, deterministic)

**Files:**
- Create: `app/reasoning/guardrail.py`
- Test: `tests/test_guardrail.py`

**Interfaces:**
- Consumes: `Evidence` from Task 7, `RouteDecision` from Task 9
- Produces: `evaluate_guardrail(evidence: Evidence, target_classes: list[str], task_type: str, saturated: bool = False) -> GuardrailVerdict` where `GuardrailVerdict` carries `triggered: bool`, `rule: str | None`, `detail: dict`

Rules, in evaluation order: `no_detections` -> `all_below_tau` (tau = 0.50) -> `ambiguous_band` (>30% of target detections in [0.25, 0.50) and `task_type == "count"`) -> `query_saturated`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_guardrail.py
from app.reasoning.guardrail import evaluate_guardrail
from app.reasoning.evidence import build_evidence
from tests.test_evidence import d

def ev(dets): return build_evidence(dets, 1025, 1025)

def test_no_detections_of_target_class_triggers():
    v = evaluate_guardrail(ev([d("Text", 9, .9)]), ["Table"], "count")
    assert v.triggered and v.rule == "no_detections"

def test_all_below_tau_triggers():
    v = evaluate_guardrail(ev([d("Table", 8, .31), d("Table", 8, .44)]), ["Table"], "count")
    assert v.triggered and v.rule == "all_below_tau"
    assert v.detail["observed_max"] == 0.44

def test_confident_detection_does_not_trigger():
    v = evaluate_guardrail(ev([d("Table", 8, .94)]), ["Table"], "count")
    assert not v.triggered and v.rule is None

def test_ambiguous_band_triggers_only_for_counting():
    dets = [d("Table", 8, .95), d("Table", 8, .30), d("Table", 8, .35)]
    assert evaluate_guardrail(ev(dets), ["Table"], "count").rule == "ambiguous_band"
    assert not evaluate_guardrail(ev(dets), ["Table"], "presence").triggered

def test_query_saturation_triggers():
    v = evaluate_guardrail(ev([d("Text", 9, .99)]), ["Text"], "count", saturated=True)
    assert v.triggered and v.rule == "query_saturated"

def test_rule_precedence_no_detections_beats_saturation():
    v = evaluate_guardrail(ev([]), ["Table"], "count", saturated=True)
    assert v.rule == "no_detections"
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `guardrail.py`.** Pure. No LLM call may appear in this file.
- [ ] **Step 4: Run to verify they pass** — 6 passed.
- [ ] **Step 5: Commit** — `git commit -m "feat: deterministic confidence guardrail"`

---

### Task 9: Intent router

**Files:**
- Create: `app/reasoning/router.py`
- Test: `tests/test_router.py`

**Interfaces:**
- Produces: `route(question: str, client) -> RouteDecision` with fields `needs_detection: bool`, `target_classes: list[str]`, `task_type: Literal["count","presence","compare","describe","out_of_scope"]`, `reason: str`, `source: Literal["prefilter","llm"]`
- Also produces: `prefilter(question: str) -> RouteDecision | None` — pure, no network

- [ ] **Step 1: Write the failing tests** for `prefilter` only (pure, no network):

```python
# tests/test_router.py
from app.reasoning.router import prefilter

def test_prefilter_short_circuits_obvious_non_visual_question():
    r = prefilter("What is the capital of France?")
    assert r is not None and r.needs_detection is False
    assert r.task_type == "out_of_scope" and r.source == "prefilter"

def test_prefilter_defers_visual_questions_to_the_llm():
    assert prefilter("How many tables are on this page?") is None

def test_prefilter_defers_ambiguous_questions_to_the_llm():
    assert prefilter("Is this page complicated?") is None
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `prefilter` then `route`.** `route` makes one `openai` call with `response_format={"type": "json_schema"}` and a strict schema, injecting `CLASS_NAMES` into the prompt so the model knows the perception vocabulary. **The prompt must state that questions about text *content* (amounts, names, dates, signatures) are `out_of_scope`, because the detector returns layout regions only.** On LLM error or timeout, fail closed: return `needs_detection=False, task_type="out_of_scope"` with the error in `reason`.
- [ ] **Step 4: Run to verify they pass** — 3 passed.
- [ ] **Step 5: Manual check with a real key** — verify `"What is the invoice total?"` routes to `out_of_scope` and `"How many tables?"` routes to `count` with `target_classes=["Table"]`. Record both for the memo.
- [ ] **Step 6: Commit** — `git commit -m "feat: hand-written intent router with fail-closed behaviour"`

---

### Task 10: Answer synthesis and pipeline

**Files:**
- Create: `app/reasoning/synthesize.py`, `app/reasoning/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `synthesize(question, evidence, verdict, route, client) -> str`
- Produces: `answer_question(image, question, detector, client) -> AskResponse`

- [ ] **Step 1: Write the failing test** using a stub LLM client and a stub detector, asserting that (a) when the guardrail triggers, `AskResponse.insufficient_information is True` **regardless of what the stub LLM returns**, and (b) when `needs_detection is False`, the detector's `predict` is never called.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `synthesize.py`.** It receives the evidence JSON and the guardrail verdict, and **never the image**. When `verdict.triggered`, the system prompt instructs the model to state plainly that it cannot answer confidently and why — and the pipeline sets `insufficient_information=True` in code, not from the LLM's output.
- [ ] **Step 4: Implement `pipeline.py`** wiring prefilter -> route -> detect -> evidence -> guardrail -> synthesize, populating `reasoning_trace`.
- [ ] **Step 5: Run to verify it passes.**
- [ ] **Step 6: Commit** — `git commit -m "feat: reasoning pipeline with code-enforced insufficiency flag"`

---

### Task 11: FastAPI app, logging, error handling

**Files:**
- Create: `app/main.py`, `app/logging_config.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests** using `TestClient` with a stubbed detector: `/health` returns 200 with `model_loaded`; `/detect` with a non-image returns 415; `/detect` with an oversized file returns 413; `/detect` with a valid image returns the spec §5 shape; `/ask` with a missing question returns 422.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `logging_config.py`** — structured JSON logs with request id, endpoint, latency ms, detection count, guardrail rule.
- [ ] **Step 4: Implement `main.py`** with both endpoints, a 10 MB upload cap, MIME validation, and exception handlers for model-not-loaded (503) and LLM timeout (504).
- [ ] **Step 5: Run to verify they pass.**
- [ ] **Step 6: Commit** — `git commit -m "feat: FastAPI app with structured logging and error handling"`

---

### Task 12: Evaluation and failure mining

**Files:**
- Create: `scripts/evaluate.py`, `scripts/mine_failures.py`

- [ ] **Step 1: Implement `evaluate.py`** — runs `model.val(split="test")`, and writes `reports/metrics.json` with overall mAP@50 / mAP@50-95, per-class AP/P/R, the confusion matrix, **per-`doc_category` mAP**, and the **query-saturation rate** (fraction of test pages whose ground-truth region count exceeds the model's query budget).
- [ ] **Step 2: Implement `mine_failures.py`** — per-image error score (unmatched GT + unmatched predictions at IoU 0.5), ranked descending, dumping the worst 25 as side-by-side GT/prediction renders into `reports/failures/`.
- [ ] **Step 3: Run both against the trained weights** once training completes.
- [ ] **Step 4: Select the five memo failure cases from `reports/failures/`** — evidence-driven, not invented. Target a spread of distinct root causes: small/thin class, dense-page saturation, semantic class confusion, annotation ambiguity, and one localisation failure.
- [ ] **Step 5: Commit** — `git commit -m "feat: evaluation and evidence-driven failure mining"`

---

### Task 13: Docker, README, memo

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `README.md`, `memo/MEMO.md`

- [ ] **Step 1: Write the `Dockerfile`** — `python:3.11-slim`, CPU torch wheel, non-root user, `HEALTHCHECK` hitting `/health`, `uvicorn` entrypoint.
- [ ] **Step 2: Build and run it** — `docker build -t rap-doclayout . && docker run -p 8000:8000 -e OPENAI_API_KEY=... rap-doclayout`, then curl both endpoints. **Paste the real responses into the README — do not invent sample payloads.**
- [ ] **Step 3: Write `README.md`** — setup, exact reproduction commands, hardware used, training time, hyperparameters, weights download path, and verified `curl` examples with real request/response payloads for both endpoints.
- [ ] **Step 4: Write `memo/MEMO.md`, max 2 pages**, covering all five required items from the brief. Source its content from the spec, `reports/metrics.json`, and `reports/failures/`.
- [ ] **Step 5: Dependency audit** — `grep -riE "langchain|langgraph|crewai|autogen" .` must return nothing outside documentation prose. Record the result.
- [ ] **Step 6: Commit and push** — `git commit -m "feat: docker packaging, README, and submission memo"`

---

## Self-review

**Spec coverage:** §1 -> Tasks 1-3. §2 -> Tasks 2-3. §3 -> Tasks 4-5. §4 -> Task 12. §5 -> Tasks 6, 11. §6 -> Tasks 7-10. §7 -> Tasks 4 (augmentation), 13 (declared in README/memo). §8 -> Tasks 11, 13. §9 -> Task 13. §10 -> Task 13 step 5. No gaps.

**Placeholder scan:** No TBDs. Every "add error handling" instance names the specific status codes and conditions.

**Type consistency:** `Detection`/`BBox` (Task 6) are consumed by `build_evidence` (Task 7). `Evidence` (Task 7) is consumed by `evaluate_guardrail` (Task 8). `RouteDecision` (Task 9) is consumed by Tasks 8 and 10. `GuardrailVerdict.rule` string literals match across Tasks 8 and 10. Consistent.

**Known risk:** Task 2 step 6 (empirical category-index verification) is the gate that protects the whole build. If the index base is wrong, every downstream metric is garbage and the error is invisible until the renders are inspected.
