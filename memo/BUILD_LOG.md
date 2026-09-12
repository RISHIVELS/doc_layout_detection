# Build log — Constrained Object Detection & Reasoning API

This is my working journal for the RAP pre-hackathon screening. I kept it as I
went rather than writing it at the end, because the brief says mid-project
pivots are useful signal and I wanted an honest record of what I actually tried
instead of a tidied-up story invented afterwards.

The formal 2-page memo (`MEMO.md`) is distilled from this. This file is the
long version: every decision, what I believed at the time, and what changed my
mind.

---

## Day 0 — Reading the brief properly before touching any code

The first thing I did was work out what is actually being graded, because the
weighting is not what you would assume from the title.

| Component | Weight |
|---|---|
| Hidden evaluation set (Part A) | 25% |
| Dataset sourcing, labelling, justification | 15% |
| Evaluation methodology and honesty | 10% |
| Failure-case analysis | 15% |
| Part B reasoning + "insufficient info" handling | 15% |
| Code quality, reproducibility, API | 10% |
| Bonus: Docker, deployment, logging | 10% |

Only 25% is model performance. Around 40% is written reasoning about my own
work. And the brief says outright that 78% mAP with three well-reasoned failure
cases beats 95% mAP with none.

That changed how I approached the whole thing. I stopped optimising for a big
number and started optimising for **a system whose limits I could describe
precisely**. Every decision below follows from that.

The other thing I pulled out of the brief was a set of hard gates that fail the
submission outright rather than costing points:

- No agentic frameworks anywhere, Part B included.
- Training must be my own code, using RT-DETR.
- At least one non-COCO class, or Part A scores **zero**.
- If they cannot reproduce my training run, Part A caps at **50%**.

I treated those four as non-negotiable design constraints rather than
guidelines, and there is a compliance table at the end of this log.

---

## Choosing the domain — three ideas, two rejected

### First idea: construction-site PPE compliance

This was my first instinct, and honestly it was partly because the brief's own
Part B examples mention helmets ("Is anyone not wearing a helmet?"). Classes
would have been person / helmet / head / vest, and the reasoning layer would do
person-to-PPE association.

I dropped it for two reasons.

The first was practical: I found it had already been taken by another
candidate, and the brief explicitly flags submissions that share dataset choice
and approach.

The second reason is the one I would actually defend, and I only saw it after
thinking about the non-COCO constraint properly. `person` **is** a COCO class.
So is most of what a PPE model detects around it. That means a meaningful
portion of the model's performance is inherited from the pretrained backbone
rather than earned by my fine-tuning. The brief wants a non-COCO class
specifically so that I cannot lean on pretrained weights — satisfying it with
one novel class bolted onto four familiar ones felt like meeting the letter of
the constraint and not the intent.

### Second idea: traffic / road scenes

Rejected faster, and for a sharper version of the same problem. Car, truck,
bus, motorcycle, person, traffic light — every one of those is COCO. I would
have been fine-tuning a model to do something it substantially already does. It
is also the single most common choice for any object detection exercise, so it
would have put me in the middle of a crowd.

### What I went with: document layout analysis

The reframe was realising that "object detection" does not have to mean
photographs of the physical world. A document page is an image, and the regions
on it — tables, figures, headings, captions, formulas — are objects with
positions and extents.

Four reasons I committed to it:

**1. The non-COCO constraint stops being an argument.** All eleven DocLayNet
classes are absent from COCO. There is no pretrained checkpoint on earth that
emits `Section-header` or `Formula`. Every point of mAP I report had to come
from my fine-tuning. I cannot accidentally score well here.

**2. It makes Part B honest instead of theatrical.** This is the reason I care
about most. In a traffic or wildlife model, the only way to trigger an
"insufficient information" response is to contrive a low-confidence case — you
are really just demonstrating that you can compare a float to a threshold. With
documents I get a structural version for free:

> *"What is the invoice total?"*

My detector knows exactly **where** the table is. It has no idea **what it
says**. The refusal is not a tuned threshold firing, it is the system
recognising the boundary of its own perception vocabulary. That is a genuinely
different and better answer to the brief's requirement for "one specific
example where it correctly outputs insufficient information", and it falls out
of the domain rather than being engineered for the demo.

**3. The failure cases have actual intellectual content.** `Text` vs
`List-item`, `Title` vs `Section-header`, `Caption` vs `Text` — these are
confusions where the pixels genuinely underdetermine the label and trained
human annotators disagree with each other. That gives me failure analysis about
annotation-protocol ambiguity rather than "the object was small and blurry".

**4. It is adjacent to what RAP actually builds.** I looked this up rather than
assume. RAP's RAPFlow is a content intelligence platform whose stated purpose is
automated data extraction from unstructured documents and forms, and their
listed capabilities include document understanding and form recognition. Layout
detection is the perception stage that has to run before any of that works —
without it a table collapses into a flat stream of characters and you cannot
tell a page footer from a contract clause.

To be precise about the claim: I am saying the **problem domain** is adjacent to
their product. I have no knowledge of their internal stack and I am not claiming
they use RT-DETR or DocLayNet.

### The risk I accepted, stated up front

DocLayNet is clean rendered PDF pages. If the hidden evaluation set contains
**phone photographs of paper documents** — perspective skew, shadows, page curl,
glare — my model will do badly, because it has never seen an image like that.
A traffic model does not face this cliff; a photo is a photo.

I decided this was worth it, and I mitigated rather than ignored it:

- Perspective and mild rotation augmentation during training.
- The operating domain is declared explicitly in the README and the memo, so
  it is a stated limitation rather than something the reviewers discover.
- The API's guardrail degrades honestly — uniformly low confidence produces a
  refusal, not a confident wrong answer.

---

## Choosing the dataset

I wanted DocLayNet from the start: 80,863 pages, hand-annotated by trained
experts at IBM Research, spanning six genuinely different document categories
(financial reports, scientific articles, laws and regulations, government
tenders, manuals, patents). Permissive licence (CDLA-Permissive-1.0).

**First attempt: the full dataset.** This does not survive contact with my
constraints. The core release is 28 GiB, plus a 7.5 GiB extras archive, and the
dataset card itself notes it takes around 45 minutes just to download on a
hosted notebook. With roughly one working day, spending an hour on a download
before writing a line of training code was not defensible.

**What I switched to:** `pierreguillou/DocLayNet-base` on Hugging Face — a
pre-processed 10% subset of the same expert annotations.

| | |
|---|---|
| Images | 8,057 (6,910 train / 648 val / 499 test) |
| Download | 3.8 GB |
| Image size | 1025 x 1025 PNG, uniform |
| Annotations | COCO-style `[x, y, w, h]` |
| Licence | CDLA-Permissive-1.0 |

Same annotation quality, same six document categories, ~8x smaller. I verified
these numbers on the dataset page rather than taking them from memory, because
the whole build plan depended on the download being small enough.

### The split

The subset ships with the author's own train/val/test split and I used it as my
primary result, for two reasons: it is reproducible by anyone who runs one line
of code, and it keeps my numbers comparable to published DocLayNet work rather
than to a split only I have.

I originally planned to re-stratify it myself by document category. I changed
my mind once I saw the splits already exist — re-rolling would have cost an
hour and made my numbers incomparable to everything else in the literature,
which is a bad trade for a cosmetic improvement.

What I am doing instead is more useful: **reporting mAP broken down per
document category** on top of the aggregate. That answers the question a single
number cannot — did the model learn document structure, or did it learn what
annual reports look like? Financial reports are the largest slice of DocLayNet,
so a model that is strong there and weak on patents would still post a
respectable headline mAP. Splitting the metric out is the only way to see it.

I also check page-level leakage explicitly. DocLayNet pages come from multi-page
source PDFs, so if a random split put pages from the same PDF on both sides of
the train/test boundary, my aggregate mAP is optimistic. I measure the overlap
using `original_filename` and report what I find rather than assuming it is
zero.

---

## Choosing the model

RT-DETR is mandated by the brief, so the interesting question is not *whether*
but *why it happens to suit this task* — which is what I would be asked to
defend verbally.

The short history: anchor-based detectors like YOLO blanket the image with
candidate boxes and then use NMS, a hand-tuned post-processing heuristic, to
delete duplicates. DETR replaced that with direct set prediction — a fixed set
of learned object queries, trained with Hungarian bipartite matching so
duplicates are penalised during training instead of cleaned up afterwards. That
removed anchors and NMS entirely but was far too slow. RT-DETR keeps the
end-to-end formulation and makes it fast, mainly by decoupling intra-scale
attention from cross-scale fusion in the encoder, and by seeding decoder
queries from high-quality encoder proposals instead of random initialisation.

Two properties of that design genuinely help me here:

**Global attention matches how layout works.** Whether a block is a `Caption`
depends on whether there is a `Picture` immediately above it. A
`Section-header` is defined largely by what follows it. These are long-range
relationships across the page, and a CNN's local receptive field models them
poorly. Self-attention over the whole page is the right shape of prior.

**No NMS is a real advantage on dense pages.** Text blocks in a two-column
layout sit tightly adjacent and their boxes overlap more than you would expect.
With NMS, an IoU threshold that is even slightly too aggressive suppresses a
legitimate neighbouring block, and I would be tuning that threshold by hand
against my validation set. Set prediction sidesteps the whole problem.

And the weaknesses I expect to show up in my results, written down before I
trained so I cannot claim afterwards that I knew:

- Transformer detectors are weaker on small objects than CNNs. `Footnote` and
  `Page-footer` are small and thin. I expect them to be my worst classes.
- The fixed query budget is a hard ceiling on dense pages (see
  `app/constants.py`).
- DETR-family models converge more slowly than YOLO, which is an awkward
  property to have on a one-day budget.

---

## Compute

I planned this for Colab initially and switched to **Kaggle** before starting.
Kaggle's free tier gives 12-hour sessions and a 30 h/week GPU quota, against
Colab's aggressive idle disconnects. On a single-day build, losing a training
run at hour two to a browser disconnect is the failure mode that actually ends
the project, so the longer, more predictable session was worth more than
anything else on offer.

**GPU: Kaggle T4 (16 GB).** Hyperparameters below are sized for it.

---

## Code decisions

### Why the class vocabulary lives in one file

My first pass had the class list written out three times — in the dataset prep
script, in the training YAML, and in the API response builder. The failure mode
that made me consolidate it is nasty: if those drift, the label ids are still
valid integers, training still runs, and the metrics still look plausible. The
model just quietly learns the wrong thing and nothing ever raises an error.
So `app/constants.py` is the only place the list exists and everything imports
from it. `tests/test_constants.py` pins the ordering.

---

*(log continues as the build progresses)*

### Dataset conversion — the part that nearly caught me out

I expected this to be a twenty-minute job: read the Hugging Face rows, convert
the boxes, write the files. It was not, and the reason is worth recording
because it is the kind of thing that decides whether a submission is real.

**What I found when I looked at the schema.** DocLayNet-base stores
`bboxes_block` aligned to *text lines*, not to regions. A paragraph spanning six
lines appears as the same block box repeated six times, each with the same
category. One sample row had over a thousand entries in `bboxes_block` for a page
that plainly contains a few dozen actual regions.

If I had written those out as labels directly — which is exactly what a
straightforward loop over the dataset does — I would have produced a training set
where the average page carries a thousand overlapping annotations. Nothing about
that crashes. The loss still goes down. The model would simply have learned a
completely wrong prior about how many objects a page contains, and I would have
spent the evening wondering why my precision was strange.

So `dedupe_annotations()` collapses on the `(box, category)` pair. I dedupe on
the pair rather than the box alone because occasionally two classes are labelled
over the same extent, and dropping one of those would be discarding real signal.
The prep report prints what fraction of the raw annotations were repeats, because
"I threw away most of the annotations" is a claim that needs a number next to it.

**The three things I tested before writing the driver.** I do not normally write
tests for a one-off conversion script. I did here because every failure mode in
this file is silent:

- *Coordinate conversion.* DocLayNet uses top-left-corner `[x, y, w, h]`;
  Ultralytics wants normalised centre coordinates. Forget the corner-to-centre
  shift and every box lands half its own size up and to the left — which reads as
  "the model localises a bit imprecisely", not as a bug.
- *Deduplication.* Described above.
- *Degenerate boxes.* Zero-area labels are accepted by Ultralytics and then
  produce NaN losses several epochs in, long after I have stopped watching the
  console.

**The check no test can do.** My class ordering is 0-indexed alphabetical, which
I inferred from the data rather than found stated unambiguously. If it is off by
one, `Table` becomes `Section-header` everywhere. The dataset stays perfectly
self-consistent, training succeeds, and every metric and failure-case analysis I
write afterwards describes a model that learned something other than what I say
it learned.

No unit test catches that, because the data is internally consistent under either
assumption. The only check that works is rendering pages with the decoded names
drawn on the boxes and looking at them. That is what `--verify` does, and running
it is a hard gate before training in my notebook. It felt paranoid to build; it is
the cheapest insurance in the project.

**One small design choice:** the script generates `doclaynet.yaml` itself instead
of me hand-writing it, so the class names in the training config physically
cannot drift from `app/constants.py`. That is the same drift bug I consolidated
the class list to avoid, and hand-writing the YAML would have reintroduced it
through the back door.

---

## Training setup — the augmentation defaults were wrong for me

Most of the training script is unremarkable: RT-DETR-L fine-tuned from COCO
weights, AdamW, cosine schedule, 30 epochs, batch 8 at 640px on the T4. Two
parts are worth explaining because they are decisions rather than defaults.

### Three augmentations I had to turn off

Ultralytics ships sensible augmentation defaults — sensible for photographs.
I went through them asking what each transform physically means when the image
is a document page, and three of them are actively harmful:

**`fliplr` defaults to 0.5.** That mirrors half of my training images. For
street scenes this is free data, because a car facing left is as valid as one
facing right. A document is not symmetric: text runs left to right, page numbers
sit in particular corners, indentation means something. A mirrored page is a
layout that cannot physically exist, and leaving this on means spending half my
training signal teaching the model that it can. Set to 0.

**`flipud`** — the same argument, just more obvious. An upside-down page.

**`mosaic` defaults on**, compositing four training images into one. For object
detection in natural scenes this is a genuinely clever trick for varying scale
and context. For documents it produces a collage of four quarter-pages, which is
not a document, and it specifically destroys whole-page spatial structure —
header at the top, footer at the bottom — which is exactly the prior I want the
model to learn. Off.

None of these would have thrown an error. I would simply have got a worse model
and had no idea which of a dozen things was responsible.

What I kept is small-angle rotation (3°) and mild perspective. Those map to real
things that happen to real documents — a page fed slightly crooked through a
scanner, or photographed at a slight angle. This is also my only hedge against
the hidden evaluation set containing camera-captured pages, which is the main
risk I knowingly took when I chose this domain.

### Learning rate

1e-4, not the 1e-2 that YOLO-style configs use. DETR-family models have
transformer components that are unstable at high learning rates, and I am
fine-tuning from COCO weights rather than training from scratch, so I want to
move the existing weights gently rather than blow them away in the first epoch.

### What starting from COCO weights actually buys me

Worth being precise about, because it interacts with the non-COCO requirement.
None of my eleven classes exist in COCO, so the detection head is effectively
relearned from nothing — I inherit zero class knowledge. What the pretrained
backbone gives me is generic visual features: edges, texture, the notion of a
coherent region. On 6,910 training images in one day, training a transformer
detector from random initialisation would not converge to anything useful, so
this is the difference between having a model and not having one.

### Resolution, and the compromise I am making

I am training at 640px on 1025px source pages. I would rather have used 800 or
1024, because thin classes — `Footnote`, `Page-footer` — lose a lot of detail
when the page is squeezed down that far, and I expect that to show up directly
in their per-class AP.

I am doing it anyway because the time budget forces it, and I would rather
report the limitation honestly than run out of session. I am noting the
prediction here, before training, so that when those classes come out worst I
can point at this paragraph rather than claim after the fact that I expected it.

---

## Switching the reasoning layer to Groq

I had planned this around OpenAI and moved to Groq. Two reasons, one practical
and one that turned out to matter more than I expected.

**The practical one:** Groq's free tier is genuinely usable, and inference is
fast enough that the two-call structure of my reasoning layer (route, then
synthesise) does not add noticeable latency to an API request. With a slower
provider I would have been tempted to collapse the two calls into one, which
would have been the wrong call architecturally — the whole point is that routing
and answering are separate decisions.

**The one I did not anticipate:** Groq supports `json_schema` with
`strict: true`, which uses constrained decoding rather than prompting. The
model is *structurally unable* to emit output that does not match my schema.

That matters for the router specifically. My original plan was to ask for JSON
in the prompt, parse it, and write retry logic for when the model returned prose
or a malformed object — that is the usual dance. With constrained decoding, the
router's output being well-formed stops being something I hope for and becomes
something the decoder guarantees. I still validate the parsed object in code,
because "well-formed" and "sensible" are different claims and I only get the
first one for free.

**One deprecation I had to work around.** My first choice was
`llama-3.3-70b-versatile`, which is the obvious Groq default. Checking the docs
rather than assuming, I found it was deprecated for free and developer tier in
June 2026, with `openai/gpt-oss-120b` given as the migration path. So I am on
the gpt-oss models. Worth recording because it is exactly the kind of thing that
silently breaks a submission a reviewer tries to run three weeks later.

**Why two different models.** The router and the synthesiser are doing different
jobs and I sized them differently:

- *Routing* is classification into a fixed schema. Small, fast model
  (`gpt-oss-20b`), pinned with strict structured output.
- *Synthesis* is open-ended writing that has to respect a guardrail verdict.
  Larger model (`gpt-oss-120b`), where the extra capability earns its latency.

**On the no-frameworks constraint:** the `groq` package is a thin HTTP client
over their chat-completions endpoint. It does not chain, plan, retry-with-tools,
or orchestrate anything. Every decision about what to call, when to call it, and
what to do with the result lives in `app/reasoning/` and is code I wrote. I did
a dependency grep for LangChain, LangGraph, CrewAI and AutoGen as a final check
and recorded the result in the README.

---

## First real failure on Kaggle — an unpinned dependency

The first thing that actually broke wasn't my code, it was a version I hadn't
pinned tightly enough. Worth recording in full because it is precisely the
"reproducibility" risk the brief warns about, and it happened to me within
minutes of running the notebook for real.

`requirements.txt` had `datasets>=2.19` — a floor, no ceiling. Kaggle's
environment installed the latest release, which turned out to be a 4.x version.
Hugging Face removed support for legacy "loading script" dataset repos entirely
in `datasets` 4.0.0, and `pierreguillou/DocLayNet-base` is exactly that kind of
repo. Result:

```
RuntimeError: Dataset scripts are no longer supported, but found DocLayNet-base.py
```

Nothing about my conversion logic was wrong. The tool I was calling into had
changed its supported input format between when I tested locally and when
Kaggle resolved the package.

**Fix:** pinned `datasets>=2.19,<4.0.0` in `requirements.txt` and in the
notebook's install cell, and wrapped the `load_dataset` call in
`prepare_dataset.py` to catch this specific `RuntimeError` and explain what
happened and how to fix it, rather than let whoever hits it next (possibly a
reviewer trying to reproduce my run six weeks from now) decode a bare stack
trace from inside the `datasets` internals.

The honest lesson: an unpinned floor-only dependency is a real reproducibility
gap, not a theoretical one. It cost me a run failure inside the first few
minutes of using the one GPU session I had.

---

## Second Kaggle snag — an interactive prompt with nothing to answer it

Right after the datasets<4.0.0 fix, the next run got as far as actually loading
the dataset and then stopped on:

```
The repository for pierreguillou/DocLayNet-base contains custom code which
must be executed to correctly load the dataset...
Do you wish to run the custom code? [y/N]
```

This is `datasets` asking permission to execute the dataset repo's own loading
script - a sensible default, since blindly running someone else's code off the
Hub is not something a library should do silently.

I answered it interactively to keep checking labels, but the real fix has to
be in code, because my actual training run does not happen interactively: the
plan is Save & Run All (Commit) specifically so the session survives me
closing the browser, per Task 5. A commit run has no terminal on the other end
to type "y" into - it would sit at that prompt until the session timed out,
and I would come back hours later to a run that never started.

Fix: pass `trust_remote_code=True` explicitly in `prepare_dataset.py`. I did
look at what I was agreeing to run before adding it - the script is
`pierreguillou`'s own conversion of IBM's DocLayNet into the `datasets`
library's structure, nothing else.

Two real dependency snags inside the first ten minutes of actually running
this on Kaggle, both invisible until I hit real infrastructure rather than my
own laptop. I am glad I budgeted slack for exactly this.

---

## Third Kaggle snag — rm -rf on the shell's own current directory

```
shell-init: error retrieving current directory: getcwd: cannot access parent
directories: No such file or directory
fatal: Unable to read current working directory
```

Kaggle-specific, not a repo problem. My clone cell did `%cd /kaggle/working/repo`
at the end, so re-running the same cell later (which happens naturally every
time I push a fix and need the latest code) executed `rm -rf
/kaggle/working/repo` while the notebook's shell was sitting inside that exact
directory. The process's cached working directory pointed at a path that had
just been deleted, and every subsequent `!` command inherited the broken state.

Fix: `%cd /kaggle/working` before the `rm -rf`, so the cell always steps out to
a stable parent directory first. Small, but it is the difference between the
clone cell being safely re-runnable and it corrupting the session on the second
run - and I am re-running it after almost every fix in this log, so it needed
to be safe to repeat.
