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
