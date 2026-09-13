# Builds a visual PDF report from reports/metrics.json and
# reports/run_metadata.json - same numbers as the memo, laid out so
# someone can understand the results without reading raw JSON.
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "text.color": "#0b0b0b",
    "axes.edgecolor": "#0b0b0b",
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#0b0b0b",
    "ytick.color": "#0b0b0b",
})

BLUE = "#2a78d6"       # validated single hue, see dataviz skill palette
GRID = "#d9d9d6"       # recessive gridline
MUTED = "#52514e"      # secondary text

ROOT = Path(__file__).resolve().parent.parent
metrics = json.loads((ROOT / "reports/metrics.json").read_text())
run = json.loads((ROOT / "reports/run_metadata.json").read_text())

CLASS_INSTANCES = {  # from the evaluate.py console output for this run
    "Caption": 149, "Footnote": 47, "Formula": 150, "List-item": 962,
    "Page-footer": 408, "Page-header": 311, "Picture": 143,
    "Section-header": 873, "Table": 252, "Text": 3002, "Title": 52,
}


def new_page(figsize=(8.5, 11)):
    fig = plt.figure(figsize=figsize, facecolor="white")
    fig.patch.set_facecolor("white")
    return fig


def bare_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def page_overview():
    fig = new_page()
    ax = bare_axes(fig, [0, 0, 1, 1])

    ax.text(0.07, 0.94, "Document Layout Detection — Evaluation Report",
            fontsize=20, fontweight="bold", ha="left")
    ax.text(0.07, 0.905, "RT-DETR-L fine-tuned on DocLayNet, evaluated on the held-out test split",
            fontsize=12, color=MUTED, ha="left")
    ax.axhline(0.885, xmin=0.07, xmax=0.93, color="#0b0b0b", linewidth=1)

    overall = metrics["overall"]
    stats = [
        ("mAP50", f"{overall['mAP50']:.3f}"),
        ("mAP50-95", f"{overall['mAP50_95']:.3f}"),
        ("Precision", f"{overall['precision']:.3f}"),
        ("Recall", f"{overall['recall']:.3f}"),
    ]
    x0 = 0.07
    box_w = 0.20
    for i, (label, value) in enumerate(stats):
        x = x0 + i * (box_w + 0.013)
        ax.add_patch(plt.Rectangle((x, 0.74), box_w, 0.12, fill=False,
                                    edgecolor="#0b0b0b", linewidth=1))
        ax.text(x + box_w / 2, 0.815, value, fontsize=22, fontweight="bold",
                ha="center", va="center", color=BLUE)
        ax.text(x + box_w / 2, 0.755, label, fontsize=10.5, ha="center",
                va="center", color=MUTED)

    ax.text(0.07, 0.68, "Test set: 499 pages, 6,349 annotated regions, 11 classes (none in COCO)",
            fontsize=10.5, color=MUTED)

    # metric bar chart - 4 distinct measures of the same model, so a small
    # categorical set (not a repeated single-series magnitude chart)
    cats = ["Precision", "Recall", "mAP50", "mAP50-95"]
    vals = [overall["precision"], overall["recall"], overall["mAP50"], overall["mAP50_95"]]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order, slots 1-4
    ax2 = fig.add_axes([0.10, 0.48, 0.80, 0.16])
    bars = ax2.bar(cats, vals, color=colors, width=0.55)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax2.spines[spine].set_visible(False)
    ax2.spines["bottom"].set_color("#0b0b0b")
    ax2.tick_params(left=False)
    ax2.set_yticklabels([])
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                  ha="center", fontsize=10, fontweight="bold")

    # training config, factual, no names/dates
    hp = run["hyperparameters"]
    hw = run["hardware"]
    config_lines = [
        f"Epochs: {hp['epochs']}    Batch: {hp['batch']}    Image size: {hp['imgsz']}px",
        f"Optimizer: {hp['optimizer']}    LR: {hp['lr0']}    Seed: {hp['seed']}",
        f"Hardware: {hw['gpu']} ({hw['gpu_memory_gb']} GB)    Wall clock: {run['wall_clock_human']}",
        f"Query-budget saturation: {metrics['query_saturation']['percent_over_budget']}% of test pages "
        f"(max {metrics['query_saturation']['max_regions_on_any_page']} regions, budget "
        f"{metrics['query_saturation']['query_budget']})",
        f"Train/test source-PDF overlap: {metrics['split_leakage']['percent_affected']}% "
        f"({metrics['split_leakage']['test_pages_total']} test pages checked)",
    ]
    ax.text(0.07, 0.40, "Training configuration", fontsize=13, fontweight="bold")
    for i, line in enumerate(config_lines):
        ax.text(0.07, 0.365 - i * 0.032, line, fontsize=10.5, color="#0b0b0b")

    return fig


def page_per_class():
    fig = new_page()
    ax_title = bare_axes(fig, [0, 0, 1, 1])
    ax_title.text(0.07, 0.955, "Per-class performance (mAP50)", fontsize=16, fontweight="bold")
    ax_title.text(0.07, 0.928, "Sorted by score. Instance count shown per class — rarity, not size alone, "
                                "drives the weakest results.", fontsize=10, color=MUTED)

    per_class = metrics["overall"]  # placeholder, real data below
    items = sorted(
        [(name, CLASS_INSTANCES[name]) for name in CLASS_INSTANCES],
        key=lambda kv: PER_CLASS_MAP50[kv[0]],
    )
    names = [n for n, _ in items]
    values = [PER_CLASS_MAP50[n] for n, _ in items]
    counts = [c for _, c in items]

    ax = fig.add_axes([0.28, 0.08, 0.62, 0.80])
    y = range(len(names))
    ax.barh(y, values, color=BLUE, height=0.6, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#0b0b0b")
    ax.spines["bottom"].set_color("#0b0b0b")
    ax.set_xlabel("mAP50", fontsize=10.5, color=MUTED)

    for yi, (v, c) in enumerate(zip(values, counts)):
        ax.text(v + 0.015, yi, f"{v:.3f}", va="center", fontsize=9.5, fontweight="bold")
        ax.text(1.02, yi, f"n={c}", va="center", fontsize=8.5, color=MUTED, transform=ax.get_yaxis_transform())

    return fig


def page_per_category():
    fig = new_page()
    ax_title = bare_axes(fig, [0, 0, 1, 1])
    ax_title.text(0.07, 0.955, "Per-document-category performance (mAP50)", fontsize=16, fontweight="bold")
    ax_title.text(0.07, 0.928, "Tests generalization across document styles, not just aggregate accuracy.",
                  fontsize=10, color=MUTED)

    cat = metrics["per_doc_category"]
    items = sorted(cat.items(), key=lambda kv: kv[1]["mAP50"])
    names = [k.replace("_", " ") for k, _ in items]
    values = [v["mAP50"] for _, v in items]
    pages = [v["pages"] for _, v in items]

    ax = fig.add_axes([0.30, 0.55, 0.60, 0.34])
    y = range(len(names))
    ax.barh(y, values, color=BLUE, height=0.55, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#0b0b0b")
    ax.spines["bottom"].set_color("#0b0b0b")
    ax.set_xlabel("mAP50", fontsize=10.5, color=MUTED)
    for yi, (v, p) in enumerate(zip(values, pages)):
        ax.text(v + 0.015, yi, f"{v:.3f}  ({p} pages)", va="center", fontsize=9.5, fontweight="bold")

    # failure notes, factual only
    ax_title.text(0.07, 0.42, "Observed failure patterns", fontsize=13, fontweight="bold")
    notes = [
        "Footnote (47 instances): recall 0.048 — rare class, near-total miss rate.",
        "Page-footer (408 instances): mAP50 0.843 — same thin shape as Footnote, but",
        "     9x more training instances and a consistent position. Rarity, not size,",
        "     is the dominant factor.",
        "Picture: precision 0.416 despite recall 0.622 — composite regions (diagrams",
        "     with embedded text) get split into multiple overlapping boxes instead",
        "     of one region.",
        "Dense repeated-entry layouts (directories, org charts) produce duplicate,",
        "     overlapping box proposals rather than one box per entry.",
    ]
    for i, line in enumerate(notes):
        ax_title.text(0.07, 0.385 - i * 0.028, line, fontsize=10, color="#0b0b0b")

    return fig


PER_CLASS_MAP50 = {
    "Caption": 0.622, "Footnote": 0.182, "Formula": 0.788, "List-item": 0.671,
    "Page-footer": 0.843, "Page-header": 0.658, "Picture": 0.495,
    "Section-header": 0.762, "Table": 0.761, "Text": 0.832, "Title": 0.224,
}

if __name__ == "__main__":
    import sys

    preview = "--preview" in sys.argv
    out_path = ROOT / "reports" / "evaluation_report.pdf"
    pages = [page_overview(), page_per_class(), page_per_category()]

    with PdfPages(out_path, metadata={
        "Title": "", "Author": "", "Subject": "", "Creator": "", "Producer": "", "CreationDate": None,
    }) as pdf:
        for i, fig in enumerate(pages, start=1):
            pdf.savefig(fig)
            if preview:
                fig.savefig(ROOT / f"reports/_preview_page{i}.png", dpi=110)
            plt.close(fig)

    print(f"wrote {out_path}")
