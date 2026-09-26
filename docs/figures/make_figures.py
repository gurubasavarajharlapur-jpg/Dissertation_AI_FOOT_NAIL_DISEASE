"""Generate the result figures used in Chapter 5 of the dissertation.

Every number here is transcribed from docs/findings_and_analysis.md, which is
the project's single source of truth for measured results. Nothing is
recomputed and nothing is invented: the confusion matrices, ablation arms and
efficiency figures are the ones reported in Chapter 5.

The two schematics (two-stage training, system architecture) describe the
design rather than a measurement.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent
ACCENT, ACCENT2, GREY = "#1F4E79", "#2F5496", "#595959"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": "#808080", "axes.labelcolor": "#262626",
                     "text.color": "#262626", "xtick.color": "#404040",
                     "ytick.color": "#404040", "savefig.dpi": 200,
                     "savefig.bbox": "tight", "figure.facecolor": "white"})

CLASSES = ["Healthy", "Nail\nfungal", "Foot\nwound", "Foot\nulcer"]
CM = {  # rows true, columns predicted; Chapter 5, Table 5.5
    "MobileNetV2": [[553, 3, 0, 0], [1, 116, 0, 0], [0, 1, 368, 7], [0, 0, 9, 190]],
    "ResNet50":    [[556, 0, 0, 0], [0, 117, 0, 0], [1, 1, 366, 8], [0, 0, 3, 196]],
}


def confusion():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    for ax, (name, m) in zip(axes, CM.items()):
        m = np.array(m, float)
        norm = m / m.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                if m[i, j]:
                    ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center",
                            fontsize=9, color="white" if norm[i, j] > 0.5 else "#1F4E79",
                            fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(4), CLASSES, fontsize=7.5)
        ax.set_yticks(range(4), CLASSES, fontsize=7.5)
        ax.set_xlabel("Predicted", fontsize=8.5)
        ax.set_ylabel("True", fontsize=8.5)
        errors = int(m.sum() - np.trace(m))
        ax.set_title(f"{name}:  {errors} errors of 1,248", fontsize=9.5,
                     color=ACCENT, fontweight="bold", pad=8)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "confusion.png")
    plt.close(fig)


def ablation():
    arms = ["Unmodified", "20px border\nmasked", "Equal-area\ninterior masked"]
    mnv2 = [0.9832, 0.9671, 0.8598]
    rn50 = [0.9896, 0.9704, 0.9030]
    floor = 0.80
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    b1 = ax.bar(x - w/2, mnv2, w, label="MobileNetV2", color=ACCENT)
    b2 = ax.bar(x + w/2, rn50, w, label="ResNet50", color="#9DC3E6")
    for bars, vals in ((b1, mnv2), (b2, rn50)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.004, f"{v:.4f}",
                    ha="center", fontsize=7.5)
    # the drop against the unmodified arm, printed inside each masked bar
    for i in range(1, 3):
        mid_a, mid_b = (floor + mnv2[i]) / 2, (floor + rn50[i]) / 2
        ax.text(i - w/2, mid_a, f"\u2212{mnv2[0]-mnv2[i]:.4f}", ha="center",
                color="white", fontsize=8, fontweight="bold")
        ax.text(i + w/2, mid_b, f"\u2212{rn50[0]-rn50[i]:.4f}", ha="center",
                color=ACCENT, fontsize=8, fontweight="bold")
    ax.set_xticks(x, arms, fontsize=8.5)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(floor, 1.008)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Masking the border costs little; masking the same area\nof the interior costs far more",
                 fontsize=9.5, color=ACCENT, pad=10)
    fig.savefig(OUT / "ablation.png"); plt.close(fig)


def efficiency():
    labels = ["Parameters\n(millions)", "Size on disk\n(MB)", "CPU inference\n(ms per image)"]
    mnv2 = [2.263108, 21.78, 248.29]
    rn50 = [23.595908, 210.55, 460.56]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0))
    for ax, lab, a, b in zip(axes, labels, mnv2, rn50):
        ax.bar([0, 1], [a, b], 0.55, color=[ACCENT, "#9DC3E6"])
        for i, v in enumerate((a, b)):
            ax.text(i, v, f"{v:,.2f}" if v < 100 else f"{v:,.1f}", ha="center",
                    va="bottom", fontsize=8)
        ax.set_xticks([0, 1], ["MobileNetV2", "ResNet50"], fontsize=8)
        ax.set_title(lab, fontsize=8.5, color=ACCENT)
        ax.set_ylim(0, max(a, b) * 1.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.5, max(a, b) * 1.12, f"{b/a:.1f}×", ha="center",
                fontsize=9, color=GREY, style="italic")
    fig.tight_layout(); fig.savefig(OUT / "efficiency.png"); plt.close(fig)


def external():
    """Confidence per external photograph, by background. Chapter 5, Table 5.16."""
    data = [  # subject, background, correct, confidence
        (1, "plain", True, 0.9988), (1, "patterned", False, 0.9219),
        (2, "plain", True, 0.9058), (2, "patterned", False, 0.6376),
        (3, "plain", True, 0.7917), (3, "patterned", False, 0.7824),
        (4, "plain", True, 0.8595), (4, "patterned", True, 0.5311),
        (5, "plain", True, 0.7528), (5, "patterned", False, 0.6074),
    ]
    PLAIN, PATT = ACCENT, "#C00000"
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.axhline(0.787, color=GREY, ls="--", lw=1)
    ax.text(0.62, 0.797, "abstention threshold 0.787", fontsize=7.5, color=GREY)
    for subj in range(1, 6):
        pair = [d for d in data if d[0] == subj]
        ax.plot([subj - 0.16, subj + 0.16], [pair[0][3], pair[1][3]],
                color="#D0D0D0", lw=1.2, zorder=1)
    for subj, bg, ok, conf in data:
        ax.scatter(subj + (-0.16 if bg == "plain" else 0.16), conf, s=95,
                   marker="o" if ok else "X", color=PLAIN if bg == "plain" else PATT,
                   zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(1, 6), [f"Subject {i}" for i in range(1, 6)], fontsize=8.5)
    ax.set_xlim(0.55, 5.45); ax.set_ylim(0.46, 1.05)
    ax.set_ylabel("Calibrated confidence")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], marker="s", ls="", color=PLAIN, label="Plain background"),
               plt.Line2D([], [], marker="s", ls="", color=PATT, label="Patterned background"),
               plt.Line2D([], [], marker="o", ls="", color=GREY, label="Classified correctly"),
               plt.Line2D([], [], marker="X", ls="", color=GREY, label="Misclassified")]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=4, columnspacing=1.1, handletextpad=0.3)
    ax.set_title("Every plain-background photograph was correct;\nfour of five patterned ones were not",
                 fontsize=9.5, color=ACCENT, pad=10)
    fig.savefig(OUT / "external.png"); plt.close(fig)


def _box(ax, xy, w, h, text, fill, edge, fontsize=8, color="#262626", bold=False):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.1, facecolor=fill, edgecolor=edge))
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=color, fontweight="bold" if bold else "normal")


def _arrow(ax, a, b, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11,
                                 linewidth=1.1, color=GREY, shrinkA=2, shrinkB=2))


def twostage():
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.25, 0.95, "Stage 1:  frozen backbone", ha="center", fontsize=9.5,
            color=ACCENT, fontweight="bold")
    ax.text(0.76, 0.95, "Stage 2:  fine-tuning", ha="center", fontsize=9.5,
            color=ACCENT, fontweight="bold")
    ax.plot([0.505, 0.505], [0.06, 0.90], color="#D9D9D9", lw=1.2, ls=":")
    # stage 1
    _box(ax, (0.03, 0.62), 0.44, 0.19, "ImageNet backbone:  FROZEN\nno weights updated",
         "#EDEDED", "#B0B0B0")
    _box(ax, (0.03, 0.36), 0.44, 0.19, "New classification head:  TRAINED\npooling, dropout 0.3, 4-way softmax",
         "#DEEAF6", ACCENT)
    _box(ax, (0.03, 0.10), 0.44, 0.17, "Learning rate 1e-3\nbest val_loss checkpointed", "white", "#B0B0B0")
    _arrow(ax, (0.25, 0.62), (0.25, 0.55))
    _arrow(ax, (0.25, 0.36), (0.25, 0.27))
    # stage 2
    _box(ax, (0.54, 0.62), 0.44, 0.19, "Top layers UNFROZEN, lower layers frozen\nBatchNorm stays frozen throughout",
         "#DEEAF6", ACCENT)
    _box(ax, (0.54, 0.36), 0.44, 0.19, "Head continues training\nfrom the stage 1 weights", "#DEEAF6", ACCENT)
    _box(ax, (0.54, 0.10), 0.44, 0.17,
         "Learning rate 1e-5, one hundred times lower\nstage 1 best val_loss passed in as the baseline",
         "white", "#B0B0B0")
    _arrow(ax, (0.76, 0.62), (0.76, 0.55))
    _arrow(ax, (0.76, 0.36), (0.76, 0.27))
    _arrow(ax, (0.47, 0.455), (0.54, 0.455))
    ax.text(0.505, 0.03,
            "A randomly initialised head would push large gradients through the pretrained filters, "
            "so the head is trained first and the backbone only then adjusted, gently.",
            ha="center", fontsize=7.5, color=GREY, style="italic")
    fig.savefig(OUT / "twostage.png"); plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, (0.14, 0.845), 0.72, 0.115,
         "src/config.py:   every path, hyper-parameter, split ratio and class name\n"
         "imported by every script, and validates itself on import",
         "#FFF2CC", "#BF8F00", fontsize=7.5)
    stages = [("download_data.py\ninspect_data.py", 0.005),
              ("extract_montages.py\npreprocessing.py", 0.257),
              ("train.py", 0.509), ("evaluate.py\ncalibrate.py", 0.761)]
    wid = 0.234
    for label, x in stages:
        _box(ax, (x, 0.585), wid, 0.145, label, "#DEEAF6", ACCENT, fontsize=7.5)
    for i in range(3):
        _arrow(ax, (stages[i][1] + wid, 0.6575), (stages[i+1][1], 0.6575))
    _arrow(ax, (0.50, 0.845), (0.50, 0.778))
    ax.plot([0.122, 0.878], [0.778, 0.778], color=GREY, lw=1.1)
    for _, x in stages:
        _arrow(ax, (x + wid/2, 0.778), (x + wid/2, 0.732))
    _box(ax, (0.02, 0.335), 0.44, 0.135,
         "saved per-image predictions\nwith full probability vectors", "white", "#B0B0B0", fontsize=7.5)
    _box(ax, (0.54, 0.335), 0.44, 0.135,
         "compare_models.py    audit_triage.py\nsweep_thresholds.py    ood_test.py\n"
         "re-analysis without a GPU", "#E2EFDA", "#548235", fontsize=7.5)
    _arrow(ax, (0.30, 0.585), (0.24, 0.47))
    _arrow(ax, (0.46, 0.4025), (0.54, 0.4025))
    _box(ax, (0.17, 0.10), 0.66, 0.135,
         "src/prototype/app.py:   screening tab and batch evaluation tab\n"
         "src/triage.py turns a prediction into a recommended action",
         "#DEEAF6", ACCENT, fontsize=7.5)
    _arrow(ax, (0.50, 0.335), (0.50, 0.235))
    ax.text(0.5, 0.025,
            "Each script is a standalone stage rather than a library; the configuration module is the only thing they share.",
            ha="center", fontsize=7.5, color=GREY, style="italic")
    fig.savefig(OUT / "architecture.png"); plt.close(fig)


for fn in (confusion, ablation, efficiency, external, twostage, architecture):
    fn(); print("wrote", fn.__name__)
