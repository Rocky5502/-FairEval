from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COL = {
    "ink": "#172033",
    "muted": "#65758b",
    "line": "#cbd5e1",
    "panel": "#f8fafc",
    "white": "#ffffff",
    "blue": "#1d4ed8",
    "blue2": "#dbeafe",
    "red": "#b42318",
    "red2": "#fee4e2",
    "green": "#15803d",
    "green2": "#dcfce7",
    "gold": "#a16207",
    "gold2": "#fef3c7",
    "violet": "#6d28d9",
    "violet2": "#ede9fe",
    "cyan": "#0e7490",
    "cyan2": "#cffafe",
}


def rr(ax, x, y, w, h, fc, ec=None, lw=1.0, radius=0.018):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec or COL["line"],
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.2, ms=10):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color or COL["muted"],
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def txt(ax, x, y, text, fs=9, weight="normal", color=None, ha="center", va="center"):
    return ax.text(
        x,
        y,
        text,
        fontsize=fs,
        fontweight=weight,
        color=color or COL["ink"],
        ha=ha,
        va=va,
    )


def icon_user(ax, x, y, size=0.018, color=None):
    color = color or COL["ink"]
    ax.add_patch(Circle((x, y + size * 0.65), size * 0.36, fill=False, ec=color, lw=1.1))
    ax.add_patch(
        FancyBboxPatch(
            (x - size * 0.56, y - size * 0.65),
            size * 1.12,
            size * 0.78,
            boxstyle="round,pad=0.001,rounding_size=0.005",
            fill=False,
            ec=color,
            lw=1.1,
        )
    )


def icon_model(ax, x, y, size=0.018, color=None):
    color = color or COL["ink"]
    ax.add_patch(Rectangle((x - size * 0.72, y - size * 0.52), size * 1.44, size * 1.04, fill=False, ec=color, lw=1.0))
    for dx in (-0.35, 0.0, 0.35):
        for dy in (-0.22, 0.22):
            ax.add_patch(Circle((x + dx * size, y + dy * size), size * 0.075, fc=color, ec="none"))


def icon_scale(ax, x, y, size=0.018, color=None):
    color = color or COL["ink"]
    ax.plot([x, x], [y - size * 0.7, y + size * 0.7], color=color, lw=1.1)
    ax.plot([x - size * 0.72, x + size * 0.72], [y + size * 0.34, y + size * 0.34], color=color, lw=1.1)
    for dx in (-0.55, 0.55):
        ax.plot(
            [x + dx * size, x + (dx - 0.18) * size, x + (dx + 0.18) * size, x + dx * size],
            [y + size * 0.34, y - size * 0.18, y - size * 0.18, y + size * 0.34],
            color=color,
            lw=0.85,
        )


def icon_shield(ax, x, y, size=0.018, color=None):
    color = color or COL["ink"]
    pts = [
        (x, y + size * 0.8),
        (x - size * 0.65, y + size * 0.35),
        (x - size * 0.48, y - size * 0.5),
        (x, y - size * 0.82),
        (x + size * 0.48, y - size * 0.5),
        (x + size * 0.65, y + size * 0.35),
    ]
    ax.add_patch(Polygon(pts, closed=True, fill=False, ec=color, lw=1.1))


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: complete FairEval framework after the local/synthetic extension.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12.6, 5.9))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

txt(ax, 0.02, 0.955, "FairEval: consequence-aware evaluation with real, synthetic, hosted, and local evidence", fs=11.5, weight="bold", ha="left")
txt(ax, 0.02, 0.918, "Recommendation change is sensitivity; harmfulness requires preference-conditioned consequence evidence.", fs=8.2, color=COL["muted"], ha="left")

# Evidence panel.
rr(ax, 0.02, 0.14, 0.205, 0.72, COL["panel"])
txt(ax, 0.04, 0.825, "1  Evidence", fs=10, weight="bold", ha="left")
icon_user(ax, 0.052, 0.766, 0.024, COL["blue"])
txt(ax, 0.078, 0.778, "6 real-world datasets", fs=8.3, weight="bold", ha="left")
txt(ax, 0.078, 0.747, "measured / observed / logged", fs=7.0, color=COL["muted"], ha="left")

real_rows = [
    ("3 personality", "movies - music - short video"),
    ("2 demographic", "MovieLens-1M - Last.fm-1K"),
    ("1 generalization", "MIND logged news impressions"),
]
for i, (head, sub) in enumerate(real_rows):
    y = 0.675 - i * 0.09
    rr(ax, 0.047, y - 0.027, 0.151, 0.056, COL["white"], radius=0.010)
    txt(ax, 0.06, y + 0.006, head, fs=7.3, weight="bold", ha="left")
    txt(ax, 0.06, y - 0.015, sub, fs=6.5, color=COL["muted"], ha="left")

rr(ax, 0.043, 0.245, 0.159, 0.126, COL["cyan2"], ec=COL["cyan"], radius=0.012)
txt(ax, 0.057, 0.34, "FairSynth-360", fs=8.0, weight="bold", color=COL["cyan"], ha="left")
txt(ax, 0.057, 0.304, "360 synthetic users", fs=6.9, ha="left")
txt(ax, 0.057, 0.278, "A/B/C identity irrelevant", fs=6.9, ha="left")
txt(ax, 0.057, 0.252, "synthetic OCEAN control", fs=6.9, ha="left")
txt(ax, 0.04, 0.185, "Real and synthetic inference remain separate", fs=6.6, weight="bold", color=COL["muted"], ha="left")

# Condition panel.
rr(ax, 0.255, 0.14, 0.205, 0.72, COL["white"])
txt(ax, 0.275, 0.825, "2  Matched conditions", fs=10, weight="bold", ha="left")
condition_rows = [
    ("C0", "preference only", COL["panel"]),
    ("C1", "observed context", COL["blue2"]),
    ("C2", "identity swap", COL["red2"]),
    ("C3", "true personality", COL["green2"]),
    ("C4", "shuffled profile", COL["gold2"]),
    ("C5", "one-trait swap", COL["violet2"]),
]
for i, (cid, label, fc) in enumerate(condition_rows):
    y = 0.735 - i * 0.082
    rr(ax, 0.28, y - 0.028, 0.154, 0.052, fc, radius=0.010)
    txt(ax, 0.293, y, cid, fs=7.6, weight="bold", ha="left")
    txt(ax, 0.326, y, label, fs=7.1, ha="left")
txt(ax, 0.277, 0.212, "LOCK", fs=7.1, weight="bold", color=COL["red"], ha="left")
txt(ax, 0.319, 0.212, "history - candidates - task - order", fs=6.5, color=COL["muted"], ha="left")
txt(ax, 0.277, 0.178, "Change one registered context factor", fs=6.7, weight="bold", ha="left")

# Model panel.
rr(ax, 0.49, 0.14, 0.245, 0.72, COL["panel"])
txt(ax, 0.51, 0.825, "3  Eight model configurations", fs=10, weight="bold", ha="left")
txt(ax, 0.51, 0.784, "6 hosted/API", fs=7.7, weight="bold", color=COL["blue"], ha="left")
hosted = ["GPT-5.6", "Claude 5", "Gemini 3.8", "DeepSeek 4.1", "Qwen 3.8", "Llama 4"]
for i, label in enumerate(hosted):
    col = i % 3
    row = i // 3
    x = 0.535 + col * 0.073
    y = 0.705 - row * 0.105
    ax.add_patch(Circle((x, y), 0.024, fc=COL["white"], ec=COL["line"], lw=0.9))
    icon_model(ax, x, y, 0.016, COL["blue"])
    txt(ax, x, y - 0.041, label, fs=6.1)

rr(ax, 0.512, 0.315, 0.197, 0.165, COL["green2"], ec=COL["green"], radius=0.012)
txt(ax, 0.526, 0.45, "2 local open-weight", fs=7.7, weight="bold", color=COL["green"], ha="left")
local_rows = [("Qwen2.5-7B", "Apache-2.0"), ("Phi-3.5-mini", "MIT")]
for i, (label, license_name) in enumerate(local_rows):
    y = 0.402 - i * 0.064
    icon_model(ax, 0.535, y, 0.015, COL["green"])
    txt(ax, 0.558, y + 0.010, label, fs=6.9, weight="bold", ha="left")
    txt(ax, 0.558, y - 0.013, license_name, fs=6.1, color=COL["muted"], ha="left")
txt(ax, 0.51, 0.268, "RTX 5090 32 GB target - one model at a time", fs=6.3, color=COL["muted"], ha="left")
txt(ax, 0.51, 0.225, "Local-only diagnostics", fs=7.1, weight="bold", color=COL["violet"], ha="left")
txt(ax, 0.51, 0.195, "token log-p - NLL - margin", fs=6.5, ha="left")
txt(ax, 0.51, 0.169, "auxiliary / uncalibrated", fs=6.2, color=COL["muted"], ha="left")

# Outcome panel.
rr(ax, 0.765, 0.14, 0.215, 0.72, COL["white"])
txt(ax, 0.785, 0.825, "4  Auditable consequences", fs=10, weight="bold", ha="left")
icon_scale(ax, 0.797, 0.755, 0.024, COL["red"])
txt(ax, 0.827, 0.759, "Identity consequence", fs=7.8, weight="bold", ha="left")
txt(ax, 0.793, 0.697, r"$CUG=U(C_1)-U(C_2)$", fs=9.2, ha="left")
txt(ax, 0.793, 0.655, "IOD - GUD - CEG*", fs=7.1, color=COL["muted"], ha="left")

icon_user(ax, 0.797, 0.585, 0.022, COL["green"])
txt(ax, 0.827, 0.59, "Personality value", fs=7.8, weight="bold", ha="left")
txt(ax, 0.793, 0.53, r"$PVA=U(C_3)-U(C_4)$", fs=9.2, ha="left")

icon_shield(ax, 0.797, 0.455, 0.022, COL["violet"])
txt(ax, 0.827, 0.46, "Contextual PAIR", fs=7.8, weight="bold", ha="left")
txt(ax, 0.793, 0.405, r"$Fit_a-\lambda Var_{cf}$", fs=9.0, ha="left")
txt(ax, 0.793, 0.363, "validation-frozen Pareto point", fs=6.4, color=COL["muted"], ha="left")

rr(ax, 0.788, 0.245, 0.166, 0.075, COL["cyan2"], radius=0.010)
txt(ax, 0.801, 0.292, "FairSynth sanity", fs=7.0, weight="bold", color=COL["cyan"], ha="left")
txt(ax, 0.801, 0.265, "known identity irrelevance", fs=6.4, ha="left")
txt(ax, 0.785, 0.19, "* CEG only with complete auditable metadata", fs=5.9, color=COL["muted"], ha="left")

for x1, x2 in ((0.225, 0.255), (0.46, 0.49), (0.735, 0.765)):
    arrow(ax, x1, 0.50, x2, 0.50, lw=1.3)

ax.plot([0.02, 0.98], [0.092, 0.092], color=COL["line"], lw=0.8)
txt(ax, 0.02, 0.055, "Primary rule: change != unfairness. Real/synthetic and hosted/local evidence are reported in separate strata.", fs=7.7, weight="bold", ha="left")
save(fig, "faireval_framework.pdf")


# ---------------------------------------------------------------------------
# Figure 2: condition geometry and inference structure.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.8, 4.9))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
txt(ax, 0.02, 0.94, "FairEval experimental geometry", fs=11.5, weight="bold", ha="left")
txt(ax, 0.02, 0.898, "Two consequence contrasts, a controlled synthetic sanity layer, and factorized reliability - no post-hoc best-prompt selection.", fs=8.0, color=COL["muted"], ha="left")

# RQ1.
rr(ax, 0.02, 0.19, 0.275, 0.62, COL["panel"])
txt(ax, 0.04, 0.765, "RQ1  demographic consequence", fs=9.1, weight="bold", ha="left")
for x, label, fc in [
    (0.065, "C0\npreference", COL["white"]),
    (0.155, "C1\nobserved", COL["blue2"]),
    (0.245, "C2\nswapped", COL["red2"]),
]:
    rr(ax, x - 0.034, 0.55, 0.068, 0.085, fc, radius=0.011)
    txt(ax, x, 0.592, label, fs=7.0, weight="bold")
arrow(ax, 0.102, 0.592, 0.115, 0.592)
arrow(ax, 0.192, 0.592, 0.205, 0.592, color=COL["red"])
txt(ax, 0.158, 0.46, r"$CUG=U(C_1)-U(C_2)$", fs=9.2)
txt(ax, 0.158, 0.405, "RBO/Jaccard = sensitivity only", fs=6.8, color=COL["muted"])
txt(ax, 0.158, 0.30, "MovieLens-1M + Last.fm-1K", fs=7.0, weight="bold")
txt(ax, 0.158, 0.258, "gender primary - age robustness", fs=6.4, color=COL["muted"])

# RQ2.
rr(ax, 0.32, 0.19, 0.275, 0.62, COL["white"])
txt(ax, 0.34, 0.765, "RQ2  measured personality value", fs=9.1, weight="bold", ha="left")
for x, label, fc in [
    (0.365, "C0\nnone", COL["panel"]),
    (0.455, "C3\ntrue", COL["green2"]),
    (0.545, "C4\nshuffled", COL["gold2"]),
]:
    rr(ax, x - 0.034, 0.55, 0.068, 0.085, fc, radius=0.011)
    txt(ax, x, 0.592, label, fs=7.0, weight="bold")
arrow(ax, 0.402, 0.592, 0.415, 0.592, color=COL["green"])
arrow(ax, 0.492, 0.592, 0.505, 0.592, color=COL["gold"])
txt(ax, 0.458, 0.46, r"$PVA=U(C_3)-U(C_4)$", fs=9.2)
txt(ax, 0.458, 0.405, "C5: one measured trait changed", fs=6.8, color=COL["muted"])
txt(ax, 0.458, 0.30, "3 measured-personality datasets", fs=7.0, weight="bold")
txt(ax, 0.458, 0.258, "whole-profile derangement preserves covariance", fs=6.3, color=COL["muted"])

# RQ3 + FairSynth.
rr(ax, 0.62, 0.19, 0.36, 0.62, COL["panel"])
txt(ax, 0.64, 0.765, "RQ3  reliability + controlled sanity", fs=9.1, weight="bold", ha="left")
factors = [
    ("Prompt", "A / B / C"),
    ("Cue", "structured / 1st-person / profile"),
    ("Order", "3 deterministic permutations"),
    ("K", "5 / 10 / 20"),
    ("Repeat", "3 main; 10 stress-test"),
]
for i, (head, detail) in enumerate(factors):
    y = 0.665 - i * 0.066
    rr(ax, 0.65, y - 0.022, 0.094, 0.043, COL["white"], radius=0.009)
    txt(ax, 0.661, y, head, fs=6.8, weight="bold", ha="left")
    txt(ax, 0.758, y, detail, fs=6.5, color=COL["muted"], ha="left")

rr(ax, 0.65, 0.255, 0.295, 0.105, COL["cyan2"], ec=COL["cyan"], radius=0.012)
txt(ax, 0.667, 0.329, "FairSynth-360", fs=7.2, weight="bold", color=COL["cyan"], ha="left")
txt(ax, 0.667, 0.298, "A/B/C has zero relevance effect by construction", fs=6.3, ha="left")
txt(ax, 0.667, 0.27, "synthetic personality tested separately", fs=6.3, ha="left")

txt(ax, 0.02, 0.108, "Inference: user is the unit - paired permutation primary - paired bootstrap CI - Wilcoxon sensitivity - Holm within pre-registered families.", fs=7.6, weight="bold", ha="left")
save(fig, "faireval_conditions.pdf")


# ---------------------------------------------------------------------------
# Figure 3: frozen-cell provenance -> auditable analysis -> contextual mitigation.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12.2, 4.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
txt(ax, 0.02, 0.93, "From frozen cell to auditable claim", fs=11.5, weight="bold", ha="left")
txt(ax, 0.02, 0.89, "Every empirical paper cell is generated from hashed artifacts; local internal diagnostics remain an auxiliary branch.", fs=7.9, color=COL["muted"], ha="left")

steps = [
    (0.025, 0.13, "Frozen\ninstance", COL["blue2"], COL["blue"]),
    (0.19, 0.14, "Immutable\nrun cell", COL["panel"], COL["ink"]),
    (0.36, 0.14, "Validated\nranking", COL["green2"], COL["green"]),
    (0.53, 0.14, "User-level\nestimand", COL["gold2"], COL["gold"]),
    (0.70, 0.14, "Inference\nartifact", COL["violet2"], COL["violet"]),
    (0.87, 0.105, "Paper\nclaim", COL["white"], COL["ink"]),
]
for x, w, label, fc, ec in steps:
    rr(ax, x, 0.49, w, 0.17, fc, ec=ec, radius=0.015)
    txt(ax, x + w / 2, 0.575, label, fs=8.0, weight="bold")

for left, right in [(0.155, 0.19), (0.33, 0.36), (0.50, 0.53), (0.67, 0.70), (0.84, 0.87)]:
    arrow(ax, left, 0.575, right, 0.575, lw=1.25)

# Provenance annotations.
txt(ax, 0.09, 0.425, "release hash\nsplit hash\ncandidate hash", fs=6.4, color=COL["muted"])
txt(ax, 0.26, 0.425, "prompt hash\nmodel/revision\nseed/order/K", fs=6.4, color=COL["muted"])
txt(ax, 0.43, 0.425, "exact K\nunique IDs\nrepair audit", fs=6.4, color=COL["muted"])
txt(ax, 0.60, 0.425, "CUG / PVA\nIOD / RBO\nuser unit", fs=6.4, color=COL["muted"])
txt(ax, 0.77, 0.425, "bootstrap CI\npermutation\nHolm / effect", fs=6.4, color=COL["muted"])
txt(ax, 0.92, 0.425, "no manual\nresult entry", fs=6.4, color=COL["muted"])

# Local diagnostic branch.
rr(ax, 0.30, 0.16, 0.31, 0.15, COL["cyan2"], ec=COL["cyan"], radius=0.012)
txt(ax, 0.317, 0.278, "Local open-weight side channel", fs=7.3, weight="bold", color=COL["cyan"], ha="left")
txt(ax, 0.317, 0.242, "token log-p - NLL - perplexity - score margin", fs=6.5, ha="left")
txt(ax, 0.317, 0.208, "descriptive only; not calibrated uncertainty", fs=6.2, color=COL["muted"], ha="left")
arrow(ax, 0.43, 0.49, 0.43, 0.315, color=COL["cyan"], lw=1.0)

# Contextual PAIR branch.
rr(ax, 0.65, 0.14, 0.31, 0.18, COL["red2"], ec=COL["red"], radius=0.012)
txt(ax, 0.667, 0.285, "RQ4 contextual PAIR", fs=7.3, weight="bold", color=COL["red"], ha="left")
txt(ax, 0.667, 0.242, r"$PAIR_a(i)=\alpha b(i)+(1-\alpha)r_a(i)-\lambda Var_{a'}r_{a'}(i)$", fs=7.3, ha="left")
txt(ax, 0.667, 0.195, "alpha/lambda validation-only -> frozen test operating point", fs=6.2, color=COL["muted"], ha="left")
arrow(ax, 0.60, 0.49, 0.73, 0.325, color=COL["red"], lw=1.0)

save(fig, "faireval_evaluation_pipeline.pdf")
