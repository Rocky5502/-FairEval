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
        "font.size": 8.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # preserve editable text in the SVG source
    }
)

C = {
    "ink": "#182235",
    "muted": "#66758C",
    "line": "#CBD5E1",
    "line2": "#DCE4ED",
    "panel": "#F7F9FC",
    "white": "#FFFFFF",
    "blue": "#1D4ED8",
    "blue2": "#DBEAFE",
    "red": "#C02518",
    "red2": "#FEE7E3",
    "green": "#15803D",
    "green2": "#DCFCE7",
    "gold": "#AD6800",
    "gold2": "#FFF1C2",
    "violet": "#6D28D9",
    "violet2": "#EEE7FF",
    "cyan": "#087A97",
    "cyan2": "#CFF5FB",
    "slate": "#475569",
}


def rr(ax, x, y, w, h, fc, ec=None, lw=0.9, r=0.018, z=1):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec or C["line"],
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(p)
    return p


def ar(ax, x1, y1, x2, y2, color=None, lw=1.05, ms=9, z=3):
    p = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color or C["muted"],
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=z,
    )
    ax.add_patch(p)
    return p


def tx(ax, x, y, s, fs=8.4, w="normal", c=None, ha="center", va="center", z=5):
    ax.text(
        x,
        y,
        s,
        fontsize=fs,
        fontweight=w,
        color=c or C["ink"],
        ha=ha,
        va=va,
        zorder=z,
    )


def user(ax, x, y, s=0.018, c=None):
    c = c or C["ink"]
    ax.add_patch(Circle((x, y + s * 0.63), s * 0.36, fill=False, ec=c, lw=1.05, zorder=4))
    ax.add_patch(
        FancyBboxPatch(
            (x - s * 0.56, y - s * 0.62),
            s * 1.12,
            s * 0.72,
            boxstyle="round,pad=0.001,rounding_size=0.004",
            fill=False,
            ec=c,
            lw=1.05,
            zorder=4,
        )
    )


def model(ax, x, y, s=0.018, c=None):
    c = c or C["ink"]
    ax.add_patch(Rectangle((x - s * 0.72, y - s * 0.52), s * 1.44, s * 1.04, fill=False, ec=c, lw=1.0, zorder=4))
    for dx in (-0.35, 0, 0.35):
        for dy in (-0.22, 0.22):
            ax.add_patch(Circle((x + dx * s, y + dy * s), s * 0.075, fc=c, ec="none", zorder=4))


def scale_icon(ax, x, y, s=0.018, c=None):
    c = c or C["ink"]
    ax.plot([x, x], [y - s * 0.72, y + s * 0.72], c=c, lw=1.05, zorder=4)
    ax.plot([x - s * 0.72, x + s * 0.72], [y + s * 0.32, y + s * 0.32], c=c, lw=1.05, zorder=4)
    for dx in (-0.55, 0.55):
        ax.plot(
            [x + dx * s, x + (dx - 0.18) * s, x + (dx + 0.18) * s, x + dx * s],
            [y + s * 0.32, y - s * 0.18, y - s * 0.18, y + s * 0.32],
            c=c,
            lw=0.78,
            zorder=4,
        )


def shield(ax, x, y, s=0.018, c=None):
    c = c or C["ink"]
    pts = [
        (x, y + s * 0.8),
        (x - s * 0.65, y + s * 0.35),
        (x - s * 0.48, y - s * 0.5),
        (x, y - s * 0.82),
        (x + s * 0.48, y - s * 0.5),
        (x + s * 0.65, y + s * 0.35),
    ]
    ax.add_patch(Polygon(pts, closed=True, fill=False, ec=c, lw=1.05, zorder=4))


def history_icon(ax, x, y, s=0.018, c=None):
    c = c or C["ink"]
    for i, width in enumerate((1.1, 0.86, 1.22)):
        yy = y + (1 - i) * s * 0.30
        ax.plot([x - s * 0.62, x - s * 0.50 + width * s], [yy, yy], c=c, lw=1.0, zorder=4)
        ax.add_patch(Circle((x - s * 0.77, yy), s * 0.055, fc=c, ec="none", zorder=4))


def ranking(ax, x, y, w, h, items, title="Ranking", edge=None):
    rr(ax, x, y, w, h, C["white"], edge or C["line"], lw=0.85, r=0.010)
    tx(ax, x + 0.013, y + h - 0.025, title, 6.7, "bold", ha="left")
    for i, item in enumerate(items):
        yy = y + h - 0.054 - i * 0.027
        tx(ax, x + 0.020, yy, f"{i+1}", 6.1, "bold", C["muted"], ha="left")
        tx(ax, x + 0.038, yy, item, 6.15, ha="left")


def tag(ax, x, y, s, fc, ec, width=0.055):
    rr(ax, x, y, width, 0.038, fc, ec, lw=0.8, r=0.009)
    tx(ax, x + width / 2, y + 0.019, s, 6.4, "bold", ec)


def setup(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, stem):
    """Write a publication PDF plus an editable SVG from the exact same drawing."""
    pdf = OUT / f"{stem}.pdf"
    svg = OUT / f"{stem}.svg"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 1 — motivating example, inspired by the comparison logic of FaiRLLM
# but redesigned around FairEval's consequence-aware distinction.
# ---------------------------------------------------------------------------
fig, ax = setup((13.0, 6.1))
tx(ax, 0.02, 0.962, "Why recommendation change is not automatically unfair", 12.0, "bold", ha="left")
tx(
    ax,
    0.02,
    0.925,
    "FairEval keeps the task and preference evidence fixed, changes one registered context factor, then asks what the ranking change does to held-out utility.",
    8.1,
    c=C["muted"],
    ha="left",
)

# Shared task / history column
rr(ax, 0.02, 0.17, 0.19, 0.69, C["panel"])
tag(ax, 0.035, 0.804, "FIXED", C["blue2"], C["blue"], 0.062)
user(ax, 0.058, 0.735, 0.025, C["blue"])
tx(ax, 0.088, 0.748, "same user task", 8.3, "bold", ha="left")
tx(ax, 0.088, 0.718, "candidate-constrained\ntop-K ranking", 6.3, c=C["muted"], ha="left")
history_icon(ax, 0.058, 0.655, 0.021, C["slate"])
tx(ax, 0.088, 0.665, "preference history", 7.4, "bold", ha="left")
tx(ax, 0.088, 0.637, "same Hᵤ + same candidates Cᵤ", 6.5, c=C["muted"], ha="left")
rr(ax, 0.040, 0.500, 0.150, 0.096, C["white"], r=0.010)
tx(ax, 0.052, 0.570, "Held-out relevance", 7.0, "bold", ha="left")
for i, t in enumerate(["A12  ★", "B04  ★", "C19", "D07  ★"]):
    tx(ax, 0.055, 0.545 - i * 0.020, t, 6.2, ha="left")
rr(ax, 0.040, 0.300, 0.150, 0.145, C["white"], r=0.010)
tx(ax, 0.052, 0.418, "Frozen controls", 7.0, "bold", ha="left")
for i, t in enumerate(["task wording", "candidate order", "sampling policy", "model / revision"]):
    tx(ax, 0.055, 0.392 - i * 0.027, f"• {t}", 6.2, ha="left")
tx(ax, 0.115, 0.215, "Only one context factor changes", 6.6, "bold", C["red"])

# Paired context lanes
lane_x = 0.245
lane_w = 0.355
for y, ctag, title, prompt, color, bg in [
    (0.630, "C1", "Observed identity", "gender: observed", C["blue"], C["blue2"]),
    (0.405, "C2", "Matched counterfactual", "gender: swapped", C["red"], C["red2"]),
    (0.180, "C3/C4", "Measured personality control", "true OCEAN\n↔ shuffled profile", C["green"], C["green2"]),
]:
    rr(ax, lane_x, y, lane_w, 0.180, C["white"], color, lw=0.95, r=0.014)
    tag(ax, lane_x + 0.014, y + 0.125, ctag, bg, color, 0.062)
    tx(ax, lane_x + 0.090, y + 0.145, title, 7.7, "bold", ha="left")
    rr(ax, lane_x + 0.014, y + 0.036, 0.128, 0.075, bg, color, lw=0.7, r=0.010)
    tx(ax, lane_x + 0.025, y + 0.086, prompt, 6.0, "bold", color, ha="left")
    tx(ax, lane_x + 0.025, y + 0.058, "same Hᵤ, Cᵤ, t, K", 6.1, c=C["muted"], ha="left")
    model(ax, lane_x + 0.172, y + 0.075, 0.021, color)
    ar(ax, lane_x + 0.145, y + 0.075, lane_x + 0.150, y + 0.075, color, 0.9, 8)
    ar(ax, lane_x + 0.194, y + 0.075, lane_x + 0.210, y + 0.075, color, 0.9, 8)

ranking(ax, 0.460, 0.656, 0.122, 0.124, ["A12 ★", "D07 ★", "B04 ★", "C19", "F03"], title="Ranking R(C1)")
ranking(ax, 0.460, 0.431, 0.122, 0.124, ["F03", "C19", "D07 ★", "A12 ★", "B04 ★"], title="Ranking R(C2)")
ranking(ax, 0.460, 0.206, 0.122, 0.124, ["A12 ★", "B04 ★", "D07 ★", "F03", "C19"], title="R(C3) / R(C4)")

# Consequence interpretation
rr(ax, 0.635, 0.17, 0.345, 0.69, C["panel"])
tx(ax, 0.655, 0.820, "FairEval interpretation layer", 9.4, "bold", ha="left")
tx(ax, 0.655, 0.786, "First measure sensitivity; then evaluate consequence.", 6.9, c=C["muted"], ha="left")
rr(ax, 0.655, 0.665, 0.145, 0.085, C["white"], C["line2"], r=0.012)
tx(ax, 0.670, 0.724, "1  Ranking change?", 7.4, "bold", ha="left")
tx(ax, 0.670, 0.694, "RBO / Jaccard", 6.6, c=C["violet"], ha="left")
tx(ax, 0.670, 0.675, "diagnostic only", 5.9, c=C["muted"], ha="left")
ar(ax, 0.808, 0.707, 0.835, 0.707, C["muted"], 1.0)
rr(ax, 0.840, 0.665, 0.118, 0.085, C["white"], C["line2"], r=0.012)
scale_icon(ax, 0.858, 0.707, 0.015, C["gold"])
tx(ax, 0.878, 0.724, "2  Consequence", 7.3, "bold", ha="left")
tx(ax, 0.878, 0.696, "held-out U(R,Y)", 6.3, ha="left")
tx(ax, 0.878, 0.675, "CUG / PVA / IOD", 5.9, c=C["muted"], ha="left")
for y, head, body, fc, ec in [
    (0.535, "Stable / low sensitivity", "lists are similar; no harm claim from change", C["white"], C["slate"]),
    (0.395, "Personalization or harmless sensitivity", "list changes, but utility is preserved or improves", C["green2"], C["green"]),
    (0.255, "Potential harmful disparity", "matched identity change produces systematic utility loss", C["red2"], C["red"]),
]:
    rr(ax, 0.655, y, 0.303, 0.104, fc, ec, lw=0.9, r=0.012)
    tx(ax, 0.670, y + 0.071, head, 7.4, "bold", ec, ha="left")
    tx(ax, 0.670, y + 0.038, body, 6.25, ha="left")
rr(ax, 0.655, 0.187, 0.303, 0.050, C["cyan2"], C["cyan"], lw=0.8, r=0.010)
tx(ax, 0.668, 0.212, "FairSynth-360 sanity: identity A/B/C is relevance-invariant\nby construction", 5.8, "bold", C["cyan"], ha="left")
for y in (0.720, 0.495, 0.270):
    ar(ax, 0.600, y, 0.632, 0.705 if y > 0.60 else (0.447 if y > 0.35 else 0.307), C["muted"], 0.85, 8)
ax.plot([0.02, 0.98], [0.115, 0.115], c=C["line"], lw=0.8)
tx(
    ax,
    0.02,
    0.073,
    "Core principle: Δ ranking ≠ unfairness. Fairness claims require matched interventions + preference-conditioned consequence evidence + robustness.",
    7.7,
    "bold",
    ha="left",
)
save(fig, "faireval_framework")

# ---------------------------------------------------------------------------
# Fig. 2 — RQ geometry. Compact, symmetric, and free of empirical results.
# ---------------------------------------------------------------------------
fig, ax = setup((12.8, 5.25))
tx(ax, 0.02, 0.955, "FairEval experimental geometry", 11.5, "bold", ha="left")
tx(
    ax,
    0.02,
    0.918,
    "Each RQ reuses the same frozen recommendation task and changes only the factor required by its registered estimand.",
    7.9,
    c=C["muted"],
    ha="left",
)

panel_specs = [
    (0.020, "RQ1", "Demographic\nconsequence", C["red"], C["red2"]),
    (0.265, "RQ2", "Grounded personality\nvalue", C["green"], C["green2"]),
    (0.510, "RQ3", "Reliability +\ngeneralization", C["blue"], C["blue2"]),
    (0.755, "RQ4", "Personalization-preserving\nmitigation", C["violet"], C["violet2"]),
]
for x, rq, title, ec, bg in panel_specs:
    rr(ax, x, 0.180, 0.225, 0.665, C["white"], C["line"], lw=0.9, r=0.013)
    tag(ax, x + 0.014, 0.782, rq, bg, ec, 0.060)
    tx(ax, x + 0.082, 0.807, title, 7.2, "bold", ha="left")

# RQ1: matched demographic intervention
x = 0.020
tx(ax, x + 0.025, 0.725, "MATCHED CONTRAST", 5.7, "bold", C["muted"], ha="left")
rr(ax, x + 0.022, 0.605, 0.070, 0.090, C["blue2"], C["blue"], r=0.010)
rr(ax, x + 0.133, 0.605, 0.070, 0.090, C["red2"], C["red"], r=0.010)
tx(ax, x + 0.057, 0.650, "C1\nobserved", 6.9, "bold")
tx(ax, x + 0.168, 0.650, "C2\nswap", 6.9, "bold")
ar(ax, x + 0.096, 0.650, x + 0.128, 0.650, C["red"], 1.0)
tx(ax, x + 0.1125, 0.557, "CUG = U(C1) − U(C2)", 7.8, "bold")
rr(ax, x + 0.022, 0.446, 0.181, 0.073, C["panel"], C["line2"], r=0.009)
tx(ax, x + 0.034, 0.493, "Scope", 5.8, "bold", C["red"], ha="left")
tx(ax, x + 0.034, 0.465, "MovieLens-1M + Last.fm-1K", 5.9, ha="left")
tx(ax, x + 0.028, 0.403, "gender confirmatory  •  age robustness", 5.7, ha="left")
tx(ax, x + 0.028, 0.363, "RBO / Jaccard = sensitivity only", 5.6, c=C["muted"], ha="left")
tx(ax, x + 0.028, 0.307, "Inference", 5.9, "bold", C["red"], ha="left")
tx(ax, x + 0.028, 0.275, "paired sign-flip + bootstrap CI", 5.8, ha="left")
tx(ax, x + 0.028, 0.239, "user is the unit", 5.6, c=C["muted"], ha="left")

# RQ2: measured psychometric value
x = 0.265
tx(ax, x + 0.025, 0.725, "MEASURED CONTROL", 5.7, "bold", C["muted"], ha="left")
rr(ax, x + 0.022, 0.605, 0.070, 0.090, C["green2"], C["green"], r=0.010)
rr(ax, x + 0.133, 0.605, 0.070, 0.090, C["gold2"], C["gold"], r=0.010)
tx(ax, x + 0.057, 0.650, "C3\ntrue", 6.9, "bold")
tx(ax, x + 0.168, 0.650, "C4\nshuffled", 6.9, "bold")
ar(ax, x + 0.096, 0.650, x + 0.128, 0.650, C["gold"], 1.0)
tx(ax, x + 0.1125, 0.557, "PVA = U(C3) − U(C4)", 7.8, "bold")
rr(ax, x + 0.022, 0.446, 0.181, 0.073, C["panel"], C["line2"], r=0.009)
tx(ax, x + 0.034, 0.493, "Scope", 5.8, "bold", C["green"], ha="left")
tx(ax, x + 0.034, 0.465, "3 measured-personality datasets", 5.9, ha="left")
tx(ax, x + 0.028, 0.403, "whole-profile derangement  •  C5 one-trait", 5.55, ha="left")
tx(ax, x + 0.028, 0.363, "dataset-native psychometrics only", 5.6, c=C["muted"], ha="left")
tx(ax, x + 0.028, 0.307, "Inference", 5.9, "bold", C["green"], ha="left")
tx(ax, x + 0.028, 0.275, "same paired user-level stack", 5.8, ha="left")
tx(ax, x + 0.028, 0.239, "true vs shuffled is primary", 5.6, c=C["muted"], ha="left")

# RQ3: registered robustness factors
x = 0.510
tx(ax, x + 0.025, 0.725, "REGISTERED FACTORS", 5.7, "bold", C["muted"], ha="left")
factor_rows = [
    ("Prompt", "3 paraphrases"),
    ("Cue", "3 realizations"),
    ("Order", "3 permutations"),
    ("K", "5 / 10 / 20"),
    ("Repeat", "3 main; 10 stress"),
]
for i, (k, v) in enumerate(factor_rows):
    yy = 0.665 - i * 0.061
    rr(ax, x + 0.022, yy - 0.022, 0.073, 0.043, C["blue2"] if i < 3 else C["panel"], C["line"], r=0.008)
    tx(ax, x + 0.0585, yy, k, 5.9, "bold", C["blue"] if i < 3 else C["ink"])
    tx(ax, x + 0.104, yy, v, 5.8, ha="left")
rr(ax, x + 0.022, 0.274, 0.181, 0.108, C["cyan2"], C["cyan"], r=0.010)
tx(ax, x + 0.036, 0.351, "FairSynth-360 sanity layer", 6.2, "bold", C["cyan"], ha="left")
tx(ax, x + 0.036, 0.322, "A/B/C identity has no relevance effect", 5.65, ha="left")
tx(ax, x + 0.036, 0.294, "real / synthetic reported separately", 5.5, c=C["muted"], ha="left")
tx(ax, x + 0.028, 0.229, "No best-prompt selection", 5.8, "bold", C["blue"], ha="left")

# RQ4: mitigation comparison and validation-only operating point
x = 0.755
tx(ax, x + 0.025, 0.725, "MITIGATION FRONTIER", 5.7, "bold", C["muted"], ha="left")
flow_y = 0.645
flow = [
    (x + 0.028, "Audit", C["panel"], C["slate"]),
    (x + 0.090, "Prompt", C["violet2"], C["violet"]),
    (x + 0.152, "PAIR", C["red2"], C["red"]),
]
for bx, label, fc, ec in flow:
    rr(ax, bx, flow_y - 0.035, 0.050, 0.070, fc, ec, r=0.009)
    tx(ax, bx + 0.025, flow_y, label, 5.8, "bold", ec)
ar(ax, x + 0.080, flow_y, x + 0.088, flow_y, C["muted"], 0.9, 7)
ar(ax, x + 0.142, flow_y, x + 0.150, flow_y, C["muted"], 0.9, 7)
rr(ax, x + 0.022, 0.450, 0.181, 0.130, C["violet2"], C["violet"], r=0.010)
tx(ax, x + 0.036, 0.548, "Validation-only selection", 6.2, "bold", C["violet"], ha="left")
tx(ax, x + 0.036, 0.514, "≥95% baseline validation nDCG", 5.7, ha="left")
tx(ax, x + 0.036, 0.484, "then minimize |CUG|", 5.7, ha="left")
tx(ax, x + 0.036, 0.455, "freeze (α, λ) before test", 5.55, c=C["muted"], ha="left")
tx(ax, x + 0.028, 0.394, "No test-set point selection", 5.9, "bold", C["red"], ha="left")
tx(ax, x + 0.028, 0.358, "report “no eligible point” if needed", 5.6, ha="left")
tx(ax, x + 0.028, 0.309, "Goal", 5.8, "bold", C["violet"], ha="left")
tx(ax, x + 0.028, 0.278, "reduce harmful gap without", 5.7, ha="left")
tx(ax, x + 0.028, 0.250, "erasing useful personalization", 5.7, ha="left")

ax.plot([0.025, 0.975], [0.115, 0.115], c=C["line"], lw=0.8)
tx(
    ax,
    0.025,
    0.073,
    "Shared contract  •  frozen prompt + candidates  •  repeated generations  •  invalid-output accounting  •  paired user-level inference  •  Holm correction",
    6.8,
    "bold",
    ha="left",
)
save(fig, "faireval_conditions")

# ---------------------------------------------------------------------------
# Fig. 3 — provenance / analysis pipeline. Main path + two explicitly separated
# side branches, with larger typography for LNCS print readability.
# ---------------------------------------------------------------------------
fig, ax = setup((12.8, 4.95))
tx(ax, 0.02, 0.952, "From frozen benchmark cell to auditable paper claim", 11.5, "bold", ha="left")
tx(
    ax,
    0.02,
    0.912,
    "Every empirical paper cell follows one hashed path; white-box internals and mitigation selection remain explicit side branches.",
    7.8,
    c=C["muted"],
    ha="left",
)

# Main six-stage path
main_y = 0.535
box_h = 0.205
box_w = 0.135
xs = [0.020, 0.180, 0.340, 0.500, 0.660, 0.820]
main_steps = [
    ("1", "Frozen\nbenchmark", "release • split\ncandidates", C["blue2"], C["blue"]),
    ("2", "Immutable\nexecution", "prompt • model\nseed • K", C["panel"], C["slate"]),
    ("3", "Strict\nvalidation", "JSON • exact K\ncandidate IDs", C["green2"], C["green"]),
    ("4", "User-level\nestimand", "CUG • PVA\nIOD • RBO", C["gold2"], C["gold"]),
    ("5", "Inference\nartifact", "bootstrap CI\npermutation • Holm", C["violet2"], C["violet"]),
    ("6", "Paper-facing\nclaim", "generated table\nor figure", C["white"], C["ink"]),
]
for x, (num, head, sub, fc, ec) in zip(xs, main_steps):
    rr(ax, x, main_y, box_w, box_h, fc, ec, lw=0.95, r=0.014)
    ax.add_patch(Circle((x + 0.020, main_y + box_h - 0.028), 0.014, fc=ec, ec="none", zorder=4))
    tx(ax, x + 0.020, main_y + box_h - 0.028, num, 5.8, "bold", C["white"])
    tx(ax, x + box_w / 2, main_y + 0.124, head, 7.6, "bold")
    tx(ax, x + box_w / 2, main_y + 0.055, sub, 5.9, c=C["muted"])
for a, b in zip(xs[:-1], xs[1:]):
    ar(ax, a + box_w + 0.004, main_y + box_h / 2, b - 0.006, main_y + box_h / 2, C["muted"], 1.0, 8)

# Small audit bar under main path
rr(ax, 0.020, 0.450, 0.935, 0.052, C["panel"], C["line2"], r=0.009)
tx(
    ax,
    0.034,
    0.476,
    "Audit trail: release / split / candidate hashes  →  model + prompt + seed provenance  →  validator log  →  analysis artifact  →  generated manuscript asset",
    5.85,
    "bold",
    C["slate"],
    ha="left",
)

# White-box branch
rr(ax, 0.150, 0.165, 0.335, 0.205, C["cyan2"], C["cyan"], lw=0.9, r=0.012)
model(ax, 0.180, 0.288, 0.018, C["cyan"])
tx(ax, 0.210, 0.332, "Local white-box diagnostics", 7.0, "bold", C["cyan"], ha="left")
tx(ax, 0.210, 0.295, "token log-p  •  NLL  •  perplexity  •  score margin", 5.9, ha="left")
tx(ax, 0.210, 0.260, "descriptive + explicitly uncalibrated", 5.7, c=C["muted"], ha="left")
tx(ax, 0.210, 0.226, "reported as a separate stratum; never imputed for hosted APIs", 5.55, c=C["muted"], ha="left")
ar(ax, 0.407, main_y - 0.004, 0.320, 0.375, C["cyan"], 0.9, 8)
tx(ax, 0.165, 0.190, "branch from validated local generations", 5.35, c=C["cyan"], ha="left")

# Mitigation branch
rr(ax, 0.535, 0.165, 0.420, 0.205, C["red2"], C["red"], lw=0.9, r=0.012)
shield(ax, 0.565, 0.288, 0.018, C["red"])
tx(ax, 0.595, 0.332, "RQ4 validation-frozen mitigation", 7.0, "bold", C["red"], ha="left")
tx(ax, 0.595, 0.295, "audit  →  identity-irrelevance prompt  →  contextual PAIR", 5.9, ha="left")
tx(ax, 0.595, 0.260, "(α, λ) chosen on validation only, then frozen before test", 5.7, c=C["muted"], ha="left")
tx(ax, 0.595, 0.226, "test outcomes never choose or relax the operating point", 5.55, c=C["muted"], ha="left")
ar(ax, 0.570, main_y - 0.004, 0.690, 0.375, C["red"], 0.9, 8)
tx(ax, 0.550, 0.190, "branch from registered user-level consequences", 5.35, c=C["red"], ha="left")

ax.plot([0.02, 0.955], [0.112, 0.112], c=C["line"], lw=0.8)
tx(
    ax,
    0.02,
    0.070,
    "No manual result entry: empirical tables and result figures are regenerated only from audited analysis artifacts.",
    6.8,
    "bold",
    ha="left",
)
save(fig, "faireval_evaluation_pipeline")

print(OUT)
