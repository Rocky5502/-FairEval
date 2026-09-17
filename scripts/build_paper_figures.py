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
# Fig. 2 — RQ geometry. Four clean panels; no empirical result values.
# ---------------------------------------------------------------------------
fig, ax = setup((12.4, 5.25))
tx(ax, 0.02, 0.955, "FairEval experimental geometry", 11.5, "bold", ha="left")
tx(ax, 0.02, 0.918, "Four research questions share one frozen task contract; each panel changes only the factor needed for its estimand.", 7.9, c=C["muted"], ha="left")

panels = [
    (0.02, 0.235, "RQ1", "Demographic consequence", C["red"], C["red2"]),
    (0.265, 0.235, "RQ2", "Grounded personality value", C["green"], C["green2"]),
    (0.510, 0.235, "RQ3", "Reliability + generalization", C["blue"], C["blue2"]),
    (0.755, 0.235, "RQ4", "Personalization-preserving mitigation", C["violet"], C["violet2"]),
]
for x, w, rq, title, ec, bg in panels:
    rr(ax, x, 0.185, w, 0.665, C["white"], C["line"], lw=0.9, r=0.013)
    tag(ax, x + 0.014, 0.790, rq, bg, ec, 0.060)
    tx(ax, x + 0.078, 0.814, title, 7.4, "bold", ha="left")

# RQ1
x = 0.02
rr(ax, x + 0.012, 0.620, 0.072, 0.095, C["blue2"], C["blue"], r=0.010)
rr(ax, x + 0.125, 0.620, 0.072, 0.095, C["red2"], C["red"], r=0.010)
tx(ax, x + 0.048, 0.668, "C1\nobserved", 7.1, "bold")
tx(ax, x + 0.161, 0.668, "C2\nswap", 7.1, "bold")
ar(ax, x + 0.086, 0.668, x + 0.120, 0.668, C["red"], 1.0)
tx(ax, x + 0.112, 0.565, "CUG = U(C1) − U(C2)", 8.0, "bold")
tx(ax, x + 0.030, 0.505, "MovieLens-1M + Last.fm-1K", 6.7, "bold", ha="left")
tx(ax, x + 0.030, 0.465, "• gender confirmatory", 6.2, ha="left")
tx(ax, x + 0.030, 0.430, "• age robustness", 6.2, ha="left")
tx(ax, x + 0.030, 0.390, "RBO / Jaccard = sensitivity only", 5.9, c=C["muted"], ha="left")
tx(ax, x + 0.030, 0.330, "Primary inference", 6.4, "bold", C["red"], ha="left")
tx(ax, x + 0.030, 0.292, "paired sign-flip + bootstrap CI", 6.1, ha="left")

# RQ2
x = 0.265
rr(ax, x + 0.012, 0.620, 0.072, 0.095, C["green2"], C["green"], r=0.010)
rr(ax, x + 0.125, 0.620, 0.072, 0.095, C["gold2"], C["gold"], r=0.010)
tx(ax, x + 0.048, 0.668, "C3\ntrue", 7.1, "bold")
tx(ax, x + 0.161, 0.668, "C4\nshuffled", 7.1, "bold")
ar(ax, x + 0.086, 0.668, x + 0.120, 0.668, C["gold"], 1.0)
tx(ax, x + 0.112, 0.565, "PVA = U(C3) − U(C4)", 8.0, "bold")
tx(ax, x + 0.030, 0.505, "3 measured-personality datasets", 6.7, "bold", ha="left")
tx(ax, x + 0.030, 0.465, "• whole-profile derangement", 6.2, ha="left")
tx(ax, x + 0.030, 0.430, "• C5 one-trait interventions", 6.2, ha="left")
tx(ax, x + 0.030, 0.390, "measured psychometrics only", 5.9, c=C["muted"], ha="left")
tx(ax, x + 0.030, 0.330, "Primary inference", 6.4, "bold", C["green"], ha="left")
tx(ax, x + 0.030, 0.292, "same paired user-level stack", 6.1, ha="left")

# RQ3
x = 0.510
tx(ax, x + 0.030, 0.695, "Registered factors", 6.8, "bold", C["blue"], ha="left")
for i, (k, v) in enumerate([("Prompt", "3 paraphrases"), ("Cue", "3 realizations"), ("Order", "3 permutations"), ("K", "5 / 10 / 20"), ("Repeat", "3 main; 10 stress")]):
    yy = 0.635 - i * 0.057
    rr(ax, x + 0.026, yy - 0.022, 0.070, 0.043, C["panel"], C["line"], r=0.008)
    tx(ax, x + 0.061, yy, k, 6.0, "bold")
    tx(ax, x + 0.102, yy, v, 5.9, ha="left")
rr(ax, x + 0.026, 0.260, 0.160, 0.105, C["cyan2"], C["cyan"], r=0.010)
tx(ax, x + 0.041, 0.340, "FairSynth-360", 6.7, "bold", C["cyan"], ha="left")
tx(ax, x + 0.041, 0.313, "identity irrelevance known", 5.9, ha="left")
tx(ax, x + 0.041, 0.286, "real / synthetic reported separately", 5.7, c=C["muted"], ha="left")

# RQ4
x = 0.755
shield(ax, x + 0.050, 0.680, 0.024, C["violet"])
tx(ax, x + 0.080, 0.697, "3-way mitigation comparison", 6.7, "bold", ha="left")
for i, label in enumerate(["unmitigated audit", "identity-irrelevance prompt", "contextual PAIR"]):
    yy = 0.620 - i * 0.052
    tx(ax, x + 0.035, yy, f"{i+1}", 6.0, "bold", C["violet"])
    tx(ax, x + 0.055, yy, label, 5.9, ha="left")
rr(ax, x + 0.026, 0.375, 0.160, 0.105, C["violet2"], C["violet"], r=0.010)
tx(ax, x + 0.041, 0.450, "Validation-only selection", 6.5, "bold", C["violet"], ha="left")
tx(ax, x + 0.041, 0.420, "≥95% baseline validation nDCG", 5.9, ha="left")
tx(ax, x + 0.041, 0.393, "then minimize |CUG|", 5.9, ha="left")
tx(ax, x + 0.030, 0.335, "No test-set point selection", 6.1, "bold", C["red"], ha="left")
tx(ax, x + 0.030, 0.302, "report ‘no eligible point’ if needed", 5.8, c=C["muted"], ha="left")

ax.plot([0.025, 0.975], [0.120, 0.120], c=C["line"], lw=0.8)
tx(ax, 0.025, 0.075, "Shared contract: user is the unit • frozen prompt/candidate task • repeated generations • invalid-output accounting • Holm within pre-registered families.", 7.0, "bold", ha="left")
save(fig, "faireval_conditions")

# ---------------------------------------------------------------------------
# Fig. 3 — provenance / analysis pipeline.
# ---------------------------------------------------------------------------
fig, ax = setup((12.8, 4.65))
tx(ax, 0.02, 0.940, "From frozen benchmark cell to auditable paper claim", 11.4, "bold", ha="left")
tx(ax, 0.02, 0.900, "All paper numbers flow through hashed artifacts; result figures remain data-generated, while the conceptual figures have editable SVG sources.", 7.7, c=C["muted"], ha="left")
steps = [
    (0.025, 0.135, "Frozen\ninstance", "release • split • candidates", C["blue2"], C["blue"]),
    (0.190, 0.145, "Immutable\nrun cell", "prompt • model • seed • K", C["panel"], C["slate"]),
    (0.360, 0.145, "Validated\nranking", "exact K • IDs • repair audit", C["green2"], C["green"]),
    (0.535, 0.145, "User-level\nestimand", "CUG • PVA • IOD • RBO", C["gold2"], C["gold"]),
    (0.710, 0.145, "Inference\nartifact", "CI • permutation • Holm", C["violet2"], C["violet"]),
    (0.885, 0.095, "Paper\nclaim", "generated table / figure", C["white"], C["ink"]),
]
for x, w, head, sub, fc, ec in steps:
    rr(ax, x, 0.470, w, 0.190, fc, ec, lw=0.95, r=0.014)
    tx(ax, x + w / 2, 0.590, head, 7.9, "bold")
    tx(ax, x + w / 2, 0.532, sub, 5.8, c=C["muted"])
for a, b in [(0.160, 0.190), (0.335, 0.360), (0.505, 0.535), (0.680, 0.710), (0.855, 0.885)]:
    ar(ax, a, 0.565, b, 0.565, C["muted"], 1.05)
rr(ax, 0.195, 0.165, 0.300, 0.155, C["cyan2"], C["cyan"], r=0.012)
model(ax, 0.225, 0.250, 0.016, C["cyan"])
tx(ax, 0.253, 0.309, "Local white-box side channel", 6.9, "bold", C["cyan"], ha="left")
tx(ax, 0.253, 0.274, "token log-p • NLL • perplexity • margin", 5.9, ha="left")
tx(ax, 0.253, 0.247, "auxiliary and explicitly uncalibrated", 5.8, c=C["muted"], ha="left")
tx(ax, 0.253, 0.216, "never pooled with unavailable hosted internals", 5.8, c=C["muted"], ha="left")
ar(ax, 0.435, 0.468, 0.350, 0.324, C["cyan"], 0.9)
rr(ax, 0.555, 0.165, 0.390, 0.155, C["red2"], C["red"], r=0.012)
shield(ax, 0.585, 0.250, 0.016, C["red"])
tx(ax, 0.618, 0.309, "RQ4 validation-frozen mitigation branch", 6.9, "bold", C["red"], ha="left")
tx(ax, 0.618, 0.274, "PAIRₐ(i) = α b(i) + (1−α) rₐ(i) − λ Varₐ′ rₐ′(i)", 5.9, ha="left")
tx(ax, 0.618, 0.247, "(α, λ) selected on validation only, then frozen for test", 5.8, c=C["muted"], ha="left")
tx(ax, 0.618, 0.216, "test outcomes never choose the operating point", 5.8, c=C["muted"], ha="left")
ar(ax, 0.610, 0.468, 0.690, 0.324, C["red"], 0.9)
save(fig, "faireval_evaluation_pipeline")

print(OUT)
