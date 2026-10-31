#!/usr/bin/env python3
"""Regenerate every figure in docs/ from the measurement tables.

All numbers are copied from the experiment tables, not from the old PNGs.
Run:  python docs/figures/make_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUT = Path(__file__).resolve().parent
DPI = 200

# ---------------------------------------------------------------- palette
C_ARC = "#1b6a8c"
C_LRU = "#8a8f98"
C_DBL = "#d9963a"
C_TWOQ = "#a83a5b"
C_V0 = "#2e7d32"
C_V1 = "#c62828"

STYLE = {C_ARC: "-", C_LRU: "--", C_DBL: "-."}

plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

TIERS3 = [50, 100, 300]
TIERS5 = [50, 100, 150, 200, 300]

V0 = {"ARC": [35.40, 43.75, 56.64], "LRU": [25.41, 36.96, 56.10], "DBL": [33.17, 41.82, 57.21]}
V1 = {"ARC": [35.12, 43.33, 56.48], "LRU": [25.17, 36.66, 55.72], "DBL": [32.83, 41.50, 56.84]}

QUALITY = {
    "ARC": [35.12, 43.33, 48.17, 51.65, 56.48],
    "LRU": [25.17, 36.66, 43.72, 48.61, 55.72],
    "DBL": [32.83, 41.50, 47.36, 51.09, 56.84],
}
HOTSPOT = {
    "ARC": [70.03, 70.14, 70.19, 70.21, 70.23],
    "LRU": [70.26, 70.33, 70.38, 70.40, 70.40],
    "DBL": [70.24, 70.29, 70.37, 70.38, 70.40],
}
TTFT_Q = {
    "ARC": [229.50, 216.39, 201.89, 202.26, 186.64],
    "LRU": [253.47, 230.59, 207.68, 208.10, 186.99],
    "DBL": [229.25, 209.00, 196.83, 187.42, 185.15],
}
TTFT_H = {
    "ARC": [38.62, 40.39, 40.31, 40.41, 40.41],
    "LRU": [38.47, 37.88, 39.28, 39.20, 39.22],
    "DBL": [37.95, 37.58, 39.01, 38.86, 39.10],
}

ABLATION = [
    ("C0  LRU", 36.968, "c0"),
    ("C1  + frequency score", 38.286, "ladder"),
    ("C2  + ghost ticket", 38.201, "ladder"),
    ("C3  + recency resonance", 38.201, "ladder"),
    ("2Q  two queues, no ghost", 42.920, "struct"),
    ("ARC  adaptive p", 43.664, "arc"),
    ("ARC  p frozen at best", 44.473, "arc"),
    ("2Q + shared ghost", 44.897, "win"),
]

TWOQ_Q = {
    "2Q+ghost": [36.15, 44.74, 49.43, 52.90, 57.76],
    "ARC": [35.12, 43.33, 48.17, 51.65, 56.48],
    "LRU": [25.17, 36.66, 43.72, 48.61, 55.72],
    "DBL": [32.83, 41.50, 47.36, 51.09, 56.84],
}
HOTSPOT4 = {"LRU": 70.33, "DBL": 70.29, "ARC": 70.14, "2Q+ghost": 51.52}


def save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path.name)


def panel_quality_v0_v1():
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9), sharey=True)
    for ax, pol in zip(axes, ["ARC", "LRU", "DBL"]):
        ax.plot(TIERS3, V0[pol], "o-", color=C_V0, label="V0")
        ax.plot(TIERS3, V1[pol], "s--", color=C_V1, label="V1")
        ax.set_title(pol)
        ax.set_xlabel("cache size (k tokens)")
        ax.set_xticks(TIERS3)
        ax.set_ylim(20, 62)
    axes[0].set_ylabel("block hit rate (%)")
    axes[0].legend(loc="lower right")
    fig.suptitle("Porting is lossless: V0 and V1 agree at every tier (QuALITY, 3072 req)",
                 fontsize=10, y=1.04)
    save(fig, "fig_v0_v1_hitrate.png")


def panel_advantage():
    d0 = [a - b for a, b in zip(V0["ARC"], V0["LRU"])]
    d1 = [a - b for a, b in zip(V1["ARC"], V1["LRU"])]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.axhline(0, color="#444", linewidth=0.8, zorder=1)
    ax.plot(TIERS3, d0, "o-", color=C_V0, label="V0  ARC − LRU")
    ax.plot(TIERS3, d1, "s--", color=C_V1, label="V1  ARC − LRU")
    for x, a, b in zip(TIERS3, d0, d1):
        ax.annotate(f"{a:+.2f}", (x, a), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=C_V0)
        ax.annotate(f"{b:+.2f}", (x, b), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color=C_V1)
    ax.set_xticks(TIERS3)
    ax.set_xlabel("cache size (k tokens)")
    ax.set_ylabel("ARC − LRU (percentage points)")
    ax.set_title("ARC's advantage over LRU, before and after the V1 port", pad=14)
    ax.set_ylim(-1.2, 12.0)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.legend(loc="upper right")
    save(fig, "fig_arc_advantage.png")


def panel_hitrate_v1():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
    for ax, (name, d, note) in zip(axes, [
        ("QuALITY", QUALITY, "ARC > LRU at every tier"),
        ("hotspot (WildChat)", HOTSPOT, "policy-independent: shared prefix"),
    ]):
        for pol in ["ARC", "LRU", "DBL"]:
            col = {"ARC": C_ARC, "LRU": C_LRU, "DBL": C_DBL}[pol]
            ax.plot(TIERS5, d[pol], marker="o", linestyle=STYLE[col], color=col, label=pol)
        ax.set_title(f"{name}\n{note}", fontsize=9)
        ax.set_xlabel("cache size (k tokens)")
        ax.set_xticks(TIERS5)
    axes[0].set_ylabel("block hit rate (%)")
    axes[0].legend(loc="lower right")
    fig.suptitle("V1, five cache tiers, 3072 requests", fontsize=10, y=1.02)
    save(fig, "fig_v1_hitrate.png")


def panel_ttft():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
    for ax, (name, d) in zip(axes, [("QuALITY", TTFT_Q), ("hotspot (WildChat)", TTFT_H)]):
        for pol in ["ARC", "LRU", "DBL"]:
            col = {"ARC": C_ARC, "LRU": C_LRU, "DBL": C_DBL}[pol]
            ax.plot(TIERS5, d[pol], marker="o", linestyle=STYLE[col], color=col, label=pol)
        ax.set_title(name)
        ax.set_xlabel("cache size (k tokens)")
        ax.set_xticks(TIERS5)
    axes[0].set_ylabel("mean TTFT (ms)")
    axes[0].legend(loc="upper right")
    axes[1].text(0.02, 0.05, "cross-batch; differences < 2 ms are not resolvable",
                 transform=axes[1].transAxes, fontsize=7, color="#666")
    fig.suptitle("TTFT tracks hit rate (QuALITY); hotspot is dominated by session drift",
                 fontsize=10, y=1.02)
    save(fig, "fig_ttft.png")


def panel_ablation():
    labels = [a[0] for a in ABLATION]
    vals = [a[1] for a in ABLATION]
    fam = [a[2] for a in ABLATION]
    cmap = {"c0": C_LRU, "ladder": "#b9a37a", "struct": "#7d8fa8",
            "arc": C_ARC, "win": C_TWOQ}
    colors = [cmap[f] for f in fam]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = list(range(len(labels)))
    # dots, not bars: a bar chart truncated at 35 would misrepresent the gaps
    ax.hlines(y, 35.5, vals, color="#cfcfcf", linewidth=1.2, zorder=1)
    ax.scatter(vals, y, s=46, c=colors, zorder=3)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}", (v, i), textcoords="offset points", xytext=(7, 0),
                    va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(35.5, 46.4)
    ax.set_xlabel("block hit rate (%)")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title("Offline ablation: where ARC's advantage actually comes from")
    save(fig, "fig_ablation.png")


def panel_twoq():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2),
                             gridspec_kw={"width_ratios": [2.0, 1.0]})
    ax = axes[0]
    order = ["2Q+ghost", "ARC", "DBL", "LRU"]
    for pol in order:
        col = {"2Q+ghost": C_TWOQ, "ARC": C_ARC, "DBL": C_DBL, "LRU": C_LRU}[pol]
        lw = 2.2 if pol == "2Q+ghost" else 1.4
        ax.plot(TIERS5, TWOQ_Q[pol], marker="o", linestyle=STYLE.get(col, "-"),
                color=col, label=pol, linewidth=lw)
    ax.set_xticks(TIERS5)
    ax.set_xlabel("cache size (k tokens)")
    ax.set_ylabel("block hit rate (%)")
    ax.set_title("QuALITY: 2Q+ghost leads at every tier", fontsize=9)
    ax.legend(loc="lower right")

    ax2 = axes[1]
    names = ["LRU", "DBL", "ARC", "2Q+ghost"]
    vals = [HOTSPOT4[n] for n in names]
    cols = [{"2Q+ghost": C_TWOQ, "ARC": C_ARC, "DBL": C_DBL, "LRU": C_LRU}[n] for n in names]
    bars = ax2.bar(names, vals, color=cols, width=0.62)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=8)
    ax2.set_ylim(0, 80)
    ax2.set_ylabel("block hit rate (%)")
    ax2.set_title("hotspot @100k: the k=0 collapse", fontsize=9)
    ax2.tick_params(axis="x", rotation=20)
    ax2.axhline(70.33, color=C_LRU, linewidth=0.8, linestyle=":")
    fig.suptitle("2Q + shared ghost: wins on frequency-tailed load, fails on short hot sets",
                 fontsize=10, y=1.02)
    save(fig, "fig_twoq.png")


if __name__ == "__main__":
    panel_quality_v0_v1()
    panel_advantage()
    panel_hitrate_v1()
    panel_ttft()
    panel_ablation()
    panel_twoq()
