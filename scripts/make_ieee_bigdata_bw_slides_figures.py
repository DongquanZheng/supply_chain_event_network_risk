"""Generate monochrome slide figures for the IEEE BigData presentation.

This reporting-only script reads existing PortWatch/GDELT/WITS result tables and
creates black-and-white, publication-style figure variants for the LaTeX slides.
It does not train models, change results, or overwrite the main paper figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "reports" / "tables"
FIG_DIR = ROOT / "reports" / "figures"

MODEL_LADDER_PATH = TABLE_DIR / "ieee_bigdata_table2_main_model_ladder.csv"
NETWORK_AUDIT_PATH = TABLE_DIR / "main_paper_network_audit_checks.csv"
DEPLOYMENT_PATH = TABLE_DIR / "main_paper_deployment_checks.csv"
GDELT_AUDIT_SUMMARY_PATH = TABLE_DIR / "ieee_bigdata_table5_gdelt_conversion_audit.csv"


def set_bw_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def clean_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.grid(axis=grid_axis, color="#d9d9d9", linewidth=0.6, alpha=0.75)


def save_fig1_framework() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=180)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        (0.16, 0.76, "PortWatch\nOperational activity\nvulnerability"),
        (0.16, 0.50, "GDELT\nExternal event\npressure"),
        (0.16, 0.24, "WITS\nTrade-dependency\nexposure"),
        (0.53, 0.50, "Network-Gated\nEvent Conversion"),
        (0.84, 0.50, "Guarded\nAlert Output"),
    ]

    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor="#f2f2f2",
                edgecolor="black",
                linewidth=1.1,
            ),
        )

    for y in [0.76, 0.50, 0.24]:
        ax.add_patch(
            FancyArrowPatch(
                (0.29, y),
                (0.42, 0.50),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.1,
                color="black",
            )
        )
    ax.add_patch(
        FancyArrowPatch(
            (0.65, 0.50),
            (0.75, 0.50),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.1,
            color="black",
        )
    )
    ax.text(
        0.50,
        0.94,
        "Reliability-aware three-source framework",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ieee_bigdata_fig1_framework_bw.png", bbox_inches="tight")
    plt.close(fig)


def save_fig2_temporal_ladder(table2: pd.DataFrame) -> None:
    subset = table2[
        table2["evaluation"].isin(
            ["temporal_main_ladder", "temporal_guarded_integration", "temporal_conversion_propensity"]
        )
    ].copy()
    subset = subset[~subset["paper_label"].str.contains("placebo", case=False, na=False)].copy()
    subset["mean_main_pr_auc"] = pd.to_numeric(subset["mean_main_pr_auc"])
    subset["main_top25_hits"] = pd.to_numeric(subset["main_top25_hits"])
    labels = ["Operational", "GDELT", "WITS\nadditive", "True\ngated", "CP\ngated", "AA9"]
    subset = subset.head(len(labels))

    x = np.arange(len(subset))
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=180)
    bars = ax.bar(
        x,
        subset["mean_main_pr_auc"],
        color="#c9c9c9",
        edgecolor="black",
        linewidth=1.0,
        hatch="//",
    )
    ax.set_ylabel("Mean PR-AUC")
    ax.set_ylim(0.185, 0.203)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    clean_axes(ax)

    for bar, value in zip(bars, subset["mean_main_pr_auc"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.00045,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax2 = ax.twinx()
    ax2.plot(
        x,
        subset["main_top25_hits"],
        color="black",
        marker="o",
        linewidth=1.6,
        markersize=5,
    )
    ax2.set_ylabel("Top-25 hits")
    ax2.set_ylim(20, 32)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)

    ax.set_title("Temporal model ladder", pad=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ieee_bigdata_fig2_temporal_ladder_bw.png", bbox_inches="tight")
    plt.close(fig)


def save_fig3_network_audit(network_audit: pd.DataFrame) -> None:
    wanted = [
        ("temporal compact", "compact_gated_true_vs_wits_additive", "Compact gated vs WITS additive"),
        ("temporal compact", "compact_gated_true_vs_equal_placebo", "Compact gated vs equal"),
        ("temporal compact", "compact_gated_true_vs_random_placebo", "Compact gated vs random"),
        ("temporal compact", "compact_gated_true_vs_shuffled_placebo", "Compact gated vs shuffled"),
        ("temporal CP", "true_cp_gated_vs_equal_placebo", "CP gated vs equal"),
        ("temporal CP", "true_cp_gated_vs_shuffled_placebo", "CP gated vs shuffled"),
        ("AA9", "guarded_policy_vs_gdelt", "AA9 vs GDELT"),
        ("AA9", "guarded_gated_vs_true_wits_additive", "AA9 vs WITS additive"),
        ("LOCO", "compact_gated_true_vs_NG6_equal_compact_gated_placebo", "LOCO compact vs equal"),
    ]

    frames = []
    for group, contrast, label in wanted:
        hit = network_audit[network_audit["contrast"] == contrast].copy()
        if hit.empty:
            continue
        hit["group"] = group
        hit["label"] = label
        frames.append(hit)
    subset = pd.concat(frames, ignore_index=True)

    y = np.arange(len(subset))[::-1]
    deltas = subset["pooled_pr_auc_delta"].astype(float)
    low = subset["ci_low"].astype(float)
    high = subset["ci_high"].astype(float)

    fig, ax = plt.subplots(figsize=(11, 6.6), dpi=180)
    ax.errorbar(
        deltas,
        y,
        xerr=[deltas - low, high - deltas],
        fmt="o",
        color="black",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
        markersize=5,
    )
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(subset["label"], fontsize=10)
    ax.set_xlabel("Pooled PR-AUC delta")
    ax.set_title("Network audit contrasts against additive and placebo controls", pad=12)
    clean_axes(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ieee_bigdata_fig3_network_audit_deltas_bw.png", bbox_inches="tight")
    plt.close(fig)


def save_fig4_shortfall() -> None:
    labels = ["WITS additive", "True gated", "AA9 guarded"]
    nonpositive = np.array([0.1513, 0.1442, 0.1467])
    positive = np.array([0.2179, 0.2203, 0.2216])
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=180)
    bars1 = ax.bar(
        x - width / 2,
        nonpositive,
        width,
        label="current shortfall <= 0",
        color="#eeeeee",
        edgecolor="black",
        linewidth=1.0,
        hatch="..",
    )
    bars2 = ax.bar(
        x + width / 2,
        positive,
        width,
        label="current shortfall > 0",
        color="#bdbdbd",
        edgecolor="black",
        linewidth=1.0,
        hatch="//",
    )
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 0.255)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    clean_axes(ax)
    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.004,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    ax.set_title("Operational shortfall conditions event conversion", pad=12)
    fig.tight_layout(rect=[0.02, 0.08, 1, 1])
    fig.savefig(FIG_DIR / "ieee_bigdata_fig4_shortfall_conversion_bw.png", bbox_inches="tight")
    plt.close(fig)


def save_fig5_gdelt_audit(table5: pd.DataFrame) -> None:
    dims = ["Event relevance", "Network audit support", "Plausible conversion path"]
    hatches = ["", "//", "..", "xx"]
    label_map = {
        "direct_or_trade_logistics_related": "Direct trade\nor logistics",
        "indirect_macro_or_risk_related": "Indirect macro\nor risk",
        "weak_or_broad_media_signal": "Weak/broad\nmedia signal",
        "supports_true_wits_over_placebo": "Supports\ntrue WITS",
        "supports_true_wits_over_placebos": "Supports\ntrue WITS",
        "partial_or_mixed_support": "Partial/mixed\nsupport",
        "placebo_or_broad_media_dominates": "Placebo/broad\nmedia",
        "strong_plausible_path": "Strong\npath",
        "yes_strong": "Strong\npath",
        "weak_or_non_network_path": "Weak/non-network\npath",
        "yes_weak_or_nonnetwork": "Weak/non-network\npath",
        "no_observed_conversion": "No observed\nconversion",
        "no_observed_conversion_false_alert": "No observed\nconversion",
        "unclear_or_missed_conversion": "Unclear/missed\nconversion",
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 5.8), dpi=180)
    for ax, dim in zip(axes, dims):
        data = table5[table5["audit_dimension"] == dim].copy()
        data = data.sort_values("cases", ascending=True)
        bars = ax.barh(
            data["label"].map(label_map).fillna(data["label"]),
            data["cases"],
            color="#d0d0d0",
            edgecolor="black",
            linewidth=0.9,
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width())}",
                va="center",
                fontsize=9,
            )
        ax.set_title(dim, fontsize=12)
        ax.set_xlabel("Cases")
        ax.set_xlim(0, max(28, data["cases"].max() + 4))
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=10)
        clean_axes(ax, grid_axis="x")
    fig.suptitle("GDELT conversion audit", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0.01, 0.01, 1, 0.94])
    fig.savefig(FIG_DIR / "ieee_bigdata_fig5_gdelt_conversion_audit_bw.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_bw_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    table2 = pd.read_csv(MODEL_LADDER_PATH)
    network_audit = pd.read_csv(NETWORK_AUDIT_PATH)
    gdelt_audit_summary = pd.read_csv(GDELT_AUDIT_SUMMARY_PATH)

    save_fig1_framework()
    save_fig2_temporal_ladder(table2)
    save_fig3_network_audit(network_audit)
    save_fig4_shortfall()
    save_fig5_gdelt_audit(gdelt_audit_summary)
    print("Wrote monochrome slide figures.")


if __name__ == "__main__":
    main()
