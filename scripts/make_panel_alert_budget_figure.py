from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE = PROJECT_ROOT / "reports" / "tables" / "panel_alert_budget_paper_table.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT = FIG_DIR / "fig_panel_alert_budget_hits.png"
REPORT = PROJECT_ROOT / "reports" / "panel_alert_budget_figure.md"

GROUP_ORDER = [
    "L3_external_only",
    "L4_own_plus_external",
    "L5_own_external_total_network",
    "L8_own_external_total_equal_placebo",
    "L9_own_external_total_shuffled_placebo",
    "L10_own_external_total_random_placebo",
]

LABELS = {
    "L3_external_only": "External\nonly",
    "L4_own_plus_external": "Own +\nexternal",
    "L5_own_external_total_network": "True total\nnetwork",
    "L8_own_external_total_equal_placebo": "Equal\nplacebo",
    "L9_own_external_total_shuffled_placebo": "Shuffled\nplacebo",
    "L10_own_external_total_random_placebo": "Random\nplacebo",
}

COLORS = {
    "L3_external_only": "#88CCEE",
    "L4_own_plus_external": "#44AA99",
    "L5_own_external_total_network": "#4477AA",
    "L8_own_external_total_equal_placebo": "#BBBBBB",
    "L9_own_external_total_shuffled_placebo": "#999999",
    "L10_own_external_total_random_placebo": "#777777",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def make_figure(table: pd.DataFrame) -> None:
    plot = table.loc[table["feature_group"].isin(GROUP_ORDER)].copy()
    budgets = sorted(plot["budget"].unique())
    x = np.arange(len(budgets))
    width = 0.12
    offsets = np.linspace(-width * 2.5, width * 2.5, len(GROUP_ORDER))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for offset, group in zip(offsets, GROUP_ORDER):
        group_df = plot.loc[plot["feature_group"].eq(group)].set_index("budget").loc[budgets]
        bars = ax.bar(
            x + offset,
            group_df["total_hit_delta_vs_operational"],
            width=width,
            label=LABELS[group].replace("\n", " "),
            color=COLORS[group],
            alpha=0.92 if group == "L5_own_external_total_network" else 0.78,
            edgecolor="#222222" if group == "L5_own_external_total_network" else "none",
            linewidth=0.8 if group == "L5_own_external_total_network" else 0.0,
        )
        for bar, hits in zip(bars, group_df["total_hits"]):
            height = bar.get_height()
            if abs(height) >= 2:
                va = "bottom" if height > 0 else "top"
                y = height + (0.25 if height > 0 else -0.25)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    f"{int(hits)}",
                    ha="center",
                    va=va,
                    fontsize=7,
                    color="#222222",
                )

    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{budget}\nalerts/fold" for budget in budgets])
    ax.set_ylabel("Hit delta versus operational baseline")
    ax.set_title("Fixed-budget alert ranking: event/network features versus operational baseline")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, -0.20))
    ax.set_ylim(
        min(-4.8, plot["total_hit_delta_vs_operational"].min() - 1.0),
        max(10.0, plot["total_hit_delta_vs_operational"].max() + 1.4),
    )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(table: pd.DataFrame) -> None:
    l5 = table.loc[table["feature_group"].eq("L5_own_external_total_network")].copy()
    content = f"""# Panel Alert-Budget Figure

## Purpose

This figure visualizes the paper-facing fixed alert-budget table. It emphasizes hit deltas versus the operational baseline while keeping placebo variants visible.

## Figure

- File: `reports/figures/fig_panel_alert_budget_hits.png`
- Source table: `reports/tables/panel_alert_budget_paper_table.csv`

## True Total-Network Summary

{l5[["budget", "total_hits", "baseline_total_hits", "total_hit_delta_vs_operational", "folds_with_more_hits", "mean_precision_at_budget"]].to_markdown(index=False)}

## Suggested Caption

Fixed-budget alert-ranking performance under the locked rolling-origin panel benchmark. Bars show total hit deltas relative to the operational baseline across the 2023, 2024, and 2025 test folds; labels show total abnormal country-week hits. The true total-network event bundle improves medium-budget alert ranking at top-25 and top-50 alerts per fold, but not at top-10. Placebo variants are shown to prevent overclaiming exact network-weight superiority.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    if not TABLE.exists():
        raise FileNotFoundError(
            f"Missing {TABLE}. Run scripts/make_panel_alert_budget_paper_table.py first."
        )
    setup_style()
    table = pd.read_csv(TABLE)
    make_figure(table)
    write_report(table)
    print(f"Saved figure: {OUTPUT}")
    print(f"Saved report: {REPORT}")


if __name__ == "__main__":
    run()
