from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry_container_event_network_benchmark.csv"
SUMMARY = PROJECT_ROOT / "reports" / "tables" / "panel_benchmark_summary.csv"
METRICS = PROJECT_ROOT / "reports" / "tables" / "panel_benchmark_metrics_by_fold.csv"
DELTAS = PROJECT_ROOT / "reports" / "tables" / "panel_benchmark_m5_deltas.csv"
MECHANISM = PROJECT_ROOT / "reports" / "tables" / "panel_network_mechanism_country_results.csv"
LEAVE_ONE_OUT = PROJECT_ROOT / "reports" / "tables" / "panel_leave_one_country_out_summary.csv"
COUNTERFACTUAL_SWAP = PROJECT_ROOT / "reports" / "tables" / "panel_counterfactual_network_swap_summary.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"


MODEL_ORDER = [
    "M1_operational",
    "M2_own_country_news",
    "M3_external_unweighted_events",
    "M4_total_import_network",
    "M5_me_strict_network",
    "M6a_total_equal_placebo",
    "M6b_total_shuffled_placebo",
    "M6c_total_random_placebo",
]

MODEL_LABELS = {
    "M1_operational": "M1 operational",
    "M2_own_country_news": "M2 own news",
    "M3_external_unweighted_events": "M3 external unweighted",
    "M4_total_import_network": "M4 total network",
    "M5_me_strict_network": "M5 ME network",
    "M6a_total_equal_placebo": "M6 equal placebo",
    "M6b_total_shuffled_placebo": "M6 shuffled placebo",
    "M6c_total_random_placebo": "M6 random placebo",
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


def save_model_comparison() -> None:
    summary = pd.read_csv(SUMMARY)
    rf = summary[summary["model"].eq("random_forest")].copy()
    rf = rf[rf["feature_group"].isin(MODEL_ORDER)].set_index("feature_group").loc[MODEL_ORDER].reset_index()
    labels = [MODEL_LABELS[x] for x in rf["feature_group"]]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(labels, rf["mean_pr_auc"], yerr=rf["std_pr_auc"], color="#4477AA", alpha=0.88)
    ax.set_ylabel("Mean PR-AUC across rolling test folds")
    ax.set_xlabel("")
    ax.set_title("Panel benchmark: Random Forest PR-AUC by feature group")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_panel_model_comparison_pr_auc.png", dpi=200)
    plt.close(fig)


def save_model_comparison_auc() -> None:
    summary = pd.read_csv(SUMMARY)
    rf = summary[summary["model"].eq("random_forest")].copy()
    rf = rf[rf["feature_group"].isin(MODEL_ORDER)].set_index("feature_group").loc[MODEL_ORDER].reset_index()
    labels = [MODEL_LABELS[x] for x in rf["feature_group"]]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.bar(x - width / 2, rf["mean_pr_auc"], width, label="PR-AUC", color="#4477AA", alpha=0.88)
    ax.bar(x + width / 2, rf["mean_roc_auc"], width, label="ROC-AUC", color="#66A61E", alpha=0.78)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean score across rolling test folds")
    ax.set_title("Panel benchmark: PR-AUC and ROC-AUC")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_panel_model_comparison_auc.png", dpi=220)
    plt.close(fig)


def save_fold_comparison() -> None:
    metrics = pd.read_csv(METRICS)
    rf = metrics[metrics["model"].eq("random_forest")].copy()
    keep = [
        "M1_operational",
        "M3_external_unweighted_events",
        "M4_total_import_network",
        "M5_me_strict_network",
        "M6b_total_shuffled_placebo",
    ]
    rf = rf[rf["feature_group"].isin(keep)]
    pivot = rf.pivot(index="fold", columns="feature_group", values="pr_auc")[keep]

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(marker="o", ax=ax)
    ax.set_ylabel("PR-AUC")
    ax.set_xlabel("Rolling-origin test fold")
    ax.set_title("PR-AUC stability across test years")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_panel_pr_auc_by_fold.png", dpi=200)
    plt.close(fig)


def save_target_and_exposure() -> None:
    df = pd.read_csv(DATASET, parse_dates=["week"])
    weekly = (
        df.groupby("week", as_index=False)
        .agg(
            positive_rate=("abnormal_next_week_container", "mean"),
            mean_container=("portcalls_container", "mean"),
            me_network=("me_network_strict_very_negative_exposure", "mean"),
            external_unweighted=("external_me_strict_very_negative_exposure", "mean"),
        )
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(weekly["week"], weekly["mean_container"], color="#4477AA")
    axes[0].set_ylabel("Mean container calls")
    axes[0].set_title("Panel target and event-network exposure")
    axes[1].plot(weekly["week"], weekly["me_network"], label="ME network exposure", color="#228833")
    axes[1].plot(weekly["week"], weekly["external_unweighted"], label="External unweighted ME exposure", color="#CC6677")
    axes[1].set_ylabel("Exposure")
    axes[1].legend()
    axes[2].bar(weekly["week"], weekly["positive_rate"], width=5, color="#999999")
    axes[2].set_ylabel("Positive rate")
    axes[2].set_xlabel("Week")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_panel_target_exposure_timeseries.png", dpi=200)
    plt.close(fig)


def save_exposure_correlation() -> None:
    df = pd.read_csv(DATASET, parse_dates=["week"])
    cols = [
        "external_very_negative_article_share",
        "network_very_negative_exposure",
        "equal_very_negative_exposure",
        "shuffled_very_negative_exposure",
        "external_me_strict_very_negative_exposure",
        "me_network_strict_very_negative_exposure",
        "me_equal_strict_very_negative_exposure",
        "me_shuffled_strict_very_negative_exposure",
    ]
    labels = {
        "external_very_negative_article_share": "external unweighted",
        "network_very_negative_exposure": "total network",
        "equal_very_negative_exposure": "total equal",
        "shuffled_very_negative_exposure": "total shuffled",
        "external_me_strict_very_negative_exposure": "ME unweighted",
        "me_network_strict_very_negative_exposure": "ME network",
        "me_equal_strict_very_negative_exposure": "ME equal",
        "me_shuffled_strict_very_negative_exposure": "ME shuffled",
    }
    corr = df[cols].rename(columns=labels).corr()

    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Correlation among event exposure features")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_panel_exposure_correlation.png", dpi=220)
    plt.close(fig)


def save_delta_chart() -> None:
    deltas = pd.read_csv(DELTAS).sort_values("pooled_pr_auc_delta")
    fig, ax = plt.subplots(figsize=(8, 5))
    err_low = deltas["pooled_pr_auc_delta"] - deltas["pr_delta_ci_low"]
    err_high = deltas["pr_delta_ci_high"] - deltas["pooled_pr_auc_delta"]
    ax.barh(deltas["baseline_group"], deltas["pooled_pr_auc_delta"], xerr=[err_low, err_high], color="#66AADD")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Pooled PR-AUC delta: M5 minus baseline")
    ax.set_ylabel("")
    ax.set_title("Directional M5 advantage with paired bootstrap intervals")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_panel_m5_delta_bootstrap.png", dpi=200)
    plt.close(fig)


def save_country_gain_chart() -> None:
    if not MECHANISM.exists():
        return
    country = pd.read_csv(MECHANISM).sort_values("m5_minus_m3_pr_auc")
    colors = ["#CC6677" if x < 0 else "#4477AA" for x in country["m5_minus_m3_pr_auc"]]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(country["ISO3"], country["m5_minus_m3_pr_auc"], color=colors, alpha=0.88)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel("Country-level PR-AUC gain: M5 ME network minus M3 unweighted")
    ax.set_ylabel("")
    ax.set_title("Where does machinery/electronics network weighting help?")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_panel_country_network_gain.png", dpi=220)
    plt.close(fig)


def save_leave_one_country_out_chart() -> None:
    if not LEAVE_ONE_OUT.exists():
        return
    loo = pd.read_csv(LEAVE_ONE_OUT)
    order = [
        "M1_operational",
        "M3_external_unweighted_events",
        "M4_total_import_network",
        "M5_me_strict_network",
    ]
    loo = loo.set_index("feature_group").loc[order].reset_index()
    labels = [MODEL_LABELS.get(x, x) for x in loo["feature_group"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(labels, loo["mean_pr_auc"], yerr=loo["std_pr_auc"], color="#88CCEE", alpha=0.9)
    ax.set_ylabel("Mean holdout-country PR-AUC")
    ax.set_title("Leave-one-country-out generalization")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_panel_leave_one_country_out.png", dpi=220)
    plt.close(fig)


def save_counterfactual_swap_chart() -> None:
    if not COUNTERFACTUAL_SWAP.exists():
        return
    swap = pd.read_csv(COUNTERFACTUAL_SWAP).sort_values("mean_pr_auc", ascending=True)
    colors = ["#4477AA" if x == "true_target_specific_me_network" else "#BBBBBB" for x in swap["variant"]]
    labels = [x.replace("true_target_specific_me_network", "true target-specific").replace("donor_", "donor ") for x in swap["variant"]]

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.barh(labels, swap["mean_pr_auc"], xerr=swap["std_pr_auc"], color=colors, alpha=0.9)
    ax.set_xlabel("Mean PR-AUC")
    ax.set_ylabel("")
    ax.set_title("Counterfactual donor-network swap")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_panel_counterfactual_network_swap.png", dpi=220)
    plt.close(fig)


def run() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_model_comparison()
    save_model_comparison_auc()
    save_fold_comparison()
    save_target_and_exposure()
    save_exposure_correlation()
    save_delta_chart()
    save_country_gain_chart()
    save_leave_one_country_out_chart()
    save_counterfactual_swap_chart()
    for path in sorted(FIG_DIR.glob("fig_panel_*.png")):
        print(path)


if __name__ == "__main__":
    run()
