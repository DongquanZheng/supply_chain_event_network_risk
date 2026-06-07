from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_benchmark_dataset import build_me_weights  # noqa: E402
from src.config import GDELT_TO_ISO3, JAPAN  # noqa: E402


DEFAULT_DATASET = (
    PROJECT_ROOT / "data" / "processed" / "japan_container_event_network_benchmark.csv"
)
DEFAULT_METRICS = PROJECT_ROOT / "reports" / "tables" / "benchmark_metrics.csv"
DEFAULT_STRICT_GKG = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "gkg_partner_me_strict_event_features_2021-01-01_2025-12-31.csv"
)
DEFAULT_ME_WEIGHTS = PROJECT_ROOT / "data" / "interim" / "japan_me_dependency_weights_2023.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

TARGET = "abnormal_next_week_container"


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["week"]).sort_values("week")


def save(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def fig_pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes: dict[str, tuple[float, float, float, float]] = {}
    colors = {
        "operational": "#E8F1F8",
        "event": "#F8EAC7",
        "trade": "#E5F3E6",
        "network": "#E5F3E6",
        "placebo": "#EFEAF5",
        "evaluation": "#F1ECF7",
        "line": "#061B3A",
        "panel": "#0A2A5E",
    }

    lanes = [
        (0.020, 0.085, 0.205, 0.805, "Data sources"),
        (0.250, 0.085, 0.225, 0.805, "Feature construction"),
        (0.500, 0.085, 0.300, 0.805, "Cumulative model design"),
        (0.820, 0.085, 0.160, 0.805, "Evaluation"),
    ]
    for x, y, w, h, title in lanes:
        panel = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.006",
            facecolor="white",
            edgecolor=colors["panel"],
            linewidth=0.9,
            alpha=0.95,
            zorder=0,
        )
        ax.add_patch(panel)
        ax.text(x + w / 2, 0.845, title, ha="center", va="center", fontsize=9.0, weight="bold", color=colors["line"])

    def box(key: str, label: str, x: float, y: float, w: float, h: float, color: str) -> None:
        boxes[key] = (x, y, w, h)
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.008",
            facecolor=color,
            edgecolor=colors["line"],
            linewidth=0.85,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.0, linespacing=1.14, color="#071B35", zorder=3)

    def left_out(key: str, frac: float = 0.5, pad: float = 0.006) -> tuple[float, float]:
        x, y, _, h = boxes[key]
        return x - pad, y + h * frac

    def right_out(key: str, frac: float = 0.5, pad: float = 0.006) -> tuple[float, float]:
        x, y, w, h = boxes[key]
        return x + w + pad, y + h * frac

    def top_out(key: str, pad: float = 0.006) -> tuple[float, float]:
        x, y, w, h = boxes[key]
        return x + w / 2, y + h + pad

    def bottom_out(key: str, pad: float = 0.006) -> tuple[float, float]:
        x, y, w, _ = boxes[key]
        return x + w / 2, y - pad

    def arrow_segment(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        dashed: bool = False,
        lw: float = 0.95,
    ) -> None:
        linestyle = (0, (4, 3)) if dashed else "-"
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "lw": lw,
                "color": colors["line"],
                "linestyle": linestyle,
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": 9.5,
            },
            zorder=1,
        )

    def bus(source: tuple[float, float], targets: list[tuple[float, float]], bus_x: float, *, dashed: bool = False) -> None:
        linestyle = (0, (4, 3)) if dashed else "-"
        ys = [source[1], *[target[1] for target in targets]]
        ax.plot([source[0], bus_x], [source[1], source[1]], color=colors["line"], lw=0.95, linestyle=linestyle, zorder=1)
        ax.plot([bus_x, bus_x], [min(ys), max(ys)], color=colors["line"], lw=0.95, linestyle=linestyle, zorder=1)
        for target in targets:
            arrow_segment((bus_x, target[1]), target, dashed=dashed)

    def vertical_arrow(start_key: str, end_key: str) -> None:
        arrow_segment(bottom_out(start_key), top_out(end_key), lw=0.9)

    ax.text(0.5, 0.955, "Event-Informed Prediction Framework", ha="center", va="center", fontsize=17, weight="bold", color=colors["line"])

    # Data sources, aligned to feature boxes to keep cross-column routing horizontal.
    box("portwatch", "PortWatch\nweekly container calls", 0.045, 0.670, 0.165, 0.115, colors["operational"])
    box("gdelt", "GDELT GKG\nown-country and\npartner events", 0.045, 0.460, 0.165, 0.115, colors["event"])
    box("wits", "WITS\n2023 import\ndependency", 0.045, 0.235, 0.165, 0.115, colors["trade"])

    box("operational_features", "Operational features\nlags, rolling means\nvolatility, trend", 0.272, 0.670, 0.185, 0.115, colors["operational"])
    box("event_features", "Event features\nvolume, tone\nnegative shares", 0.272, 0.460, 0.185, 0.115, colors["event"])
    box("network_features", "GDELT x WITS\nnetwork-weighted\nexposure", 0.272, 0.235, 0.185, 0.115, colors["network"])

    model_x, model_w, model_h = 0.535, 0.215, 0.075
    box("m1", "M1\nOperational baseline", model_x, 0.690, model_w, model_h, colors["operational"])
    box("m2", "M2\nM1 + own-country news", model_x, 0.585, model_w, model_h, colors["event"])
    box("m3", "M3\nM2 + external events", model_x, 0.480, model_w, model_h, colors["event"])
    box("m4", "M4\nM3 + total-import\nnetwork exposure", model_x, 0.375, model_w, model_h, colors["network"])
    box("m5", "M5\nM3 + sector-specific\nnetwork exposure", model_x, 0.270, model_w, model_h, colors["network"])
    box("m6", "M6\nPlacebo / shuffled\nexposure variants", model_x, 0.145, model_w, 0.075, colors["placebo"])

    box("validation", "Temporal out-of-sample\nvalidation\nPR-AUC, ROC-AUC, F1\nprecision, recall", 0.845, 0.4425, 0.125, 0.150, colors["evaluation"])
    box("robustness", "Robustness checks\nablation, placebo\nshuffled exposure", 0.845, 0.120, 0.125, 0.125, colors["placebo"])

    # Data to feature construction.
    arrow_segment(right_out("portwatch"), left_out("operational_features"))
    arrow_segment(right_out("gdelt"), left_out("event_features"))
    arrow_segment(right_out("wits"), left_out("network_features"))
    arrow_segment(bottom_out("event_features"), top_out("network_features"))

    # Feature construction to model ladder.
    arrow_segment(right_out("operational_features"), left_out("m1"))
    arrow_segment(right_out("event_features"), left_out("m3"))
    bus(right_out("network_features"), [left_out("m4"), left_out("m5")], bus_x=0.498)
    bus(right_out("network_features", 0.36), [left_out("m6")], bus_x=0.498, dashed=True)

    # Main cumulative ladder and evaluation bus.
    vertical_arrow("m1", "m2")
    vertical_arrow("m2", "m3")
    vertical_arrow("m3", "m4")
    vertical_arrow("m4", "m5")

    eval_bus_x = 0.775
    ax.plot([eval_bus_x, eval_bus_x], [0.3075, 0.7275], color=colors["line"], lw=0.95, zorder=1)
    for key in ["m1", "m2", "m3", "m4", "m5"]:
        start = right_out(key)
        ax.plot([start[0], eval_bus_x], [start[1], start[1]], color=colors["line"], lw=0.95, zorder=1)
    arrow_segment(right_out("m3"), left_out("validation"))

    arrow_segment(right_out("m6"), left_out("robustness"), dashed=True)

    save(fig, "fig_pipeline_overview.png")

def fig_target_timeseries(df: pd.DataFrame) -> None:
    positives = df[df[TARGET].eq(1)]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["week"], df["portcalls_container"], color="#1f77b4", lw=1.4)
    ax.scatter(
        positives["week"],
        positives["portcalls_container"],
        color="#d62728",
        s=24,
        label="Positive next-week abnormal label",
        zorder=3,
    )
    ax.axvspan(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"), color="#f2c94c", alpha=0.15)
    ax.axvspan(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"), color="#eb5757", alpha=0.10)
    ax.set_title("Japan weekly container port calls and positive labels")
    ax.set_xlabel("Week")
    ax.set_ylabel("Weekly container port calls")
    ax.legend(loc="upper right")
    save(fig, "fig_target_timeseries.png")


def fig_model_comparison(metrics_path: Path) -> None:
    metrics = pd.read_csv(metrics_path)
    rf = metrics[metrics["model"].eq("random_forest")].copy()
    order = [
        "M1_operational",
        "M2_simple_news",
        "M3_unweighted_me_event",
        "M4_total_import_network",
        "M5_me_network",
        "M6_equal_placebo",
        "M6_shuffled_placebo",
        "M6_random_placebo",
    ]
    rf["feature_group"] = pd.Categorical(rf["feature_group"], categories=order, ordered=True)
    rf = rf.sort_values("feature_group")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    sns.barplot(data=rf, y="feature_group", x="pr_auc", ax=axes[0], color="#4c78a8")
    sns.barplot(data=rf, y="feature_group", x="roc_auc", ax=axes[1], color="#72b7b2")
    axes[0].set_title("Random Forest PR-AUC")
    axes[1].set_title("Random Forest ROC-AUC")
    axes[0].set_ylabel("")
    axes[1].set_ylabel("")
    axes[0].set_xlabel("Test PR-AUC")
    axes[1].set_xlabel("Test ROC-AUC")
    save(fig, "fig_model_comparison_auc.png")


def fig_network_exposure_timeseries(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(df["week"], df["me_strict_unweighted_exposure"], label="M3 unweighted ME", color="#ff7f0e")
    axes[0].plot(df["week"], df["me_strict_network_exposure"], label="M5 ME network", color="#1f77b4")
    axes[0].set_ylabel("ME event exposure")
    axes[0].legend(loc="upper right")

    axes[1].plot(df["week"], df["total_network_very_negative_exposure"], label="M4 total network", color="#9467bd")
    axes[1].plot(df["week"], df["unweighted_very_negative_exposure"], label="General unweighted", color="#2ca02c")
    axes[1].set_ylabel("General exposure")
    axes[1].legend(loc="upper right")

    axes[2].bar(df["week"], df[TARGET], width=5, color="#777")
    axes[2].set_ylabel("Positive label")
    axes[2].set_xlabel("Week")
    axes[0].set_title("Event exposure time series")
    save(fig, "fig_network_exposure_timeseries.png")


def fig_exposure_correlation(df: pd.DataFrame) -> None:
    cols = [
        "unweighted_very_negative_exposure",
        "total_network_very_negative_exposure",
        "me_strict_unweighted_exposure",
        "me_strict_network_exposure",
        "me_strict_equal_exposure",
        "me_strict_shuffled_exposure",
        "me_strict_random_exposure",
    ]
    labels = {
        "unweighted_very_negative_exposure": "general unweighted",
        "total_network_very_negative_exposure": "total network",
        "me_strict_unweighted_exposure": "ME unweighted",
        "me_strict_network_exposure": "ME network",
        "me_strict_equal_exposure": "ME equal",
        "me_strict_shuffled_exposure": "ME shuffled",
        "me_strict_random_exposure": "ME random",
    }
    corr = df[cols].rename(columns=labels).corr()

    fig, ax = plt.subplots(figsize=(8.5, 6))
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="vlag", annot=True, fmt=".2f", square=True, ax=ax)
    ax.set_title("Exposure feature correlation")
    save(fig, "fig_exposure_correlation.png")


def load_or_build_me_weights(path: Path, partner_iso3: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    weights = build_me_weights(partner_iso3)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(path, index=False)
    return weights


def fig_partner_contributions(df: pd.DataFrame, strict_gkg_path: Path, weights_path: Path) -> None:
    strict = pd.read_csv(strict_gkg_path, parse_dates=["event_week"])
    strict["partner_iso3"] = strict["code"].map(GDELT_TO_ISO3)
    strict = strict.dropna(subset=["partner_iso3"])
    strict = strict[strict["partner_iso3"].ne(JAPAN.iso3)].copy()

    partner_iso3 = sorted(strict["partner_iso3"].unique())
    weights = load_or_build_me_weights(weights_path, partner_iso3)

    merged = strict.merge(
        weights[["partner_iso3", "me_dependency_share"]],
        on="partner_iso3",
        how="inner",
    )
    signal = "machinery_electronics_disruption_very_negative_share"
    merged["contribution"] = merged[signal] * merged["me_dependency_share"]

    positive_weeks = df[df[TARGET].eq(1) & (df["week"] >= "2025-01-01")]["week"].head(3)
    if positive_weeks.empty:
        positive_weeks = df[df[TARGET].eq(1)]["week"].tail(3)

    focus = merged[merged["event_week"].isin(positive_weeks)].copy()
    if focus.empty:
        return

    top_partners = (
        focus.groupby("partner_iso3")["contribution"]
        .sum()
        .sort_values(ascending=False)
        .head(6)
        .index
    )
    focus["partner_group"] = focus["partner_iso3"].where(
        focus["partner_iso3"].isin(top_partners),
        "Other",
    )
    pivot = (
        focus.groupby(["event_week", "partner_group"])["contribution"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )
    ordered_cols = [c for c in top_partners if c in pivot.columns]
    if "Other" in pivot.columns:
        ordered_cols.append("Other")
    pivot = pivot[ordered_cols]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    pivot.plot(kind="bar", stacked=True, ax=ax, width=0.8)
    ax.set_title("Partner contribution to ME network exposure in selected positive weeks")
    ax.set_xlabel("Event week")
    ax.set_ylabel("Weighted ME exposure contribution")
    ax.legend(title="Partner", bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, "fig_partner_contributions.png")


def run(args: argparse.Namespace) -> None:
    setup_style()
    df = load_dataset(Path(args.dataset))
    fig_pipeline_overview()
    fig_target_timeseries(df)
    fig_model_comparison(Path(args.metrics))
    fig_network_exposure_timeseries(df)
    fig_exposure_correlation(df)
    fig_partner_contributions(df, Path(args.strict_gkg), Path(args.me_weights))

    print(f"Saved figures to: {FIGURE_DIR}")
    for path in sorted(FIGURE_DIR.glob("fig_*.png")):
        print(path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--strict-gkg", default=str(DEFAULT_STRICT_GKG))
    parser.add_argument("--me-weights", default=str(DEFAULT_ME_WEIGHTS))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
