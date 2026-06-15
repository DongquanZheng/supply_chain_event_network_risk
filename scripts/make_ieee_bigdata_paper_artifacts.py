"""Build paper-ready IEEE BigData tables and figures.

This script is reporting-only. It reads existing PortWatch/GDELT/WITS panel and
consolidated result artifacts, then writes compact tables and figures for the
main paper draft. It does not train models and does not introduce fourth-source
data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "reports" / "tables"
FIG_DIR = ROOT / "reports" / "figures"

PANEL_PATH = ROOT / "data" / "processed" / "multicountry32_container_event_network_benchmark.csv"
MODEL_LADDER_PATH = TABLE_DIR / "main_paper_consolidated_model_ladder.csv"
NETWORK_AUDIT_PATH = TABLE_DIR / "main_paper_network_audit_checks.csv"
DEPLOYMENT_PATH = TABLE_DIR / "main_paper_deployment_checks.csv"
GDELT_AUDIT_PATH = TABLE_DIR / "gdelt_conversion_audit.csv"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def write_table(df: pd.DataFrame, stem: str, caption: str) -> None:
    csv_path = TABLE_DIR / f"{stem}.csv"
    md_path = TABLE_DIR / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {caption}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")


def fmt_float(value: float | int | str, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def build_table1() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_PATH, parse_dates=["week"])
    deploy = pd.read_csv(DEPLOYMENT_PATH)
    severe_rows = deploy[
        (deploy["evaluation"] == "target_sensitivity")
        & (deploy["subgroup"] == "sigma_2p0_severe")
    ]
    severe_positives = int(severe_rows["positives"].dropna().iloc[0]) if not severe_rows.empty else None
    test_rows = int(severe_rows["rows"].dropna().iloc[0]) if not severe_rows.empty else None

    rows = [
        ("Panel scope", "Countries", panel["ISO3"].nunique(), "Expanded32 country-week panel"),
        ("Panel scope", "Country-week rows", len(panel), "Full processed PortWatch/GDELT/WITS panel"),
        (
            "Panel scope",
            "Week range",
            f"{panel['week'].min().date()} to {panel['week'].max().date()}",
            "Weekly country panel",
        ),
        (
            "Prediction target",
            "Main positives",
            int(panel["abnormal_next_week_container"].sum()),
            "Next-week abnormal container activity",
        ),
        (
            "Prediction target",
            "Main positive rate",
            fmt_float(panel["abnormal_next_week_container"].mean(), 3),
            "Positive weeks / country-week rows",
        ),
        (
            "Prediction target",
            "Severe positives in stress set",
            severe_positives,
            "2.0-sigma severe-label guardrail from deployment checks",
        ),
        (
            "Evaluation",
            "Temporal stress rows",
            test_rows,
            "Rows used in AA9 target-sensitivity checks",
        ),
        (
            "Data sources",
            "Main sources",
            "PortWatch; GDELT; WITS",
            "No fourth-source data in main claims",
        ),
    ]
    return pd.DataFrame(rows, columns=["section", "item", "value", "note"])


def build_table2(model_ladder: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        ("temporal_main_ladder", "NG0_portwatch_operational", "Operational only"),
        ("temporal_main_ladder", "NG1_portwatch_gdelt_additive", "Operational + GDELT"),
        ("temporal_main_ladder", "NG2_portwatch_gdelt_wits_additive", "Operational + GDELT + WITS additive"),
        ("temporal_main_ladder", "NG5_compact_network_gated_true", "Compact true WITS gated"),
        ("temporal_main_ladder", "NG6_equal_compact_gated_placebo", "Equal WITS gated placebo"),
        ("temporal_main_ladder", "NG6_random_compact_gated_placebo", "Random WITS gated placebo"),
        ("temporal_main_ladder", "NG6_shuffled_compact_gated_placebo", "Shuffled WITS gated placebo"),
        ("temporal_conversion_propensity", "GCL4_true_wits_conversion_propensity_gated", "Conversion-propensity gated"),
        ("temporal_guarded_integration", "AA9_gated_if_validation_safe_else_additive", "AA9 guarded integration"),
        ("loco_transfer", "NG0_portwatch_operational", "LOCO operational"),
        ("loco_transfer", "NG2_portwatch_gdelt_wits_additive", "LOCO WITS additive"),
        ("loco_transfer", "NG5_compact_network_gated_true", "LOCO compact true gated"),
        ("loco_transfer", "NG6_equal_compact_gated_placebo", "LOCO equal gated placebo"),
    ]
    frames = []
    for evaluation, artifact_id, label in wanted:
        hit = model_ladder[
            (model_ladder["evaluation"] == evaluation)
            & (model_ladder["artifact_id"] == artifact_id)
        ].copy()
        if hit.empty:
            continue
        hit["paper_label"] = label
        frames.append(hit)
    out = pd.concat(frames, ignore_index=True)
    out = out[
        [
            "evaluation",
            "paper_label",
            "mean_main_pr_auc",
            "mean_severe_pr_auc",
            "main_top10_hits",
            "main_top25_hits",
            "severe_top25_hits",
        ]
    ].copy()
    for col in ["mean_main_pr_auc", "mean_severe_pr_auc"]:
        out[col] = out[col].map(lambda x: fmt_float(x, 4))
    return out


def build_table3(network_audit: pd.DataFrame) -> pd.DataFrame:
    wanted_contrasts = [
        "compact_gated_true_vs_wits_additive",
        "compact_gated_true_vs_equal_placebo",
        "compact_gated_true_vs_random_placebo",
        "compact_gated_true_vs_shuffled_placebo",
        "compact_gated_true_vs_NG2_portwatch_gdelt_wits_additive",
        "compact_gated_true_vs_NG6_equal_compact_gated_placebo",
        "true_cp_gated_vs_true_wits_additive",
        "true_cp_gated_vs_equal_placebo",
        "true_cp_gated_vs_shuffled_placebo",
        "guarded_gated_vs_true_wits_additive",
        "guarded_policy_vs_equal_placebo",
        "guarded_policy_vs_gdelt",
    ]
    out = network_audit[network_audit["contrast"].isin(wanted_contrasts)].copy()
    out = out[
        [
            "evaluation",
            "contrast",
            "pooled_pr_auc_delta",
            "ci_low",
            "ci_high",
            "top25_hit_delta",
            "audit_result",
        ]
    ]
    for col in ["pooled_pr_auc_delta", "ci_low", "ci_high"]:
        out[col] = out[col].map(lambda x: fmt_float(x, 4))
    return out


def build_table4(deployment: pd.DataFrame) -> pd.DataFrame:
    target = deployment[
        (deployment["evaluation"] == "target_sensitivity")
        & (deployment["policy"].isin(["AA0_fixed_operational", "AA2_fixed_true_wits_additive", "AA3_fixed_true_gated", "AA9_gated_if_validation_safe_else_additive"]))
    ].copy()
    sub = deployment[
        (deployment["evaluation"] == "shortfall_subgroup")
        & (deployment["policy"].isin(["AA2_fixed_true_wits_additive", "AA3_fixed_true_gated", "AA9_gated_if_validation_safe_else_additive"]))
    ].copy()
    out = pd.concat([target, sub], ignore_index=True)
    out["context"] = out["subgroup"]
    out = out[["evaluation", "context", "policy", "rows", "positives", "main_or_label_pr_auc", "top25_hits"]]
    out["main_or_label_pr_auc"] = out["main_or_label_pr_auc"].map(lambda x: fmt_float(x, 4))
    return out


def build_table5(gdelt_audit: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for field, label in [
        ("logistics_event_relevance", "Event relevance"),
        ("network_audit_support", "Network audit support"),
        ("plausible_conversion_path", "Plausible conversion path"),
    ]:
        counts = gdelt_audit[field].value_counts().rename_axis("label").reset_index(name="cases")
        counts.insert(0, "audit_dimension", label)
        pieces.append(counts)
    return pd.concat(pieces, ignore_index=True)


def save_fig1_framework() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        (0.16, 0.72, "PortWatch\nOperational vulnerability\nactivity, shortfall, labels"),
        (0.16, 0.48, "GDELT\nExternal event pressure\nown + partner events"),
        (0.16, 0.24, "WITS\nTrade dependency exposure\ntrue/equal/random/shuffled"),
        (0.53, 0.48, "Network-Gated Event Conversion\nvulnerability x event x exposure"),
        (0.84, 0.48, "Deployment-Aware Alert Output\nAA9 guarded fallback"),
    ]
    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f7f7", edgecolor="#333333", linewidth=1.1),
        )
    arrowprops = dict(arrowstyle="->", color="#333333", linewidth=1.4)
    for y in [0.72, 0.48, 0.24]:
        ax.annotate("", xy=(0.40, 0.48), xytext=(0.29, y), arrowprops=arrowprops)
    ax.annotate("", xy=(0.74, 0.48), xytext=(0.65, 0.48), arrowprops=arrowprops)
    ax.text(0.5, 0.92, "Three-source main-paper framework: PortWatch + GDELT + WITS", ha="center", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ieee_bigdata_fig1_framework.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fig2_ladder(table2: pd.DataFrame) -> None:
    subset = table2[table2["evaluation"].isin(["temporal_main_ladder", "temporal_guarded_integration", "temporal_conversion_propensity"])].copy()
    subset = subset[~subset["paper_label"].str.contains("placebo", case=False, na=False)]
    subset["mean_main_pr_auc_num"] = subset["mean_main_pr_auc"].astype(float)
    subset["main_top25_hits_num"] = pd.to_numeric(subset["main_top25_hits"], errors="coerce")
    labels = subset["paper_label"].tolist()
    x = range(len(subset))
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.bar(x, subset["mean_main_pr_auc_num"], color="#4c78a8", alpha=0.85)
    ax1.set_ylabel("Mean PR-AUC")
    ax1.set_ylim(0.17, 0.205)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(list(x), subset["main_top25_hits_num"], color="#f58518", marker="o", linewidth=2)
    ax2.set_ylabel("Top-25 hits")
    ax1.set_title("Main temporal ladder: PR-AUC and alert-budget behavior")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ieee_bigdata_fig2_temporal_ladder.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fig3_network_audit(network_audit: pd.DataFrame) -> None:
    subset = network_audit[
        network_audit["contrast"].isin(
            [
                "compact_gated_true_vs_wits_additive",
                "compact_gated_true_vs_equal_placebo",
                "compact_gated_true_vs_random_placebo",
                "compact_gated_true_vs_shuffled_placebo",
                "compact_gated_true_vs_NG2_portwatch_gdelt_wits_additive",
                "compact_gated_true_vs_NG6_equal_compact_gated_placebo",
                "true_cp_gated_vs_true_wits_additive",
                "true_cp_gated_vs_equal_placebo",
                "true_cp_gated_vs_shuffled_placebo",
                "guarded_gated_vs_true_wits_additive",
            ]
        )
    ].copy()
    label_map = {
        "compact_gated_true_vs_wits_additive": "Temporal compact gated\nvs WITS additive",
        "compact_gated_true_vs_equal_placebo": "Temporal compact gated\nvs equal placebo",
        "compact_gated_true_vs_random_placebo": "Temporal compact gated\nvs random placebo",
        "compact_gated_true_vs_shuffled_placebo": "Temporal compact gated\nvs shuffled placebo",
        "compact_gated_true_vs_NG2_portwatch_gdelt_wits_additive": "LOCO compact gated\nvs WITS additive",
        "compact_gated_true_vs_NG6_equal_compact_gated_placebo": "LOCO compact gated\nvs equal placebo",
        "true_cp_gated_vs_true_wits_additive": "Conversion-propensity gated\nvs WITS additive",
        "true_cp_gated_vs_equal_placebo": "Conversion-propensity gated\nvs equal placebo",
        "true_cp_gated_vs_shuffled_placebo": "Conversion-propensity gated\nvs shuffled placebo",
        "guarded_gated_vs_true_wits_additive": "AA9 guarded integration\nvs WITS additive",
    }
    subset["label"] = subset["contrast"].map(label_map)
    subset = subset.sort_values(["evaluation", "contrast"])
    y = range(len(subset))
    fig, ax = plt.subplots(figsize=(10, 7))
    deltas = subset["pooled_pr_auc_delta"].astype(float)
    low = subset["ci_low"].astype(float)
    high = subset["ci_high"].astype(float)
    ax.errorbar(deltas, list(y), xerr=[deltas - low, high - deltas], fmt="o", color="#4c78a8", ecolor="#8da0cb", capsize=3)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(subset["label"], fontsize=8)
    ax.set_xlabel("Pooled PR-AUC delta")
    ax.set_title("Network audit contrasts: true gated versus additive/placebo controls")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ieee_bigdata_fig3_network_audit_deltas.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fig4_shortfall(deployment: pd.DataFrame) -> None:
    subset = deployment[
        (deployment["evaluation"] == "shortfall_subgroup")
        & (deployment["policy"].isin(["AA2_fixed_true_wits_additive", "AA3_fixed_true_gated", "AA9_gated_if_validation_safe_else_additive"]))
    ].copy()
    pivot = subset.pivot(index="policy", columns="subgroup", values="main_or_label_pr_auc")
    pivot = pivot.loc[["AA2_fixed_true_wits_additive", "AA3_fixed_true_gated", "AA9_gated_if_validation_safe_else_additive"]]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    pivot.plot(kind="bar", ax=ax, color=["#72b7b2", "#e45756"])
    ax.set_ylabel("PR-AUC")
    ax.set_xlabel("")
    ax.set_xticklabels(["WITS additive", "True gated", "AA9 guarded"], rotation=0)
    ax.set_title("Event conversion is conditioned by current operational shortfall")
    ax.legend(title="Shortfall regime", loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ieee_bigdata_fig4_shortfall_conversion.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fig5_gdelt_audit(table5: pd.DataFrame) -> None:
    dims = ["Event relevance", "Network audit support", "Plausible conversion path"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for ax, dim in zip(axes, dims):
        data = table5[table5["audit_dimension"] == dim].copy()
        data = data.sort_values("cases", ascending=True)
        ax.barh(data["label"], data["cases"], color=colors[: len(data)])
        ax.set_title(dim)
        ax.set_xlabel("Cases")
        ax.set_xlim(0, max(28, data["cases"].max() + 2))
    fig.suptitle("GDELT conversion audit: relevance, network support, and conversion paths", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ieee_bigdata_fig5_gdelt_conversion_audit.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    model_ladder = pd.read_csv(MODEL_LADDER_PATH)
    network_audit = pd.read_csv(NETWORK_AUDIT_PATH)
    deployment = pd.read_csv(DEPLOYMENT_PATH)
    gdelt_audit = pd.read_csv(GDELT_AUDIT_PATH)

    table1 = build_table1()
    table2 = build_table2(model_ladder)
    table3 = build_table3(network_audit)
    table4 = build_table4(deployment)
    table5 = build_table5(gdelt_audit)

    write_table(table1, "ieee_bigdata_table1_data_task", "Table 1. Data and task definition")
    write_table(table2, "ieee_bigdata_table2_main_model_ladder", "Table 2. Main model ladder")
    write_table(table3, "ieee_bigdata_table3_network_audit", "Table 3. WITS network audit and placebo checks")
    write_table(table4, "ieee_bigdata_table4_deployment_subgroups", "Table 4. Deployment and subgroup checks")
    write_table(table5, "ieee_bigdata_table5_gdelt_conversion_audit", "Table 5. GDELT conversion audit summary")

    save_fig1_framework()
    save_fig2_ladder(table2)
    save_fig3_network_audit(network_audit)
    save_fig4_shortfall(deployment)
    save_fig5_gdelt_audit(table5)

    print("Wrote IEEE BigData paper tables and figures.")


if __name__ == "__main__":
    main()
