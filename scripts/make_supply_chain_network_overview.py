from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GDELT_TO_ISO3
from src.wits import build_partner_dependency_weights, fetch_partner_trade


DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry_container_event_network_benchmark.csv"
ME_WEIGHTS = PROJECT_ROOT / "data" / "interim" / "panel_me_dependency_weights_2023.csv"
TOTAL_WEIGHTS = PROJECT_ROOT / "data" / "interim" / "panel_total_dependency_weights_2023.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

COUNTRY_LABELS = {
    "ARE": "UAE",
    "AUS": "Australia",
    "CHN": "China",
    "DEU": "Germany",
    "IDN": "Indonesia",
    "JPN": "Japan",
    "KOR": "Korea",
    "SAU": "Saudi Arabia",
    "THA": "Thailand",
    "USA": "United States",
    "VNM": "Vietnam",
}


def build_or_load_total_weights() -> pd.DataFrame:
    if TOTAL_WEIGHTS.exists():
        return pd.read_csv(TOTAL_WEIGHTS)

    countries = sorted(set(GDELT_TO_ISO3.values()))
    frames = []
    for target_iso3 in countries:
        partners = [iso3 for iso3 in countries if iso3 != target_iso3]
        trade = fetch_partner_trade(target_iso3, year=2023)
        weights = build_partner_dependency_weights(trade, partner_iso3=partners)
        weights["ISO3"] = target_iso3
        frames.append(
            weights[
                ["ISO3", "partner_iso3", "value_thousand_usd", "import_dependency_share"]
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    TOTAL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TOTAL_WEIGHTS, index=False)
    return out


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv(DATASET, parse_dates=["week"])
    me_weights = pd.read_csv(ME_WEIGHTS)
    total_weights = build_or_load_total_weights()
    return dataset, me_weights, total_weights


def top_edges(weights: pd.DataFrame, top_k: int) -> pd.DataFrame:
    return (
        weights.sort_values(["ISO3", "import_dependency_share"], ascending=[True, False])
        .groupby("ISO3", as_index=False)
        .head(top_k)
        .copy()
    )


def make_graph(me_weights: pd.DataFrame, total_weights: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    countries = sorted(set(GDELT_TO_ISO3.values()))
    graph.add_nodes_from(countries)

    me_top = top_edges(me_weights, top_k=4)
    total_top = top_edges(total_weights, top_k=3)

    for _, row in total_top.iterrows():
        graph.add_edge(
            row["partner_iso3"],
            row["ISO3"],
            total_weight=row["import_dependency_share"],
            me_weight=0.0,
        )

    for _, row in me_top.iterrows():
        source = row["partner_iso3"]
        target = row["ISO3"]
        if graph.has_edge(source, target):
            graph[source][target]["me_weight"] = row["import_dependency_share"]
        else:
            graph.add_edge(
                source,
                target,
                total_weight=0.0,
                me_weight=row["import_dependency_share"],
            )

    return graph


def node_attributes(dataset: pd.DataFrame, me_weights: pd.DataFrame) -> pd.DataFrame:
    recent = dataset[dataset["week"].dt.year.eq(2025)]
    exposure = (
        recent.groupby("ISO3", as_index=False)
        .agg(
            avg_me_exposure=("me_network_strict_very_negative_exposure", "mean"),
            positive_rate=("abnormal_next_week_container", "mean"),
        )
    )
    incoming = (
        me_weights.groupby("partner_iso3", as_index=False)["import_dependency_share"]
        .sum()
        .rename(columns={"partner_iso3": "ISO3", "import_dependency_share": "me_partner_importance"})
    )
    out = exposure.merge(incoming, on="ISO3", how="outer").fillna(0)
    out["label"] = out["ISO3"].map(COUNTRY_LABELS).fillna(out["ISO3"])
    return out


def draw_network(dataset: pd.DataFrame, me_weights: pd.DataFrame, total_weights: pd.DataFrame) -> None:
    graph = make_graph(me_weights, total_weights)
    attrs = node_attributes(dataset, me_weights).set_index("ISO3")

    undirected = nx.Graph()
    for source, target, data in graph.edges(data=True):
        weight = data["me_weight"] + 0.6 * data["total_weight"]
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)

    pos = nx.spring_layout(undirected, seed=42, weight="weight", k=1.18, iterations=600)
    visual_offsets = {
        "CHN": np.array([0.10, -0.02]),
        "USA": np.array([-0.12, 0.08]),
        "JPN": np.array([-0.04, -0.04]),
        "KOR": np.array([-0.02, -0.08]),
        "AUS": np.array([0.07, 0.00]),
    }
    for node, offset in visual_offsets.items():
        if node in pos:
            pos[node] = pos[node] + offset

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
        }
    )
    fig, ax = plt.subplots(figsize=(13.6, 8.6))
    ax.set_title(
        "Import-Dependency Network with 2025 Event-Exposure Scores",
        fontsize=15,
        pad=14,
    )
    ax.set_axis_off()

    exposures = attrs.loc[list(graph.nodes), "avg_me_exposure"]
    node_colors = exposures.to_numpy()
    importance = attrs.loc[list(graph.nodes), "me_partner_importance"].to_numpy()
    node_sizes = 900 + 5200 * importance

    me_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["me_weight"] > 0]
    total_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["total_weight"] > 0]

    total_edge_artists = nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=total_edges,
        width=[0.35 + 4.2 * graph[u][v]["total_weight"] for u, v in total_edges],
        edge_color="#4C78A8",
        alpha=0.18,
        arrows=True,
        arrowsize=8,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )
    me_edge_artists = nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=me_edges,
        width=[0.60 + 5.25 * graph[u][v]["me_weight"] for u, v in me_edges],
        edge_color="#F58518",
        alpha=0.64,
        arrows=True,
        arrowsize=10,
        connectionstyle="arc3,rad=-0.10",
        ax=ax,
    )
    for artist_group, zorder in [(total_edge_artists, 1), (me_edge_artists, 2)]:
        if artist_group is None:
            continue
        if isinstance(artist_group, list):
            for artist in artist_group:
                artist.set_zorder(zorder)
        else:
            artist_group.set_zorder(zorder)

    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap="YlOrRd",
        linewidths=1.3,
        edgecolors="white",
        alpha=0.92,
        ax=ax,
    )
    nodes.set_zorder(3)

    labels = {node: attrs.loc[node, "label"] for node in graph.nodes}
    label_artists = nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=9,
        font_weight="bold",
        ax=ax,
    )
    for artist in label_artists.values():
        artist.set_zorder(4)
        artist.set_path_effects([pe.withStroke(linewidth=1.35, foreground="white")])

    cbar = fig.colorbar(nodes, ax=ax, fraction=0.020, pad=0.022, shrink=0.72)
    cbar.set_label("Event exposure score, 2025", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_linewidth(0.6)

    legend_items = [
        Line2D([0], [0], color="#4C78A8", lw=3.5, alpha=0.60, label="Total-import dependency, top 3/target"),
        Line2D([0], [0], color="#F58518", lw=3.5, alpha=0.85, label="Machinery/electronics dependency, top 4/target"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#999999",
            markeredgecolor="white",
            markersize=14,
            label="Node size: dependency importance",
        ),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower left",
        frameon=True,
        framealpha=0.72,
        facecolor="white",
        edgecolor="#E6E6E6",
        prop={"size": 7.5},
        borderpad=0.35,
        labelspacing=0.35,
        handlelength=2.0,
        handletextpad=0.55,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig_all_country_supply_chain_network.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_edge_table(me_weights: pd.DataFrame, total_weights: pd.DataFrame) -> None:
    me = top_edges(me_weights, top_k=4).copy()
    me["network_layer"] = "machinery_electronics"
    total = top_edges(total_weights, top_k=3).copy()
    total["network_layer"] = "total_import"
    edges = pd.concat([me, total], ignore_index=True)
    edges["source_partner"] = edges["partner_iso3"].map(COUNTRY_LABELS).fillna(edges["partner_iso3"])
    edges["target_country"] = edges["ISO3"].map(COUNTRY_LABELS).fillna(edges["ISO3"])
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    edges[
        [
            "network_layer",
            "source_partner",
            "target_country",
            "partner_iso3",
            "ISO3",
            "import_dependency_share",
            "value_thousand_usd",
        ]
    ].to_csv(TABLE_DIR / "all_country_network_overview_edges.csv", index=False)


def run() -> None:
    dataset, me_weights, total_weights = load_inputs()
    draw_network(dataset, me_weights, total_weights)
    save_edge_table(me_weights, total_weights)
    print(FIG_DIR / "fig_all_country_supply_chain_network.png")
    print(TABLE_DIR / "all_country_network_overview_edges.csv")


if __name__ == "__main__":
    run()
