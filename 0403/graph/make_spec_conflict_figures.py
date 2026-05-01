#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from matplotlib.patches import FancyArrowPatch
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def add_curved_arrow(ax, pos, u, v, width=2.0, color="black", rad=0.2,
                     linestyle="-", alpha=1.0, mutation_scale=34,
                     shrinkA=24, shrinkB=24, zorder=1):
    x1, y1 = pos[u]
    x2, y2 = pos[v]

    patch = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle='-|>',
        mutation_scale=mutation_scale,
        linewidth=width,
        color=color,
        linestyle=linestyle,
        alpha=alpha,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
        zorder=zorder,
    )
    ax.add_patch(patch)

VALID_BASIS = {"A-dominant", "B-dominant", "Both", "Underdetermined"}
VALID_REL = {"conflict-like", "consensus-like", "indifference-like"}


def short_basis_id(full_name: str) -> str:
    return full_name.split("_", 1)[0] if full_name else full_name


def normalize_basis(x: str) -> str:
    return x if x in VALID_BASIS else "Underdetermined"


def normalize_relation(x: str) -> str:
    return x if x in VALID_REL else "indifference-like"


def load_monitor_results(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of item records.")
    return data


def aggregate_pair_stats(data):
    rows = []

    for row in data:
        basis_A = row.get("basis_A")
        basis_B = row.get("basis_B")
        if not basis_A or not basis_B:
            continue

        final = row.get("final_decision", {}) or {}
        basis_decision = normalize_basis(final.get("final_basis_decision", "Underdetermined"))
        relation = normalize_relation(final.get("final_relation", "indifference-like"))
        confidence = (final.get("final_confidence") or "unknown").lower().strip()

        rows.append({
            "item_id": row.get("item_id", ""),
            "pair": f"{basis_A}__{basis_B}",
            "basis_A": basis_A,
            "basis_B": basis_B,
            "basis_A_short": short_basis_id(basis_A),
            "basis_B_short": short_basis_id(basis_B),
            "basis_decision": basis_decision,
            "relation": relation,
            "confidence": confidence,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid rows found in input JSON.")

    agg_rows = []
    for pair, g in df.groupby("pair", sort=False):
        basis_A = g["basis_A"].iloc[0]
        basis_B = g["basis_B"].iloc[0]
        n = len(g)

        bcounts = g["basis_decision"].value_counts().to_dict()
        rcounts = g["relation"].value_counts().to_dict()

        a_count = bcounts.get("A-dominant", 0)
        b_count = bcounts.get("B-dominant", 0)
        both_count = bcounts.get("Both", 0)
        under_count = bcounts.get("Underdetermined", 0)

        denom = a_count + b_count
        asym = ((a_count - b_count) / denom) if denom else 0.0

        overall = "Underdetermined"
        if a_count > b_count and a_count >= both_count and a_count >= under_count:
            overall = "A-dominant"
        elif b_count > a_count and b_count >= both_count and b_count >= under_count:
            overall = "B-dominant"
        elif both_count > max(a_count, b_count, under_count):
            overall = "Both"

        agg_rows.append({
            "pair": pair,
            "basis_A": basis_A,
            "basis_B": basis_B,
            "basis_A_short": short_basis_id(basis_A),
            "basis_B_short": short_basis_id(basis_B),
            "n_items": n,
            "conflict_prevalence": rcounts.get("conflict-like", 0) / n,
            "consensus_prevalence": rcounts.get("consensus-like", 0) / n,
            "indifference_prevalence": rcounts.get("indifference-like", 0) / n,
            "a_dominant_rate": a_count / n,
            "b_dominant_rate": b_count / n,
            "both_rate": both_count / n,
            "underdetermined_rate": under_count / n,
            "attribution_asymmetry": asym,
            "overall_basis_direction": overall,
        })

    return pd.DataFrame(agg_rows).sort_values(
        ["conflict_prevalence", "pair"], ascending=[False, True]
    ).reset_index(drop=True)


def plot_top_conflict_pairs(agg: pd.DataFrame, outpath: Path, topn: int = 12):
    top = agg.head(min(topn, len(agg))).copy()
    top["pair_label"] = top["basis_A_short"] + " vs " + top["basis_B_short"]

    plt.figure(figsize=(9, 5.5))
    bars = plt.barh(list(top["pair_label"])[::-1], list(top["conflict_prevalence"])[::-1])
    plt.xlabel("Conflict prevalence")
    plt.ylabel("Principle pair")
    plt.title("Top principle pairs by conflict prevalence")

    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.1f}", va="center", fontsize=9)

    plt.xlim(0, min(1.08, max(top["conflict_prevalence"].max() + 0.08, 0.5)))
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


def grouped_positions():
    pos = {}

    safe = ["S1", "S2", "S3", "S4"]
    ethics = ["E1", "E2", "E3", "E4", "E5"]
    guideline = ["G1", "G2", "G3", "G4", "G5"]
    helpful = ["H1", "H2", "H3", "H4", "H5", "H6"]

    # left, top, right, bottom
    for i, n in enumerate(safe):
        pos[n] = (-2.0, 1.5 - i)
    for i, n in enumerate(ethics):
        pos[n] = (-1.0 + i * 0.5, 2.2)
    for i, n in enumerate(guideline):
        pos[n] = (2.0, 1.5 - i)
    for i, n in enumerate(helpful):
        pos[n] = (-1.25 + i * 0.5, -2.2)

    return pos

# def plot_principle_graph(agg: pd.DataFrame, outpath: Path, topk: int = 8):
#     # 상위 8개 = conflict_prevalence가 가장 큰 8개 pair
#     top_edges = agg.sort_values(
#         ["conflict_prevalence", "pair"], ascending=[False, True]
#     ).head(topk).copy()

#     drawable_edges = []
#     for _, r in top_edges.iterrows():
#         a = r["basis_A_short"]
#         b = r["basis_B_short"]
#         w = float(r["conflict_prevalence"])
#         direction = r["overall_basis_direction"]

#         if w <= 0:
#             continue

#         drawable_edges.append((a, b, w, direction))

#     used_nodes = sorted(
#         set([x[0] for x in drawable_edges] + [x[1] for x in drawable_edges]),
#         key=lambda x: (x[0], int(x[1:]))
#     )

#     full_pos = grouped_positions()
#     pos = {n: full_pos[n] for n in used_nodes if n in full_pos}

#     plt.figure(figsize=(10, 8))
#     ax = plt.gca()

#     color_map = {}
#     for n in used_nodes:
#         if n.startswith("S"):
#             color_map[n] = "#d9edf7"
#         elif n.startswith("E"):
#             color_map[n] = "#dff0d8"
#         elif n.startswith("G"):
#             color_map[n] = "#fcf8e3"
#         else:
#             color_map[n] = "#f2dede"

#     G_nodes = nx.DiGraph()
#     for n in used_nodes:
#         G_nodes.add_node(n)

#     nx.draw_networkx_nodes(
#         G_nodes, pos,
#         node_size=850,
#         node_color=[color_map[n] for n in used_nodes],
#         edgecolors="black",
#         linewidths=0.8,
#         ax=ax
#     )
#     nx.draw_networkx_labels(G_nodes, pos, font_size=11, ax=ax)

#     for a, b, w, direction in drawable_edges:
#         width = 1.5 + 6 * w

#         if direction == "A-dominant":
#             # arrow points toward A
#             nx.draw_networkx_edges(
#                 G_nodes, pos,
#                 edgelist=[(b, a)],
#                 width=width,
#                 edge_color="black",
#                 arrows=True,
#                 arrowstyle='-|>',
#                 arrowsize=34,
#                 min_source_margin=22,
#                 min_target_margin=26,
#                 connectionstyle="arc3,rad=0.16",
#                 ax=ax
#             )

#         elif direction == "B-dominant":
#             # arrow points toward B
#             nx.draw_networkx_edges(
#                 G_nodes, pos,
#                 edgelist=[(a, b)],
#                 width=width,
#                 edge_color="black",
#                 arrows=True,
#                 arrowstyle='-|>',
#                 arrowsize=34,
#                 min_source_margin=22,
#                 min_target_margin=26,
#                 connectionstyle="arc3,rad=0.16",
#                 ax=ax
#             )

#         elif direction == "Both":
#         # bidirectional with separated curves
#             nx.draw_networkx_edges(
#                 G_nodes, pos,
#                 edgelist=[(a, b)],
#                 width=width,
#                 edge_color="black",
#                 arrows=True,
#                 arrowstyle='-|>',
#                 arrowsize=30,
#                 min_source_margin=22,
#                 min_target_margin=26,
#                 connectionstyle="arc3,rad=0.20",
#                 ax=ax
#             )
#             nx.draw_networkx_edges(
#                 G_nodes, pos,
#                 edgelist=[(b, a)],
#                 width=width,
#                 edge_color="black",
#                 arrows=True,
#                 arrowstyle='-|>',
#                 arrowsize=30,
#                 min_source_margin=22,
#                 min_target_margin=26,
#                 connectionstyle="arc3,rad=-0.20",
#                 ax=ax
#             )

#         elif direction == "Underdetermined":
#             nx.draw_networkx_edges(
#                 G_nodes, pos,
#                 edgelist=[(a, b)],
#                 width=width,
#                 edge_color="gray",
#                 style="dashed",
#                 arrows=False,
#                 alpha=0.9,
#                 connectionstyle="arc3,rad=0.0",
#                 ax=ax
#             )

#     # 간단한 legend
#     from matplotlib.lines import Line2D
#     legend_elements = [
#         Line2D([0], [0], color="black", lw=2, label="single-arrow: dominant basis"),
#         Line2D([0], [0], color="black", lw=2, linestyle="-", label="thicker edge: higher conflict prevalence"),
#         Line2D([0], [0], color="gray", lw=2, linestyle="--", label="underdetermined"),
#     ]
#     ax.legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=9)

#     plt.title(f"Top {topk} principle conflict edges")
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=220, bbox_inches="tight")
#     plt.close()

def plot_principle_graph(agg: pd.DataFrame, outpath: Path, topk: int = 8):
    top_edges = agg.sort_values(
        ["conflict_prevalence", "pair"], ascending=[False, True]
    ).head(topk).copy()

    drawable_edges = []
    for _, r in top_edges.iterrows():
        a = r["basis_A_short"]
        b = r["basis_B_short"]
        w = float(r["conflict_prevalence"])
        direction = r["overall_basis_direction"]
        if w <= 0:
            continue
        drawable_edges.append((a, b, w, direction))

    used_nodes = sorted(
        set([x[0] for x in drawable_edges] + [x[1] for x in drawable_edges]),
        key=lambda x: (x[0], int(x[1:]))
    )

    full_pos = grouped_positions()
    pos = {n: full_pos[n] for n in used_nodes if n in full_pos}

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    color_map = {}
    for n in used_nodes:
        if n.startswith("S"):
            color_map[n] = "#d9edf7"
        elif n.startswith("E"):
            color_map[n] = "#dff0d8"
        elif n.startswith("G"):
            color_map[n] = "#fcf8e3"
        else:
            color_map[n] = "#f2dede"

    G_nodes = nx.DiGraph()
    for n in used_nodes:
        G_nodes.add_node(n)

    nx.draw_networkx_nodes(
        G_nodes, pos,
        node_size=850,
        node_color=[color_map[n] for n in used_nodes],
        edgecolors="black",
        linewidths=0.8,
        ax=ax
    )
    nx.draw_networkx_labels(G_nodes, pos, font_size=11, ax=ax)

    for a, b, w, direction in drawable_edges:
        width = 1.5 + 6 * w

        if direction == "A-dominant":
            # arrow points toward A
            add_curved_arrow(
                ax, pos, b, a,
                width=width, color="black", rad=0.20,
                mutation_scale=36, shrinkA=24, shrinkB=24
            )

        elif direction == "B-dominant":
            # arrow points toward B
            add_curved_arrow(
                ax, pos, a, b,
                width=width, color="black", rad=0.20,
                mutation_scale=36, shrinkA=24, shrinkB=24
            )

        elif direction == "Both":
            add_curved_arrow(
                ax, pos, a, b,
                width=width, color="black", rad=0.20,
                mutation_scale=32, shrinkA=24, shrinkB=24
            )
            add_curved_arrow(
                ax, pos, b, a,
                width=width, color="black", rad=-0.20,
                mutation_scale=32, shrinkA=24, shrinkB=24
            )

        elif direction == "Underdetermined":
            patch = FancyArrowPatch(
                posA=pos[a], posB=pos[b],
                connectionstyle="arc3,rad=0.0",
                arrowstyle='-',
                linewidth=width,
                color="gray",
                linestyle="--",
                alpha=0.9,
                shrinkA=24,
                shrinkB=24,
                zorder=1,
            )
            ax.add_patch(patch)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black", lw=2, label="single arrow: dominant basis"),
        Line2D([0], [0], color="black", lw=2, label="thicker edge: higher conflict prevalence"),
        Line2D([0], [0], color="gray", lw=2, linestyle="--", label="underdetermined"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=9)

    plt.title(f"Top {topk} principle conflict edges")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


def plot_top_pair_item_bars(data: list, agg: pd.DataFrame, outpath: Path, topk: int = 8):
    # 상위 8개 pair = conflict_prevalence 상위
    top_pairs = agg.sort_values(
        ["conflict_prevalence", "pair"], ascending=[False, True]
    ).head(topk)["pair"].tolist()

    # relation -> height
    relation_score = {
        "conflict-like": 1.0,
        "consensus-like": 0.5,
        "indifference-like": 0.0,
    }

    # basis -> color
    basis_color = {
        "A-dominant": "#4C78A8",
        "B-dominant": "#E45756",
        "Both": "#7A4EAB",
        "Underdetermined": "#9E9E9E",
    }

    # filter rows
    rows = []
    for row in data:
        basis_A = row.get("basis_A")
        basis_B = row.get("basis_B")
        if not basis_A or not basis_B:
            continue
        pair = f"{basis_A}__{basis_B}"
        if pair not in top_pairs:
            continue

        final = row.get("final_decision", {}) or {}
        item_id = row.get("item_id", "")
        rows.append({
            "pair": pair,
            "pair_label": f"{short_basis_id(basis_A)} vs {short_basis_id(basis_B)}",
            "item_id": item_id,
            "item_short": item_id.split("_")[-1],
            "basis_decision": final.get("final_basis_decision", "Underdetermined"),
            "relation": final.get("final_relation", "indifference-like"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No rows found for top pairs.")

    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharey=True)
    axes = axes.flatten()

    for ax, pair in zip(axes, top_pairs):
        g = df[df["pair"] == pair].copy().sort_values("item_short")
        if g.empty:
            ax.axis("off")
            continue

        heights = [relation_score.get(r, 0.0) for r in g["relation"]]
        colors = [basis_color.get(b, "#9E9E9E") for b in g["basis_decision"]]

        ax.bar(g["item_short"], heights, color=colors, edgecolor="black")
        ax.set_title(g["pair_label"].iloc[0], fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Relation score")

        # relation labels on top
        for x, h, rel in zip(g["item_short"], heights, g["relation"]):
            ax.text(x, h + 0.03, rel.replace("-like", ""), ha="center", va="bottom", fontsize=8, rotation=45)

    # 남는 subplot 끄기
    for ax in axes[len(top_pairs):]:
        ax.axis("off")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C78A8", edgecolor="black", label="A-dominant"),
        Patch(facecolor="#E45756", edgecolor="black", label="B-dominant"),
        Patch(facecolor="#7A4EAB", edgecolor="black", label="Both"),
        Patch(facecolor="#9E9E9E", edgecolor="black", label="Underdetermined"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4, frameon=False)

    plt.suptitle(f"Per-scenario monitor results for top {topk} principle pairs", y=0.98, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()

def plot_top_edge_bar_chart(agg: pd.DataFrame, outpath: Path, topk: int = 8):
    # 그래프와 동일한 기준: conflict_prevalence 상위 8개
    top_edges = agg.sort_values(
        ["conflict_prevalence", "pair"], ascending=[False, True]
    ).head(topk).copy()

    top_edges["pair_label"] = top_edges["basis_A_short"] + " vs " + top_edges["basis_B_short"]

    plt.figure(figsize=(8, 4.8))
    bars = plt.barh(
        top_edges["pair_label"][::-1],
        top_edges["conflict_prevalence"][::-1],
        color="#4C78A8",
        edgecolor="black"
    )

    plt.xlabel("Conflict prevalence")
    plt.ylabel("Principle pair")
    plt.title(f"Conflict prevalence for top {topk} graph edges")
    plt.xlim(0, 1.05)

    # 숫자 라벨
    for bar in bars:
        w = bar.get_width()
        plt.text(
            w + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}",
            va="center",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to spec_conflict_monitor_results.json")
    parser.add_argument("--outdir", default="output_img", help="Directory to save figures and CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_monitor_results(input_path)
    agg = aggregate_pair_stats(data)

    agg.to_csv(outdir / "spec_conflict_pair_stats.csv", index=False)
    plot_top_conflict_pairs(agg, outdir / "top_conflict_pairs.png")
    plot_principle_graph(agg, outdir / "principle_conflict_graph.png")
    # plot_top_pair_item_bars(data, agg, outdir / "top_pair_item_bars.png", topk=8)
    plot_top_edge_bar_chart(agg, outdir / "top_8_edge_bar_chart.png", topk=8)

    print("Saved:")
    print(outdir / "spec_conflict_pair_stats.csv")
    print(outdir / "top_conflict_pairs.png")
    print(outdir / "principle_conflict_graph.png")


if __name__ == "__main__":
    main()