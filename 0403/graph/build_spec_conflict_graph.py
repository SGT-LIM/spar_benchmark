#!/usr/bin/env python3
"""
Aggregate monitor results from the spec-conflict benchmark into graph-ready outputs.

Input:
    A JSON file like spec_conflict_monitor_results.json, where each item contains:
      - item_id
      - basis_A
      - basis_B
      - final_decision.final_basis_decision
      - final_decision.final_relation
      - final_decision.final_confidence
      - optional rule_profile / monitor / generations / prompts

Outputs:
    1) pair_stats.json   : detailed per-pair statistics
    2) edges.csv         : graph-ready edge table
    3) nodes.csv         : graph-ready node table
    4) graph.json        : {nodes: [...], edges: [...]}
    5) summary.json      : corpus-level summary

Conventions:
    - conflict_prevalence = fraction of items labeled conflict-like
    - edge weight defaults to conflict_prevalence
    - arrow points TOWARD the dominant basis
      * if A-dominant overall: source=basis_B, target=basis_A
      * if B-dominant overall: source=basis_A, target=basis_B
    - if overall attribution is Both / Underdetermined / tied, the edge is treated as undirected
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VALID_BASIS = {"A-dominant", "B-dominant", "Both", "Underdetermined"}
VALID_REL = {"conflict-like", "consensus-like", "indifference-like"}
VALID_CONF = {"high", "medium", "low"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def normalize_confidence(conf: Optional[str]) -> str:
    if not conf:
        return "unknown"
    conf = conf.strip().lower()
    return conf if conf in VALID_CONF else "unknown"


def get_final_fields(row: Dict[str, Any]) -> Tuple[str, str, str]:
    final = row.get("final_decision", {}) or {}
    basis = final.get("final_basis_decision", "Underdetermined")
    relation = final.get("final_relation", "indifference-like")
    confidence = normalize_confidence(final.get("final_confidence"))

    if basis not in VALID_BASIS:
        basis = "Underdetermined"
    if relation not in VALID_REL:
        relation = "indifference-like"
    return basis, relation, confidence


def short_basis_id(full_name: str) -> str:
    return full_name.split("_", 1)[0] if full_name else full_name


def pretty_basis_name(full_name: str) -> str:
    parts = full_name.split("_", 1)
    if len(parts) == 1:
        return full_name
    text = parts[1].replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else full_name


def node_group(node_id: str) -> str:
    if node_id.startswith("S"):
        return "Safe"
    if node_id.startswith("E"):
        return "Ethics"
    if node_id.startswith("G"):
        return "Guideline"
    if node_id.startswith("H"):
        return "Helpfulness"
    return "Unknown"


def direction_bucket_for_item(basis_decision: str) -> str:
    if basis_decision == "A-dominant":
        return "A"
    if basis_decision == "B-dominant":
        return "B"
    if basis_decision == "Both":
        return "Both"
    return "Underdetermined"


@dataclass
class PairAggregate:
    basis_A: str
    basis_B: str

    def __post_init__(self) -> None:
        self.item_ids: List[str] = []
        self.basis_counter: Counter = Counter()
        self.relation_counter: Counter = Counter()
        self.conf_counter: Counter = Counter()
        self.direction_counter: Counter = Counter()
        self.decision_source_counter: Counter = Counter()
        self.trigger_counter: Counter = Counter()

    def add(self, row: Dict[str, Any]) -> None:
        item_id = row.get("item_id", "")
        basis_decision, relation, confidence = get_final_fields(row)

        self.item_ids.append(item_id)
        self.basis_counter[basis_decision] += 1
        self.relation_counter[relation] += 1
        self.conf_counter[confidence] += 1
        self.direction_counter[direction_bucket_for_item(basis_decision)] += 1

        final = row.get("final_decision", {}) or {}
        self.decision_source_counter[final.get("decision_source", "unknown")] += 1

        rule = row.get("rule_profile", {}) or {}
        trigger = rule.get("trigger", "none")
        self.trigger_counter[trigger] += 1

    @property
    def n(self) -> int:
        return len(self.item_ids)

    def dominant_relation(self) -> str:
        if not self.relation_counter:
            return "indifference-like"
        top = self.relation_counter.most_common()
        if len(top) >= 2 and top[0][1] == top[1][1]:
            return "mixed"
        return top[0][0]

    def overall_basis_direction(self) -> str:
        a = self.basis_counter["A-dominant"]
        b = self.basis_counter["B-dominant"]
        both = self.basis_counter["Both"]
        under = self.basis_counter["Underdetermined"]

        if a > b and a >= both and a >= under:
            return "A-dominant"
        if b > a and b >= both and b >= under:
            return "B-dominant"
        if both > max(a, b, under):
            return "Both"
        return "Underdetermined"

    def directed_edge_endpoints(self) -> Tuple[Optional[str], Optional[str]]:
        overall = self.overall_basis_direction()
        if overall == "A-dominant":
            return self.basis_B, self.basis_A
        if overall == "B-dominant":
            return self.basis_A, self.basis_B
        return None, None

    def to_stats_dict(self) -> Dict[str, Any]:
        n = self.n
        source, target = self.directed_edge_endpoints()
        overall_basis = self.overall_basis_direction()

        a_rate = safe_div(self.basis_counter["A-dominant"], n)
        b_rate = safe_div(self.basis_counter["B-dominant"], n)
        both_rate = safe_div(self.basis_counter["Both"], n)
        under_rate = safe_div(self.basis_counter["Underdetermined"], n)

        dominant_mass = self.basis_counter["A-dominant"] + self.basis_counter["B-dominant"]
        asymmetry = safe_div(
            self.basis_counter["A-dominant"] - self.basis_counter["B-dominant"],
            dominant_mass,
        ) if dominant_mass else 0.0

        conflict_prev = safe_div(self.relation_counter["conflict-like"], n)
        consensus_prev = safe_div(self.relation_counter["consensus-like"], n)
        indiff_prev = safe_div(self.relation_counter["indifference-like"], n)

        high_conf = safe_div(self.conf_counter["high"], n)
        medium_conf = safe_div(self.conf_counter["medium"], n)
        low_conf = safe_div(self.conf_counter["low"], n)

        return {
            "pair_key": f"{self.basis_A}__{self.basis_B}",
            "basis_A": self.basis_A,
            "basis_B": self.basis_B,
            "basis_A_short": short_basis_id(self.basis_A),
            "basis_B_short": short_basis_id(self.basis_B),
            "n_items": n,
            "item_ids": self.item_ids,
            "basis_counts": dict(self.basis_counter),
            "relation_counts": dict(self.relation_counter),
            "confidence_counts": dict(self.conf_counter),
            "decision_source_counts": dict(self.decision_source_counter),
            "rule_trigger_counts": dict(self.trigger_counter),
            "a_dominant_rate": a_rate,
            "b_dominant_rate": b_rate,
            "both_rate": both_rate,
            "underdetermined_rate": under_rate,
            "conflict_prevalence": conflict_prev,
            "consensus_prevalence": consensus_prev,
            "indifference_prevalence": indiff_prev,
            "high_confidence_rate": high_conf,
            "medium_confidence_rate": medium_conf,
            "low_confidence_rate": low_conf,
            "attribution_asymmetry": asymmetry,
            "overall_basis_direction": overall_basis,
            "dominant_relation": self.dominant_relation(),
            "graph_weight": conflict_prev,
            "graph_source": source,
            "graph_target": target,
            "graph_directed": source is not None and target is not None,
            "graph_edge_type": (
                "directed" if source is not None and target is not None
                else ("bidirectional" if overall_basis == "Both" else "undirected")
            ),
        }


def aggregate_pairs(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], PairAggregate]:
    pair_map: Dict[Tuple[str, str], PairAggregate] = {}
    for row in rows:
        basis_A = row.get("basis_A")
        basis_B = row.get("basis_B")
        if not basis_A or not basis_B:
            continue
        key = (basis_A, basis_B)
        if key not in pair_map:
            pair_map[key] = PairAggregate(basis_A=basis_A, basis_B=basis_B)
        pair_map[key].add(row)
    return pair_map


def build_nodes(rows: List[Dict[str, Any]], pair_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nodes: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        for full_basis in (row.get("basis_A"), row.get("basis_B")):
            if not full_basis:
                continue
            node_id = short_basis_id(full_basis)
            nodes.setdefault(node_id, {
                "id": node_id,
                "full_name": full_basis,
                "label": node_id,
                "title": pretty_basis_name(full_basis),
                "group": node_group(node_id),
                "degree_pairs": 0,
                "total_conflict_weight": 0.0,
                "incoming_dominance": 0,
                "outgoing_dominance": 0,
                "undirected_incidents": 0,
            })

    for edge in pair_stats:
        a = edge["basis_A_short"]
        b = edge["basis_B_short"]
        weight = edge["graph_weight"]
        directed = edge["graph_directed"]
        src = edge["graph_source"]
        tgt = edge["graph_target"]

        if a in nodes:
            nodes[a]["degree_pairs"] += 1
            nodes[a]["total_conflict_weight"] += weight
        if b in nodes:
            nodes[b]["degree_pairs"] += 1
            nodes[b]["total_conflict_weight"] += weight

        if directed and src and tgt:
            src_short = short_basis_id(src)
            tgt_short = short_basis_id(tgt)
            if src_short in nodes:
                nodes[src_short]["outgoing_dominance"] += 1
            if tgt_short in nodes:
                nodes[tgt_short]["incoming_dominance"] += 1
        else:
            if a in nodes:
                nodes[a]["undirected_incidents"] += 1
            if b in nodes:
                nodes[b]["undirected_incidents"] += 1

    return sorted(nodes.values(), key=lambda x: x["id"])


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def make_summary(rows: List[Dict[str, Any]], pair_stats: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    basis_counts = Counter()
    relation_counts = Counter()
    conf_counts = Counter()

    for row in rows:
        basis, relation, conf = get_final_fields(row)
        basis_counts[basis] += 1
        relation_counts[relation] += 1
        conf_counts[conf] += 1

    return {
        "n_items": len(rows),
        "n_pairs": len(pair_stats),
        "n_nodes": len(nodes),
        "basis_decision_counts": dict(basis_counts),
        "relation_counts": dict(relation_counts),
        "confidence_counts": dict(conf_counts),
        "mean_conflict_prevalence_across_pairs": (sum(e["conflict_prevalence"] for e in pair_stats) / len(pair_stats) if pair_stats else 0.0),
        "mean_consensus_prevalence_across_pairs": (sum(e["consensus_prevalence"] for e in pair_stats) / len(pair_stats) if pair_stats else 0.0),
        "mean_indifference_prevalence_across_pairs": (sum(e["indifference_prevalence"] for e in pair_stats) / len(pair_stats) if pair_stats else 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to monitor results JSON")
    parser.add_argument("--outdir", default="output", help="Directory for graph outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_json(input_path)
    if not isinstance(rows, list):
        raise ValueError("Input JSON must be a list of item records.")

    pair_map = aggregate_pairs(rows)
    pair_stats = [agg.to_stats_dict() for _, agg in sorted(pair_map.items(), key=lambda x: x[0])]
    nodes = build_nodes(rows, pair_stats)
    summary = make_summary(rows, pair_stats, nodes)

    graph_json = {"nodes": nodes, "edges": pair_stats}

    dump_json(pair_stats, outdir / "pair_stats.json")
    dump_json(graph_json, outdir / "graph.json")
    dump_json(summary, outdir / "summary.json")

    edge_fields = [
        "pair_key", "basis_A", "basis_B", "basis_A_short", "basis_B_short", "n_items",
        "a_dominant_rate", "b_dominant_rate", "both_rate", "underdetermined_rate",
        "conflict_prevalence", "consensus_prevalence", "indifference_prevalence",
        "high_confidence_rate", "medium_confidence_rate", "low_confidence_rate",
        "attribution_asymmetry", "overall_basis_direction", "dominant_relation",
        "graph_weight", "graph_source", "graph_target", "graph_directed", "graph_edge_type",
    ]
    write_csv(outdir / "edges.csv", pair_stats, edge_fields)

    node_fields = [
        "id", "full_name", "label", "title", "group",
        "degree_pairs", "total_conflict_weight", "incoming_dominance", "outgoing_dominance", "undirected_incidents",
    ]
    write_csv(outdir / "nodes.csv", nodes, node_fields)

    print(f"Wrote graph outputs to: {outdir}")


if __name__ == "__main__":
    main()
