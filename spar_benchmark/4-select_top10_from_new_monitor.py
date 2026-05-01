#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

DOMINANT = {"A-dominant", "B-dominant"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select top canonical scenarios from new monitor results and derive desired-basis map."
    )
    p.add_argument("--input_json", type=str, help="Monitor results JSON file")
    p.add_argument("--input_csv", type=str, help="Monitor results CSV file")
    p.add_argument("--k", type=int, default=15, help="Number of canonical scenarios to select")
    p.add_argument("--min_conflict", type=int, default=1, help="Minimum number of conflict-like rows")
    p.add_argument("--out_dir", type=str, default="top10_from_new_monitor2")
    return p.parse_args()


def canonical_from_item_id(item_id: str) -> str:
    parts = str(item_id).split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return str(item_id)


def load_rows_from_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must contain a list")

    rows: List[Dict[str, Any]] = []
    for row in data:
        item_id = row.get("item_id")
        final = row.get("final_decision", {}) or {}
        monitor = row.get("monitor", {}) or {}
        rows.append(
            {
                "item_id": item_id,
                "canonical_item_id": canonical_from_item_id(item_id),
                "basis_A": row.get("basis_A"),
                "basis_B": row.get("basis_B"),
                "final_basis_decision": final.get("final_basis_decision"),
                "final_relation": final.get("final_relation"),
                "final_confidence": final.get("final_confidence"),
                "final_parse_ok": final.get("final_parse_ok"),
                "monitor_rationale": monitor.get("rationale"),
            }
        )
    return rows


def load_rows_from_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row.get("item_id")
            rows.append(
                {
                    "item_id": item_id,
                    "canonical_item_id": canonical_from_item_id(item_id),
                    "basis_A": row.get("basis_A"),
                    "basis_B": row.get("basis_B"),
                    "final_basis_decision": row.get("final_basis_decision"),
                    "final_relation": row.get("final_relation"),
                    "final_confidence": row.get("final_confidence"),
                    "final_parse_ok": str(row.get("final_parse_ok", "")).lower() == "true",
                    "monitor_rationale": row.get("monitor_rationale"),
                }
            )
    return rows


def confidence_to_num(x: str | None) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(str(x).lower(), -1)


def summarize(rows: List[Dict[str, Any]], min_conflict: int = 1) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["canonical_item_id"]].append(row)

    summaries: List[Dict[str, Any]] = []
    for cid, items in grouped.items():
        basis_A = items[0].get("basis_A")
        basis_B = items[0].get("basis_B")

        parse_ok_items = [r for r in items if r.get("final_parse_ok")]
        conflict_items = [
            r for r in parse_ok_items
            if r.get("final_relation") == "conflict-like" and r.get("final_basis_decision") in DOMINANT
        ]
        if len(conflict_items) < min_conflict:
            continue

        basis_counts = Counter(r.get("final_basis_decision") for r in conflict_items)
        conf_counts = Counter(str(r.get("final_confidence")).lower() for r in conflict_items)
        total_conflict = len(conflict_items)
        a_count = basis_counts.get("A-dominant", 0)
        b_count = basis_counts.get("B-dominant", 0)
        majority_count = max(a_count, b_count)
        majority_basis = None
        desired_basis = None
        if a_count > b_count:
            majority_basis = "A"
            desired_basis = "B"
        elif b_count > a_count:
            majority_basis = "B"
            desired_basis = "A"

        score = (
            100 * total_conflict
            + 20 * majority_count
            + 5 * conf_counts.get("high", 0)
            + 2 * conf_counts.get("medium", 0)
            - 10 * abs(a_count - b_count == 0)
        )

        example_ids = [r.get("item_id") for r in conflict_items]
        summaries.append(
            {
                "canonical_item_id": cid,
                "basis_A": basis_A,
                "basis_B": basis_B,
                "n_rows_total": len(items),
                "n_parse_ok": len(parse_ok_items),
                "conflict_like_count": total_conflict,
                "A_dominant_count": a_count,
                "B_dominant_count": b_count,
                "high_conflict_count": conf_counts.get("high", 0),
                "medium_conflict_count": conf_counts.get("medium", 0),
                "low_conflict_count": conf_counts.get("low", 0),
                "majority_basis": majority_basis,
                "desired_basis": desired_basis,
                "selection_score": score,
                "example_item_ids": " | ".join(example_ids),
            }
        )

    summaries.sort(
        key=lambda r: (
            r["conflict_like_count"],
            max(r["A_dominant_count"], r["B_dominant_count"]),
            r["high_conflict_count"],
            r["selection_score"],
        ),
        reverse=True,
    )
    return summaries


def write_outputs(summaries: List[Dict[str, Any]], out_dir: str, k: int) -> Tuple[str, str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    topk = summaries[:k]

    csv_path = out / "top10_canonical_scenarios.csv"
    fieldnames = [
        "canonical_item_id",
        "basis_A",
        "basis_B",
        "n_rows_total",
        "n_parse_ok",
        "conflict_like_count",
        "A_dominant_count",
        "B_dominant_count",
        "high_conflict_count",
        "medium_conflict_count",
        "low_conflict_count",
        "majority_basis",
        "desired_basis",
        "selection_score",
        "example_item_ids",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(topk)

    desired_map = {
        row["canonical_item_id"]: row["desired_basis"]
        for row in topk
        if row.get("desired_basis") in {"A", "B"}
    }
    map_path = out / "desired_basis_map_top10.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(desired_map, f, ensure_ascii=False, indent=2)

    report_path = out / "top10_report.txt"
    lines = ["Selected top canonical scenarios from new monitor results", ""]
    for idx, row in enumerate(topk, 1):
        lines.append(f"{idx}. {row['canonical_item_id']}")
        lines.append(f"   basis_A: {row['basis_A']}")
        lines.append(f"   basis_B: {row['basis_B']}")
        lines.append(
            f"   conflict_like={row['conflict_like_count']}, A-dominant={row['A_dominant_count']}, B-dominant={row['B_dominant_count']}, majority={row['majority_basis']}, desired={row['desired_basis']}"
        )
        lines.append(f"   example_ids: {row['example_item_ids']}")
        lines.append("")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return str(csv_path), str(map_path), str(report_path)


def main() -> None:
    args = parse_args()
    if bool(args.input_json) == bool(args.input_csv):
        raise ValueError("Provide exactly one of --input_json or --input_csv")

    rows = load_rows_from_json(args.input_json) if args.input_json else load_rows_from_csv(args.input_csv)
    summaries = summarize(rows, min_conflict=args.min_conflict)
    csv_path, map_path, report_path = write_outputs(summaries, args.out_dir, args.k)
    print(f"Loaded rows: {len(rows)}")
    print(f"Eligible canonical scenarios: {len(summaries)}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved desired-basis map: {map_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
