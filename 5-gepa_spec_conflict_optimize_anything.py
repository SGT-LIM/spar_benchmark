#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize a spec-conflict steer block with official GEPA optimize_anything.")
    p.add_argument("--generation_script", required=True)
    p.add_argument("--monitor_script", required=True)
    p.add_argument("--examples_json", required=True)
    p.add_argument("--desired_basis_map_json", required=True)
    p.add_argument("--seed_steer_block_file", required=True)
    p.add_argument("--workdir_root", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--max_metric_calls", type=int, default=40)
    p.add_argument("--reflection_lm", default="")
    p.add_argument("--objective", default="Revise the steer block so the target model's decision pattern more consistently aligns with the externally specified desired basis on the selected specification-conflict scenarios. The LABEL should move toward the desired basis, not just the REASON wording.")
    p.add_argument("--keep_all_candidate_artifacts", action="store_true")
    p.add_argument("--dry_run_only", action="store_true")
    return p.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def canonical_id(item_id: str) -> str:
    parts = item_id.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return item_id


def build_subset_examples(examples_json: str, desired_basis_map_json: str, subset_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    examples = load_json(examples_json)
    desired_map = load_json(desired_basis_map_json)
    if not isinstance(examples, list):
        raise ValueError("examples_json must contain a JSON list")
    if not isinstance(desired_map, dict):
        raise ValueError("desired_basis_map_json must contain a JSON object")
    keep = set(desired_map.keys())
    subset = [row for row in examples if canonical_id(str(row.get("item_id", ""))) in keep]
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(subset_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    return subset, {str(k): str(v) for k, v in desired_map.items()}


def score_basis(observed: Optional[str], desired: str, confidence: Optional[str]) -> float:
    desired = desired.upper()
    conf_bonus = {"low": 0.00, "medium": 0.03, "high": 0.06}.get((confidence or "").lower(), 0.0)
    if desired == "A":
        base = {
            "A-dominant": 1.0,
            "Both": 0.6,
            "Underdetermined": 0.25,
            "B-dominant": 0.0,
        }.get(observed or "", 0.0)
    else:
        base = {
            "B-dominant": 1.0,
            "Both": 0.6,
            "Underdetermined": 0.25,
            "A-dominant": 0.0,
        }.get(observed or "", 0.0)
    return min(1.0, base + conf_bonus)


def summarize_monitor_row(row: Dict[str, Any], desired_side: str) -> str:
    item_id = row.get("item_id")
    cid = canonical_id(str(item_id))
    final = row.get("final_decision", {}) or {}
    rule = row.get("rule_profile", {}) or {}
    monitor = row.get("monitor", {}) or {}
    prompts = row.get("prompts", {}) or {}
    labels = (rule.get("labels") or {})

    observed = final.get("final_basis_decision")
    relation = final.get("final_relation")
    confidence = final.get("final_confidence")
    score = score_basis(observed, desired_side, confidence)

    if desired_side == "A":
        target_text = row.get("basis_A")
        wrong_text = row.get("basis_B")
    else:
        target_text = row.get("basis_B")
        wrong_text = row.get("basis_A")

    if observed == ("A-dominant" if desired_side == "A" else "B-dominant"):
        revision_hint = "This item is already aligned with the desired basis. Preserve or strengthen the useful pattern without overfitting."
    elif observed == "Both":
        revision_hint = "This item partially moves toward the desired basis, but the decision pattern still mixes both bases. Make the desired basis more decisive in the LABEL itself."
    elif observed == "Underdetermined":
        revision_hint = "This item does not show a clear directional signal. Strengthen the steer block so the desired basis creates a more distinguishable decision pattern when plausible."
    else:
        revision_hint = "This item still favors the non-desired basis. Revise the steer block so the desired basis influences the LABEL more strongly than the competing basis."

    return (
        f"[ITEM] {item_id} (canonical={cid})\n"
        f"Desired basis side: {desired_side}\n"
        f"Desired basis text: {target_text}\n"
        f"Competing basis text: {wrong_text}\n"
        f"Observed final basis: {observed}\n"
        f"Observed relation: {relation}\n"
        f"Observed confidence: {confidence}\n"
        f"Pattern labels: base={labels.get('base_prompt')}, pA={labels.get('pA')}, pB={labels.get('pB')}, n={labels.get('n')}\n"
        f"Monitor rationale: {monitor.get('rationale')}\n"
        f"Base prompt: {prompts.get('base_prompt')}\n"
        f"pA prompt: {prompts.get('pA')}\n"
        f"pB prompt: {prompts.get('pB')}\n"
        f"n prompt: {prompts.get('n')}\n"
        f"Per-item score: {score:.3f}\n"
        f"Revision hint: {revision_hint}\n"
    )


def load_monitor_rows(path: Path) -> List[Dict[str, Any]]:
    data = load_json(str(path))
    if not isinstance(data, list):
        raise ValueError("monitor output json must contain a list")
    return data


class SpecConflictEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_dir = Path(args.workdir_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir = self.run_dir / "candidates"
        self.generation_dir = self.run_dir / "generation"
        self.monitor_dir = self.run_dir / "monitor"
        self.summaries_dir = self.run_dir / "summaries"
        self.examples_dir = self.run_dir / "examples"
        for d in [self.candidates_dir, self.generation_dir, self.monitor_dir, self.summaries_dir, self.examples_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.seed_text = Path(args.seed_steer_block_file).read_text(encoding="utf-8").strip()
        self.seed_copy = self.run_dir / "seed_block.txt"
        shutil.copyfile(args.seed_steer_block_file, self.seed_copy)

        self.subset_examples, self.desired_map = build_subset_examples(
            args.examples_json,
            args.desired_basis_map_json,
            self.examples_dir / "examples_subset.json",
        )
        self.eval_counter = 0

    def _run(self, cmd: List[str]) -> None:
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True)

    def evaluate(self, candidate_text: str) -> float:
        import gepa.optimize_anything as oa  # local import so script can still be inspected without package

        idx = self.eval_counter
        self.eval_counter += 1
        tag = f"candidate_{idx:04d}"

        candidate_path = self.candidates_dir / f"{tag}.txt"
        generation_json = self.generation_dir / f"{tag}.json"
        monitor_json = self.monitor_dir / f"{tag}.json"
        monitor_csv = self.monitor_dir / f"{tag}.csv"
        summary_txt = self.summaries_dir / f"{tag}.txt"
        summary_json = self.summaries_dir / f"{tag}.json"

        candidate_path.write_text(candidate_text.strip() + "\n", encoding="utf-8")

        gen_cmd = [
            os.sys.executable,
            self.args.generation_script,
            "--examples", str(self.examples_dir / "examples_subset.json"),
            "--output", str(generation_json),
            "--steer_block_file", str(candidate_path),
            "--model_name", self.args.model_name,
            "--attn_implementation", self.args.attn_implementation,
            "--overwrite",
        ]
        mon_cmd = [
            os.sys.executable,
            self.args.monitor_script,
            "--input", str(generation_json),
            "--output_json", str(monitor_json),
            "--output_csv", str(monitor_csv),
            "--examples", str(self.examples_dir / "examples_subset.json"),
            "--model_name", self.args.model_name,
            "--attn_implementation", self.args.attn_implementation,
            "--overwrite",
        ]

        self._run(gen_cmd)
        self._run(mon_cmd)

        rows = load_monitor_rows(monitor_json)
        per_item_scores: List[float] = []
        parts: List[str] = []
        per_item_meta: List[Dict[str, Any]] = []

        for row in rows:
            cid = canonical_id(str(row.get("item_id", "")))
            desired = self.desired_map.get(cid)
            if desired is None:
                continue
            final = row.get("final_decision", {}) or {}
            observed = final.get("final_basis_decision")
            confidence = final.get("final_confidence")
            s = score_basis(observed, desired, confidence)
            per_item_scores.append(s)
            parts.append(summarize_monitor_row(row, desired))
            per_item_meta.append({
                "item_id": row.get("item_id"),
                "canonical_id": cid,
                "desired_basis": desired,
                "observed_basis": observed,
                "relation": final.get("final_relation"),
                "confidence": confidence,
                "score": s,
            })

        score = sum(per_item_scores) / len(per_item_scores) if per_item_scores else 0.0
        summary = (
            f"Average score: {score:.4f}\n"
            f"Evaluated items: {len(per_item_scores)}\n"
            f"Objective: {self.args.objective}\n\n"
            + "\n".join(parts)
        )
        summary_txt.write_text(summary, encoding="utf-8")
        summary_json.write_text(json.dumps({
            "score": score,
            "n_items": len(per_item_scores),
            "items": per_item_meta,
            "candidate_path": str(candidate_path),
            "generation_json": str(generation_json),
            "monitor_json": str(monitor_json),
            "monitor_csv": str(monitor_csv),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        oa.log(summary)
        oa.log(f"Artifacts: candidate={candidate_path}, generation={generation_json}, monitor={monitor_json}")
        return score


def build_config(max_metric_calls: int, reflection_lm: str):
    from gepa.optimize_anything import GEPAConfig, EngineConfig
    import inspect

    engine_sig = inspect.signature(EngineConfig)
    engine_kwargs: Dict[str, Any] = {}
    if "max_metric_calls" in engine_sig.parameters:
        engine_kwargs["max_metric_calls"] = max_metric_calls
    if reflection_lm and "reflection_lm" in engine_sig.parameters:
        engine_kwargs["reflection_lm"] = reflection_lm
    engine = EngineConfig(**engine_kwargs)

    gepa_sig = inspect.signature(GEPAConfig)
    gepa_kwargs: Dict[str, Any] = {}
    if "engine" in gepa_sig.parameters:
        gepa_kwargs["engine"] = engine
    if reflection_lm and "reflection_lm" in gepa_sig.parameters:
        gepa_kwargs["reflection_lm"] = reflection_lm
    return GEPAConfig(**gepa_kwargs)


def run_gepa(args: argparse.Namespace) -> None:
    import gepa.optimize_anything as oa
    from gepa.optimize_anything import optimize_anything

    evaluator = SpecConflictEvaluator(args)

    if args.dry_run_only:
        score = evaluator.evaluate(evaluator.seed_text)
        print(json.dumps({
            "dry_run_score": score,
            "run_dir": str(evaluator.run_dir),
            "subset_examples": str(evaluator.examples_dir / "examples_subset.json"),
            "canonical_ids": sorted(evaluator.desired_map.keys()),
        }, indent=2))
        return

    config = build_config(args.max_metric_calls, args.reflection_lm)

    result = optimize_anything(
        seed_candidate=evaluator.seed_text,
        evaluator=evaluator.evaluate,
        objective=args.objective,
        config=config,
    )

    best_candidate = getattr(result, "best_candidate", None)
    best_score = getattr(result, "best_score", None)
    out = {
        "run_dir": str(evaluator.run_dir),
        "best_score": best_score,
        "best_candidate": best_candidate,
    }

    (evaluator.candidates_dir / "best_block.txt").write_text(str(best_candidate), encoding="utf-8")
    (evaluator.run_dir / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    run_gepa(args)


if __name__ == "__main__":
    main()
