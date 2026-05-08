#!/usr/bin/env python3
"""
GEPA 실행 결과 분석 도구

사용법:
    python analyze_gepa_run.py --run_dir /path/to/run_20251106_143022 --output_dir ./analysis_out

분석 내용:
    1. Candidate 점수 추이 (시간순)
    2. 항목별 점수 히트맵 (candidate × scenario)
    3. 시나리오별 응답 변화 추적
    4. Candidate 텍스트 diff (이웃 candidate 간)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import difflib


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze GEPA optimization run results")
    p.add_argument("--run_dir", required=True, help="GEPA run directory (run_YYYYMMDD_HHMMSS)")
    p.add_argument("--output_dir", required=True, help="Where to save analysis outputs")
    p.add_argument("--top_k_diffs", type=int, default=5, 
                   help="How many candidate-to-candidate diffs to show in detail")
    p.add_argument("--track_scenarios", nargs="*", default=None,
                   help="Specific canonical_ids to track (default: all)")
    return p.parse_args()


# ============================================================
# 1. 데이터 로딩
# ============================================================

def load_candidates(run_dir: Path) -> List[Dict[str, Any]]:
    """
    candidates/, summaries/, generation/, monitor/ 디렉토리에서 
    각 candidate의 모든 정보를 모아서 리스트로 반환.
    
    각 항목 구조:
    {
        "idx": 0,
        "tag": "candidate_0000",
        "candidate_text": "...",       # steer block 텍스트
        "score": 0.45,                 # 평균 점수
        "n_items": 8,                  # 평가된 항목 수
        "items": [...],                # 항목별 상세 결과
        "summary_text": "...",         # 자연어 요약
        "generation": [...],           # 모델 응답들
        "monitor": [...],              # 평가 결과들
    }
    """
    candidates_dir = run_dir / "candidates"
    summaries_dir = run_dir / "summaries"
    generation_dir = run_dir / "generation"
    monitor_dir = run_dir / "monitor"

    candidate_files = sorted(candidates_dir.glob("candidate_*.txt"))
    
    results = []
    for cand_file in candidate_files:
        # candidate_0000.txt -> "candidate_0000", 0
        tag = cand_file.stem
        match = re.match(r"candidate_(\d+)", tag)
        if not match:
            continue
        idx = int(match.group(1))
        
        entry = {
            "idx": idx,
            "tag": tag,
            "candidate_text": cand_file.read_text(encoding="utf-8"),
        }
        
        # summary json 로드 (점수, 항목별 결과)
        summary_json = summaries_dir / f"{tag}.json"
        if summary_json.exists():
            data = json.loads(summary_json.read_text(encoding="utf-8"))
            entry["score"] = data.get("score")
            entry["n_items"] = data.get("n_items")
            entry["items"] = data.get("items", [])
        
        # summary txt (자연어 요약)
        summary_txt = summaries_dir / f"{tag}.txt"
        if summary_txt.exists():
            entry["summary_text"] = summary_txt.read_text(encoding="utf-8")
        
        # generation json (모델 응답)
        gen_json = generation_dir / f"{tag}.json"
        if gen_json.exists():
            try:
                entry["generation"] = json.loads(gen_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                entry["generation"] = None
        
        # monitor json (평가 결과)
        mon_json = monitor_dir / f"{tag}.json"
        if mon_json.exists():
            try:
                entry["monitor"] = json.loads(mon_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                entry["monitor"] = None
        
        results.append(entry)
    
    return results


def canonical_id(item_id: str) -> str:
    """item_id에서 scenario-level canonical_id 추출.

    예: s4_e4_gpt54mini_001 -> s4_e4
    이미 canonical_id가 있으면 그대로 쓰고, 없을 때 fallback으로 사용.
    """
    parts = str(item_id).split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return str(item_id)


def get_item_canonical_id(item: Dict[str, Any]) -> str:
    """summary item에서 canonical_id를 안전하게 가져옴.

    기존 코드가 item['canonical_id']가 항상 있다고 가정해서,
    실제 파일에 canonical_id가 없으면 heatmap/trace가 None으로 묶이는 문제가 있었음.
    """
    cid = item.get("canonical_id")
    if cid:
        return str(cid)
    return canonical_id(str(item.get("item_id", "")))




def sanity_check_scores(candidates: List[Dict], tolerance: float = 1e-5) -> None:
    """저장된 candidate score와 item score 평균이 맞는지만 조용히 확인."""
    n_warn = 0
    for c in candidates:
        saved = c.get("score")
        item_scores = [item.get("score") for item in c.get("items", []) if isinstance(item.get("score"), (int, float))]
        if saved is None or not item_scores:
            continue
        recomputed = sum(item_scores) / len(item_scores)
        if abs(float(saved) - recomputed) > tolerance:
            n_warn += 1
            print(f"[WARN] {c.get('tag')} score mismatch: saved={saved:.6f}, recomputed={recomputed:.6f}")
    if n_warn == 0:
        print("[OK] Score sanity check passed")

# ============================================================
# 2. 점수 추이 분석
# ============================================================

def analyze_score_trajectory(candidates: List[Dict], output_dir: Path) -> Dict[str, Any]:
    """
    Candidate 점수 추이를 분석하고 그래프 그림.
    """
    import matplotlib.pyplot as plt
    
    indices = [c["idx"] for c in candidates if c.get("score") is not None]
    scores = [c["score"] for c in candidates if c.get("score") is not None]
    
    if not scores:
        print("[WARN] No scores found")
        return {}
    
    # 누적 최고 점수 (best-so-far)
    best_so_far = []
    current_best = -float("inf")
    for s in scores:
        current_best = max(current_best, s)
        best_so_far.append(current_best)
    
    # 그래프 그리기
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(indices, scores, "o-", label="Candidate score", alpha=0.6, color="steelblue")
    ax.plot(indices, best_so_far, "s--", label="Best so far", color="darkred", linewidth=2)
    ax.axhline(y=scores[0], color="gray", linestyle=":", label=f"Seed score ({scores[0]:.3f})")
    
    # 최고 점수 지점 표시
    best_idx_pos = scores.index(max(scores))
    ax.scatter([indices[best_idx_pos]], [max(scores)], 
               color="gold", s=200, zorder=5, edgecolors="black",
               label=f"Best: candidate_{indices[best_idx_pos]:04d} ({max(scores):.3f})")
    
    ax.set_xlabel("Candidate Index (chronological)", fontsize=12)
    ax.set_ylabel("Average Score", fontsize=12)
    ax.set_title("GEPA Optimization Trajectory", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = output_dir / "01_score_trajectory.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[OK] Score trajectory saved to {out_path}")
    
    # 요약 통계
    summary = {
        "n_candidates": len(scores),
        "seed_score": scores[0],
        "final_best_score": max(scores),
        "best_candidate_idx": indices[best_idx_pos],
        "improvement": max(scores) - scores[0],
        "improvement_pct": (max(scores) - scores[0]) / max(abs(scores[0]), 1e-6) * 100,
    }
    
    return summary


# ============================================================
# 3. 항목별 점수 히트맵
# ============================================================

def _item_cid(item: Dict[str, Any]) -> str:
    """Return a robust canonical_id for one item."""
    cid = item.get("canonical_id")
    if cid:
        return str(cid)
    return canonical_id(item.get("item_id", ""))


def analyze_per_item_scores(candidates: List[Dict], output_dir: Path,
                            track_scenarios: Optional[List[str]] = None) -> None:
    """
    각 candidate × 각 시나리오의 점수를 히트맵으로 시각화.

    중요: candidate score는 보통 모든 item_id 점수의 평균입니다.
    그런데 canonical_id 하나(e.g., s4_e4)에 여러 item_id가 있을 수 있습니다.
    따라서 heatmap도 canonical_id별로 마지막 item을 덮어쓰지 않고 평균(mean)으로 집계합니다.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    # 모든 canonical_id 수집
    all_cids = set()
    for c in candidates:
        for item in c.get("items", []):
            all_cids.add(_item_cid(item))

    if track_scenarios:
        cid_list = [cid for cid in track_scenarios if cid in all_cids]
    else:
        cid_list = sorted(all_cids)

    if not cid_list:
        print("[WARN] No scenarios to analyze")
        return

    # 점수 매트릭스 만들기 (rows=candidates, cols=scenarios)
    # canonical_id에 여러 item이 있으면 평균으로 집계
    n_cands = len(candidates)
    n_scens = len(cid_list)
    score_matrix = np.full((n_cands, n_scens), np.nan)
    count_matrix = np.zeros((n_cands, n_scens), dtype=int)

    for i, c in enumerate(candidates):
        grouped_scores = defaultdict(list)
        for item in c.get("items", []):
            score = item.get("score")
            if score is None:
                continue
            grouped_scores[_item_cid(item)].append(float(score))

        for j, cid in enumerate(cid_list):
            vals = grouped_scores.get(cid, [])
            if vals:
                score_matrix[i, j] = float(np.mean(vals))
                count_matrix[i, j] = len(vals)

    # 디버깅용: canonical_id별 item 개수 저장
    counts_path = output_dir / "02_per_item_heatmap_counts.csv"
    with open(counts_path, "w", encoding="utf-8") as f:
        f.write("candidate_idx,canonical_id,n_items_used,mean_score\n")
        for i, c in enumerate(candidates):
            for j, cid in enumerate(cid_list):
                val = score_matrix[i, j]
                val_str = "" if np.isnan(val) else f"{val:.10f}"
                f.write(f"{c['idx']},{cid},{count_matrix[i, j]},{val_str}\n")

    # 히트맵 그리기: 원래 스타일 유지
    fig_height = max(6, 0.3 * n_cands)
    fig_width = max(8, 0.6 * n_scens)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    im = ax.imshow(score_matrix, aspect="auto", cmap="RdYlGn",
                   vmin=0.0, vmax=1.0, interpolation="nearest")

    ax.set_xticks(range(n_scens))
    ax.set_xticklabels(cid_list, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_cands))
    ax.set_yticklabels([f"c{c['idx']:04d}" for c in candidates], fontsize=8)
    ax.set_xlabel("Scenario (canonical_id)", fontsize=12)
    ax.set_ylabel("Candidate", fontsize=12)
    ax.set_title("Per-scenario Mean Score Heatmap (Candidate × Scenario)",
                 fontsize=13, fontweight="bold")

    # 점수 값 표시 (작은 매트릭스일 때만)
    if n_cands * n_scens <= 200:
        for i in range(n_cands):
            for j in range(n_scens):
                val = score_matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color="black" if 0.3 < val < 0.7 else "white")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean score within canonical_id", fontsize=11)

    plt.tight_layout()
    out_path = output_dir / "02_per_item_heatmap.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[OK] Heatmap saved to {out_path}")
    print(f"[OK] Heatmap aggregation counts saved to {counts_path}")


# ============================================================
# 4. 시나리오별 응답 변화 추적
# ============================================================

def find_response_in_generation(gen_data: Any, item_id: str) -> Optional[str]:
    """generation json에서 특정 item의 응답 텍스트 찾기.

    기존 스타일은 유지하되, GEPA 결과에서 자주 나오는 구조를 조금 더 robust하게 처리:
    - list[dict] 형태
    - dict 안에 results/items/generations 등이 들어있는 형태
    - base/pA/pB/n variant별 응답이 들어있는 형태
    """
    if gen_data is None:
        return None

    rows = []
    if isinstance(gen_data, list):
        rows = gen_data
    elif isinstance(gen_data, dict):
        for key in ["results", "items", "generations", "data"]:
            if isinstance(gen_data.get(key), list):
                rows = gen_data[key]
                break
        if not rows:
            rows = [gen_data]

    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("item_id", "")) != str(item_id):
            continue

        # variant별 결과가 있으면 label pattern을 먼저 보여줌
        variant_lines = []
        for variant in ["base", "pA", "pB", "n"]:
            val = row.get(variant)
            if isinstance(val, dict):
                label = val.get("label") or val.get("decision") or val.get("LABEL")
                text = val.get("response") or val.get("output") or val.get("text") or val.get("reason") or val.get("REASON")
                if label or text:
                    variant_lines.append(f"{variant}: {label or ''} {text or ''}".strip())
            elif isinstance(val, str):
                variant_lines.append(f"{variant}: {val}")
        if variant_lines:
            return "\n".join(variant_lines)[:2000]

        for key in ["response", "output", "generation", "completion",
                    "answer", "text", "model_output", "raw_response"]:
            if key in row:
                return str(row[key])[:2000]

        return json.dumps(row, ensure_ascii=False, indent=2)[:2000]

    return None


def find_monitor_row(mon_data: Any, item_id: str) -> Optional[Dict]:
    """monitor json에서 특정 item의 평가 결과 찾기."""
    if mon_data is None or not isinstance(mon_data, list):
        return None
    for row in mon_data:
        if isinstance(row, dict) and str(row.get("item_id", "")) == str(item_id):
            return row
    return None


def trace_scenario_evolution(candidates: List[Dict], output_dir: Path,
                             scenarios_to_track: Optional[List[str]] = None) -> None:
    """
    특정 시나리오들에 대해, candidate가 진화하면서 응답이 어떻게 변했는지 추적.
    """
    # 추적할 시나리오 결정
    all_cids = set()
    for c in candidates:
        for item in c.get("items", []):
            all_cids.add(get_item_canonical_id(item))
    
    if scenarios_to_track:
        cids = [c for c in scenarios_to_track if c in all_cids]
    else:
        cids = sorted(all_cids)
    
    if not cids:
        print("[WARN] No scenarios to trace")
        return
    
    # 각 시나리오마다 별도 파일 작성
    trace_dir = output_dir / "03_scenario_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    
    for cid in cids:
        lines = []
        lines.append("=" * 80)
        lines.append(f"SCENARIO TRACE: {cid}")
        lines.append("=" * 80)
        
        # 이 시나리오의 desired_basis 확인 (첫번째 candidate에서)
        desired = None
        for c in candidates:
            for item in c.get("items", []):
                if get_item_canonical_id(item) == cid:
                    desired = item.get("desired_basis")
                    break
            if desired:
                break
        lines.append(f"Desired basis: {desired}")
        lines.append("")
        
        for c in candidates:
            # 이 candidate의 이 시나리오에 대한 항목 찾기
            target_item = None
            for item in c.get("items", []):
                if get_item_canonical_id(item) == cid:
                    target_item = item
                    break
            
            if target_item is None:
                continue
            
            lines.append("-" * 80)
            lines.append(f"Candidate {c['idx']:04d} (avg score: {c.get('score', 'N/A')})")
            lines.append("-" * 80)
            lines.append(f"  Item ID:        {target_item.get('item_id')}")
            lines.append(f"  Observed basis: {target_item.get('observed_basis')}")
            lines.append(f"  Relation:       {target_item.get('relation')}")
            lines.append(f"  Confidence:     {target_item.get('confidence')}")
            score_val = target_item.get("score")
            lines.append(f"  Score:          {score_val:.3f}" if isinstance(score_val, (int, float)) else f"  Score:          {score_val}")
            
            # generation에서 실제 응답 찾기
            response = find_response_in_generation(
                c.get("generation"), target_item.get("item_id"))
            if response:
                lines.append(f"\n  Response (truncated):")
                for rline in response.split("\n")[:30]:  # 최대 30줄
                    lines.append(f"    {rline}")
            
            # monitor 결과의 rationale
            mon_row = find_monitor_row(c.get("monitor"), target_item.get("item_id"))
            if mon_row:
                rationale = (mon_row.get("monitor") or {}).get("rationale")
                if rationale:
                    lines.append(f"\n  Monitor rationale:")
                    lines.append(f"    {rationale[:500]}")
            
            lines.append("")
        
        out_file = trace_dir / f"trace_{cid}.txt"
        out_file.write_text("\n".join(lines), encoding="utf-8")
    
    print(f"[OK] Scenario traces saved to {trace_dir}/ ({len(cids)} files)")


# ============================================================
# 5. Candidate 텍스트 diff
# ============================================================

def make_text_diff(text_a: str, text_b: str, label_a: str, label_b: str) -> str:
    """두 텍스트의 unified diff 생성"""
    lines_a = text_a.splitlines(keepends=True)
    lines_b = text_b.splitlines(keepends=True)
    diff = difflib.unified_diff(
        lines_a, lines_b,
        fromfile=label_a, tofile=label_b,
        n=2  # context 줄 수
    )
    return "".join(diff)


def analyze_candidate_diffs(candidates: List[Dict], output_dir: Path,
                            top_k: int = 5) -> None:
    """
    Candidate 텍스트가 어떻게 변화했는지 분석.
    
    1. 모든 이웃(consecutive) diff 저장
    2. 점수가 가장 크게 향상된 top_k 변화는 별도로 강조
    3. seed → best diff도 따로 저장
    """
    diffs_dir = output_dir / "04_candidate_diffs"
    diffs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 모든 이웃 candidate 간 diff
    all_diffs_path = diffs_dir / "all_consecutive_diffs.txt"
    with open(all_diffs_path, "w", encoding="utf-8") as f:
        for i in range(1, len(candidates)):
            prev = candidates[i-1]
            curr = candidates[i]
            score_delta = (curr.get("score") or 0) - (prev.get("score") or 0)
            
            f.write(f"\n{'='*80}\n")
            f.write(f"Candidate {prev['idx']:04d} (score={prev.get('score', 'N/A')}) "
                    f"-> {curr['idx']:04d} (score={curr.get('score', 'N/A')})\n")
            f.write(f"Score delta: {score_delta:+.4f}\n")
            f.write(f"{'='*80}\n")
            
            diff = make_text_diff(
                prev["candidate_text"], curr["candidate_text"],
                f"candidate_{prev['idx']:04d}", f"candidate_{curr['idx']:04d}"
            )
            f.write(diff if diff else "(no text difference)\n")
    print(f"[OK] All consecutive diffs saved to {all_diffs_path}")
    
    # 2. 점수 향상이 가장 큰 변화 top_k
    score_jumps = []
    for i in range(1, len(candidates)):
        prev = candidates[i-1]
        curr = candidates[i]
        ps = prev.get("score") or 0
        cs = curr.get("score") or 0
        score_jumps.append((cs - ps, i, prev, curr))
    
    score_jumps.sort(key=lambda x: x[0], reverse=True)
    top_jumps = score_jumps[:top_k]
    
    top_jumps_path = diffs_dir / f"top_{top_k}_improvements.txt"
    with open(top_jumps_path, "w", encoding="utf-8") as f:
        f.write(f"Top {top_k} score improvements\n")
        f.write("=" * 80 + "\n\n")
        for rank, (delta, i, prev, curr) in enumerate(top_jumps, 1):
            f.write(f"\n#{rank}: Candidate {prev['idx']:04d} -> {curr['idx']:04d} "
                    f"(delta = {delta:+.4f})\n")
            f.write("-" * 80 + "\n")
            f.write(f"Score: {prev.get('score', 'N/A')} -> {curr.get('score', 'N/A')}\n\n")
            
            diff = make_text_diff(
                prev["candidate_text"], curr["candidate_text"],
                f"candidate_{prev['idx']:04d}", f"candidate_{curr['idx']:04d}"
            )
            f.write(diff if diff else "(no text difference)\n")
            f.write("\n")
    print(f"[OK] Top {top_k} improvements saved to {top_jumps_path}")
    
    # 3. Seed -> Best diff
    if candidates:
        scored = [c for c in candidates if c.get("score") is not None]
        if scored:
            best = max(scored, key=lambda c: c["score"])
            seed = candidates[0]
            
            seed_to_best_path = diffs_dir / "seed_to_best.txt"
            with open(seed_to_best_path, "w", encoding="utf-8") as f:
                f.write(f"Seed (candidate_{seed['idx']:04d}, score={seed.get('score', 'N/A')}) "
                        f"-> Best (candidate_{best['idx']:04d}, score={best.get('score', 'N/A')})\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("--- SEED TEXT ---\n")
                f.write(seed["candidate_text"])
                f.write("\n\n--- BEST TEXT ---\n")
                f.write(best["candidate_text"])
                f.write("\n\n--- DIFF ---\n")
                
                diff = make_text_diff(
                    seed["candidate_text"], best["candidate_text"],
                    "seed", "best"
                )
                f.write(diff if diff else "(no text difference)\n")
            print(f"[OK] Seed-to-best diff saved to {seed_to_best_path}")


# ============================================================
# 6. CSV 익스포트 (다른 도구로 추가 분석할 수 있게)
# ============================================================

def export_summary_csv(candidates: List[Dict], output_dir: Path) -> None:
    """모든 candidate 정보를 CSV로 익스포트"""
    import csv
    
    # candidate-level summary
    summary_path = output_dir / "05_candidates_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "tag", "score", "n_items", "candidate_text_length"])
        for c in candidates:
            writer.writerow([
                c["idx"], c["tag"], 
                c.get("score", ""), c.get("n_items", ""),
                len(c.get("candidate_text", ""))
            ])
    print(f"[OK] Candidate summary CSV: {summary_path}")
    
    # item-level (long format)
    items_path = output_dir / "05_items_long.csv"
    with open(items_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate_idx", "candidate_score",
            "item_id", "canonical_id", "desired_basis",
            "observed_basis", "relation", "confidence", "item_score"
        ])
        for c in candidates:
            for item in c.get("items", []):
                writer.writerow([
                    c["idx"], c.get("score", ""),
                    item.get("item_id", ""), get_item_canonical_id(item),
                    item.get("desired_basis", ""),
                    item.get("observed_basis", ""), item.get("relation", ""),
                    item.get("confidence", ""), item.get("score", ""),
                ])
    print(f"[OK] Items long-format CSV: {items_path}")


# ============================================================
# 메인
# ============================================================

def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    print(f"[INFO] Loading candidates from {run_dir}")
    candidates = load_candidates(run_dir)
    print(f"[INFO] Loaded {len(candidates)} candidates")
    sanity_check_scores(candidates)
    
    if not candidates:
        print("[ERROR] No candidates found")
        return
    
    # 분석 실행
    print("\n[1/5] Analyzing score trajectory...")
    summary = analyze_score_trajectory(candidates, output_dir)
    print(f"  -> {json.dumps(summary, indent=2)}")
    
    print("\n[2/5] Analyzing per-item scores...")
    analyze_per_item_scores(candidates, output_dir, args.track_scenarios)
    
    print("\n[3/5] Tracing scenario evolution...")
    trace_scenario_evolution(candidates, output_dir, args.track_scenarios)
    
    print("\n[4/5] Analyzing candidate diffs...")
    analyze_candidate_diffs(candidates, output_dir, top_k=args.top_k_diffs)
    
    print("\n[5/5] Exporting CSV summaries...")
    export_summary_csv(candidates, output_dir)
    
    # 최종 요약
    final_summary_path = output_dir / "00_run_summary.json"
    final_summary_path.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_candidates": len(candidates),
        **summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"\n[DONE] All outputs saved to {output_dir}")
    print(f"       Summary: {final_summary_path}")


if __name__ == "__main__":
    main()
