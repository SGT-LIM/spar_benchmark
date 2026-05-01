import json
from collections import Counter
from pathlib import Path

# ===== 경로 설정 =====
file_a = Path("/home/coder/SPAR/0416-GEPAbased/2-spec_conflict_results_with_steer.json")   # 1회 결과 파일
file_b = Path("/home/coder/SPAR/0423/2-multiple-spec_conflict_results_with_steer.json")   # 2회 결과 파일
output_file = Path("merged_3samples.json")

PROMPT_KEYS = ["base_prompt", "pA", "pB", "n"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_summary(samples):
    labels = [s.get("label", "CLARIFY") for s in samples]
    counter = Counter(labels)

    # 가장 많은 label을 대표 label로 사용
    # 동률이면 CLARIFY 우선, 없으면 사전순
    max_count = max(counter.values())
    candidates = [lab for lab, cnt in counter.items() if cnt == max_count]
    if "CLARIFY" in candidates:
        final_label = "CLARIFY"
    else:
        final_label = sorted(candidates)[0]

    consistency = max_count / len(samples) if samples else 0.0

    return final_label, consistency, dict(counter)


def merge_one_item(item_a, item_b):
    merged = {
        "item_id": item_b["item_id"],
        "basis_A": item_b.get("basis_A", item_a.get("basis_A")),
        "basis_B": item_b.get("basis_B", item_a.get("basis_B")),
        "steer_block_used": item_b.get("steer_block_used", item_a.get("steer_block_used")),
        "num_samples": 3,
        "generations": {}
    }

    for key in PROMPT_KEYS:
        # b 쪽 기존 samples 2개
        b_gen = item_b["generations"][key]
        old_samples = b_gen.get("samples", [])

        # a 쪽 단일 결과 1개를 sample 형식으로 변환
        a_gen = item_a["generations"][key]
        new_sample = {
            "raw_text": a_gen["raw_text"],
            "label": a_gen["label"]
        }

        merged_samples = old_samples + [new_sample]

        final_label, consistency, label_dist = compute_summary(merged_samples)

        merged["generations"][key] = {
            "label": final_label,
            "consistency": consistency,
            "label_dist": label_dist,
            "samples": merged_samples
        }

    return merged


def main():
    data_a = load_json(file_a)
    data_b = load_json(file_b)

    a_map = {item["item_id"]: item for item in data_a}
    b_map = {item["item_id"]: item for item in data_b}

    merged_data = []

    common_ids = sorted(set(a_map.keys()) & set(b_map.keys()))
    missing_in_a = sorted(set(b_map.keys()) - set(a_map.keys()))
    missing_in_b = sorted(set(a_map.keys()) - set(b_map.keys()))

    if missing_in_a:
        print(f"[WARN] b에는 있지만 a에는 없는 item_id 수: {len(missing_in_a)}")
    if missing_in_b:
        print(f"[WARN] a에는 있지만 b에는 없는 item_id 수: {len(missing_in_b)}")

    for item_id in common_ids:
        merged_item = merge_one_item(a_map[item_id], b_map[item_id])
        merged_data.append(merged_item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"[DONE] merged file saved to: {output_file}")
    print(f"[INFO] merged items: {len(merged_data)}")


if __name__ == "__main__":
    main()