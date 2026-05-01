import json
from collections import Counter, defaultdict
from statistics import mean, median
from pathlib import Path

json_path = Path("/home/coder/SPAR/0423-0430/3-monitor_results.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

metrics = [
    "basis_consistency",
    "relation_consistency",
    "confidence_consistency",
    "parse_ok_rate",
]

values_by_metric = defaultdict(list)

basis_dist_all = Counter()
relation_dist_all = Counter()
confidence_dist_all = Counter()

low_consistency_items = []

def print_summary(values, title):
    print(f"\n=== {title}: summary ===")
    print(f"count   : {len(values)}")
    if values:
        print(f"mean    : {mean(values):.6f}")
        print(f"median  : {median(values):.6f}")
        print(f"min     : {min(values):.6f}")
        print(f"max     : {max(values):.6f}")

def print_exact_distribution(values, title):
    counter = Counter(values)
    print(f"\n=== {title}: exact distribution ===")
    for k in sorted(counter.keys()):
        print(f"{k:.6f}: {counter[k]}")

def print_bucket_distribution(values, title):
    buckets = {
        "1.0": 0,
        "[0.8,1.0)": 0,
        "[0.6,0.8)": 0,
        "[0.4,0.6)": 0,
        "[0.2,0.4)": 0,
        "[0.0,0.2)": 0,
    }

    for v in values:
        if v == 1.0:
            buckets["1.0"] += 1
        elif 0.8 <= v < 1.0:
            buckets["[0.8,1.0)"] += 1
        elif 0.6 <= v < 0.8:
            buckets["[0.6,0.8)"] += 1
        elif 0.4 <= v < 0.6:
            buckets["[0.4,0.6)"] += 1
        elif 0.2 <= v < 0.4:
            buckets["[0.2,0.4)"] += 1
        elif 0.0 <= v < 0.2:
            buckets["[0.0,0.2)"] += 1

    print(f"\n=== {title}: bucket distribution ===")
    for k, count in buckets.items():
        print(f"{k}: {count}")

for item in data:
    item_id = item.get("item_id")
    monitor = item.get("monitor", {})

    for metric in metrics:
        v = monitor.get(metric)
        if v is not None:
            values_by_metric[metric].append(v)

    basis_dist_all.update(monitor.get("basis_dist", {}))
    relation_dist_all.update(monitor.get("relation_dist", {}))
    confidence_dist_all.update(monitor.get("confidence_dist", {}))

    basis_c = monitor.get("basis_consistency")
    relation_c = monitor.get("relation_consistency")

    if basis_c is not None and relation_c is not None:
        if basis_c < 1.0 or relation_c < 1.0:
            low_consistency_items.append({
                "item_id": item_id,
                "basis_A": item.get("basis_A"),
                "basis_B": item.get("basis_B"),
                "monitor_basis_decision": monitor.get("basis_decision"),
                "monitor_relation": monitor.get("relation"),
                "basis_consistency": basis_c,
                "relation_consistency": relation_c,
                "basis_dist": monitor.get("basis_dist", {}),
                "relation_dist": monitor.get("relation_dist", {}),
            })

print("\n==============================")
print("MONITOR CONSISTENCY ANALYSIS")
print("==============================")

for metric in metrics:
    vals = values_by_metric[metric]
    print_summary(vals, metric)
    print_exact_distribution(vals, metric)
    print_bucket_distribution(vals, metric)

print("\n=== Monitor basis decision distribution across all monitor samples ===")
for k, v in basis_dist_all.items():
    print(f"{k}: {v}")

print("\n=== Monitor relation distribution across all monitor samples ===")
for k, v in relation_dist_all.items():
    print(f"{k}: {v}")

print("\n=== Monitor confidence distribution across all monitor samples ===")
for k, v in confidence_dist_all.items():
    print(f"{k}: {v}")

print("\n=== Items with imperfect monitor consistency ===")
print(f"count: {len(low_consistency_items)}")

for x in low_consistency_items[:50]:
    print(
        f"{x['item_id']} | "
        f"{x['basis_A']} vs {x['basis_B']} | "
        f"basis={x['monitor_basis_decision']} "
        f"basis_c={x['basis_consistency']} "
        f"basis_dist={x['basis_dist']} | "
        f"relation={x['monitor_relation']} "
        f"relation_c={x['relation_consistency']} "
        f"relation_dist={x['relation_dist']}"
    )