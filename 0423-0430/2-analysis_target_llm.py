import json
from collections import Counter, defaultdict
from statistics import mean, median
from pathlib import Path

json_path = Path("merged_3samples.json")
PROMPT_KEYS = ["base_prompt", "pA", "pB", "n"]

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

all_consistencies = []
by_prompt = defaultdict(list)

for item in data:
    generations = item.get("generations", {})
    for key in PROMPT_KEYS:
        if key in generations:
            c = generations[key].get("consistency")
            if c is not None:
                all_consistencies.append(c)
                by_prompt[key].append(c)

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
    for k, v in buckets.items():
        print(f"{k}: {v}")

def print_summary(values, title):
    print(f"\n=== {title}: summary ===")
    print(f"count   : {len(values)}")
    if values:
        print(f"mean    : {mean(values):.6f}")
        print(f"median  : {median(values):.6f}")
        print(f"min     : {min(values):.6f}")
        print(f"max     : {max(values):.6f}")

# 전체
print_summary(all_consistencies, "ALL")
print_exact_distribution(all_consistencies, "ALL")
print_bucket_distribution(all_consistencies, "ALL")

# prompt별
for key in PROMPT_KEYS:
    vals = by_prompt[key]
    print_summary(vals, key)
    print_exact_distribution(vals, key)
    print_bucket_distribution(vals, key)