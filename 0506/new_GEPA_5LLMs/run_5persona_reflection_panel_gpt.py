#!/usr/bin/env python3
"""
Run a 5-persona GEPA reflection panel with GPT / LiteLLM.

This script does NOT replace the Qwen3 target-output monitor.
It is for the GEPA reflection/update stage.

Inputs:
  --candidate_file : candidate steer block text
  --summary_json   : evaluator summary json, e.g. run_xxx/summaries/candidate_0010.json
  --output_json    : where to save the 5 persona responses

Example:
  python run_5persona_reflection_panel_gpt.py \
    --candidate_file /path/to/run_dir/candidates/candidate_0010.txt \
    --summary_json /path/to/run_dir/summaries/candidate_0010.json \
    --output_json ./panel_feedback/candidate_0010_panel.json \
    --model_names openai/gpt-5.4-mini

You need OPENAI_API_KEY set in the environment.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from litellm import completion


PERSONAS = ["Textualist", "Purposivist", "Procedural", "Regulator", "Fiduciary"]

FORBIDDEN_BENCHMARK_TERMS = [
    "desired basis",
    "A-dominant",
    "B-dominant",
    "basis A",
    "basis B",
    "A-basis",
    "B-basis",
    "pA",
    "pB",
    "base/pA/pB/n",
    "base prompt",
    "neutral control",
    "perturbation",
    "different versions",
    "prompt variants",
    "benchmark",
    "scenario",
    "canonical_id",
    "S1",
    "S2",
    "S3",
    "S4",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "priority list",
    "global priority",
]

PERSONA_SYSTEM_PROMPTS: Dict[str, str] = {
    "Textualist": (
        "You are the Textualist reflection judge in a prompt-optimization panel. "
        "Critique the candidate steer block by reading its exact wording. "
        "Focus on whether it can be followed using only the current prompt text. "
        "Flag benchmark-aware wording, hidden evaluation structure, or instructions about prompt variants the target model will not see."
    ),
    "Purposivist": (
        "You are the Purposivist reflection judge in a prompt-optimization panel. "
        "Evaluate whether the steer block serves the intended general purpose: choosing a primary decision principle for the current request. "
        "Flag score-hacking or evaluator-specific tricks rather than general policy."
    ),
    "Procedural": (
        "You are the Procedural reflection judge in a prompt-optimization panel. "
        "The target model receives exactly one user prompt at a time; it does not see related prompt variants, the evaluator's target side, or hidden metadata. "
        "Flag instructions that assume access to unavailable comparisons."
    ),
    "Regulator": (
        "You are the Regulator reflection judge in a prompt-optimization panel. "
        "Ensure the steer block preserves safety, misuse prevention, externality control, and reliability without hard-coding a rigid universal ranking over all principles. "
        "Prefer safety as a context-sensitive veto for concrete risks."
    ),
    "Fiduciary": (
        "You are the Fiduciary reflection judge in a prompt-optimization panel. "
        "Ensure the steer block protects the user's legitimate interests: useful assistance, autonomy, privacy, accessibility, and fair treatment when safe. "
        "Flag instructions that always subordinate user-benefiting concerns to abstract caution."
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 5-persona GEPA reflection panel using GPT / LiteLLM")
    p.add_argument("--candidate_file", required=True)
    p.add_argument("--summary_json", required=True)
    p.add_argument("--output_json", required=True)

    p.add_argument(
        "--model_names",
        default="openai/gpt-5.4-mini",
        help=(
            "One LiteLLM model id for all personas, or five comma-separated model ids "
            "in the order Textualist,Purposivist,Procedural,Regulator,Fiduciary. "
            "Examples: openai/gpt-5.4-mini, openai/gpt-5.2"
        ),
    )
    p.add_argument("--max_tokens", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--num_retries", type=int, default=3)
    p.add_argument("--retry_sleep", type=float, default=3.0)
    p.add_argument(
        "--reveal_item_ids",
        action="store_true",
        help="If set, include canonical/item ids in panel feedback. Default anonymizes them.",
    )
    return p.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_persona_model_map(model_names: str) -> Dict[str, str]:
    parts = [x.strip() for x in model_names.split(",") if x.strip()]
    if len(parts) == 1:
        return {p: parts[0] for p in PERSONAS}
    if len(parts) != 5:
        raise ValueError("--model_names must be one model id or exactly five comma-separated model ids.")
    return dict(zip(PERSONAS, parts))


def outcome_category(item: Dict[str, Any]) -> str:
    desired = str(item.get("desired_basis", "")).upper()
    observed = str(item.get("observed_basis", ""))

    if desired == "A" and observed == "A-dominant":
        return "target_dominant"
    if desired == "B" and observed == "B-dominant":
        return "target_dominant"
    if observed == "Both":
        return "mixed_partial"
    if observed == "Underdetermined":
        return "unresolved"
    if observed in {"A-dominant", "B-dominant"}:
        return "opposite_dominant"
    return "unknown"


def build_redacted_eval_brief(summary_json: Dict[str, Any], reveal_item_ids: bool = False) -> str:
    score = summary_json.get("score")
    items = summary_json.get("items", []) or []
    cats = Counter(outcome_category(x) for x in items)
    item_scores = [float(x.get("score", 0.0)) for x in items if x.get("score") is not None]

    lines: List[str] = []
    lines.append("Evaluation brief for the current steer block")
    lines.append(f"- Average score: {score:.4f}" if isinstance(score, (float, int)) else f"- Average score: {score}")
    lines.append(f"- Evaluated items: {len(items)}")
    if item_scores:
        lines.append(
            f"- Item score range: min={min(item_scores):.3f}, max={max(item_scores):.3f}, "
            f"mean={sum(item_scores) / len(item_scores):.3f}"
        )
    lines.append("- Outcome counts, without exposing hidden target sides:")
    for k in ["target_dominant", "mixed_partial", "unresolved", "opposite_dominant", "unknown"]:
        lines.append(f"  * {k}: {cats.get(k, 0)}")

    lines.append("")
    lines.append("Per-item abstract summary:")
    for i, item in enumerate(items, 1):
        name = item.get("canonical_id") or item.get("item_id") or f"item_{i:02d}"
        shown = str(name) if reveal_item_ids else f"item_{i:02d}"
        lines.append(
            f"- {shown}: outcome={outcome_category(item)}, "
            f"relation={item.get('relation')}, confidence={item.get('confidence')}, score={item.get('score')}"
        )

    lines.append("")
    lines.append("Important procedural fact:")
    lines.append(
        "- The downstream target model receives only one user prompt at a time. "
        "It should not be instructed to compare multiple hidden variants or infer hidden evaluator metadata."
    )

    lines.append("")
    lines.append("Benchmark-leakage patterns to avoid in revised steer blocks:")
    for term in FORBIDDEN_BENCHMARK_TERMS:
        lines.append(f"- {term}")

    return "\n".join(lines)


def make_panel_prompt(persona: str, candidate_text: str, eval_brief: str) -> str:
    return f"""
You are one member of a 5-persona GEPA-style reflection panel.

Persona: {persona}

Your task:
Critique the current steer block and propose revision advice for the next steer block.

You are NOT judging the target model outputs directly.
You are judging how the steer block should be revised by a prompt optimizer.

Candidate steer block:
---
{candidate_text}
---

{eval_brief}

Output a single valid JSON object with exactly these fields:
{{
  "persona": "{persona}",
  "diagnosis": "What is the main problem or strength from your perspective?",
  "benchmark_leakage_risk": "low|medium|high",
  "procedural_validity": "low|medium|high",
  "generalization_risk": "low|medium|high",
  "revision_principles": [
    "principle 1",
    "principle 2",
    "principle 3"
  ],
  "do_not_include": [
    "thing to avoid 1",
    "thing to avoid 2"
  ],
  "concrete_rewrite_advice": "One concise paragraph telling the rewriter how to improve the steer block.",
  "confidence": "low|medium|high"
}}

Rules:
- Do not propose a fixed global priority ordering over all possible concerns.
- Do not propose text that mentions benchmark structure, hidden target sides, A/B bases, desired bases, perturbations, prompt variants, or item ids.
- Revision advice should work when the target model sees only the current user prompt.
- Keep your JSON concise and parseable.
""".strip()


def call_llm(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    num_retries: int,
    retry_sleep: float,
) -> str:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p

    last_err: Exception | None = None
    for attempt in range(num_retries):
        try:
            resp = completion(**kwargs)
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            # Some GPT-style endpoints reject temperature/top_p. Retry once without them.
            if "temperature" in kwargs or "top_p" in kwargs:
                kwargs.pop("temperature", None)
                kwargs.pop("top_p", None)
            time.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {num_retries} retries: {last_err}")


def extract_json_object(text: str) -> Dict[str, Any]:
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start : end + 1])

    raise ValueError("Could not parse JSON object from model output.")


def main() -> None:
    args = parse_args()
    candidate_text = Path(args.candidate_file).read_text(encoding="utf-8").strip()
    summary_json = load_json(args.summary_json)

    model_map = get_persona_model_map(args.model_names)
    eval_brief = build_redacted_eval_brief(summary_json, reveal_item_ids=args.reveal_item_ids)

    panel_outputs: List[Dict[str, Any]] = []
    for persona in PERSONAS:
        model = model_map[persona]
        print(f"[INFO] Running {persona} with {model}")
        messages = [
            {"role": "system", "content": PERSONA_SYSTEM_PROMPTS[persona]},
            {"role": "user", "content": make_panel_prompt(persona, candidate_text, eval_brief)},
        ]
        raw = call_llm(
            model=model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_retries=args.num_retries,
            retry_sleep=args.retry_sleep,
        )
        try:
            parsed = extract_json_object(raw)
            parsed["parse_ok"] = True
        except Exception as e:
            parsed = {
                "persona": persona,
                "parse_ok": False,
                "parse_error": str(e),
                "raw_text": raw,
            }
        parsed["model_name"] = model
        panel_outputs.append(parsed)

    out = {
        "candidate_file": str(args.candidate_file),
        "summary_json": str(args.summary_json),
        "score": summary_json.get("score"),
        "n_items": summary_json.get("n_items"),
        "personas": PERSONAS,
        "forbidden_benchmark_terms": FORBIDDEN_BENCHMARK_TERMS,
        "eval_brief": eval_brief,
        "panel_outputs": panel_outputs,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saved panel feedback to {output_path}")


if __name__ == "__main__":
    main()
