#!/usr/bin/env python3
"""
Revise a steer block using 5-persona reflection-panel outputs, with GPT / LiteLLM.

Reads:
  1) candidate steer block
  2) panel JSON from run_5persona_reflection_panel_gpt.py

Then:
  1) adjudicator GPT synthesizes a benchmark-agnostic revision plan
  2) rewriter GPT produces the revised steer block
  3) optional repair removes benchmark-leakage terms

Example:
  python revise_steer_block_from_panel_gpt.py \
    --candidate_file /path/to/run_dir/candidates/candidate_0010.txt \
    --panel_json ./panel_feedback/candidate_0010_panel.json \
    --output_steer_block ./new_blocks/candidate_0010_panel_rewrite.txt \
    --output_plan_json ./new_blocks/candidate_0010_revision_plan.json \
    --model_name openai/gpt-5.4-mini

You need OPENAI_API_KEY set in the environment.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from litellm import completion


DEFAULT_FORBIDDEN_TERMS = [
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

ADJUDICATOR_SYSTEM = """
You are an adjudicator for a 5-persona prompt-optimization reflection panel.

Your job is to synthesize the panel's critiques into a benchmark-agnostic revision plan.
The plan will be given to a separate rewriter that writes the next steer block.

Important:
- Do NOT encode hidden evaluator details.
- Do NOT propose a fixed global priority ordering over all decision principles.
- Do NOT refer to prompt variants, perturbations, A/B bases, desired bases, benchmark items, or hidden metadata.
- The target model sees only one current user prompt at a time.
- The revised steer block should be a general decision policy for choosing ALLOW / REFUSE / CLARIFY.
""".strip()

REWRITER_SYSTEM = """
You are a constrained prompt rewriter.

Your job is to rewrite a steer block for a target LLM classifier.
Return only the revised steer block, with no analysis, markdown, title, or commentary.

The revised steer block must:
- Work when the target model sees only the current user prompt.
- Identify one primary concern or principle that drives the LABEL.
- Make the LABEL follow that primary concern.
- Use safety, ethics, legality, and reliability as context-sensitive constraints.
- Avoid a fixed global priority ranking over all concerns.
- Avoid benchmark-aware language.
- Avoid hidden metadata, prompt variants, A/B bases, desired bases, perturbations, item ids, or benchmark terminology.
""".strip()

REPAIR_SYSTEM = """
You are a prompt sanitizer.
Rewrite the steer block to remove benchmark-aware or forbidden terms while preserving the intended general policy.
Return only the cleaned steer block.
""".strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Revise steer block from 5-persona panel output using GPT / LiteLLM")
    p.add_argument("--candidate_file", required=True)
    p.add_argument("--panel_json", required=True)
    p.add_argument("--output_steer_block", required=True)
    p.add_argument("--output_plan_json", required=True)

    p.add_argument("--model_name", default="openai/gpt-5.4-mini")
    p.add_argument("--adjudicator_model_name", default=None)
    p.add_argument("--rewriter_model_name", default=None)

    p.add_argument("--max_tokens_plan", type=int, default=1200)
    p.add_argument("--max_tokens_rewrite", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--num_retries", type=int, default=3)
    p.add_argument("--retry_sleep", type=float, default=3.0)
    p.add_argument("--repair_attempts", type=int, default=1)
    return p.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
            if "temperature" in kwargs or "top_p" in kwargs:
                kwargs.pop("temperature", None)
                kwargs.pop("top_p", None)
            time.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {num_retries} retries: {last_err}")


def strip_fences(text: str) -> str:
    x = text.strip()
    x = re.sub(r"^```(?:text|markdown)?\s*", "", x, flags=re.IGNORECASE)
    x = re.sub(r"\s*```$", "", x)
    return x.strip()


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


def get_forbidden_terms(panel_json: Dict[str, Any]) -> List[str]:
    terms = panel_json.get("forbidden_benchmark_terms") or DEFAULT_FORBIDDEN_TERMS
    out = []
    for t in terms:
        s = str(t).strip()
        if s and s not in out:
            out.append(s)
    return out


def leakage_terms(text: str, forbidden_terms: List[str]) -> List[str]:
    low = text.lower()
    hits = []
    for term in forbidden_terms:
        if term.lower() in low:
            hits.append(term)
    return hits


def summarize_panel(panel_json: Dict[str, Any]) -> str:
    outputs = panel_json.get("panel_outputs", []) or []
    lines = []
    for row in outputs:
        persona = row.get("persona", "UNKNOWN")
        lines.append(f"## {persona}")
        for key in [
            "diagnosis",
            "benchmark_leakage_risk",
            "procedural_validity",
            "generalization_risk",
            "revision_principles",
            "do_not_include",
            "concrete_rewrite_advice",
            "confidence",
        ]:
            if key in row:
                lines.append(f"{key}: {json.dumps(row[key], ensure_ascii=False)}")
        if not row.get("parse_ok", True):
            lines.append(f"parse_error: {row.get('parse_error')}")
            lines.append(f"raw_text: {row.get('raw_text')}")
        lines.append("")
    return "\n".join(lines).strip()


def build_adjudicator_prompt(candidate_text: str, panel_json: Dict[str, Any], forbidden_terms: List[str]) -> str:
    panel_summary = summarize_panel(panel_json)
    forbidden_text = "\n".join(f"- {x}" for x in forbidden_terms)

    return f"""
Current steer block:
---
{candidate_text}
---

5-persona panel feedback:
---
{panel_summary}
---

Forbidden terms / patterns for the revised steer block:
{forbidden_text}

Create a single benchmark-agnostic revision plan.

Output a valid JSON object with exactly these fields:
{{
  "main_revision_goal": "one sentence",
  "preserve": ["point 1", "point 2"],
  "change": ["point 1", "point 2"],
  "avoid": ["point 1", "point 2"],
  "rewriter_instructions": "one concise paragraph for the rewriter",
  "quality_checks": ["check 1", "check 2", "check 3"]
}}

Rules:
- Do not include forbidden terms in the plan unless quoting them under the avoid field as things to avoid.
- Do not tell the rewriter to create a fixed hierarchy over all possible concerns.
- The plan must make the next steer block usable for a single current prompt only.
""".strip()


def build_rewriter_prompt(candidate_text: str, revision_plan: Dict[str, Any], forbidden_terms: List[str]) -> str:
    forbidden_text = "\n".join(f"- {x}" for x in forbidden_terms)

    return f"""
Current steer block:
---
{candidate_text}
---

Revision plan:
---
{json.dumps(revision_plan, ensure_ascii=False, indent=2)}
---

Forbidden terms / patterns:
{forbidden_text}

Rewrite the steer block.

Output requirements:
- Return only the revised steer block.
- Do not include markdown fences.
- Do not include analysis.
- Do not include any forbidden terms.
- Do not include a numbered global ranking of all possible concerns.
- The target model must be able to apply the instruction to one current prompt at a time.
""".strip()


def build_repair_prompt(block_text: str, forbidden_terms: List[str], hits: List[str]) -> str:
    return f"""
The following steer block contains forbidden benchmark-aware terms.

Forbidden hits:
{json.dumps(hits, ensure_ascii=False, indent=2)}

Full forbidden list:
{json.dumps(forbidden_terms, ensure_ascii=False, indent=2)}

Steer block:
---
{block_text}
---

Rewrite it to remove the forbidden terms and any benchmark-aware language.
Preserve the general current-prompt-only decision policy.
Return only the cleaned steer block.
""".strip()


def main() -> None:
    args = parse_args()

    candidate_text = Path(args.candidate_file).read_text(encoding="utf-8").strip()
    panel_json = load_json(args.panel_json)
    forbidden_terms = get_forbidden_terms(panel_json)

    adjudicator_model = args.adjudicator_model_name or args.model_name
    rewriter_model = args.rewriter_model_name or args.model_name

    raw_plan = call_llm(
        model=adjudicator_model,
        messages=[
            {"role": "system", "content": ADJUDICATOR_SYSTEM},
            {"role": "user", "content": build_adjudicator_prompt(candidate_text, panel_json, forbidden_terms)},
        ],
        max_tokens=args.max_tokens_plan,
        temperature=args.temperature,
        top_p=args.top_p,
        num_retries=args.num_retries,
        retry_sleep=args.retry_sleep,
    )

    try:
        revision_plan = extract_json_object(raw_plan)
        plan_parse_ok = True
    except Exception as e:
        revision_plan = {
            "main_revision_goal": "Improve the steer block using panel feedback while avoiding benchmark-aware language.",
            "preserve": ["single primary concern", "LABEL follows the primary concern"],
            "change": ["make the instruction current-prompt-only", "avoid fixed global rankings"],
            "avoid": forbidden_terms,
            "rewriter_instructions": "Rewrite the block as a general decision policy for one current prompt at a time.",
            "quality_checks": ["no forbidden terms", "no hidden metadata", "no global priority list"],
            "parse_error": str(e),
            "raw_plan": raw_plan,
        }
        plan_parse_ok = False

    revised = call_llm(
        model=rewriter_model,
        messages=[
            {"role": "system", "content": REWRITER_SYSTEM},
            {"role": "user", "content": build_rewriter_prompt(candidate_text, revision_plan, forbidden_terms)},
        ],
        max_tokens=args.max_tokens_rewrite,
        temperature=args.temperature,
        top_p=args.top_p,
        num_retries=args.num_retries,
        retry_sleep=args.retry_sleep,
    )
    revised = strip_fences(revised)

    repair_history = []
    for attempt in range(args.repair_attempts):
        hits = leakage_terms(revised, forbidden_terms)
        if not hits:
            break
        repaired = call_llm(
            model=rewriter_model,
            messages=[
                {"role": "system", "content": REPAIR_SYSTEM},
                {"role": "user", "content": build_repair_prompt(revised, forbidden_terms, hits)},
            ],
            max_tokens=args.max_tokens_rewrite,
            temperature=args.temperature,
            top_p=args.top_p,
            num_retries=args.num_retries,
            retry_sleep=args.retry_sleep,
        )
        repair_history.append({"attempt": attempt + 1, "hits_before": hits, "raw_repair": repaired})
        revised = strip_fences(repaired)

    final_hits = leakage_terms(revised, forbidden_terms)

    out_block_path = Path(args.output_steer_block)
    out_block_path.parent.mkdir(parents=True, exist_ok=True)
    out_block_path.write_text(revised.strip() + "\n", encoding="utf-8")

    meta = {
        "candidate_file": str(args.candidate_file),
        "panel_json": str(args.panel_json),
        "adjudicator_model": adjudicator_model,
        "rewriter_model": rewriter_model,
        "plan_parse_ok": plan_parse_ok,
        "raw_plan": raw_plan,
        "revision_plan": revision_plan,
        "final_forbidden_hits": final_hits,
        "repair_history": repair_history,
        "output_steer_block": str(out_block_path),
    }

    out_plan_path = Path(args.output_plan_json)
    out_plan_path.parent.mkdir(parents=True, exist_ok=True)
    out_plan_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_steer_block": str(out_block_path),
        "output_plan_json": str(out_plan_path),
        "plan_parse_ok": plan_parse_ok,
        "final_forbidden_hits": final_hits,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
