from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .output_protocol import analyze_ranking_output
from .prompts import build_ranking_prompt, prompt_sha256
from .providers.base import GenerationRequest, ProviderAdapter
from .schema import PromptCondition, UserInstance


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_one(
    *,
    instance: UserInstance,
    condition: PromptCondition,
    provider: ProviderAdapter,
    model_id: str,
    k: int,
    repetition: int,
    output_jsonl: Path,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    template_id: str = "field_v2_a",
    prompt_mode: str = "audit",
    cue_id: str = "structured_key_value",
    candidate_order_seed: int | None = None,
    seed: int | None = None,
    reasoning_or_thinking_setting: str | None = None,
    code_commit_sha: str | None = None,
    planned_cell_id: str | None = None,
    allow_format_repair: bool = False,
) -> dict[str, Any]:
    """Execute and persist one V5 benchmark cell with exactly one model call.

    Canonical V5 forbids generative output repair. The first response is retained
    verbatim and classified by the deterministic output protocol. Envelope-only
    normalization may recover a ranking for semantic evaluation, while strict
    serialization compliance is logged independently.
    """
    if allow_format_repair:
        raise ValueError(
            "generative format repair is disabled in FairEval V5; "
            "use deterministic output-protocol normalization only"
        )

    prompt = build_ranking_prompt(
        instance,
        condition,
        k=k,
        template_id=template_id,
        prompt_mode=prompt_mode,
        cue_id=cue_id,
        candidate_order_seed=candidate_order_seed,
    )
    request_utc = datetime.now(timezone.utc).isoformat()
    request = GenerationRequest(
        prompt=prompt,
        model_id=model_id,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        seed=seed if provider.supports_seed() else None,
        reasoning_or_thinking_setting=reasoning_or_thinking_setting,
    )
    response = provider.generate(request)
    provider_metadata = dict(response.provider_metadata)

    protocol = analyze_ranking_output(
        response.text,
        candidate_ids=instance.candidate_ids(),
        k=k,
    )
    if protocol.candidate_id_mutation_detected:
        raise RuntimeError("deterministic output protocol reported candidate-ID mutation")

    row: dict[str, Any] = {
        "schema_version": "faireval-run-v5",
        "planned_cell_id": planned_cell_id,
        "dataset": instance.dataset,
        "user_id": instance.user_id,
        "condition_id": condition.condition_id,
        "condition_name": condition.condition_name,
        "template_id": template_id,
        "prompt_template_id": template_id,
        "prompt_mode": prompt_mode,
        "cue_id": cue_id,
        "candidate_order_seed": candidate_order_seed,
        "k": int(k),
        "repetition": repetition,
        "provider": provider.provider_name,
        "model_family": provider.family,
        "requested_model_id": model_id,
        "resolved_model_version": response.resolved_model_version,
        "request_utc": request_utc,
        "temperature": temperature,
        "temperature_requested": temperature,
        "top_p": top_p,
        "top_p_requested": top_p,
        "max_output_tokens": max_output_tokens,
        "reasoning_or_thinking_setting": reasoning_or_thinking_setting,
        "reasoning_or_thinking_applied": provider_metadata.get(
            "reasoning_or_thinking_applied"
        ),
        "sampling_controls_applied": provider_metadata.get("sampling_controls_applied"),
        "sampling_policy": provider_metadata.get("sampling_policy"),
        "output_token_parameter": provider_metadata.get("output_token_parameter"),
        "seed_requested": seed,
        "seed_supported": provider.supports_seed(),
        "prompt_sha256": prompt_sha256(prompt),
        "request_sha256": _sha256(json.dumps(asdict(request), sort_keys=True)),
        "raw_response": response.text,
        "response_sha256": _sha256(response.text),
        "provider_metadata": provider_metadata,
        # Legacy aliases retained so downstream invalid-output metrics can be
        # migrated without silently changing their meaning. In V5 they both mean
        # semantic exact-k candidate validity after deterministic envelope parsing.
        "initial_valid": protocol.semantic_ranking_valid,
        "initial_errors": list(protocol.semantic_errors),
        "repair": None,
        "final_valid": protocol.semantic_ranking_valid,
        "final_errors": list(protocol.semantic_errors),
        "ranking": None if protocol.ranking is None else list(protocol.ranking),
        "output_protocol": protocol.as_dict(),
        "strict_format_valid": protocol.strict_format_valid,
        "semantic_ranking_valid": protocol.semantic_ranking_valid,
        "format_violations": list(protocol.format_violations),
        "semantic_errors": list(protocol.semantic_errors),
        "deterministic_normalization_applied": protocol.normalization_applied,
        "normalization_actions": list(protocol.normalization_actions),
        "parser_ambiguity": protocol.parser_ambiguity,
        "candidate_id_mutation_detected": protocol.candidate_id_mutation_detected,
        "generative_format_repair_enabled": False,
        "provider_generation_calls_for_cell": 1,
        "code_commit_sha": code_commit_sha,
    }
    _append_jsonl(output_jsonl, row)
    return row
