from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluator import build_format_repair_prompt, validate_ranking_output
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
    allow_format_repair: bool = True,
) -> dict[str, Any]:
    """Execute and persist one benchmark cell.

    The raw first response is always retained. A repair call, if used, is stored
    separately and is allowed to change formatting only. Persistent invalidity
    remains an explicit outcome rather than being silently excluded.

    ``prompt_mode='audit'`` is mandatory for RQ1--RQ3 unmitigated evaluation.
    RQ4 may additionally run named mitigation modes such as
    ``identity_irrelevance``. Template, cue representation, and candidate-order
    seed are logged so presentation choices cannot be hidden from analysis.

    ``planned_cell_id`` links the persisted response to an immutable run-plan
    row. Resumable execution should always provide it; direct unit/pilot calls may
    leave it ``None``.
    """
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
    validation = validate_ranking_output(response.text, instance, k=k)

    repair_record: dict[str, Any] | None = None
    final_validation = validation
    if not validation.valid and allow_format_repair:
        repair_prompt = build_format_repair_prompt(response.text, k=k)
        repair_request = GenerationRequest(
            prompt=repair_prompt,
            model_id=model_id,
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=max_output_tokens,
            seed=seed if provider.supports_seed() else None,
            reasoning_or_thinking_setting=reasoning_or_thinking_setting,
        )
        repair_response = provider.generate(repair_request)
        repaired_validation = validate_ranking_output(
            repair_response.text,
            instance,
            k=k,
            repaired_format=True,
        )
        repair_record = {
            "prompt_sha256": prompt_sha256(repair_prompt),
            "raw_response": repair_response.text,
            "response_sha256": _sha256(repair_response.text),
            "resolved_model_version": repair_response.resolved_model_version,
            "provider_metadata": dict(repair_response.provider_metadata),
            "valid": repaired_validation.valid,
            "errors": list(repaired_validation.errors),
        }
        if repaired_validation.valid:
            final_validation = repaired_validation

    row: dict[str, Any] = {
        "schema_version": "faireval-run-v4",
        "planned_cell_id": planned_cell_id,
        "dataset": instance.dataset,
        "user_id": instance.user_id,
        "condition_id": condition.condition_id,
        "condition_name": condition.condition_name,
        # Keep legacy template_id/temperature/top_p fields for backwards
        # compatibility while also exposing explicit requested-value names used
        # by the reproducibility manifest.
        "template_id": template_id,
        "prompt_template_id": template_id,
        "prompt_mode": prompt_mode,
        "cue_id": cue_id,
        "candidate_order_seed": candidate_order_seed,
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
        "initial_valid": validation.valid,
        "initial_errors": list(validation.errors),
        "repair": repair_record,
        "final_valid": final_validation.valid,
        "final_errors": list(final_validation.errors),
        "ranking": (
            list(final_validation.ranking.ranked_item_ids)
            if final_validation.ranking is not None
            else None
        ),
        "code_commit_sha": code_commit_sha,
    }
    _append_jsonl(output_jsonl, row)
    return row
