from __future__ import annotations

import math
import os
from typing import Any

from .base import GenerationRequest, GenerationResponse, ProviderAdapter


class LocalTransformersAdapter(ProviderAdapter):
    """Direct local Hugging Face inference with auxiliary white-box diagnostics.

    This adapter deliberately loads model weights in-process rather than calling
    an OpenAI-compatible server. That gives FairEval access to generation scores
    for the two open-weight models while keeping the same prompt/output validator
    used by hosted APIs.

    The reported log-probability/margin values are *generation-score diagnostics*.
    They are not assumed to be calibrated probabilities or directly comparable
    to provider-side confidence for closed models.
    """

    output_token_parameter = "max_new_tokens"

    def __init__(
        self,
        *,
        family: str,
        model_id: str,
        revision: str | None = None,
        trust_remote_code: bool = False,
        dtype_preference: str = "bfloat16",
    ) -> None:
        self.family = family
        self.provider_name = "local_transformers"
        self.model_id = model_id
        self.revision = None if revision in {None, "", "pin_exact_huggingface_commit_before_pilot"} else revision
        self.trust_remote_code = bool(trust_remote_code)
        self.dtype_preference = dtype_preference
        self._tokenizer = None
        self._model = None

    def supports_seed(self) -> bool:
        return True

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Install a CUDA-matched PyTorch build and requirements-local-gpu.txt"
            ) from exc

        if not torch.cuda.is_available() and os.environ.get("FAIREVAL_ALLOW_LOCAL_CPU") != "1":
            raise RuntimeError(
                "Local open-weight track requires CUDA by default. Set "
                "FAIREVAL_ALLOW_LOCAL_CPU=1 only for tiny smoke tests."
            )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )
        dtype = getattr(torch, self.dtype_preference, None)
        if dtype is None:
            raise ValueError(f"unsupported torch dtype preference {self.dtype_preference!r}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=dtype,
            device_map="auto",
        )
        model.eval()
        self._tokenizer, self._model = tokenizer, model
        return tokenizer, model

    @staticmethod
    def _chat_text(tokenizer, prompt: str) -> str:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        import torch

        tokenizer, model = self._load()
        text = self._chat_text(tokenizer, request.prompt)
        encoded = tokenizer(text, return_tensors="pt")
        first_device = next(model.parameters()).device
        encoded = {key: value.to(first_device) for key, value in encoded.items()}
        input_len = int(encoded["input_ids"].shape[1])

        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=first_device)
            generator.manual_seed(int(request.seed))

        do_sample = float(request.temperature) > 0.0
        kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": int(request.max_output_tokens),
            "do_sample": do_sample,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = float(request.temperature)
            kwargs["top_p"] = float(request.top_p)
        if generator is not None:
            kwargs["generator"] = generator

        with torch.inference_mode():
            generated = model.generate(**kwargs)

        sequence = generated.sequences[0]
        new_ids = sequence[input_len:]
        decoded = tokenizer.decode(new_ids, skip_special_tokens=True)

        selected_logprobs: list[float] = []
        margins: list[float] = []
        # `generated.scores` are the model generation scores after generation
        # processors/warpers. They are useful internal diagnostics, but are not
        # advertised as calibrated uncertainty.
        for token_id, scores in zip(new_ids.tolist(), generated.scores, strict=False):
            row = scores[0].float()
            log_probs = torch.log_softmax(row, dim=-1)
            selected_logprobs.append(float(log_probs[int(token_id)].item()))
            top2 = torch.topk(row, k=2).values
            margins.append(float((top2[0] - top2[1]).item()))

        mean_logprob = (
            sum(selected_logprobs) / len(selected_logprobs) if selected_logprobs else None
        )
        nll = None if mean_logprob is None else -mean_logprob
        perplexity = None if nll is None else float(math.exp(min(nll, 50.0)))
        mean_margin = sum(margins) / len(margins) if margins else None

        commit_hash = getattr(getattr(model, "config", None), "_commit_hash", None)
        gpu_name = None
        vram_total = None
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_total = int(props.total_memory)

        metadata = {
            "model_id": self.model_id,
            "model_revision_requested": self.revision,
            "model_commit_hash": commit_hash,
            "sampling_controls_requested": {
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
            "sampling_controls_applied": True,
            "sampling_policy": "explicit_temperature_and_top_p",
            "output_token_parameter": self.output_token_parameter,
            "reasoning_or_thinking_applied": request.reasoning_or_thinking_setting,
            "local_white_box": True,
            "white_box_diagnostic_semantics": "post_processor_generation_scores_uncalibrated",
            "generated_tokens": int(new_ids.numel()),
            "mean_generated_token_logprob": mean_logprob,
            "generated_token_nll": nll,
            "generated_token_perplexity": perplexity,
            "mean_top1_top2_logit_margin": mean_margin,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": gpu_name,
            "gpu_vram_bytes": vram_total,
            "dtype": str(next(model.parameters()).dtype),
        }
        return GenerationResponse(
            text=decoded,
            requested_model_id=request.model_id,
            resolved_model_version=str(commit_hash or self.revision or request.model_id),
            provider_metadata=metadata,
        )
