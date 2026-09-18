from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MAIN = PAPER / "main.tex"


def _bib_keys(text: str) -> set[str]:
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", text))


def _cite_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", text):
        keys.update(key.strip() for key in group.split(",") if key.strip())
    return keys


def _input_targets(text: str) -> list[Path]:
    targets = []
    for name in re.findall(r"\\input\{([^}]+)\}", text):
        candidate = PAPER / name
        if candidate.suffix == "":
            candidate = candidate.with_suffix(".tex")
        targets.append(candidate)
    return targets


def _bibliography_targets(text: str) -> list[Path]:
    targets: list[Path] = []
    for group in re.findall(r"\\bibliography\{([^}]+)\}", text):
        for name in group.split(","):
            name = name.strip()
            if not name:
                continue
            candidate = PAPER / name
            if candidate.suffix == "":
                candidate = candidate.with_suffix(".bib")
            targets.append(candidate)
    return targets


def main() -> int:
    main_text = MAIN.read_text(encoding="utf-8")

    bib_targets = _bibliography_targets(main_text)
    if not bib_targets:
        raise SystemExit("main.tex declares no bibliography")
    missing_bibs = [str(path.relative_to(ROOT)) for path in bib_targets if not path.is_file()]
    if missing_bibs:
        raise SystemExit(f"missing bibliography files: {missing_bibs}")
    bib_keys: set[str] = set()
    for path in bib_targets:
        bib_keys.update(_bib_keys(path.read_text(encoding="utf-8")))

    input_targets = _input_targets(main_text)
    missing_inputs = [str(path.relative_to(ROOT)) for path in input_targets if not path.is_file()]
    if missing_inputs:
        raise SystemExit(f"missing LaTeX input files: {missing_inputs}")

    input_text = "\n".join(path.read_text(encoding="utf-8") for path in input_targets)
    cited = _cite_keys(main_text) | _cite_keys(input_text)
    missing_cites = sorted(cited - bib_keys)
    if missing_cites:
        raise SystemExit(f"missing bibliography keys: {missing_cites}")

    required_assets = [
        PAPER / "figures" / "faireval_framework.pdf",
        PAPER / "figures" / "faireval_conditions.pdf",
        PAPER / "figures" / "faireval_evaluation_pipeline.pdf",
        PAPER / "related_work_table.tex",
        PAPER / "benchmark_model_table.tex",
        PAPER / "rq_design_table.tex",
        PAPER / "results_contract_table.tex",
    ]
    missing_assets = [str(path.relative_to(ROOT)) for path in required_assets if not path.is_file()]
    if missing_assets:
        raise SystemExit(f"missing manuscript assets: {missing_assets}")

    stale_phrases = [
        "DeepSeek V4 Flash, Qwen 3.8 Max",
        "DeepSeek V4 Flash,",
        "qwen3.8-max-0902",
        "deepseek-flash",
        "Pending exact host/revision",
        "spans six recommendation datasets and six LLM families",
        "across six datasets, six LLM families",
        "one frozen model configuration from six independent families",
        "optional local open-weight replication stratum",
        "optional local transparency replication",
        "current hosted-only phase does not require local execution",
        "250 RMB hard ceiling",
        "250 RMB emergency hard ceiling",
        "250 RMB non-bypassable emergency ceiling",
    ]
    stale = [phrase for phrase in stale_phrases if phrase in main_text]
    if stale:
        raise SystemExit(f"stale pre-gateway/white-box/budget manuscript wording found: {stale}")

    if "\\author{Anonymous Authors}" not in main_text:
        raise SystemExit("double-blind author placeholder missing")
    integrity_markers = (
        "audited frozen experiment artifacts",
        "No empirical value is typed into the manuscript by hand",
    )
    if not all(marker in main_text for marker in integrity_markers):
        raise SystemExit("artifact-only result-integrity policy missing from manuscript")
    if "persistent invalid output" not in main_text.lower():
        raise SystemExit("primary invalid-output policy is missing from manuscript text")
    budget_markers = (
        "200 RMB normal stop target",
        "250 RMB client-side emergency stop threshold",
        "does not represent the 250 RMB threshold as an atomic provider-side spending cap",
    )
    if not all(marker in main_text for marker in budget_markers):
        raise SystemExit("hosted API client-side budget-safety wording missing from manuscript")
    if "Zhizengzeng" not in main_text:
        raise SystemExit("hosted gateway provenance missing from manuscript")
    if "12,960" not in main_text or "full-scale local white-box" not in main_text:
        raise SystemExit("full-scale local white-box campaign is missing from manuscript")

    related = (PAPER / "related_work_table.tex").read_text(encoding="utf-8")
    if related.count("\\\\") < 16:
        raise SystemExit("related-work comparison table appears incomplete")
    if "do not imply empirical superiority" not in related:
        raise SystemExit("comparison-table interpretation guard missing")

    benchmark = (PAPER / "benchmark_model_table.tex").read_text(encoding="utf-8")
    required_benchmark_tokens = (
        "Personality 2018",
        "MIND",
        "FairSynth-360",
        "gpt-5.6-terra",
        "claude-sonnet-5",
        "gemini-3.8-flash",
        "deepseek-v4.1-flash",
        "qwen3.8-max",
        "llama-4-maverick",
        "Qwen/Qwen2.5-7B-Instruct",
        "microsoft/Phi-3.5-mini-instruct",
        "generation-score diagnostics",
        "application unverified",
        "12,960",
        "1,080",
        "200 RMB",
        "250 RMB client-side emergency stop threshold",
        "not an atomic provider-side spend cap",
    )
    for token in required_benchmark_tokens:
        if token not in benchmark:
            raise SystemExit(f"benchmark/model table missing required token: {token}")

    for token in ("figures/faireval_evaluation_pipeline.pdf", r"\label{fig:audit-pipeline}"):
        if token not in main_text:
            raise SystemExit(
                f"auditable artifact-to-claim pipeline missing from manuscript: {token}"
            )

    if "FairSynth-360" not in main_text:
        raise SystemExit("manuscript must explain the FairSynth-360 auxiliary scope")
    if "Qwen2.5" not in main_text or "Phi-3.5" not in main_text:
        raise SystemExit("manuscript must explain the full-scale local open-weight models")

    results_entry = (PAPER / "results_contract_table.tex").read_text(encoding="utf-8")
    for token in (
        "generated/rq12_inference_table.tex",
        "generated/rq34_results_table.tex",
        "generated/fairsynth_hosted_table.tex",
        "generated/whitebox_summary_table.tex",
        "Do not type empirical values",
    ):
        if token not in results_entry:
            raise SystemExit(f"results entrypoint missing artifact-only token: {token}")

    rq_design = (PAPER / "rq_design_table.tex").read_text(encoding="utf-8")
    rq4_required = (
        "20\\% validation split",
        "\\geq95\\%",
        "one global",
        "test outcomes never select",
        "no eligible PAIR point",
    )
    for token in rq4_required:
        if token not in rq_design:
            raise SystemExit(f"RQ4 design table missing frozen selection guard: {token}")

    print(
        f"paper preflight: {len(cited)} citation keys resolved across "
        f"{len(bib_targets)} bibliography files"
    )
    print("paper preflight: LaTeX inputs and required vector/table assets present")
    print("paper preflight: six hosted gateway IDs + lean 1,080-call hosted FairSynth + full-scale local white-box stratum synchronized")
    print("paper preflight: 200 RMB normal stop / 250 RMB client-side emergency threshold documented")
    print("paper preflight: no atomic provider-side spend-cap claim")
    for token in (
        "Hosted gateway operational pilot (not inferential evidence)",
        "99 of 1,080 planned cells",
        "64.9362 RMB",
        "generated/hosted_pilot_operational_table.tex",
        "figures/hosted_pilot_cost_profile.pdf",
    ):
        if token not in main_text:
            raise SystemExit(f"hosted pilot operational reporting missing: {token}")

    print("paper preflight: generated hosted/local result tables are artifact-only")
    print("paper preflight: hosted pilot operational reporting is outcome-blind and explicitly non-inferential")
    print("paper preflight: RQ4 validation/no-test-selection rule synchronized")
    print("paper preflight: double-blind and invalid-output guards present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
