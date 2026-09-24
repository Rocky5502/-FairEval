from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render LaTeX macros from the deterministic FairSynth oracle sanity artifact."
    )
    parser.add_argument(
        "--input",
        default="results/smoke/fairsynth_oracle_v1.json",
    )
    parser.add_argument(
        "--output",
        default="paper/generated/fairsynth_oracle_sanity_macros.tex",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "faireval-fairsynth-oracle-sanity-v1":
        raise ValueError("unexpected FairSynth oracle schema")
    guards = payload.get("guards", {})
    if guards.get("llm_result") is not False:
        raise ValueError("oracle sanity artifact must not be an LLM result")
    if payload.get("identity_oracle_counterfactual_rank_change") != 0.0:
        raise ValueError("identity oracle should be structurally invariant")

    personality = payload["personality_proxy"]
    true = float(personality["mean_true_ndcg_at_k"])
    shuffled = float(personality["mean_shuffled_ndcg_at_k"])
    delta = float(personality["mean_true_minus_shuffled_ndcg"])
    fraction = float(personality["fraction_users_true_gt_shuffled"])

    lines = [
        "% AUTO-GENERATED from deterministic FairSynth oracle sanity artifact.",
        "% Non-LLM structural validation only; not a human psychometric result.",
        f"\\newcommand{{\\FairSynthOracleTrueNDCG}}{{{true:.3f}}}",
        f"\\newcommand{{\\FairSynthOracleShuffledNDCG}}{{{shuffled:.3f}}}",
        f"\\newcommand{{\\FairSynthOracleDeltaNDCG}}{{{delta:+.3f}}}",
        f"\\newcommand{{\\FairSynthOracleTrueBetterPct}}{{{100.0 * fraction:.1f}\\%}}",
        "\\newcommand{\\FairSynthOracleIdentityChange}{0}",
        "",
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "input": args.input,
        "output": str(out),
        "llm_result": False,
        "true_ndcg": true,
        "shuffled_ndcg": shuffled,
        "delta_ndcg": delta,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
