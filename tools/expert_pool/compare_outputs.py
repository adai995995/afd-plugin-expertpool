#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Locate generation divergence in two private, identical offered traces.

Token equality and numerical accuracy are separate checks. A small top-two
margin is evidence about sensitivity, not proof of an error's cause. Neither
this report nor an isolated prefix probe recovers missing pressure-time logits.
"""

import argparse
import json
import math
from pathlib import Path


def distribution(step: list[dict]) -> dict:
    if not step or any(not math.isfinite(item["logprob"]) for item in step):
        raise ValueError("Diagnostic logprobs must be nonempty and finite")
    ranked = sorted(step, key=lambda item: -item["logprob"])
    return {
        "candidates": ranked,
        "top_two_margin": ranked[0]["logprob"] - ranked[1]["logprob"]
        if len(ranked) > 1
        else None,
    }


def first_difference(reference: list[int], candidate: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(reference, candidate, strict=False)):
        if left != right:
            return index
    return (
        None
        if len(reference) == len(candidate)
        else min(len(reference), len(candidate))
    )


def compare_reports(reference: dict, candidate: dict) -> dict:
    if reference["trace_sha256"] != candidate["trace_sha256"]:
        raise ValueError("Output comparison requires the same offered requests")
    indexed = []
    for report in (reference, candidate):
        records = {item["request_id"]: item for item in report["records"]}
        if len(records) != len(report["records"]):
            raise ValueError("Duplicate request identity")
        indexed.append(records)
    left, right = indexed
    if left.keys() != right.keys():
        raise ValueError("Request coverage differs")
    matched = 0
    differences = []
    incomplete = []
    for identity in sorted(left):
        a, b = left[identity], right[identity]
        if any(a[key] != b[key] for key in ("domain", "length_class")):
            raise ValueError("Request metadata differs")
        if any(item["status"] != "completed" for item in (a, b)):
            incomplete.append(identity)
            continue
        if any("token_ids" not in item for item in (a, b)):
            raise ValueError(
                "Enable --capture-output-tokens; a hash cannot locate divergence"
            )
        offset = first_difference(a["token_ids"], b["token_ids"])
        if offset is None:
            matched += 1
            continue
        detail = {
            "request_id": identity,
            "domain": a["domain"],
            "length_class": a["length_class"],
            "first_difference_step": offset + 1,
            "common_generated_prefix": a["token_ids"][:offset],
        }
        for name, record in (("reference", a), ("candidate", b)):
            tokens = record["token_ids"]
            detail[name + "_token"] = tokens[offset] if offset < len(tokens) else None
            steps = record.get("top_logprobs", [])
            if steps:
                if len(steps) != len(tokens):
                    raise ValueError(
                        "Diagnostic probabilities do not cover the sequence"
                    )
                if offset < len(steps):
                    detail[name + "_distribution"] = distribution(steps[offset])
        differences.append(detail)
    return {
        "trace_sha256": reference["trace_sha256"],
        "requests": len(left),
        "identical_sequences": matched,
        "differences": differences,
        "incomplete_requests": incomplete,
        "all_sequences_identical": not differences and not incomplete,
        "scope": "free-generation; common prefix ends at first difference",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_reports(
        json.loads(args.reference.read_text()), json.loads(args.candidate.read_text())
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "requests": result["requests"],
                "identical_sequences": result["identical_sequences"],
                "differences": len(result["differences"]),
                "incomplete_requests": len(result["incomplete_requests"]),
            }
        )
    )


if __name__ == "__main__":
    main()
