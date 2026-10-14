from __future__ import annotations

import json
from pathlib import Path

from weft.runtime import Study, TuningPolicy, load_configurations

from weft_runner import from_arguments, parser_for


def main():
    parser = parser_for("Select complete Config bindings through the shared Weft tuner")
    parser.add_argument("--configs", type=Path, required=True,
                        help="JSON list of complete Config or finite ConfigSpace objects")
    parser.add_argument("--output", type=Path, required=True, help="public TuningResult JSON outside the repository")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--deadline", type=float, default=600)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    from kernel_configuration import PROJECT

    if args.output.resolve().is_relative_to(PROJECT):
        parser.error("generated selections must remain outside the repository")
    workload = from_arguments(args, parser)
    configs = load_configurations(args.configs, max_candidates=args.max_candidates)
    # The driver already performs warmup and repeated numerical/timing calls.
    # The public study samples that same driver boundary; no nested Cartesian
    # loop, environment-variable overrides, or second winner format remains.
    policy = TuningPolicy(max_candidates=args.max_candidates, deadline_s=args.deadline,
                          warmup=0, samples=args.samples, repetitions=1)
    result = Study(configs, identity=workload.identity(), policy=policy, cache=args.cache,
                   timing_scope="numerical-driver-native-median-ns").run(workload.trial, refresh=args.refresh)
    result.save(args.output)
    print(json.dumps({"winner": result.winner.to_dict(), "cache_hit": result.cache_hit,
                      "budget_exhausted": result.budget_exhausted,
                      "candidates": [item.to_dict() for item in result.candidates],
                      "selection": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
