from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .environment import check_environment
from .runner import ConfigError, RunnerError, build_config, render_status_report, run_loop


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be 0 or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dialoop",
        description="Run the Dialoop novel dialogue labeling workflow.",
    )
    parser.add_argument("novel_path", type=Path, help="Path to the source novel text file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("labeled.txt"),
        help="Path for speaker labels. Relative paths resolve from the novel directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=1,
        help="Maximum dialogue count to request per batch.",
    )
    parser.add_argument(
        "--threshold",
        type=non_negative_int,
        default=10,
        help="Maximum line gap treated as continuous dialogue.",
    )
    parser.add_argument(
        "--max-iterations",
        type=positive_int,
        default=100,
        help="Maximum OpenCode continuation iterations before stopping.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved paths and environment checks without starting OpenCode.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = build_config(
            novel_path=args.novel_path,
            output_path=args.output,
            batch_size=args.batch_size,
            threshold=args.threshold,
            max_iterations=args.max_iterations,
        )
    except ConfigError as error:
        parser.exit(2, f"dialoop: error: {error}\n")

    environment = check_environment()

    if args.dry_run:
        print(render_status_report(config, environment, title="Dialoop dry run"))
        return 0

    print(render_status_report(config, environment, title="Dialoop run"))
    print()

    try:
        result = run_loop(config, environment)
    except RunnerError as error:
        parser.exit(1, f"dialoop: error: {error}\n")

    print()
    print(f"Dialoop completed after {result.iterations} iteration(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
