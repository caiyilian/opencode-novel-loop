from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .quality import (
    DEFAULT_SCAN_PATHS,
    QualityError,
    evaluate_labels,
    load_terms,
    render_annotation_summary,
    render_error_labels,
    render_evaluation_report,
    render_term_scan_report,
    scan_terms,
    summarize_annotations,
)


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
        prog="dialoop-quality",
        description="Quality evaluation helpers for Dialoop speaker labeling.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Compare labels against an annotated answer file.")
    evaluate.add_argument("--answers", type=Path, required=True, help="Annotated answer text file.")
    evaluate.add_argument("--labels", type=Path, required=True, help="One-speaker-per-line label file.")
    evaluate.add_argument("--novel", type=Path, help="Optional source novel file for dialogue count reporting.")
    evaluate.add_argument(
        "--max-errors",
        type=positive_int,
        default=50,
        help="Maximum mismatches to print in the console report.",
    )
    evaluate.add_argument(
        "--all-errors",
        action="store_true",
        help="Print all mismatches in the console report.",
    )
    evaluate.add_argument(
        "--error-output",
        type=Path,
        help="Optional path to write error_labels.txt style mismatch lines.",
    )

    scan = subparsers.add_parser(
        "scan-terms",
        help="Scan production paths for project-specific names, phrases, or character voice terms.",
    )
    scan.add_argument(
        "--path",
        type=Path,
        action="append",
        dest="paths",
        help="Path to scan. Defaults to dialoop/ and pyproject.toml.",
    )
    scan.add_argument(
        "--term",
        action="append",
        dest="terms",
        default=[],
        help="Term to search for. Can be repeated.",
    )
    scan.add_argument("--terms-file", type=Path, help="UTF-8 file with one scan term per line.")

    annotations = subparsers.add_parser(
        "annotations-summary",
        help="Summarize risk and verifier metadata from annotations.jsonl.",
    )
    annotations.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to .dialoop/annotations.jsonl.",
    )
    annotations.add_argument(
        "--show-problems",
        type=non_negative_int,
        default=0,
        help="Print the first N verifier or structural problems.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            return run_evaluate(args)
        if args.command == "scan-terms":
            return run_scan_terms(args)
        if args.command == "annotations-summary":
            return run_annotations_summary(args)
    except QualityError as error:
        parser.exit(2, f"dialoop-quality: error: {error}\n")
    raise AssertionError(f"unhandled command: {args.command}")


def run_evaluate(args: argparse.Namespace) -> int:
    report = evaluate_labels(answer_path=args.answers, labels_path=args.labels, novel_path=args.novel)
    max_errors = None if args.all_errors else args.max_errors
    print(render_evaluation_report(report, max_errors=max_errors))
    if args.error_output is not None:
        args.error_output.parent.mkdir(parents=True, exist_ok=True)
        args.error_output.write_text(render_error_labels(report), encoding="utf-8")
    return 0


def run_scan_terms(args: argparse.Namespace) -> int:
    paths = args.paths or list(DEFAULT_SCAN_PATHS)
    terms = load_terms(terms=args.terms, terms_file=args.terms_file)
    matches = scan_terms(paths, terms)
    print(render_term_scan_report(matches, terms, paths))
    return 1 if matches else 0


def run_annotations_summary(args: argparse.Namespace) -> int:
    summary = summarize_annotations(args.annotations)
    print(render_annotation_summary(summary, show_problems=args.show_problems))
    return 1 if summary.has_structural_errors else 0


def evaluate_main(argv: Optional[Sequence[str]] = None) -> int:
    return main(["evaluate", *(argv if argv is not None else sys.argv[1:])])


def scan_terms_main(argv: Optional[Sequence[str]] = None) -> int:
    return main(["scan-terms", *(argv if argv is not None else sys.argv[1:])])


def annotations_summary_main(argv: Optional[Sequence[str]] = None) -> int:
    return main(["annotations-summary", *(argv if argv is not None else sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
