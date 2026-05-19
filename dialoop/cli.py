from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .environment import check_environment
from .model_client import DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL, ModelConfig, OpenAICompatibleClient
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


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
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
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible model endpoint base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key for the OpenAI-compatible endpoint. Ollama accepts a placeholder value.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--protocol",
        choices=["auto", "tools", "json"],
        default="auto",
        help="Model tool protocol to use in the future agent loop.",
    )
    parser.add_argument(
        "--model-timeout",
        type=positive_float,
        default=30.0,
        help="Timeout in seconds for model endpoint requests.",
    )
    parser.add_argument(
        "--check-model",
        action="store_true",
        help="During --dry-run, send one small request to the configured model endpoint.",
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
    model_config = ModelConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.model_timeout,
    )

    if args.dry_run:
        print(render_status_report(config, environment, title="Dialoop dry run"))
        print()
        print(render_model_report(model_config, protocol=args.protocol, check_model=args.check_model))
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


def render_model_report(model_config: ModelConfig, protocol: str, check_model: bool) -> str:
    lines = [
        "Model backend:",
        f"  base_url: {model_config.base_url}",
        f"  model: {model_config.model}",
        f"  protocol: {protocol}",
        f"  timeout: {model_config.timeout:g}s",
    ]
    if not check_model:
        lines.append("  connection: skipped (use --check-model to test)")
        return "\n".join(lines)

    status = OpenAICompatibleClient(model_config).check_connection()
    state = "ok" if status.ok else "failed"
    lines.append(f"  connection: {state}")
    lines.append(f"  message: {status.message}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
