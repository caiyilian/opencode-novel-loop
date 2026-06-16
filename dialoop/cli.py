from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .agent_loop import AgentBatchResult, AgentLoopConfig, AgentLoopError, AgentRunner
from .annotations import AnnotationStore
from .environment import CommandStatus, check_python
from .local_tools import DialoopLocalTools
from .model_client import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ModelClientError,
    ModelConfig,
    OpenAICompatibleClient,
)
from .runner import ConfigError, DialoopConfig, build_config


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


def non_negative_float(value: str) -> float:
    parsed = float(value)
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
        "--reset-output",
        action="store_true",
        help="Truncate the speaker label output before starting a labeling run.",
    )
    parser.add_argument(
        "--annotations-output",
        type=Path,
        default=Path(".dialoop") / "annotations.jsonl",
        help="Path for JSONL annotation evidence. Relative paths resolve from the novel directory.",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Disable annotations.jsonl writing for this run.",
    )
    parser.add_argument(
        "--reset-annotations",
        action="store_true",
        help="Truncate the annotations output before starting a labeling run.",
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
        help="Maximum dialogue batches to process in this run.",
    )
    parser.add_argument(
        "--max-tool-steps",
        type=positive_int,
        default=20,
        help="Maximum model/tool steps for one dialogue batch.",
    )
    parser.add_argument(
        "--context-window-lines",
        type=positive_int,
        default=80,
        help="Novel lines before and after the active batch to request in the first read_novel call.",
    )
    parser.add_argument(
        "--read-window-limit",
        type=positive_int,
        default=300,
        help="Maximum novel lines returned by one read_novel tool call.",
    )
    parser.add_argument(
        "--search-limit",
        type=positive_int,
        default=20,
        help="Default maximum matches returned by one search_novel tool call.",
    )
    parser.add_argument(
        "--previous-context-dialogues",
        type=non_negative_int,
        default=8,
        help="Previously labeled neighboring dialogues to include in each batch prompt.",
    )
    parser.add_argument(
        "--following-context-dialogues",
        type=non_negative_int,
        default=8,
        help="Following unlabeled neighboring dialogues to include in each batch prompt.",
    )
    parser.add_argument(
        "--identity-lookahead-lines",
        type=positive_int,
        default=120,
        help="Maximum later novel lines scanned by one identity lookahead helper call.",
    )
    parser.add_argument(
        "--identity-lookahead-rounds",
        type=non_negative_int,
        default=2,
        help="Soft maximum identity lookahead rounds for one temporary speaker.",
    )
    parser.add_argument(
        "--identity-mode",
        choices=["off", "auto"],
        default="auto",
        help="Identity Locator/Resolver Coordinator mode: auto runs bounded identity review for temporary speakers.",
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
        default=60.0,
        help="Timeout in seconds for model endpoint requests.",
    )
    parser.add_argument(
        "--model-retries",
        type=non_negative_int,
        default=2,
        help="Retry count for timed out model endpoint requests.",
    )
    parser.add_argument(
        "--model-retry-delay",
        type=non_negative_float,
        default=5.0,
        help="Delay in seconds before retrying a timed out model endpoint request.",
    )
    parser.add_argument(
        "--verifier-mode",
        choices=["off", "risk", "all"],
        default="risk",
        help=(
            "Verifier Agent mode: off disables it, risk verifies only high-risk annotations, "
            "all verifies every submitted annotation."
        ),
    )
    parser.add_argument(
        "--verifier-max-tokens",
        type=positive_int,
        default=1200,
        help="Maximum tokens for one Verifier Agent response.",
    )
    parser.add_argument(
        "--verifier-retries",
        type=non_negative_int,
        default=1,
        help="Retry count for invalid or failed Verifier Agent responses.",
    )
    parser.add_argument(
        "--check-model",
        action="store_true",
        help="During --dry-run, send one small request to the configured model endpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved paths and environment/model checks without starting a labeling run.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the initial system/user prompt for the current batch to stdout.",
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

    model_config = ModelConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.model_timeout,
        retries=args.model_retries,
        retry_delay=args.model_retry_delay,
    )
    annotations_path = _resolve_path_from_workdir(args.annotations_output, config.workdir)

    if args.dry_run:
        print(
            render_dry_run_report(
                config,
                check_python(),
                title="Dialoop dry run",
                annotations_path=None if args.no_annotations else annotations_path,
                max_tool_steps=args.max_tool_steps,
                context_window_lines=args.context_window_lines,
                read_window_limit=args.read_window_limit,
                search_limit=args.search_limit,
                previous_context_dialogues=args.previous_context_dialogues,
                following_context_dialogues=args.following_context_dialogues,
                identity_lookahead_lines=args.identity_lookahead_lines,
                identity_lookahead_rounds=args.identity_lookahead_rounds,
                identity_mode=args.identity_mode,
                verifier_mode=args.verifier_mode,
                verifier_max_tokens=args.verifier_max_tokens,
                verifier_retries=args.verifier_retries,
            )
        )
        print()
        print(render_model_report(model_config, protocol=args.protocol, check_model=args.check_model))
        return 0

    if args.reset_output:
        reset_output_for_run(config.output_path)

    tools = DialoopLocalTools.from_paths(
        novel_path=config.novel_path,
        labels_path=config.output_path,
        batch_size=config.batch_size,
        max_line_gap=config.threshold,
        read_window_limit=args.read_window_limit,
        search_limit=args.search_limit,
        previous_context_dialogues=args.previous_context_dialogues,
        following_context_dialogues=args.following_context_dialogues,
        identity_lookahead_lines=args.identity_lookahead_lines,
        identity_lookahead_rounds=args.identity_lookahead_rounds,
    )
    if not args.no_annotations:
        try:
            prepare_annotations_for_run(
                annotations_path=annotations_path,
                labeled_count=tools.get_progress()["labeled"],
                reset_annotations=args.reset_annotations,
            )
        except ConfigError as error:
            parser.exit(2, f"dialoop: error: {error}\n")

    agent = AgentRunner(
        model_client=OpenAICompatibleClient(model_config),
        tools=tools,
        config=AgentLoopConfig(
            protocol=args.protocol,
            max_tool_steps=args.max_tool_steps,
            context_window_lines=args.context_window_lines,
            identity_mode=args.identity_mode,
            verifier_mode=args.verifier_mode,
            verifier_max_tokens=args.verifier_max_tokens,
            verifier_retries=args.verifier_retries,
        ),
        prompt_output=sys.stdout if args.show_prompt else None,
        annotation_store=None if args.no_annotations else AnnotationStore(annotations_path),
    )

    try:
        for iteration in range(1, args.max_iterations + 1):
            result = agent.run_one_batch()
            print(render_agent_result(result, iteration=iteration, max_iterations=args.max_iterations))
            if result.done:
                print()
                print("Dialoop run complete.")
                return 0
            print()
    except KeyboardInterrupt:
        print()
        print(render_interrupted_result(tools.get_progress()))
        return 130
    except (AgentLoopError, ModelClientError) as error:
        parser.exit(1, f"dialoop: error: {error}\n")

    print(render_iteration_limit_result(args.max_iterations, tools.get_progress()))
    return 0


def _format_command_status(status: CommandStatus) -> list[str]:
    state = "found" if status.available else "missing"
    lines = [f"{status.name}: {state}"]
    if status.executable:
        lines.append(f"  executable: {status.executable}")
    if status.version:
        lines.append(f"  version: {status.version}")
    if status.install_hint:
        lines.append(f"  install: {status.install_hint}")
    return lines


def _resolve_path_from_workdir(path: Path, workdir: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (workdir / expanded).resolve()


def prepare_annotations_for_run(
    annotations_path: Path,
    labeled_count: int,
    reset_annotations: bool,
) -> None:
    if reset_annotations:
        annotations_path.parent.mkdir(parents=True, exist_ok=True)
        annotations_path.write_text("", encoding="utf-8")
        return

    annotation_count = count_annotation_records(annotations_path)
    if annotation_count <= labeled_count:
        return

    raise ConfigError(
        f"annotations output already has {annotation_count} record(s), but the label output has "
        f"{labeled_count} label(s). This would append duplicate or stale annotation rows. "
        "Clear the annotations file, choose a different --annotations-output path, or pass "
        "--reset-annotations when intentionally starting a fresh annotation run."
    )


def reset_output_for_run(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")


def count_annotation_records(annotations_path: Path) -> int:
    if not annotations_path.exists():
        return 0
    with annotations_path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def render_dry_run_report(
    config: DialoopConfig,
    python_status: CommandStatus,
    title: str,
    annotations_path: Optional[Path] = None,
    max_tool_steps: Optional[int] = None,
    context_window_lines: Optional[int] = None,
    read_window_limit: Optional[int] = None,
    search_limit: Optional[int] = None,
    previous_context_dialogues: Optional[int] = None,
    following_context_dialogues: Optional[int] = None,
    identity_lookahead_lines: Optional[int] = None,
    identity_lookahead_rounds: Optional[int] = None,
    identity_mode: Optional[str] = None,
    verifier_mode: Optional[str] = None,
    verifier_max_tokens: Optional[int] = None,
    verifier_retries: Optional[int] = None,
) -> str:
    lines = [
        title,
        "",
        "Paths:",
        f"  novel: {config.novel_path}",
        f"  output: {config.output_path}",
        f"  annotations: {annotations_path if annotations_path is not None else 'disabled'}",
        f"  workdir: {config.workdir}",
        "",
        "Local tools:",
        f"  get_dialogue.py: {config.get_dialogue_path}",
        f"  write_label.py: {config.write_label_path}",
        "",
        "Dialogue options:",
        f"  batch_size: {config.batch_size}",
        f"  threshold: {config.threshold}",
        f"  max_iterations: {config.max_iterations}",
    ]
    if max_tool_steps is not None:
        lines.append(f"  max_tool_steps: {max_tool_steps}")
    if context_window_lines is not None:
        lines.append(f"  context_window_lines: {context_window_lines}")
    if read_window_limit is not None:
        lines.append(f"  read_window_limit: {read_window_limit}")
    if search_limit is not None:
        lines.append(f"  search_limit: {search_limit}")
    if previous_context_dialogues is not None:
        lines.append(f"  previous_context_dialogues: {previous_context_dialogues}")
    if following_context_dialogues is not None:
        lines.append(f"  following_context_dialogues: {following_context_dialogues}")
    if identity_lookahead_lines is not None:
        lines.append(f"  identity_lookahead_lines: {identity_lookahead_lines}")
    if identity_lookahead_rounds is not None:
        lines.append(f"  identity_lookahead_rounds: {identity_lookahead_rounds}")
    if identity_mode is not None:
        lines.append(f"  identity_mode: {identity_mode}")
    if verifier_mode is not None:
        lines.append(f"  verifier_mode: {verifier_mode}")
    if verifier_max_tokens is not None:
        lines.append(f"  verifier_max_tokens: {verifier_max_tokens}")
    if verifier_retries is not None:
        lines.append(f"  verifier_retries: {verifier_retries}")
    lines.extend(["", "Environment:"])
    lines.extend(f"  {line}" for line in _format_command_status(python_status))
    return "\n".join(lines)


def render_agent_result(
    result: AgentBatchResult,
    iteration: Optional[int] = None,
    max_iterations: Optional[int] = None,
) -> str:
    lines = [
        "Dialoop batch result:",
        f"  submitted: {str(result.submitted).lower()}",
        f"  done: {str(result.done).lower()}",
        f"  tool_steps: {result.tool_steps}",
        f"  message: {result.message}",
        f"  annotations_written: {result.annotations_written}",
    ]
    if iteration is not None and max_iterations is not None:
        lines.append(f"  iteration: {iteration}/{max_iterations}")
    if result.batch_dialogues:
        lines.append("  batch:")
        for dialogue in result.batch_dialogues:
            lines.append(f"    - index: {dialogue['index']}, line: {dialogue['line_number']}")
    lines.extend(
        [
            "  progress:",
            f"    labeled: {result.progress['labeled']}",
            f"    total: {result.progress['total']}",
            f"    remaining: {result.progress['remaining']}",
        ]
    )
    return "\n".join(lines)


def render_iteration_limit_result(max_iterations: int, progress: dict) -> str:
    lines = [
        f"Dialoop stopped after reaching --max-iterations={max_iterations}.",
        "  progress:",
        f"    labeled: {progress['labeled']}",
        f"    total: {progress['total']}",
        f"    remaining: {progress['remaining']}",
    ]
    return "\n".join(lines)


def render_interrupted_result(progress: dict) -> str:
    lines = [
        "Dialoop interrupted. Progress already written to the output file is preserved.",
        "  progress:",
        f"    labeled: {progress['labeled']}",
        f"    total: {progress['total']}",
        f"    remaining: {progress['remaining']}",
    ]
    return "\n".join(lines)


def render_model_report(model_config: ModelConfig, protocol: str, check_model: bool) -> str:
    lines = [
        "Model backend:",
        f"  base_url: {model_config.base_url}",
        f"  model: {model_config.model}",
        f"  protocol: {protocol}",
        f"  timeout: {model_config.timeout:g}s",
        f"  retries: {model_config.retries}",
        f"  retry_delay: {model_config.retry_delay:g}s",
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
