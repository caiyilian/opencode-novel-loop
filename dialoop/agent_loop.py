from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, TextIO

from .annotations import AnnotationStore, build_annotation_records
from .coordinator import Coordinator
from .local_tools import DialoopLocalTools, SpeakerCountMismatchError, ToolValidationError
from .model_client import ChatMessage, ChatResult, ToolCall
from .protocol import JsonAction, ProtocolError, local_tool_specs, parse_json_action
from .risk import assess_annotation_risk
from .verifier import VerifierAgent


class AgentLoopError(RuntimeError):
    """Raised when a batch cannot be completed by the agent loop."""


SUBMIT_LABEL_ALIASES = (
    "speakers",
    "speaker",
    "speaker_names",
    "speaker_name",
    "labels",
    "label",
    "names",
    "name",
)
IDENTITY_LOOKUP_TOOL_NAMES = frozenset({"locate_identity", "resolve_identity"})
TEMPORARY_IDENTITY_SPEAKERS = frozenset(
    {
        "少女",
        "女孩",
        "姑娘",
        "少年",
        "老人",
        "老者",
    }
)


@dataclass(frozen=True)
class AgentLoopConfig:
    protocol: str = "auto"
    max_tool_steps: int = 20
    context_window_lines: int = 80
    temperature: float = 0.0
    require_context_before_submit: bool = True
    require_identity_tool_for_temporary_speaker: bool = True
    verifier_mode: str = "off"
    verifier_temperature: float = 0.0
    verifier_max_tokens: int = 1200
    verifier_retries: int = 1

    def __post_init__(self) -> None:
        if self.protocol not in {"auto", "tools", "json"}:
            raise AgentLoopError(f"unsupported protocol: {self.protocol}")
        if self.verifier_mode not in {"off", "risk", "all"}:
            raise AgentLoopError(f"unsupported verifier_mode: {self.verifier_mode}")
        if self.max_tool_steps <= 0:
            raise AgentLoopError("max_tool_steps must be greater than 0")
        if self.context_window_lines <= 0:
            raise AgentLoopError("context_window_lines must be greater than 0")
        if self.verifier_max_tokens <= 0:
            raise AgentLoopError("verifier_max_tokens must be greater than 0")
        if self.verifier_retries < 0:
            raise AgentLoopError("verifier_retries must be 0 or greater")


@dataclass(frozen=True)
class ToolExecution:
    name: str
    result: dict[str, Any]
    arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted_submit(self) -> bool:
        return self.name == "submit_labels" and self.result.get("accepted") is True


@dataclass(frozen=True)
class AgentBatchResult:
    submitted: bool
    done: bool
    tool_steps: int
    progress: dict[str, Any]
    message: str
    batch_dialogues: list[dict[str, Any]] = field(default_factory=list)
    tool_history: list[ToolExecution] = field(default_factory=list)
    annotations_written: int = 0


def system_prompt(protocol: str) -> str:
    json_instruction = ""
    if protocol in {"auto", "json"}:
        json_instruction = (
            "\n如果原生 tool calling 不可用，请严格只输出一个 JSON object，例如："
            '{"action":"read_novel","args":{"start_line":1,"end_line":20}}. '
            "准备好后用 submit_labels 提交。"
        )

    return (
        "你是小说对话说话人标注助手。"
        "你的任务是判断当前 batch 中每一句引号对话的说话人。"
        "提交前必须先用 read_novel 读取目标行附近的原文上下文；如果仍不确定，再用 search_novel 搜索相关人名、称呼或关键词。"
        "标签必须使用简体中文，不要输出英文，也不要把身份翻译成 Customer、Merchant、Clerk 这类英文词。"
        "优先使用原文中出现的人名、称呼或稳定身份。"
        "如果没有姓名但上下文有身份或群体，请用中文身份词，例如：村民、骑士、店员、商人、众人、未知。"
        "不要用临时行为关系替代更稳定身份；例如上下文说明是村落居民时，用“村民”而不是“顾客”。"
        "如果当前上下文只给出“女孩”“少年”“老人”等临时描述，但这是一个可追踪的具体人物，且后文在有限范围内揭示其姓名或稳定称呼，请使用后文揭示的姓名或稳定称呼。"
        "身份工具触发规则是强规则：遇到少女、女孩、姑娘、少年、老人、老者这类可追踪具体人物的临时身份词时，不要直接 submit_labels，必须先调用 locate_identity 查找后文候选区域；如果返回候选，再调用 resolve_identity 细读候选区域。"
        "如果 resolve_identity 得到稳定姓名或称呼，使用该结果作为候选 speaker，并用 record_character 记录显示名、别名和证据；如果有界查找没有证据，可以保留临时身份词，但必须在 reason 里说明查找结果。"
        "如果已有角色库条目，提交前凡是 speaker 可能是别名、简称或临时称呼，必须调用 normalize_speaker 获取显示名归一建议；该建议只辅助判断，不能自动覆盖原文证据或 submit_labels 的最终 speaker。"
        "当 Labeler、Verifier、Identity Resolver 或 Normalizer 结论冲突时，必须调用 arbitrate_identity 获取裁决建议；最终仍必须用 submit_labels 明确提交。"
        "不要把“我”“咱”“汝”“你”“您”等代词或口癖当成身份后置词；如果上下文已经能判断这是某个已知角色的自称，直接用该角色名，不要对代词调用 locate_identity。"
        "如果当前引号内容是在某个角色讲故事、转述戏曲、举例或复述别人说过的话，speaker 应该是外层正在讲述的角色，不要把故事内部的男孩、恶魔、商人等人物当成当前真实说话人。"
        "不要为了无名群体、路人或临时职能角色无限寻找姓名；只有具体人物明显会继续参与场景时，才进行有限的后文确认。"
        "如果引号内容明显不是人物说话，而是叙述中的环境声、物体声音、心理比喻声或声音效果，请标注为“非人物发声”；如果文本明确说明某个角色发出该声音，如喊叫、叹息、笑声或嚎叫，仍标注该角色。"
        "短句、追问、省略号、沉默或半句话要重点参考相邻对话和最近已标注结果；不要机械沿用上一句说话人。"
        "相邻对话标签只是连续性线索，如果原文中的“某某说/问/回答”等强证据与标签冲突，以原文为准。"
        "不要直接编辑文件。"
        "提交时必须调用 submit_labels，且 speaker 数量必须等于当前 batch 的对话数量，顺序必须一致。"
        "如果可以判断依据，请在 submit_labels 参数中同时提供 evidence_lines、reason、rejected_candidates 和 confidence。"
        "confidence 只能是 high、medium 或 low。"
        + json_instruction
    )


def batch_prompt(batch_result: dict[str, Any], context_window_lines: int) -> str:
    progress = batch_result["progress"]
    dialogues = batch_result["dialogues"]
    first_line = min(dialogue["line_number"] for dialogue in dialogues)
    last_line = max(dialogue["line_number"] for dialogue in dialogues)
    context_start = max(1, first_line - context_window_lines)
    context_end = last_line + context_window_lines
    lines = [
        "请标注当前对话 batch。",
        f"进度：已标注 {progress['labeled']}/{progress['total']}，剩余 {progress['remaining']}。",
        (
            f"本次 submit_labels 必须且只能提交 {len(dialogues)} 个 speaker；"
            "不要提交最近已标注或后续未标注对话的 speaker。"
        ),
        "",
        "当前对话：",
    ]
    for offset, dialogue in enumerate(dialogues, start=1):
        lines.append(f"{offset}. index={dialogue['index']} 行号={dialogue['line_number']} 文本={dialogue['text']}")
    previous_dialogues = batch_result.get("previous_dialogues", [])
    if previous_dialogues:
        lines.extend(
            [
                "",
                "最近已标注对话（来自输出文件，仅作为连续性线索；如果原文强证据冲突，以原文为准）：",
            ]
        )
        for dialogue in previous_dialogues:
            lines.append(
                f"- index={dialogue['index']} 行号={dialogue['line_number']} "
                f"speaker={dialogue['speaker']} 文本={dialogue['text']}"
            )
    following_dialogues = batch_result.get("following_dialogues", [])
    if following_dialogues:
        lines.extend(
            [
                "",
                "后续未标注对话（只用于判断当前 batch，不要为这些对话提交标签）：",
            ]
        )
        for dialogue in following_dialogues:
            lines.append(f"- index={dialogue['index']} 行号={dialogue['line_number']} 文本={dialogue['text']}")
    known_characters = batch_result.get("known_characters", [])
    if known_characters:
        lines.extend(["", "轻量角色库（仅作显示名/别名线索，不可替代原文证据）："])
        for character in known_characters:
            aliases = ", ".join(character.get("aliases", [])) or "none"
            lines.append(
                f"- {character.get('display_name')} aliases={aliases} "
                f"confidence={character.get('confidence')} last_seen={character.get('last_seen_dialogue_index')}"
            )
    lines.extend(
        [
            "",
            f"第一步请调用 read_novel(start_line={context_start}, end_line={context_end}) 读取上下文。",
            "根据上下文判断说话人；如果遇到可追踪具体人物的身份后置介绍，可以有限读取后文确认姓名或稳定称呼。",
            "身份工具检查：如果候选 speaker 是少女、女孩、姑娘、少年、老人、老者等临时身份词，且像是会继续参与场景的具体人物，必须先调用 locate_identity；有候选范围时继续调用 resolve_identity。",
            "角色库检查：如果已知角色库非空，且候选 speaker 可能是别名、简称或临时称呼，提交前必须调用 normalize_speaker；解析出新稳定姓名时调用 record_character 记录证据。",
            "冲突检查：如果 Labeler、Verifier、Identity Resolver 或 Normalizer 结论不一致，必须调用 arbitrate_identity 后再提交。",
            "排除项：不要对“我/咱/汝/你/您”等代词或口癖做身份后置查找；不要把故事、戏曲、传闻里的内部人物当成当前引号的真实 speaker。",
            "对很短的追问、沉默、省略号或半句话，务必结合前后相邻对话轮次和最近已标注 speaker 判断。",
            "如果需要更多线索，再调用 read_novel 或 search_novel；不要为了普通无名群体无限查找姓名。",
            (
                "最后调用 submit_labels，按上述对话顺序提交简体中文 speaker 标签；"
                "尽量一并提供 evidence_lines、reason、rejected_candidates 和 confidence。"
            ),
        ]
    )
    return "\n".join(lines)


class AgentRunner:
    def __init__(
        self,
        model_client: Any,
        tools: DialoopLocalTools,
        config: Optional[AgentLoopConfig] = None,
        prompt_output: Optional[TextIO] = None,
        annotation_store: Optional[AnnotationStore] = None,
        verifier_client: Optional[Any] = None,
    ):
        self.model_client = model_client
        self.tools = tools
        self.config = config or AgentLoopConfig()
        self.prompt_output = prompt_output
        self.annotation_store = annotation_store
        self.verifier_agent = (
            None
            if self.config.verifier_mode == "off"
            else VerifierAgent(
                verifier_client or model_client,
                temperature=self.config.verifier_temperature,
                max_tokens=self.config.verifier_max_tokens,
                retries=self.config.verifier_retries,
            )
        )
        self.coordinator = Coordinator(
            verifier_agent=self.verifier_agent,
            verifier_mode=self.config.verifier_mode,
            verifier_context_budget=self.config.verifier_max_tokens,
        )
        self._used_context_tool = False
        self._used_identity_lookup_tool = False
        self._blocked_review_attempts: list[dict[str, Any]] = []

    def run_one_batch(self) -> AgentBatchResult:
        self._used_context_tool = False
        self._used_identity_lookup_tool = False
        self._blocked_review_attempts = []
        initial_batch = self.tools.get_next_dialogue()
        if initial_batch["done"]:
            return AgentBatchResult(
                submitted=False,
                done=True,
                tool_steps=0,
                progress=initial_batch["progress"],
                message="all dialogues are already labeled",
                batch_dialogues=[],
            )

        messages = [
            ChatMessage(role="system", content=system_prompt(self.config.protocol)),
            ChatMessage(role="user", content=batch_prompt(initial_batch, self.config.context_window_lines)),
        ]
        if self.prompt_output is not None:
            print(format_prompt_messages(messages), file=self.prompt_output)
        history: list[ToolExecution] = []

        for step in range(1, self.config.max_tool_steps + 1):
            response = self._chat(messages)
            executions = self._handle_response(messages, response)
            history.extend(executions)

            review_blocked_submit = False
            for execution in executions:
                if execution.accepted_submit:
                    finalized = self._finalize_accepted_submit(
                        dialogues=initial_batch["dialogues"],
                        accepted_submit=execution,
                        history=history,
                        messages=messages,
                    )
                    if finalized is None:
                        review_blocked_submit = True
                        break
                    progress = finalized["progress"]
                    annotations_written = finalized["annotations_written"]
                    return AgentBatchResult(
                        submitted=True,
                        done=progress["remaining"] == 0,
                        tool_steps=step,
                        progress=progress,
                        message=accepted_submit_message(execution.result),
                        batch_dialogues=initial_batch["dialogues"],
                        tool_history=history,
                        annotations_written=annotations_written,
                    )
            if review_blocked_submit:
                self._maybe_warn_before_step_limit(messages, step)
                continue
            self._maybe_warn_before_step_limit(messages, step)

        progress = self.tools.get_progress()
        raise AgentLoopError(
            f"model did not submit labels within {self.config.max_tool_steps} tool step(s); "
            f"progress remains {progress['labeled']}/{progress['total']}; "
            f"active batch: {format_batch_summary(initial_batch['dialogues'])}; "
            f"recent tools: {format_recent_tools(history)}"
        )

    def _maybe_warn_before_step_limit(self, messages: list[ChatMessage], completed_step: int) -> None:
        remaining_steps = self.config.max_tool_steps - completed_step
        if remaining_steps != 1:
            return
        messages.append(
            ChatMessage(
                role="user",
                content=(
                    "Only one model response remains for this batch. "
                    "If you have enough evidence, call submit_labels now. "
                    "If evidence is imperfect, submit the best concise speaker label such as 未知 or 非人物发声 "
                    "instead of continuing to search."
                ),
            )
        )

    def _chat(self, messages: list[ChatMessage]) -> ChatResult:
        submit_label_count = self.tools.active_batch_size or None
        tools = (
            None
            if self.config.protocol == "json"
            else local_tool_specs(submit_label_count=submit_label_count)
        )
        return self.model_client.chat(
            messages=messages,
            tools=tools,
            temperature=self.config.temperature,
        )

    def _handle_response(self, messages: list[ChatMessage], response: ChatResult) -> list[ToolExecution]:
        if response.tool_calls:
            if self.config.protocol == "json":
                messages.append(
                    ChatMessage(
                        role="user",
                        content="Native tool calls were returned while protocol=json. Reply with a JSON action object instead.",
                    )
                )
                return []

            messages.append(
                ChatMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=[call.to_openai_tool_call() for call in response.tool_calls],
                )
            )
            executions = [self._execute_native_tool_call(messages, call) for call in response.tool_calls]
            self._append_retry_messages(messages, executions)
            return executions

        messages.append(ChatMessage(role="assistant", content=response.content))

        if self.config.protocol in {"auto", "json"}:
            action = self._parse_json_action_or_remind(messages, response.content)
            if action is not None:
                execution = self._execute_json_action(action)
                messages.append(
                    ChatMessage(
                        role="user",
                        content=f"Tool result for {execution.name}: {format_tool_result(execution.result)}",
                    )
                )
                self._append_retry_messages(messages, [execution])
                return [execution]

        messages.append(
            ChatMessage(
                role="user",
                content="You must use a tool or JSON action and finish by calling submit_labels for the active batch.",
            )
        )
        return []

    def _parse_json_action_or_remind(self, messages: list[ChatMessage], content: str) -> Optional[JsonAction]:
        try:
            return parse_json_action(content)
        except ProtocolError as error:
            messages.append(
                ChatMessage(
                    role="user",
                    content=f"Your response was not a valid JSON action: {error}. Reply with exactly one action object.",
                )
            )
            return None

    def _execute_native_tool_call(self, messages: list[ChatMessage], call: ToolCall) -> ToolExecution:
        execution = self._execute_tool(call.name, call.arguments)
        messages.append(
            ChatMessage(
                role="tool",
                content=format_tool_result(execution.result),
                tool_call_id=call.id,
            )
        )
        return execution

    def _execute_json_action(self, action: JsonAction) -> ToolExecution:
        return self._execute_tool(action.action, action.args)

    def _append_retry_messages(self, messages: list[ChatMessage], executions: list[ToolExecution]) -> None:
        for execution in executions:
            retry_message = submit_retry_message(execution)
            if retry_message is not None:
                messages.append(ChatMessage(role="user", content=retry_message))

    def _execute_tool(self, name: str, args: dict[str, Any]) -> ToolExecution:
        try:
            if name == "get_next_dialogue":
                result = self.tools.get_next_dialogue(batch_size=optional_int(args, "batch_size"))
            elif name == "read_novel":
                result = self.tools.read_novel(
                    start_line=required_int(args, "start_line"),
                    end_line=required_int(args, "end_line"),
                )
            elif name == "search_novel":
                result = self.tools.search_novel(
                    keyword=required_str(args, "keyword"),
                    limit=optional_int(args, "limit"),
                )
            elif name == "locate_identity":
                result = self.tools.locate_identity(
                    speaker=required_str(args, "speaker"),
                    dialogue_index=optional_int(args, "dialogue_index"),
                    search_after_line=optional_int(args, "search_after_line"),
                    lookahead_lines=optional_int(args, "lookahead_lines"),
                    max_candidates=optional_int(args, "max_candidates") or 3,
                )
            elif name == "resolve_identity":
                result = self.tools.resolve_identity(
                    speaker=required_str(args, "speaker"),
                    start_line=required_int(args, "start_line"),
                    end_line=required_int(args, "end_line"),
                    dialogue_index=optional_int(args, "dialogue_index"),
                )
            elif name == "record_character":
                result = self.tools.record_character(
                    display_name=required_str(args, "display_name"),
                    aliases=optional_str_list(args, "aliases"),
                    summary=optional_str(args, "summary") or "",
                    evidence_lines=optional_int_list(args, "evidence_lines"),
                    last_seen_dialogue_index=optional_int(args, "last_seen_dialogue_index"),
                    last_seen_line_number=optional_int(args, "last_seen_line_number"),
                    confidence=optional_str(args, "confidence") or "medium",
                )
            elif name == "normalize_speaker":
                result = self.tools.normalize_speaker(speaker=required_str(args, "speaker"))
            elif name == "arbitrate_identity":
                result = self.tools.arbitrate_identity(
                    labeler_speaker=required_str(args, "labeler_speaker"),
                    verifier_verdict=optional_str(args, "verifier_verdict"),
                    resolver_verdict=optional_str(args, "resolver_verdict"),
                    resolver_speaker=optional_str(args, "resolver_speaker"),
                    normalizer_speaker=optional_str(args, "normalizer_speaker"),
                )
            elif name == "submit_labels":
                result = self._execute_submit_labels(args)
            else:
                raise AgentLoopError(f"unknown tool call: {name}")
        except KeyError as error:
            result = {"accepted": False, "error": f"missing required argument: {error.args[0]}"}
        except SpeakerCountMismatchError as error:
            result = error.to_result()
        except ToolValidationError as error:
            result = {"accepted": False, "error": str(error)}

        if name in {"read_novel", "search_novel", "locate_identity", "resolve_identity"} and "error" not in result:
            self._used_context_tool = True
        if name in IDENTITY_LOOKUP_TOOL_NAMES and "error" not in result:
            self._used_identity_lookup_tool = True

        return ToolExecution(name=name, result=result, arguments=dict(args))

    def _finalize_accepted_submit(
        self,
        dialogues: list[dict[str, Any]],
        accepted_submit: ToolExecution,
        history: list[ToolExecution],
        messages: list[ChatMessage],
    ) -> Optional[dict[str, Any]]:
        speakers = accepted_submit.result.get("speakers", [])
        if not isinstance(speakers, list) or any(not isinstance(item, str) for item in speakers):
            return self._commit_without_annotations(accepted_submit)

        records = []
        if self.annotation_store is not None or self.verifier_agent is not None:
            records = build_annotation_records(
                dialogues=dialogues,
                speakers=speakers,
                submit_args=accepted_submit.arguments,
                tool_summary=summarize_tool_history(history),
                recovery=submit_recovery_info(
                    accepted_submit.result,
                    blocked_reviews=self._blocked_review_attempts,
                ),
            )
            records = self._review_annotation_records(records)
            blocking_review = first_blocking_verifier_review(records)
            if blocking_review is not None:
                self._mark_submit_blocked_by_verifier(accepted_submit, blocking_review)
                messages.append(ChatMessage(role="user", content=verifier_retry_message(blocking_review)))
                return None

        commit_result = self.tools.commit_labels(speakers)
        accepted_submit.result.update(commit_result)
        accepted_submit.result["pending_review"] = False
        annotations_written = 0
        if self.annotation_store is not None and records:
            annotations_written = self.annotation_store.append(records)
        return {
            "progress": commit_result["progress"],
            "annotations_written": annotations_written,
        }

    def _commit_without_annotations(self, accepted_submit: ToolExecution) -> dict[str, Any]:
        speakers = accepted_submit.result.get("speakers", [])
        commit_result = self.tools.commit_labels(speakers)
        accepted_submit.result.update(commit_result)
        accepted_submit.result["pending_review"] = False
        return {
            "progress": commit_result["progress"],
            "annotations_written": 0,
        }

    def _review_annotation_records(self, records: list[Any]) -> list[Any]:
        reviewed = []
        for record in records:
            risk = assess_annotation_risk(record)
            decision = self.coordinator.review(record, risk)
            arbiter = decision.arbiter
            trace = decision.trace_dicts()
            if repeated_fragile_review_block(record, arbiter, self._blocked_review_attempts):
                arbiter = unblock_repeated_fragile_arbiter(arbiter)
                trace.append(repeated_fragile_unblock_trace_event(len(trace) + 1, arbiter))
            reviewed.append(
                record.with_review(
                    risk=decision.risk,
                    verifier=decision.verifier,
                    arbiter=arbiter,
                    coordinator_trace=trace,
                )
            )
        return reviewed

    def _mark_submit_blocked_by_verifier(
        self,
        accepted_submit: ToolExecution,
        review: dict[str, Any],
    ) -> None:
        accepted_submit.result["accepted"] = False
        accepted_submit.result["pending_review"] = False
        accepted_submit.result["error"] = "verifier rejected the submitted speaker before label write"
        accepted_submit.result["verifier"] = review
        if isinstance(review.get("arbiter"), dict):
            accepted_submit.result["arbiter"] = review["arbiter"]
        self._blocked_review_attempts.append(blocked_review_attempt(review))

    def _execute_submit_labels(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            speakers = submit_speakers_from_args(args, self.tools.active_batch_size)
        except ToolValidationError as error:
            return submit_labels_argument_error(
                error=str(error),
                args=args,
                expected_count=self.tools.active_batch_size,
            )

        if self.config.require_context_before_submit and not self._used_context_tool:
            return self._reject_premature_submit_with_context()
        result = self.tools.validate_labels(speakers=speakers)
        if (
            self.config.require_identity_tool_for_temporary_speaker
            and not self._used_identity_lookup_tool
        ):
            temporary_speakers = temporary_identity_speakers(result["speakers"])
            if temporary_speakers:
                return reject_temporary_identity_submit(temporary_speakers)
        result["pending_review"] = True
        return result

    def _reject_premature_submit_with_context(self) -> dict[str, Any]:
        context = self.tools.read_active_context(self.config.context_window_lines)
        self._used_context_tool = True
        return {
            "accepted": False,
            "error": (
                "call read_novel or search_novel before submit_labels; "
                "automatic_context is included for this batch, so review it and call submit_labels again"
            ),
            "automatic_context": context,
        }


def temporary_identity_speakers(speakers: list[str]) -> list[str]:
    result = []
    for speaker in speakers:
        cleaned = speaker.strip()
        if cleaned in TEMPORARY_IDENTITY_SPEAKERS:
            result.append(cleaned)
    return result


def reject_temporary_identity_submit(temporary_speakers: list[str]) -> dict[str, Any]:
    return {
        "accepted": False,
        "error": "temporary identity speaker submitted without identity tool lookup",
        "temporary_speakers": temporary_speakers,
        "identity_tools_required": ["locate_identity", "resolve_identity"],
        "instruction": (
            "Before submitting a trackable temporary identity speaker, call locate_identity for that speaker. "
            "If locate_identity returns candidates, call resolve_identity on the best candidate range. "
            "If a stable identity is resolved, submit that speaker and consider record_character. "
            "If the bounded lookup finds no evidence or reaches its limit, submit the temporary identity with "
            "evidence_lines, reason, rejected_candidates, and confidence."
        ),
    }


def format_tool_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def accepted_submit_message(result: dict[str, Any]) -> str:
    if "warning" not in result:
        return "submitted labels for one batch"
    return f"submitted labels for one batch with recovery: {result['warning']}"


def submit_retry_message(execution: ToolExecution) -> Optional[str]:
    if execution.name != "submit_labels":
        return None
    if execution.result.get("accepted") is not False:
        return None
    instruction = execution.result.get("instruction")
    if not isinstance(instruction, str) or not instruction:
        return None
    return instruction


def summarize_tool_history(history: list[ToolExecution]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "read_novel": [],
        "search_novel": [],
        "locate_identity": [],
        "resolve_identity": [],
        "record_character": [],
        "normalize_speaker": [],
        "arbitrate_identity": [],
        "submit_labels": [],
    }

    for execution in history:
        if execution.name == "read_novel":
            summary["read_novel"].append(
                {
                    "requested_start_line": execution.arguments.get("start_line"),
                    "requested_end_line": execution.arguments.get("end_line"),
                    "start_line": execution.result.get("start_line"),
                    "end_line": execution.result.get("end_line"),
                    "truncated": execution.result.get("truncated", False),
                }
            )
        elif execution.name == "search_novel":
            summary["search_novel"].append(
                {
                    "keyword": execution.arguments.get("keyword"),
                    "limit": execution.arguments.get("limit"),
                    "total_matches": execution.result.get("total_matches"),
                    "truncated": execution.result.get("truncated", False),
                }
            )
        elif execution.name == "locate_identity":
            summary["locate_identity"].append(
                {
                    "speaker": execution.arguments.get("speaker"),
                    "dialogue_index": execution.arguments.get("dialogue_index"),
                    "search_start_line": execution.result.get("search_start_line"),
                    "search_end_line": execution.result.get("search_end_line"),
                    "round": execution.result.get("round"),
                    "round_limit": execution.result.get("round_limit"),
                    "round_limit_reached": execution.result.get("round_limit_reached", False),
                    "candidate_count": len(execution.result.get("candidates", []))
                    if isinstance(execution.result.get("candidates"), list)
                    else 0,
                }
            )
        elif execution.name == "resolve_identity":
            summary["resolve_identity"].append(
                {
                    "speaker": execution.arguments.get("speaker"),
                    "start_line": execution.arguments.get("start_line"),
                    "end_line": execution.arguments.get("end_line"),
                    "verdict": execution.result.get("verdict"),
                    "recommended_speaker": execution.result.get("recommended_speaker"),
                    "evidence_lines": execution.result.get("evidence_lines"),
                }
            )
        elif execution.name == "record_character":
            record = execution.result.get("record") if isinstance(execution.result.get("record"), dict) else {}
            summary["record_character"].append(
                {
                    "display_name": record.get("display_name"),
                    "aliases": record.get("aliases"),
                    "confidence": record.get("confidence"),
                }
            )
        elif execution.name == "normalize_speaker":
            summary["normalize_speaker"].append(
                {
                    "speaker": execution.arguments.get("speaker"),
                    "suggested_display_name": execution.result.get("suggested_display_name"),
                    "matched": execution.result.get("matched"),
                    "confidence": execution.result.get("confidence"),
                }
            )
        elif execution.name == "arbitrate_identity":
            summary["arbitrate_identity"].append(
                {
                    "labeler_speaker": execution.arguments.get("labeler_speaker"),
                    "decision": execution.result.get("decision"),
                    "recommended_speaker": execution.result.get("recommended_speaker"),
                    "reason": execution.result.get("reason"),
                }
            )
        elif execution.name == "submit_labels":
            summary["submit_labels"].append(
                {
                    "accepted": execution.result.get("accepted"),
                    "error": execution.result.get("error"),
                    "warning": execution.result.get("warning"),
                    "expected_count": execution.result.get("expected_count"),
                    "received_count": execution.result.get("received_count"),
                    "ignored_speakers": execution.result.get("ignored_speakers"),
                }
            )

    return summary


def submit_recovery_info(
    result: dict[str, Any],
    blocked_reviews: Optional[list[dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    recovery = {
        key: result[key]
        for key in ("warning", "expected_count", "received_count", "ignored_speakers")
        if key in result
    }
    if blocked_reviews:
        recovery["blocked_reviews"] = [dict(review) for review in blocked_reviews]
    return recovery or None


def blocked_review_attempt(review: dict[str, Any]) -> dict[str, Any]:
    data = {
        key: review[key]
        for key in ("index", "line_number", "speaker", "verdict", "reason", "risk_signal_codes")
        if key in review
    }
    arbiter = review.get("arbiter")
    if isinstance(arbiter, dict):
        data["arbiter"] = {
            key: arbiter[key]
            for key in (
                "decision",
                "verdict",
                "reason",
                "blocks_submission",
                "confidence",
                "block_reason_code",
            )
            if key in arbiter
        }
    return data


def repeated_fragile_review_block(
    record: Any,
    arbiter: Optional[dict[str, Any]],
    blocked_attempts: list[dict[str, Any]],
) -> bool:
    if not is_fragile_review_block(arbiter):
        return False
    for attempt in blocked_attempts:
        if attempt.get("index") != record.index:
            continue
        if attempt.get("line_number") != record.line_number:
            continue
        if attempt.get("speaker") != record.speaker:
            continue
        if is_fragile_review_block(attempt.get("arbiter")):
            return True
    return False


def is_fragile_review_block(arbiter: Any) -> bool:
    if not isinstance(arbiter, dict):
        return False
    if arbiter.get("blocks_submission") is not True:
        return False
    if arbiter.get("decision") != "needs_more_evidence":
        return False
    if arbiter.get("block_reason_code") == "fragile_high_risk_pass":
        return True
    reason = arbiter.get("reason")
    return isinstance(reason, str) and "short dialogue has no rejected speaker candidates" in reason


def unblock_repeated_fragile_arbiter(arbiter: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not isinstance(arbiter, dict):
        return arbiter
    unblocked = dict(arbiter)
    unblocked["blocks_submission"] = False
    unblocked["unblocked_after_repeated_review"] = True
    unblocked["unblock_reason"] = (
        "Repeated fragile verifier block for the same dialogue and speaker; "
        "allow progress after one retry while preserving the blocked review in recovery."
    )
    return unblocked


def repeated_fragile_unblock_trace_event(step: int, arbiter: Optional[dict[str, Any]]) -> dict[str, Any]:
    return {
        "step": step,
        "agent": "arbiter",
        "action": "unblocked",
        "reason": (
            "Repeated fragile verifier block matched a previous blocked review; "
            "submission was allowed to avoid a retry loop."
        ),
        "metadata": {
            "block_reason_code": "fragile_high_risk_pass",
            "arbiter_decision": arbiter.get("decision") if isinstance(arbiter, dict) else None,
        },
    }


def first_blocking_verifier_review(records: list[Any]) -> Optional[dict[str, Any]]:
    for record in records:
        verifier = record.verifier
        arbiter = getattr(record, "arbiter", None)
        if isinstance(arbiter, dict) and arbiter.get("blocks_submission") is True:
            review = dict(verifier) if isinstance(verifier, dict) else {}
            return {
                **review,
                "arbiter": arbiter,
                "index": record.index,
                "line_number": record.line_number,
                "speaker": record.speaker,
            }
        if not isinstance(verifier, dict):
            continue
        if verifier.get("enabled") is True and verifier.get("verdict") == "fail":
            return {
                **verifier,
                "arbiter": arbiter if isinstance(arbiter, dict) else None,
                "index": record.index,
                "line_number": record.line_number,
                "speaker": record.speaker,
            }
    return None


def verifier_retry_message(review: dict[str, Any]) -> str:
    risk_codes = review.get("risk_signal_codes")
    if not isinstance(risk_codes, list):
        risk_codes = []
    reason = review.get("reason") if isinstance(review.get("reason"), str) else "no reason"
    arbiter = review.get("arbiter") if isinstance(review.get("arbiter"), dict) else {}
    arbiter_reason = arbiter.get("reason") if isinstance(arbiter.get("reason"), str) else None
    arbiter_text = f" Arbiter reason: {arbiter_reason}." if arbiter_reason else ""
    fragile_instruction = ""
    if arbiter.get("block_reason_code") == "fragile_high_risk_pass" or (
        isinstance(arbiter_reason, str) and "short dialogue has no rejected speaker candidates" in arbiter_reason
    ):
        fragile_instruction = (
            " For this short dialogue, do not repeat the same bare submit_labels call. "
            "If the same speaker is still best, resubmit with evidence_lines, reason, "
            "rejected_candidates comparing at least one plausible alternative speaker, and confidence."
        )
    return (
        "Verifier rejected the submitted label before writing it to the output file. "
        f"Dialogue index={review.get('index')} line={review.get('line_number')} "
        f"speaker={review.get('speaker')} was not accepted. "
        f"Verifier reason: {reason}.{arbiter_text} "
        f"Risk signals: {', '.join(str(code) for code in risk_codes) or 'none'}. "
        "Read or search more context if needed, then call submit_labels again with a corrected speaker."
        f"{fragile_instruction}"
    )


def format_batch_summary(dialogues: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"index={dialogue['index']} line={dialogue['line_number']} text={dialogue['text']}"
        for dialogue in dialogues
    )


def format_recent_tools(history: list[ToolExecution], limit: int = 5) -> str:
    if not history:
        return "none"
    return ", ".join(format_tool_execution_summary(execution) for execution in history[-limit:])


def format_tool_execution_summary(execution: ToolExecution) -> str:
    if "error" in execution.result:
        return f"{execution.name}(error={shorten_text(str(execution.result['error']), 80)})"
    if execution.result.get("accepted") is True:
        return f"{execution.name}(accepted=true)"
    if execution.result.get("accepted") is False:
        return f"{execution.name}(accepted=false)"
    return execution.name


def shorten_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def format_prompt_messages(messages: list[ChatMessage]) -> str:
    lines = ["Dialoop prompt:"]
    for message in messages:
        lines.extend(
            [
                f"--- {message.role} ---",
                message.content,
            ]
        )
    lines.append("--- end prompt ---")
    return "\n".join(lines)


def required_int(args: dict[str, Any], name: str) -> int:
    value = args[name]
    if type(value) is not int:
        raise ToolValidationError(f"{name} must be an integer")
    return value


def optional_int(args: dict[str, Any], name: str) -> Optional[int]:
    if name not in args or args[name] is None:
        return None
    return required_int(args, name)


def required_str(args: dict[str, Any], name: str) -> str:
    value = args[name]
    if not isinstance(value, str):
        raise ToolValidationError(f"{name} must be a string")
    return value


def optional_str(args: dict[str, Any], name: str) -> Optional[str]:
    if name not in args or args[name] is None:
        return None
    return required_str(args, name)


def required_str_list(args: dict[str, Any], name: str) -> list[str]:
    value = args[name]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ToolValidationError(f"{name} must be a list of strings")
    return value


def optional_str_list(args: dict[str, Any], name: str) -> Optional[list[str]]:
    if name not in args or args[name] is None:
        return None
    return required_str_list(args, name)


def optional_int_list(args: dict[str, Any], name: str) -> Optional[list[int]]:
    if name not in args or args[name] is None:
        return None
    value = args[name]
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ToolValidationError(f"{name} must be a list of integers")
    return value


def submit_speakers_from_args(args: dict[str, Any], expected_count: int) -> list[str]:
    for name in SUBMIT_LABEL_ALIASES:
        if name in args:
            return coerce_speaker_argument(args[name], name, expected_count)
    raise ToolValidationError("missing required argument: speakers")


def coerce_speaker_argument(value: Any, name: str, expected_count: int) -> list[str]:
    if isinstance(value, list):
        if any(not isinstance(item, str) for item in value):
            raise ToolValidationError(f"{name} must be a list of strings")
        return value
    if isinstance(value, str) and expected_count == 1:
        return [value]
    if isinstance(value, str):
        raise ToolValidationError(f"{name} must be a list of strings for {expected_count} speakers")
    raise ToolValidationError(f"{name} must be a list of strings")


def submit_labels_argument_error(error: str, args: dict[str, Any], expected_count: int) -> dict[str, Any]:
    example_speakers = ["<speaker>"] * max(1, expected_count)
    example_args = {"speakers": example_speakers}
    return {
        "accepted": False,
        "error": error,
        "expected_count": expected_count,
        "received_arguments": args,
        "expected_arguments": example_args,
        "instruction": (
            "Retry submit_labels now with arguments exactly like "
            f"{json.dumps(example_args, ensure_ascii=False)}. "
            "Use the key `speakers`; it must contain one speaker per active dialogue."
        ),
    }
