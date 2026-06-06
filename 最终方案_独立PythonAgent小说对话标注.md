# 最终方案：独立 Python Agent 小说对话说话人标注工具

> 本文用于替代 README 中早期的 OpenCode wrapper 路线，作为后续实现的主方案。
> 结论先行：本项目最终不把 OpenCode / openloop 作为运行时依赖，而是借鉴它们的 agent loop 和自动继续思想，做一个独立 Python CLI，后端兼容 OpenAI-compatible API，重点支持 Ollama 本地模型。

## 1. 最终决策

### 1.1 主线选择

采用：

```text
独立 Python CLI
  + OpenAI-compatible / Ollama 模型后端
  + 内置小说阅读工具
  + 内置 agent tool loop
  + labeled.txt 断点续跑
```

不采用：

```text
运行时依赖 OpenCode
运行时依赖 openloop 插件
复制 OpenCode 大量源码
V2 的完整 chunk / 摘要 / 角色库流水线作为 v1 主架构
```

### 1.2 为什么这样选

当前已经验证过的成功工作流是：

```text
OpenCode AI 自己读小说原文
  -> 判断当前对话说话人
  -> 写入 labeled.txt
  -> openloop 继续唤醒
```

这个工作流成功的关键不是 OpenCode 的 TUI，也不是 openloop 插件本身，而是：

1. LLM 能主动使用工具读取原文。
2. LLM 不被固定上下文窗口限制。
3. 进度由简单可靠的外部文件保存。
4. 未完成时可以继续下一轮。

所以最终产品应该保留这些机制，而不是把 OpenCode 整体打包进去。

### 1.3 对现有路线的判断

#### OpenCode wrapper 路线

当前仓库中已有的 `dialoop` OpenCode wrapper 实现可以保留为实验代码，但不作为最终主线。

原因：

- 仍要求用户安装 OpenCode。
- 仍依赖 OpenCode 的模型配置和命令行为。
- 对非技术用户来说，分发门槛没有根本降低。
- 难以稳定兼容 Ollama 等本地模型的不同 tool calling 行为。

#### V2 路线

`speaker_labeling_plan_v2.md` 中的记忆、review queue、实体类型设计有价值，但不适合作为 v1 主循环。

原因：

- Python 预先决定模型能看到什么，容易把模型降级成被动分类器。
- chunk、摘要、角色卡、实体合并会带来大量自研复杂度。
- 摘要漂移和角色库脏增长会成为主要风险。
- v1 阶段应先验证最小 agent loop 的标注质量。

V2 的部分能力应作为后续增强层逐步引入，而不是一开始进入核心架构。

## 2. 项目目标

### 2.1 v1 目标

做一个用户可以直接运行的小说对话说话人标注 CLI：

```bash
dialoop novel.txt
```

它应当：

- 自动提取 `「」` 对话。
- 按 `labeled.txt` 行数断点续跑。
- 调用本地或远程模型判断说话人。
- 允许模型主动读取小说任意行范围。
- 允许模型搜索关键词或角色名。
- 校验并追加写入说话人。
- 一直运行到全部对话标注完成，或达到安全上限。

### 2.2 优先级

本项目优先级：

1. 标注质量。
2. 可长时间无人值守运行。
3. 能使用 Ollama 本地模型降低 token 成本。
4. 简单分发和简单命令行使用。
5. 速度。

速度不是 v1 的核心目标。一卷小说跑很久可以接受。

### 2.3 非目标

v1 不做：

- GUI。
- OpenCode 插件安装。
- 自动下载或打包大模型。
- 全套 V2 记忆系统。
- 自动保证 100% 正确。
- 多小说项目管理。
- 复杂人工标注平台。

## 3. 总体架构

```text
用户
  |
  | dialoop novel.txt --model qwen3:32b --base-url http://localhost:11434/v1
  v
CLI 层
  |
  | 构建运行配置
  v
Runner / Orchestrator
  |
  | 管理进度、重试、最大轮数、日志
  v
Agent Loop
  |
  | 给模型提供工具
  v
Model Backend
  |
  | OpenAI-compatible API / Ollama
  v
Tool Executor
  |
  | read_novel / search_novel / get_next_dialogue / submit_labels
  v
novel.txt + labeled.txt
```

### 3.1 Python 负责什么

Python 是可靠控制层，负责：

- 解析小说和对话。
- 计算总对话数。
- 根据 `labeled.txt` 判断当前进度。
- 暴露工具给模型。
- 执行工具调用。
- 校验模型提交的 speaker 数量。
- 写入 `labeled.txt`。
- 判断是否完成。
- 控制每轮最大 tool steps。
- 失败重试和安全退出。

### 3.2 LLM 负责什么

LLM 是判断层，负责：

- 决定需要读哪些上下文行。
- 决定是否需要搜索角色名、称呼、关键词。
- 根据原文判断说话人。
- 在证据不足时继续读取更多上下文。
- 最终调用 `submit_labels` 提交结果。

LLM 不直接编辑文件，不直接运行 shell，不直接决定任务是否全局完成。

## 4. 核心数据模型

### 4.1 Dialogue

每个对话最少包含：

```json
{
  "index": 0,
  "line_number": 23,
  "text": "这是最后一件了吧？"
}
```

说明：

- `index` 是全书对话序号，从 0 开始。
- `line_number` 是小说原文行号，从 1 开始。
- `text` 是 `「」` 内的对话内容。

### 4.2 Progress

v1 不需要单独数据库。进度由 `labeled.txt` 决定：

```text
已标注数量 = labeled.txt 非空行数量
下一条对话 index = 已标注数量
```

如果 `labeled.txt` 有 400 行，则下一轮从第 401 条对话开始。

### 4.3 Label

v1 输出仍保持简单：

```text
罗伦斯
赫萝
商人
```

每行一个说话人，顺序与对话出现顺序一致。

后续增强可以增加结构化输出，但 `labeled.txt` 必须作为最小稳定输出保留。

## 5. 内置工具

v1 的核心是给模型提供少量高质量工具。

### 5.1 get_next_dialogue

用途：获取下一批待标注对话。

输入：

```json
{
  "batch_size": 1
}
```

输出：

```json
{
  "done": false,
  "progress": {
    "labeled": 400,
    "total": 2340
  },
  "dialogues": [
    {
      "index": 400,
      "line_number": 1208,
      "text": "这是最后一件了吧？"
    }
  ]
}
```

规则：

- 如果全部完成，返回 `done: true`。
- 不让模型自己推断进度。
- batch 内对话必须按原文顺序返回。

### 5.2 read_novel

用途：按行号读取小说原文。

输入：

```json
{
  "start_line": 1190,
  "end_line": 1225
}
```

输出：

```text
1190: ...
1191: ...
...
1225: ...
```

规则：

- 输出必须带行号。
- 行范围要有最大限制，例如一次最多 300 行。
- 超出文件范围时自动裁剪。
- 模型可以多次调用它读取更多上下文。

### 5.3 search_novel

用途：搜索关键词、角色名、称呼、口癖。

输入：

```json
{
  "keyword": "赫萝",
  "limit": 20
}
```

输出：

```json
{
  "matches": [
    {
      "line_number": 87,
      "line": "87: ..."
    }
  ],
  "truncated": false
}
```

规则：

- 默认最多返回 20 条。
- 返回结果只用于定位，模型需要再用 `read_novel` 读取完整上下文。
- 后续可支持正则或模糊搜索，但 v1 只需要普通子串搜索。

### 5.4 submit_labels

用途：提交当前 batch 的说话人。

输入：

```json
{
  "speakers": ["罗伦斯"]
}
```

输出：

```json
{
  "accepted": true,
  "written": 1,
  "progress": {
    "labeled": 401,
    "total": 2340
  }
}
```

规则：

- `speakers` 数量必须等于当前 batch 对话数量。
- 校验失败时不写入。
- 写入只追加到 `labeled.txt`。
- 不允许模型跳过当前 batch 写后面的对话。
- 写入成功后，本轮 agent loop 结束，进入下一轮。

### 5.5 get_progress

可选工具，用于调试和恢复。

输出：

```json
{
  "labeled": 401,
  "total": 2340,
  "remaining": 1939,
  "output_path": "labeled.txt"
}
```

v1 可以实现，也可以先不暴露给模型。

## 6. Agent Loop

### 6.1 外层循环

Python 外层循环：

```text
while not all_dialogues_labeled:
  batch = get_next_dialogue(batch_size)
  run_one_agent_turn(batch)
```

完成判断由 Python 做：

```text
len(labeled.txt lines) >= len(all_dialogues)
```

不再依赖 `<promise>DONE</promise>`。

### 6.2 内层工具循环

单个 batch 的内层循环：

```text
messages = [system_prompt, user_prompt_for_current_batch]

for step in range(max_tool_steps):
  response = model.chat(messages, tools)

  if response requests tool:
    result = execute_tool(response.tool_call)
    messages.append(tool_result)

    if tool == submit_labels and accepted:
      return success

    continue

  if response gives final text without submit_labels:
    append reminder asking it to call submit_labels
    continue

fail current batch if max_tool_steps exceeded
```

### 6.3 为什么按 batch 结束一轮

模型每标完一个 batch 就结束本轮，而不是在同一次模型上下文里一直标完整卷。

原因：

- 上下文不会无限增长。
- 每个 batch 写入后天然 checkpoint。
- 失败后最多重跑当前 batch。
- 适合本地模型长时间慢速运行。

## 7. 模型后端

### 7.1 统一使用 OpenAI-compatible API

v1 使用 `openai` Python SDK 或等价 HTTP 客户端。

远程模型示例：

```bash
dialoop novel.txt \
  --base-url https://api.deepseek.com/v1 \
  --api-key $DEEPSEEK_API_KEY \
  --model deepseek-chat
```

Ollama 示例：

```bash
dialoop novel.txt \
  --base-url http://localhost:11434/v1 \
  --api-key ollama \
  --model qwen3:32b
```

说明：

- Ollama 的 OpenAI-compatible API 通常要求传一个占位 API key。
- 默认可以设置为 `ollama`，不需要真实密钥。

### 7.2 Tool calling 两级协议

不同本地模型的 tool calling 稳定性不同。v1 必须设计两级协议。

#### 优先：原生 tools / function calling

如果后端支持 OpenAI tools：

- Python 发送 `tools` schema。
- 模型返回 `tool_calls`。
- Python 执行工具并回传结果。

这是首选路径。

#### 兜底：JSON action 协议

如果模型不能稳定返回原生 tool calls，则让模型输出严格 JSON：

```json
{
  "action": "read_novel",
  "args": {
    "start_line": 100,
    "end_line": 140
  }
}
```

或：

```json
{
  "action": "submit_labels",
  "args": {
    "speakers": ["罗伦斯"]
  }
}
```

Python 解析 JSON、执行工具、把结果作为下一条消息返回给模型。

v1 可以先实现一种协议，但最终方案必须保留这两个通道的设计位置。

## 8. CLI 设计

推荐命令：

```bash
dialoop novel.txt
```

常用参数：

```bash
dialoop novel.txt \
  --output labeled.txt \
  --model qwen3:32b \
  --base-url http://localhost:11434/v1 \
  --api-key ollama \
  --batch-size 1 \
  --max-tool-steps 20 \
  --max-iterations 100000
```

建议参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `novel_path` | 必填 | 小说 txt 路径 |
| `--output` | `labeled.txt` | 标注输出 |
| `--model` | `qwen3:32b` 或配置文件默认 | 模型名 |
| `--base-url` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `--api-key` | `ollama` | API key 或占位值 |
| `--batch-size` | `1` | 每轮标注几句 |
| `--max-tool-steps` | `20` | 单个 batch 最多工具调用步数 |
| `--max-iterations` | `100000` | 外层最大 batch 轮数 |
| `--context-window-lines` | `80` | 初始建议模型读取目标对话前后多少行 |
| `--read-window-limit` | `300` | 单次 read_novel 最大行数 |
| `--search-limit` | `20` | 单次 search_novel 最大结果数 |
| `--previous-context-dialogues` | `8` | 每轮 prompt 中附带多少条最近已标注对话 |
| `--following-context-dialogues` | `8` | 每轮 prompt 中附带多少条后续未标注对话 |
| `--dry-run` | false | 只检查配置、索引和模型连接 |
| `--protocol` | `auto` | `tools` / `json` / `auto` |

## 9. Prompt 设计

### 9.1 System Prompt 原则

System prompt 应该短、稳定、规则明确：

```text
你是小说对话说话人标注助手。
你的任务是判断当前 batch 中每句「」对话的说话人。

你可以使用工具读取小说原文、搜索关键词、提交标注。
如果上下文不够，继续读取更多原文，不要猜。
提交时必须调用 submit_labels，speaker 数量必须等于当前 batch 对话数量。
不要修改小说原文。
不要直接写文件。
```

### 9.2 判断规则

v1 prompt 中应包含少量通用规则：

- 叙述中的“某某说 / 某某道 / 某某问”是强证据。
- 被称呼者通常不是说话者。
- 两人连续对话时可参考轮换，但不能覆盖明确叙述。
- 短句、追问、省略号、沉默或半句话要重点参考相邻对话和最近已标注结果，但不能机械沿用上一句说话人。
- 如果是群体呼喊或无名路人，允许输出“路人”“众人”“店员”等通用称呼。
- 角色名要保持一致，优先使用原文中出现过的名字或稳定称呼。
- 如果当前只出现“女孩”“少年”“老人”等临时描述，但这是可追踪的具体人物，且后文在有限范围内揭示姓名或稳定称呼，应回填为后文揭示的姓名或稳定称呼。
- 不要为了无名群体、路人或临时职能角色无限寻找姓名；普通低重要度群体可以保留稳定身份词。
- 如果引号内容明显不是人物说话，而是叙述中的环境声、物体声音、心理比喻声或声音效果，统一标注为“非人物发声”；如果文本明确说明某个角色发出该声音，如喊叫、叹息、笑声或嚎叫，仍标注该角色。
- 不确定时可以输出“未知”，但应先尝试读取更多上下文。

### 9.3 User Prompt for Batch

每个 batch 的 user prompt 包含：

```text
当前需要标注的对话：
1. 第 1208 行：「这是最后一件了吧？」

最近已标注对话：
- 第 1204 行 speaker=甲：「……」

后续未标注对话：
- 第 1210 行：「……」

请使用工具读取必要上下文，判断说话人，并调用 submit_labels。
```

不要把大段原文直接塞进初始 prompt。让模型自己调用 `read_novel`。

## 10. 输出和日志

### 10.1 必需输出

```text
labeled.txt
```

每行一个说话人。

### 10.2 建议输出

为了可调试，建议增加：

```text
.dialoop/
  run.log
  calls.jsonl
  errors.jsonl
  config.json
```

其中：

- `run.log`: 人类可读运行日志。
- `calls.jsonl`: 每次模型调用、工具调用、结果摘要。
- `errors.jsonl`: 失败 batch、异常、重试记录。
- `config.json`: 本次运行配置快照。

v1 可以先只写控制台日志，后续再加文件日志。

## 11. 错误处理

### 11.1 submit_labels 数量不匹配

如果当前 batch 有 2 条对话，但模型提交 1 个或 3 个 speaker：

- Python 拒绝写入。
- 返回错误给模型。
- 要求模型重新提交。

### 11.2 模型没有调用 submit_labels

如果模型只输出解释，没有调用工具：

- Python 追加一条提醒。
- 要求它必须调用 `submit_labels`。
- 超过 `max_tool_steps` 后当前 batch 失败。

### 11.3 模型请求过大行范围

如果 `read_novel(1, 10000)`：

- Python 裁剪到 `read_window_limit`。
- 返回提示：结果已裁剪，请继续分段读取。

### 11.4 模型过早提交

如果模型没有先调用 `read_novel` / `search_novel` 就调用 `submit_labels`：

- Python 拒绝本次写入。
- 自动读取当前 batch 附近上下文，并把 `automatic_context` 返回给模型。
- 当前 batch 视为已经提供过上下文；模型下一步重新提交时不应被同一规则反复拒绝。

### 11.5 中断恢复

用户 Ctrl+C：

- 已写入的 `labeled.txt` 保留。
- 下次运行从已写入行数继续。
- 不需要额外恢复步骤。

### 11.6 API 失败

API 网络失败或模型错误：

- 当前 batch 不写入。
- 按配置重试。
- 重试仍失败则退出，并提示下次可从 `labeled.txt` 继续。

## 12. 与现有文件的关系

### 12.1 `get_dialogue.py`

现有对话提取逻辑可以复用，但最终应内聚到 Python 包中，例如：

```text
dialoop/dialogue.py
```

保留脚本入口作为兼容层可以接受。

### 12.2 `write_label.py`

最终主流程不应让模型调用 shell 执行 `write_label.py`。

应改为 Python 内部工具：

```text
submit_labels(speakers)
```

`write_label.py` 可以保留为人工调试工具。

### 12.3 `opencode-dev/`

只作为参考源码。

参考点：

- tool schema 设计。
- read_file 的 offset / limit 思路。
- agent tool loop。
- 工具结果带行号。

不作为运行时依赖。

### 12.4 `openloop-main/`

只作为参考源码。

参考点：

- 最大迭代上限。
- 未完成继续。
- 状态文件思想。

不作为运行时依赖。

### 12.5 当前 `dialoop` OpenCode wrapper

当前已经提交的 OpenCode wrapper 可以：

- 暂时保留。
- 后续移动到 `experiments/opencode-wrapper/`。
- 或在新主线稳定后删除。

不要继续在它上面叠最终功能。

## 13. 实施阶段

### 阶段 0：文档决策

目标：确认本文作为最终方向。

产物：

- 本文档。
- README 后续可更新为指向本文。

### 阶段 1：核心数据和工具

目标：不接模型，先实现可靠的本地工具。

内容：

- `DialogueIndex`: 提取所有 `「」` 对话和行号。
- `LabelStore`: 读取和追加 `labeled.txt`。
- `read_novel(start_line, end_line)`。
- `search_novel(keyword, limit)`。
- `get_next_dialogue(batch_size)`。
- `submit_labels(speakers)`。

验收：

- 单元测试覆盖进度、越界、数量校验。
- 不调用模型也能模拟完整标注写入。

### 阶段 2：模型适配层

目标：接入 OpenAI-compatible API。

内容：

- `ModelClient`。
- 支持 `base_url`、`api_key`、`model`。
- 支持至少一种协议：原生 tools 或 JSON action。
- dry-run 检查模型连接。

验收：

- 能对 Ollama endpoint 发起简单请求。
- API 失败有清晰错误。

### 阶段 3：Agent Loop

目标：跑通单个 batch。

内容：

- system prompt。
- batch prompt。
- tool loop。
- `max_tool_steps`。
- `submit_labels` 成功后结束当前 batch。

验收：

- 用假模型测试 tool loop。
- 用真实模型测试前 1 到 3 条对话。
- `novel.txt` 不被修改。
- `labeled.txt` 正确追加。

### 阶段 4：长跑能力

目标：支持整卷无人值守运行。

内容：

- 外层循环。
- 最大迭代次数。
- Ctrl+C 安全退出。
- 错误重试。
- 运行日志。

验收：

- 可从中断的 `labeled.txt` 继续。
- 模型失败不会写入半条 batch。
- 日志足够定位最后一次失败。

### 阶段 5：质量增强

目标：在阶段 4 已经能整卷无人值守运行的基础上，把优化重心从“能跑完”转为“尽可能少错”。阶段 5 不应回到 V2 那种由 Python 预先切 chunk、维护复杂 `unknown` / `mystery` / `npc_group` 实体体系的路线；当前已经验证模型可以主动阅读原文，下一步应该让模型读得更有目的，并用独立校验降低单个 agent 的误判。

阶段 5 的核心原则：

- 仍以 `labeled.txt` 作为稳定主输出，每行一个最终 speaker。
- 不硬编码具体小说的人名、地点、剧情或关系。
- 不把 `unknown` / `mystery` / `npc_group` 作为主线标签体系；大多数情况应通过继续阅读原文得到姓名、稳定身份或通用身份词。
- 不强制模型读取固定次数；改为要求给出足够证据，证据不足时再继续读。
- 前文已标注结果只能作为弱线索，不能作为事实来源；如果原文证据冲突，以原文为准。
- 接受运行时间变长，用更多模型轮次换准确率。
- 代码、prompt 模板、默认配置中不得出现特定小说的人名、专有地点、专有剧情或只属于某个角色的口癖。测试样本和人工答案文件可以包含这些文本，但生产逻辑不能依赖它们。
- 不催促模型“快速标注”。`max_tool_steps`、查找上限和上下文预算是防止死循环的安全边界，不是速度优化目标。
- 单个 agent 的任务不能无限变重；即使模型支持较长上下文，也应把定位、细读、校验、归一拆给不同角色，避免一个上下文同时承担全书记忆、局部推理和质量审查。

建议把阶段 5 拆成以下子阶段。

#### 阶段 5A：评估基线和错误分类

目标：先把“质量是否变好”变成可重复衡量的问题。

内容：

- 增加评估脚本，对比 `labeled_test.txt` 和人工答案文件，输出准确率、错误行号、模型标签、参考答案。
- 把错误按通用类型分类，而不是按具体作品分类：
  - 对话轮次错判：短句、追问、省略号、连续问答被归到上一位说话人。
  - 被称呼者误判：把话中被问到、被称呼的人当成说话人。
  - 身份后置：先出现临时描述，后文才揭示姓名或稳定身份。
  - 场景参与者误判：新场景中多个无名或低重要度人物混淆。
  - 非人物发声：环境声、物体声、心理比喻声被标成角色。
  - 角色名归一：同一角色被标成多个等价名称。
- `error_labels.txt` 可以作为人工记录格式继续保留，但后续应由脚本生成候选错误报告，人工只做确认和补充。
- 增加生产逻辑专有词扫描。扫描对象应限制在 `dialoop/`、prompt 模板、默认配置等生产路径；测试语料、人工答案、样例小说和文档可以包含小说文本。专有词列表由评测语料临时配置，不写进生产代码。

验收：

- 能复现“前 N 条准确率”和错误明细。
- 每次 prompt 或 agent 策略变更后，能用同一批样本做对照。
- 能证明生产代码和默认 prompt 没有混入当前评测小说的专有角色名、地点、剧情或口癖。

#### 阶段 5B：证据化标注

目标：避免模型“看起来答对但没有证据”，也避免模型过早提交。

内容：

- 内部结果从单纯 speaker 扩展为临时结构化结果，再由 Python 只把 speaker 写入 `labeled.txt`：

```json
{
  "speaker": "角色A",
  "evidence_lines": [681, 682],
  "reason": "681 行说明角色A开口，682 行是紧接的提问",
  "rejected_candidates": [
    {"speaker": "角色B", "reason": "该句是在询问角色B，且前后轮次显示不是角色B发问"}
  ],
  "confidence": "high|medium|low"
}
```

- `submit_labels` 仍保持只写 speaker，避免破坏现有输出。
- 新增调试输出建议写入 `.dialoop/annotations.jsonl`，用于保存证据、反证、工具调用摘要和最终 speaker。
- 对短句、追问、省略号这类高风险文本，prompt 应明确要求说明“为什么不是上一句说话人 / 为什么不是被称呼者”。

验收：

- 每条标注能追溯到原文行号证据。
- 失败样本不只看到错标签，还能看到模型当时依据了哪些证据。

#### 阶段 5C：风险门控

目标：明确“哪些样本进入复核”由谁判断，避免后续实现时把规则写死成一堆脆弱 if。

风险门控分两层：

```text
Programmatic Risk Gate
  Python 负责便宜、确定、可测试的信号。

Risk Judge / Verifier Agent
  模型负责语义性强、程序难以灵活判断的信号。
```

Python 可以判断的信号：

- 对话文本长度很短，或主要由省略号、语气词、标点组成。
- 当前 batch 附近存在连续多句相邻对话。
- Labeler 输出 `confidence=low` 或缺少 `evidence_lines`。
- Labeler 输出“非人物发声”。
- Labeler 输出通用身份词，而附近上下文出现了具体人物介绍或称呼。
- 当前 speaker 和最近已标注 speaker 的轮次关系异常，例如连续多句同一 speaker 但没有叙述动作支持。

模型更适合判断的信号：

- 这句话是否像是在回应上一句话。
- 这句话中的“你 / 您 / 先生 / 小姐 / 对方称呼 / 人名称呼”是否说明被称呼者不是说话者。
- 当前证据是否真的支持 Labeler 的 speaker。
- 是否需要向前或向后继续读取。
- 临时身份词是否可能在后文被揭示为具体姓名或稳定身份。

Risk Judge / Verifier 的结构化输出建议：

```json
{
  "needs_review": true,
  "risk_reasons": ["short_dialogue", "possible_addressee_confusion"],
  "recommended_action": "verify|identity_lookup|accept|arbiter",
  "notes": "简短说明，不复制大段原文"
}
```

处理方式：

- 普通样本：Labeler 通过，且 Programmatic Risk Gate 没有触发高风险，可以直接写入。
- 程序高风险样本：必须交给 Verifier。
- Verifier 认为证据不足：触发重读、身份查找、或进入 Arbiter。
- Verifier 认为 Labeler 错误：进入 Arbiter 或要求 Labeler 按指定证据重做。

验收：

- 风险触发原因要写入 `.dialoop/annotations.jsonl`，例如 `risk_reasons: ["short_dialogue", "turn_pattern_suspicious"]`。
- 每条复核都有明确原因，不能只记录“模型觉得不确定”。

#### 阶段 5D：主 agent 与辅助工具分工（阶段 6 前置）

目标：把“主 agent / 子 agent”的分工边界定义清楚，但阶段 5 不要求所有角色都已经是独立 LLM 会话。阶段 5 先把 Locator、Resolver、Normalizer、Librarian、Arbiter 做成主 agent 可调用的工具、轻量逻辑或已存在 Verifier 这类小范围模型复核，用真实长跑验证触发时机、输入输出和日志价值。等阶段 5 调好后，再在阶段 6 把最值得模型化的工具升级为真正独立 LLM agent。

阶段 5 的建议结构：

```text
Python Runner
  负责进度、文件、工具执行、写入和安全边界。

Quality Coordinator Agent（主 agent，阶段 5 可先由 Runner / Prompt / 风险门控共同承担）
  负责看 Labeler 的结构化结果和风险信号，决定是否：
  - 接受结果。
  - 调用 Verifier。
  - 调用 Identity Locator / Identity Resolver。
  - 调用 Name Normalizer。
  - 调用 Arbiter。

Labeler Agent（阶段 5 的主标注模型）
  负责当前 batch 的初始 speaker 判断和正证据。

Verifier Agent（阶段 5 已可作为独立模型复核 agent）
  负责找反证、轮次冲突、被称呼者误判和证据不足。

Identity Locator Agent（阶段 5 先作为有界查找工具或轻量逻辑）
  粗略查找临时身份后文是否出现姓名或稳定身份。

Identity Resolver Agent（阶段 5 先作为候选范围细读工具或轻量逻辑）
  在 Locator 给出的候选行范围内细读，判断是否真是同一人物。

Name Normalizer / Character Librarian Agent（阶段 5 先作为轻量角色库工具）
  负责角色显示名归一、轻量角色库更新和重复候选检查。

Arbiter Agent（阶段 5 先作为冲突裁决工具）
  只在多个 agent 结论冲突时做最终裁决。
```

说明：

- 多个 agent 可以使用同一个模型；这通常只增加调用时间，不需要加载多份模型显存。
- 阶段 5 的“agent”允许先表现为可调用工具、轻量程序逻辑或单一 Verifier 模型会话；这一步的重点是确认触发策略和证据结构是否有效。
- 真正多个独立 LLM 会话的协作不在阶段 5 里强行完成，单独放到阶段 6。
- 主 agent 负责调度和裁决流程，不应携带过大的小说上下文。
- 子 agent 的 prompt 必须单一明确，避免一个 agent 同时承担定位、细读、归一和仲裁。
- 不做简单多数投票。投票无法解决高度相关错误，必须比较证据和反证。
- 每个 agent 都要有上下文预算；超过预算时应返回“需要更多定位信息”，而不是把越来越多原文塞进同一个 prompt。

验收：

- 能从日志看出某条标注经过了哪些 agent。
- 每个 agent 的输入输出结构稳定，方便单独测试。
- 相比单 agent，错误样本的复核命中率提升。

#### 阶段 5E：身份后置查找 agent

目标：解决“当前只知道是一个女孩、少年、老人、客人、村民等临时身份，但后文可能揭示姓名或稳定身份”的问题，同时避免为了普通路人无限往后找。

建议拆成两个子 agent：

```text
Identity Locator Agent
  输入：当前对话、当前行号、临时身份词、已读上下文、起始搜索行号。
  任务：粗略向后查找可能揭示身份的区域。
  输出：候选行号范围、简短原因、是否建议继续查找。

Identity Resolver Agent
  输入：Locator 给出的候选行范围、当前对话和必要上下文。
  任务：细读候选区域，判断是否确实是同一人物。
  输出：resolved | not_same_person | not_enough_evidence，以及推荐 speaker。
```

查找策略：

- Locator 不需要读大量细节，只需要找可能的人物介绍区域。
- Resolver 只读 Locator 给出的较小范围，避免主标注 agent 浪费上下文。
- 每次查找要带 `search_after_line`，如果第一轮没找到，下一轮从上次检查范围之后继续。
- 设置安全边界，例如 `--identity-lookahead-lines` 和 `--identity-lookahead-rounds`，默认值应保守，用户可调大。
- 查找上限表示“防止死循环”，不是催促模型快速结束；上限内没找到时，可以保留稳定身份词，不输出臆造姓名。

何时触发：

- Labeler 输出的是临时身份词。
- 文本或上下文显示这是可追踪具体人物，而不是普通群体。
- 后续对话或叙述暗示该人物继续参与场景。
- Verifier 认为当前身份词可能可以被后文更稳定命名。

何时不触发：

- 明显是无名群体、路人、临时职能角色。
- 已有稳定身份词足够满足标注目标。
- 查找成本已经达到用户设置的上限。

验收：

- 对身份后置样本，能输出“在哪些行找到介绍、为什么是同一人”。
- 对普通路人样本，不会无限向后搜索。

#### 阶段 5F：轻量角色库和显示名归一

目标：实时维护“已出现角色和稳定身份”的轻量库，用于减少同一对象多标签问题，但不走 V2 的完整角色卡数据库路线。

建议数据结构：

```json
{
  "id": "char_0001",
  "display_name": "角色A",
  "aliases": ["角色A的别称", "某个稳定身份词"],
  "summary": "80 到 160 字以内，只记录当前小说文本已经证明的事实",
  "evidence_lines": [120, 135],
  "last_seen_dialogue_index": 42,
  "confidence": "high|medium|low"
}
```

角色库维护分工：

- Name Normalizer Agent：
  - 输入 Labeler 的 speaker、证据、当前轻量角色库候选。
  - 判断是否应映射到已有 display_name。
  - 只能提出归一建议，不能直接覆盖说话人归属。
- Character Librarian Agent：
  - 负责新增角色条目、更新 summary、补充 aliases。
  - 每次新增后检查是否可能与旧条目重复。
  - 对重复候选给出证据和建议。
- Arbiter Agent：
  - 当 Normalizer / Librarian 与 Labeler 结论冲突时裁决。

原则：

- 角色库只记录当前输入文本和运行中形成的证据。
- 不允许根据外部作品知识预置角色。
- 不允许把“看起来类似”的身份强行合并；必须有证据行支持。
- 角色库是辅助线索，不是最终真相。原文证据优先。
- 如果旧库条目来自错误标注，Verifier/Arbiter 可以要求降权或修正。

验收：

- 同一重要角色的显示名更稳定。
- 临时身份词能在证据充分时归一到已有角色。
- 无名低重要度人物不会被过度合并成重要角色。

#### 阶段 5G：阶段 5 不做什么

阶段 5 暂不做：

- 完整 V2 角色卡数据库。
- 长期 `mystery` 实体追踪。
- 把所有无名 NPC 都结构化入库。
- 把 Locator / Resolver / Normalizer / Librarian / Arbiter 全部升级为独立 LLM 会话；这放到阶段 6。
- GUI 或人工标注平台。
- 用投票替代证据判断。
- 在生产代码或 prompt 模板中写入某部小说的角色名、地点、剧情、口癖。

这些能力可以作为更后面的研究方向，但不应阻塞当前目标：在现有整卷长跑 agent loop 上显著提高标注准确率。

### 阶段 6：真正多 LLM Agent 协作

目标：在阶段 5 的风险门控、身份工具、角色库和触发策略验证后，把已经证明需要独立上下文的辅助环节升级为真正独立的 LLM agent 会话。阶段 6 不在单个 issue 中一次性完成，而是拆成多个可独立开发、独立验证、独立合并的子阶段，避免一个阶段 6 分支长期偏离主分支。

启动条件：

- 阶段 5 的整卷长跑能稳定完成。
- `.dialoop/annotations.jsonl` 中能看到 identity 或 verifier 等辅助工具在合适样本上被触发；如果 normalizer / arbiter 一直没有触发，也应视为阶段 6 调度改造的输入信号，而不是阻塞条件。
- mismatch attribution 能说明哪些错误仍然来自身份后置、显示名归一、Verifier false pass 或多结论冲突。
- 当前工具层的输入输出结构已经足够稳定，可以作为独立 agent 的协议。

整体目标结构：

```text
Python Runner
  负责进度、文件、预算、工具执行、agent 调度和安全边界。

Coordinator Agent
  只看结构化候选、风险信号和各子 agent 结论，决定下一步。

Labeler Agent
  负责当前 batch 的初始 speaker、正证据、反证候选和 confidence。

Verifier Agent
  负责找反证、轮次冲突、被称呼者误判和证据不足。

Identity Locator Agent
  独立模型会话，负责粗略定位身份后置候选区域。

Identity Resolver Agent
  独立模型会话，负责细读候选范围并判断是否同一人物。

Name Normalizer / Character Librarian Agent
  独立模型会话，负责显示名归一、别名维护和角色库去重建议。

Arbiter Agent
  独立模型会话，只在多方结论冲突时比较证据并裁决。
```

原则：

- 多个 agent 可以共享同一个模型 endpoint，不要求加载多份模型。
- 不是每条对话都调用所有 agent；仍由风险门控和触发规则决定。
- 每个 agent 都必须有单一职责、稳定 JSON 输出和上下文预算。
- 不做简单投票，必须比较证据、反证和原文行号。
- 角色库仍是辅助线索，不能覆盖原文证据。

#### 阶段 6-1：Coordinator 调度骨架 + 子 agent 协议

目标：先把“谁决定调用哪个 agent、每个 agent 输入输出是什么、日志如何记录”做成稳定骨架，不急着把所有子 agent 都实现完整能力。

内容：

- 新增 Coordinator 层，放在 Python Runner 与各子 agent 之间。
- 定义统一 agent result schema，例如：

```json
{
  "agent": "labeler|verifier|identity_locator|identity_resolver|normalizer|arbiter",
  "verdict": "accept|reject|uncertain|resolved|not_same_person|not_enough_evidence",
  "recommended_speaker": "可选",
  "evidence_lines": [123],
  "counter_evidence_lines": [],
  "reason": "简短说明",
  "confidence": "high|medium|low"
}
```

- Coordinator 根据 risk signals、Verifier 结果、identity lookup 结果和 mismatch 类型决定下一步，而不是依赖主 agent 自觉调用工具。
- 每个子 agent 有独立 prompt 构造函数和独立上下文预算。
- `.dialoop/annotations.jsonl` 中增加 coordinator trace，记录某条标注经过了哪些 agent、为什么调用、为什么接受或拒绝。
- 先用 fake model / deterministic stub 覆盖调度分支，不要求阶段 6-1 就显著提升准确率。

验收：

- 单元测试能证明 Coordinator 会按风险信号调用正确子 agent。
- annotations 能记录 coordinator trace。
- 现有长跑流程仍能完成，`labeled.txt` 输出格式不变。
- 不把具体评测小说的人名、地点、剧情或口癖写进生产逻辑。

#### 阶段 6-2：独立 Verifier + Arbiter 闭环

目标：优先处理阶段 5 mismatch attribution 中数量最大的 `high_risk_verifier_pass` 和 `high_confidence_wrong`，让 Verifier 不再只是被动复核，而是能与 Labeler 结论形成可裁决的闭环。

内容：

- Verifier 使用独立模型会话和独立 prompt，只负责找反证、轮次冲突、被称呼者误判和证据不足。
- Verifier 输出必须包含 `verdict / counter_evidence_lines / reason / confidence`。
- 当 Labeler 与 Verifier 冲突时，Coordinator 调用 Arbiter。
- Arbiter 只比较结构化证据和反证，不重新长篇阅读全局上下文。
- Verifier false pass 样本要能进入回归测试或至少进入可复查报告。

验收：

- `high_risk_verifier_pass` 数量相对阶段 5-4 基线下降。
- 被 Verifier 否决的样本不会直接写入 `labeled.txt`。
- Arbiter 的裁决原因写入 annotations。

#### 阶段 6-3：独立 Identity Locator / Resolver

目标：把阶段 5-4 的身份后置工具升级为独立上下文的模型子 agent，解决主 agent 上下文有限、误触发和候选解释能力弱的问题。

内容：

- Identity Locator 独立读取有限后文，只负责找候选身份揭示区域。
- Identity Resolver 独立细读候选范围，判断是否同一人物。
- Coordinator 负责决定什么时候触发 identity 流程，避免把代词、口癖、故事内部人物和普通群体送入 identity lookup。
- 保留有界查找参数，例如 `--identity-lookahead-lines`、`--identity-lookahead-rounds`。
- identity 子 agent 输出必须包含候选范围、同一人物判断、推荐 speaker 和证据行。

验收：

- 身份后置样本能稳定给出“在哪些行找到身份、为什么是同一人”。
- `男子 / 咱 / 戏曲故事里的男孩` 这类误触发不回归。
- identity 相关错误相对阶段 5-4 基线下降。

#### 阶段 6-4：Name Normalizer + Character Librarian

目标：让角色显示名归一和轻量角色库更新脱离主 agent 自觉调用，改为 Coordinator 驱动的独立子 agent 或确定性工具链。

内容：

- Character Librarian 维护轻量角色库，只记录当前文本证据支持的 display_name、aliases、summary、evidence_lines、last_seen 和 confidence。
- Name Normalizer 根据 Labeler speaker、证据和角色库候选判断是否应映射到已有 display_name。
- 新增或更新角色库时必须保存证据行。
- 角色库只能辅助归一，不能覆盖原文强证据。
- 对疑似重复角色条目输出冲突建议，必要时交给 Arbiter。

验收：

- annotations 中能看到 record/normalize 的触发原因和结果。
- 同一重要角色显示名更稳定。
- 无名低重要度人物不会被过度合并。

#### 阶段 6-5：整合回归与阶段 6 收敛

目标：把阶段 6 各子 agent 的收益用统一评估证明，决定是否进入更大规模角色记忆或人工复核系统。

内容：

- 固定阶段 5-4 作为对照基线。
- 每个阶段 6 子阶段都运行相同的 evaluate、mismatch-attribution、专有词扫描。
- 报告按错误类型比较变化：
  - `high_risk_verifier_pass`
  - `medium_or_lower_risk_no_verifier`
  - `adjacent_turn_order`
  - `same_line_multiple_dialogues`
  - `high_confidence_wrong`
  - identity / normalization 相关错误
- 记录 token/调用次数/失败率，避免质量提升完全靠不可控成本堆出来。

验收：

- 日志能清楚显示某条标注经过了哪些独立 LLM agent。
- 每个 agent 的输入输出可以单独测试和回放。
- 与阶段 5-4 最终版本相比，身份后置、显示名归一和 Verifier false pass 相关错误有可量化下降。
- 生产逻辑仍不包含具体评测小说的人名、地点、剧情或口癖。

## 14. 测试策略

### 14.1 不调用模型的测试

必须优先覆盖：

- 对话提取。
- 行号保留。
- `labeled.txt` 进度计算。
- batch 获取。
- `submit_labels` 数量校验。
- `read_novel` 越界和裁剪。
- `search_novel` limit。
- fake model 的 tool loop。

### 14.2 调用模型的小样本测试

先跑：

```bash
dialoop novel.txt --batch-size 1 --max-iterations 3
```

人工检查：

- 是否读取了合理上下文。
- 是否把说话人写入正确行。
- 是否没有修改小说。
- 如果不确定，是否继续读取更多上下文。

### 14.3 长跑测试

小样本通过后再跑：

```bash
dialoop novel.txt --batch-size 1
```

长跑测试关注：

- 是否能稳定续跑。
- 是否出现重复标注。
- 是否出现批量数量错位。
- 是否出现模型沉迷解释不提交。
- 本地模型长上下文是否稳定。

## 15. 风险和控制

### 15.1 Ollama tool calling 不稳定

控制：

- 保留 JSON action 协议。
- `--protocol auto` 自动降级。
- 对模型输出做严格 JSON 提取和错误提示。

### 15.2 模型读太多上下文

控制：

- `read_window_limit`。
- 每个 batch 的 `max_tool_steps`。
- 日志记录每轮读取行数。

### 15.3 模型写入数量错位

控制：

- `submit_labels` 必须校验数量。
- 写入前绑定当前 batch id。
- 每次只追加当前 batch。

### 15.4 角色名不统一

v1 接受一定程度不统一，但阶段 5 应把“同一对象显示名稳定”作为质量增强重点之一。

后续控制：

- 增加轻量 alias map，只保存当前小说中有证据支持的等价项。
- 增加角色名归一 agent，把“姓名 / 全称 / 稳定身份词”统一到一个显示名。
- 归一只处理显示名，不直接改变说话人归属；发现归属疑似错误时交给 Verifier/Arbiter。
- 保留人工抽样评估，用错误样本反推归一规则是否过度合并。

### 15.5 普通路人消耗过多推理

控制目标不是给每个路人建立实体，而是在不牺牲重要角色准确率的前提下，使用稳定、可解释的通用身份词。

可接受标签示例：

- `路人`
- `众人`
- `店员`
- `村民`
- `丈夫`
- `商人`
- `非人物发声`

阶段 5 不默认引入 `npc_group`。只有当后续确实需要结构化统计无名群体时，再作为独立输出层讨论，不进入 v1 主标签体系。

## 16. README 后续应调整的内容

README 当前仍描述 OpenCode + Ralph Loop wrapper 路线。后续应更新为：

- OpenCode/openloop 是灵感来源，不是运行依赖。
- 默认使用 Ollama 本地模型。
- 提供最短启动命令。
- 提供远程 OpenAI-compatible 模型配置示例。
- 说明 `labeled.txt` 断点续跑。
- 说明已有 OpenCode wrapper 是实验代码或历史实现。

## 17. 最终结论

本项目最终应做成：

```text
一个独立的 Python 小说对话说话人标注 agent。
```

它的核心不是自研复杂角色记忆系统，也不是打包 OpenCode，而是：

```text
给模型可靠的读小说工具，
让模型自己找上下文，
由 Python 控制进度和写入。
```

这是当前成功经验、DeepSeek 新方案和 V2 失败经验共同指向的最稳妥路线。
