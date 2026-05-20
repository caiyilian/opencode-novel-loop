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
| `--read-window-limit` | `300` | 单次 read_novel 最大行数 |
| `--search-limit` | `20` | 单次 search_novel 最大结果数 |
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

### 11.4 中断恢复

用户 Ctrl+C：

- 已写入的 `labeled.txt` 保留。
- 下次运行从已写入行数继续。
- 不需要额外恢复步骤。

### 11.5 API 失败

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

在 v1 agent loop 稳定后，再逐步引入：

- `unknown` / `mystery` / `npc_group`。
- review queue。
- evidence 记录。
- 角色名归一。
- prompt 长度报告。
- 人工抽样评估。

这些来自 V2，但不能阻塞 v1。

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

v1 接受一定程度不统一。

后续控制：

- 增加 alias map。
- 增加角色名归一工具。
- 增加人工 review。

### 15.5 普通路人消耗过多推理

v1 prompt 先允许输出通用称呼：

- `路人`
- `众人`
- `店员`
- `商人`
- `未知`

后续再引入 `npc_group`。

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
