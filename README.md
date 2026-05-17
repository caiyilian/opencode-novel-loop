# opencode-novel-loop

> 结合 [OpenCode](https://github.com/anomalyco/opencode) + [Ralph Loop](https://github.com/tylerhaaven/openloop) 的零配置小说对话说话人标注工具。

---

## 背景

我有一批小说文本需要标注——把文中所有 `「对话内容」` 标注上对应的说话角色名。这件事交给 AI 做非常合适：读上下文、判断谁在说话、写入结果。

我找到了一套能完美完成这个任务的方案：

1. 在服务器上安装 [OpenCode](https://github.com/anomalyco/opencode)
2. 安装 [openloop](https://github.com/tylerhaaven/openloop) 插件（ralph-loop）
3. 一条提示词：`/ralph-loop 运行 python get_dialogue.py 然后按照程序输出要求完成任务`

然后它就自己跑起来了。提取对话 → 分析上下文 → 判断说话人 → 写入 labeled.txt → 继续下一个。一直循环。标注了 400 多条对话还在继续，准确率很高。

**ralph-loop 的核心设计**非常巧妙：每次 session 空闲时自动检查 assistant 输出有没有 `<promise>DONE</promise>`，没有就注入一条 continuation prompt，让 AI 继续干活。不需要人工介入，不需要写复杂的调度逻辑，就是单纯的「没干完就继续」。

## 痛点

但这套方案对别人来说门槛太高了：

- 要先装 OpenCode（虽然一条命令就能装，但很多人不知道）
- 要手动安装 openloop 插件并配置到正确的目录
- 要配置模型和 API key（即使 OpenCode 内置了免费 Qwen Plus，默认也不一定选中）
- 要知道 `/ralph-loop` 这个命令怎么用，提示词怎么写

**我的目标**：别人拿到这个项目后，不需要了解 OpenCode 插件系统，不需要配置模型 API key，不需要学 ralph-loop 的用法，**一行命令就能开始标注**。

## 项目目标

做一个独立 CLI 工具，结合 OpenCode（底层 AI 引擎）和 Ralph Loop（自动循环机制），专门用于小说对话说话人标注：

- **零配置** — 无需 API key，默认使用 OpenCode 内置的免费 Qwen Plus（`alibaba/qwen-plus`）
- **一键启动** — `./dialoop novel.txt` 即可开始标注
- **继承 ralph-loop 思路** — 不重复造轮子，把 openloop 的核心循环逻辑内嵌到工具中
- **平台无关** — setup 脚本自动检测环境、安装 OpenCode、初始化配置

## 技术思路

核心依赖：

| 组件 | 来源 | 角色 |
|------|------|------|
| OpenCode | [anomalyco/opencode](https://github.com/anomalyco/opencode) | AI 引擎，提供模型推理和工具调用 |
| Ralph Loop | [tylerhaaven/openloop](https://github.com/tylerhaaven/openloop) | 借鉴其循环机制（session idle → 检查 DONE → 继续） |
| `get_dialogue.py` | 本项目 | 从 novel.txt 提取 `「」` 格式对话，按已标注进度取下一句 |
| `write_label.py` | 本项目 | 将说话角色名追加写入 labeled.txt |

关键设计：

- 用 OpenCode 非交互模式（`opencode run --dangerously-skip-permissions`）代替 TUI + 插件模式
- 用 Python 主循环脚本实现 ralph-loop 的空闲检测和 continuation 逻辑
- 预设 OpenCode 配置（模型、权限），用户无需手动编辑任何配置文件

## 当前状态

🚧 项目刚启动，还未开始开发。

目前仓库中只有两个参考项目的源码：

- `opencode-dev/` — OpenCode 官方源码（v1.15.3）
- `openloop-main/` — openloop（ralph-loop 插件）源码

以及已有的标注脚本：

- `get_dialogue.py` — 对话提取脚本
- `write_label.py` — 标注写入脚本

## 路线图

- [ ] 搭建项目骨架（CLI 入口、setup 脚本、配置模板）
- [ ] 实现主循环逻辑（替代 openloop 插件的 session idle 检测 + continuation）
- [ ] 集成 OpenCode 自动安装与环境检测
- [ ] 内嵌模型配置（默认 alibaba/qwen-plus，无需额外 API key）
- [ ] 测试端到端流程：`novel.txt` → `labeled.txt`
- [ ] 编写使用文档和贡献指南

## 许可

MIT License
