import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { Plugin } from "@opencode-ai/plugin";
import { tool } from "@opencode-ai/plugin/tool";
import type { Event, Message, Part, TextPart } from "@opencode-ai/sdk";

type RalphState = {
  active: boolean;
  iteration: number;
  maxIterations: number;
  sessionId?: string;
  prompt?: string;
};

type SessionTranscriptEntry = {
  info: Message;
  parts: Part[];
};

const STATE_FILE = join(".opencode", "ralph-loop.local.md");
const DEFAULT_MAX_ITERATIONS = 100;
const COMPLETION_PROMISE = /<promise>\s*DONE\s*<\/promise>/is;

function pluginRoot(): string {
  return dirname(dirname(fileURLToPath(import.meta.url)));
}

function opencodeConfigDir(): string | undefined {
  if (process.env.OPENCODE_CONFIG_DIR) {
    return process.env.OPENCODE_CONFIG_DIR;
  }

  if (process.env.HOME) {
    return join(process.env.HOME, ".config", "opencode");
  }

  return undefined;
}

function preferredConfigSubdir(baseDir: string, plural: string, singular: string): string {
  const pluralDir = join(baseDir, plural);
  if (existsSync(pluralDir)) {
    return pluralDir;
  }

  const singularDir = join(baseDir, singular);
  if (existsSync(singularDir)) {
    return singularDir;
  }

  return pluralDir;
}

function copyMissingDirectory(sourceDir: string, targetDir: string): void {
  if (!existsSync(sourceDir) || existsSync(targetDir)) {
    return;
  }

  mkdirSync(dirname(targetDir), { recursive: true });
  cpSync(sourceDir, targetDir, { recursive: true });
}

function copyMissingFile(sourceFile: string, targetFile: string): void {
  if (!existsSync(sourceFile) || existsSync(targetFile)) {
    return;
  }

  mkdirSync(dirname(targetFile), { recursive: true });
  cpSync(sourceFile, targetFile);
}

function installAssets(): void {
  const configDir = opencodeConfigDir();
  if (!configDir) {
    return;
  }

  const root = pluginRoot();
  const skillTargetRoot = preferredConfigSubdir(configDir, "skills", "skill");
  const commandTargetRoot = preferredConfigSubdir(configDir, "commands", "command");

  for (const skillName of ["ralph-loop", "cancel-ralph", "ralph-help"]) {
    copyMissingDirectory(join(root, "skills", skillName), join(skillTargetRoot, skillName));
  }

  for (const commandName of ["ralph-loop.md", "cancel-ralph.md", "ralph-help.md"]) {
    copyMissingFile(join(root, "commands", commandName), join(commandTargetRoot, commandName));
  }
}

function statePath(directory: string): string {
  return join(directory, STATE_FILE);
}

function parseState(markdown: string): RalphState {
  const match = markdown.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) {
    return {
      active: false,
      iteration: 0,
      maxIterations: DEFAULT_MAX_ITERATIONS,
    };
  }

  const state: RalphState = {
    active: false,
    iteration: 0,
    maxIterations: DEFAULT_MAX_ITERATIONS,
  };

  for (const rawLine of match[1].split(/\r?\n/)) {
    const [rawKey, ...rawValue] = rawLine.split(":");
    const key = rawKey.trim();
    const value = rawValue.join(":").trim();

    if (key === "active") {
      state.active = value === "true";
    }

    if (key === "iteration") {
      state.iteration = Number.parseInt(value, 10) || 0;
    }

    if (key === "maxIterations") {
      state.maxIterations = Number.parseInt(value, 10) || DEFAULT_MAX_ITERATIONS;
    }

    if (key === "sessionId") {
      state.sessionId = value || undefined;
    }
  }

  const prompt = markdown.slice(match[0].length).trim();
  if (prompt) {
    state.prompt = prompt;
  }

  return state;
}

function serializeState(state: RalphState): string {
  const lines = [
    "---",
    `active: ${state.active}`,
    `iteration: ${state.iteration}`,
    `maxIterations: ${state.maxIterations}`,
  ];

  if (state.sessionId) {
    lines.push(`sessionId: ${state.sessionId}`);
  }

  lines.push("---");

  if (state.prompt) {
    lines.push("", state.prompt);
  }

  return `${lines.join("\n")}\n`;
}

function readState(directory: string): RalphState {
  const filePath = statePath(directory);
  if (!existsSync(filePath)) {
    return {
      active: false,
      iteration: 0,
      maxIterations: DEFAULT_MAX_ITERATIONS,
    };
  }

  return parseState(readFileSync(filePath, "utf8"));
}

function writeState(directory: string, state: RalphState): void {
  const filePath = statePath(directory);
  mkdirSync(dirname(filePath), { recursive: true });
  writeFileSync(filePath, serializeState(state), "utf8");
}

function clearState(directory: string): void {
  const filePath = statePath(directory);
  if (existsSync(filePath)) {
    rmSync(filePath);
  }
}

function latestAssistantText(entries: SessionTranscriptEntry[]): string {
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const entry = entries[index];
    if (entry.info.role !== "assistant") {
      continue;
    }

    return entry.parts
      .filter((part): part is TextPart => part.type === "text")
      .map((part) => part.text)
      .join("\n");
  }

  return "";
}

async function sessionIsComplete(client: Parameters<Plugin>[0]["client"], directory: string, sessionID: string): Promise<boolean> {
  const response = await client.session.messages({
    path: {
      id: sessionID,
    },
    query: {
      directory,
      limit: 20,
    },
  });

  if (response.error) {
    return false;
  }

  const transcript = (response.data ?? []) as SessionTranscriptEntry[];
  return COMPLETION_PROMISE.test(latestAssistantText(transcript));
}

function continuationPrompt(state: RalphState): string {
  return [
    `[RALPH LOOP ${state.iteration}/${state.maxIterations}]`,
    "",
    "Continue from where you left off.",
    "Keep making concrete progress on the original task.",
    "Only output `<promise>DONE</promise>` when the task is completely and verifiably finished.",
    "If you are blocked, explain the blocker instead of claiming completion.",
    "",
    "Original task:",
    state.prompt ?? "(no task recorded)",
  ].join("\n");
}

function isSessionIdle(event: Event): event is Extract<Event, { type: "session.idle" }> {
  return event.type === "session.idle";
}

function isSessionDeleted(event: Event): event is Extract<Event, { type: "session.deleted" }> {
  return event.type === "session.deleted";
}

function isCommandExecuted(event: Event): event is Extract<Event, { type: "command.executed" }> {
  return event.type === "command.executed";
}

const RalphLoopPlugin: Plugin = async ({ client, directory }) => {
  installAssets();

  return {
    tool: {
      "ralph-loop": tool({
        description: "Start Ralph Loop and keep continuing the current session until the task is done.",
        args: {
          task: tool.schema.string().min(1),
          maxIterations: tool.schema.number().int().positive().optional(),
        },
        async execute({ task, maxIterations }, context) {
          writeState(context.directory, {
            active: true,
            iteration: 0,
            maxIterations: maxIterations ?? DEFAULT_MAX_ITERATIONS,
            sessionId: context.sessionID,
            prompt: task,
          });

          return [
            `Ralph Loop started for this session (${maxIterations ?? DEFAULT_MAX_ITERATIONS} max iterations).`,
            "",
            `Task: ${task}`,
            "",
            "When the work is fully complete, output `<promise>DONE</promise>`.",
            "Use `/cancel-ralph` to stop early.",
          ].join("\n");
        },
      }),
      "cancel-ralph": tool({
        description: "Cancel the active Ralph Loop for the current project directory.",
        args: {},
        async execute(_args, context) {
          const state = readState(context.directory);
          if (!state.active) {
            return "No active Ralph Loop to cancel.";
          }

          clearState(context.directory);
          return `Ralph Loop cancelled after ${state.iteration} iteration(s).`;
        },
      }),
      "ralph-help": tool({
        description: "Show Ralph Loop usage and commands.",
        args: {},
        async execute() {
          return [
            "Ralph Loop commands:",
            "- `/ralph-loop <task>` starts the loop",
            "- `/cancel-ralph` stops the active loop",
            "- `/ralph-help` shows this help",
            "",
            "The loop ends only when the assistant truthfully emits `<promise>DONE</promise>`.",
          ].join("\n");
        },
      }),
    },
    event: async ({ event }) => {
      if (isCommandExecuted(event)) {
        if (event.properties.name === "ralph-loop") {
          const current = readState(directory);
          if (current.active && !current.sessionId) {
            writeState(directory, {
              ...current,
              sessionId: event.properties.sessionID,
            });
          }
        }

        if (event.properties.name === "cancel-ralph") {
          clearState(directory);
        }

        return;
      }

      if (isSessionDeleted(event)) {
        const current = readState(directory);
        if (current.sessionId && current.sessionId === event.properties.info.id) {
          clearState(directory);
        }
        return;
      }

      if (!isSessionIdle(event)) {
        return;
      }

      const current = readState(directory);
      if (!current.active) {
        return;
      }

      if (current.sessionId && current.sessionId !== event.properties.sessionID) {
        return;
      }

      if (await sessionIsComplete(client, directory, event.properties.sessionID)) {
        clearState(directory);
        return;
      }

      if (current.iteration >= current.maxIterations) {
        clearState(directory);
        return;
      }

      const nextState: RalphState = {
        ...current,
        iteration: current.iteration + 1,
        sessionId: event.properties.sessionID,
      };
      writeState(directory, nextState);

      await client.session.promptAsync({
        path: {
          id: event.properties.sessionID,
        },
        query: {
          directory,
        },
        body: {
          parts: [
            {
              type: "text",
              text: continuationPrompt(nextState),
            },
          ],
        },
      });
    },
  };
};

export default RalphLoopPlugin;
