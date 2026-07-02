import { constants as fsConstants } from "node:fs";
import { access } from "node:fs/promises";
import path from "node:path";

import { ManagerError } from "./errors";

export interface CommandResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

export interface CommandRunOptions {
  cwd?: string;
  env?: Record<string, string | undefined>;
}

export interface CommandStreamOptions extends CommandRunOptions {
  onStdoutChunk?: (chunk: string) => void | Promise<void>;
  onStderrChunk?: (chunk: string) => void | Promise<void>;
  stdio?: "pipe" | "inherit";
}

export interface CommandRunner {
  run(
    command: string,
    args?: string[],
    options?: CommandRunOptions,
  ): Promise<CommandResult>;
  runStreaming(
    command: string,
    args?: string[],
    options?: CommandStreamOptions,
  ): Promise<CommandResult>;
  commandExists(command: string): Promise<boolean>;
}

export class BunCommandRunner implements CommandRunner {
  private async consumeStream(
    stream: ReadableStream<Uint8Array> | null | undefined,
    onChunk?: (chunk: string) => void | Promise<void>,
  ): Promise<string> {
    if (!stream) {
      return "";
    }

    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let output = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      const chunk = decoder.decode(value, { stream: true });
      if (!chunk) {
        continue;
      }

      output += chunk;
      await onChunk?.(chunk);
    }

    const trailing = decoder.decode();
    if (trailing) {
      output += trailing;
      await onChunk?.(trailing);
    }

    return output;
  }

  async run(
    command: string,
    args: string[] = [],
    options?: CommandRunOptions,
  ): Promise<CommandResult> {
    return this.runStreaming(command, args, options);
  }

  async runStreaming(
    command: string,
    args: string[] = [],
    options?: CommandStreamOptions,
  ): Promise<CommandResult> {
    if (options?.stdio === "inherit") {
      const proc = Bun.spawn([command, ...args], {
        cwd: options?.cwd,
        env: {
          ...process.env,
          ...options?.env,
        },
        stdin: "inherit",
        stderr: "inherit",
        stdout: "inherit",
      });

      const exitCode = await proc.exited;
      return {
        stdout: "",
        stderr: "",
        exitCode,
      };
    }

    const proc = Bun.spawn([command, ...args], {
      cwd: options?.cwd,
      env: {
        ...process.env,
        ...options?.env,
      },
      stderr: "pipe",
      stdout: "pipe",
    });

    const [stdout, stderr, exitCode] = await Promise.all([
      this.consumeStream(proc.stdout, options?.onStdoutChunk),
      this.consumeStream(proc.stderr, options?.onStderrChunk),
      proc.exited,
    ]);

    return { stdout, stderr, exitCode };
  }

  async commandExists(command: string): Promise<boolean> {
    const extensions =
      process.platform === "win32"
        ? (process.env.PATHEXT ?? ".EXE")
            .split(";")
            .map((e) => e.toLowerCase())
        : [""];

    if (command.includes(path.sep)) {
      for (const ext of extensions) {
        const candidate =
          process.platform === "win32" && path.extname(command) === ""
            ? command + ext
            : command;
        try {
          await access(candidate);
          return true;
        } catch {}
      }
      return false;
    }

    const pathEnv = process.env.PATH ?? "";
    for (const segment of pathEnv.split(path.delimiter)) {
      for (const ext of extensions) {
        const candidate =
          process.platform === "win32"
            ? path.join(segment, command + ext)
            : path.join(segment, command);
        try {
          await access(candidate);
          return true;
        } catch {}
      }
    }

    return false;
  }
}

export async function runOrThrow(
  runner: CommandRunner,
  command: string,
  args: string[],
  options?: CommandRunOptions,
): Promise<CommandResult> {
  const result = await runner.run(command, args, options);
  if (result.exitCode !== 0) {
    throw new ManagerError(
      `Command failed: ${command} ${args.join(" ")}`.trim(),
      {
        code: "command-failed",
        cause: result.stderr || result.stdout,
        exitCode: 1,
      },
    );
  }

  return result;
}
