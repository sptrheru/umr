import { existsSync } from "node:fs";
import { link } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";

import { ManagerError } from "../errors";
import {
  ensureDir,
  fileSize,
  pathExists,
  removeEmptyParents,
  removeFileIfExists,
  sha256File,
} from "../fs";
import type { DataPaths } from "../paths";
import { emitInfo } from "../progress";
import type { CommandRunner } from "../shell";
import type {
  ModelDetails,
  ModelManifestMember,
  ModelRecord,
  OperationContext,
  RegistrarAdapter,
  RegistrationHealth,
  RegistrationRecord,
  RegistrationResult,
} from "../types";

function slugifySegment(input: string): string {
  return input
    .trim()
    .replace(/\.gguf$/i, "")
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-{2,}/g, "-");
}

async function tryLink(from: string, to: string): Promise<boolean> {
  try {
    await link(from, to);
    return true;
  } catch {
    return false;
  }
}

function getEntryMember(model: ModelRecord): ModelManifestMember | null {
  return (
    model.manifest.find((member) => member.relPath === model.entryRelPath) ??
    null
  );
}

function assertGGUFEntry(model: ModelRecord): void {
  if (!model.entryFilename.toLowerCase().endsWith(".gguf")) {
    throw new ManagerError(
      "UMR currently supports GGUF models only. Support for other model formats is coming soon.",
      {
        code: "unsupported-model-format",
        exitCode: 2,
      },
    );
  }
}

export class LMStudioRegistrarAdapter implements RegistrarAdapter {
  constructor(
    private readonly runner: CommandRunner,
    private readonly dataPaths: DataPaths,
    private readonly env: Record<string, string | undefined> = process.env,
  ) {}

  client(): string {
    return "lmstudio";
  }

  private resolveModelsDir(): string {
    const override = this.env.UMR_LMSTUDIO_MODELS_DIR?.trim();
    if (override) {
      return path.resolve(override);
    }

    const home = this.env.HOME ?? process.env.HOME ?? homedir();
    const candidates = [
      home ? path.join(home, ".lmstudio", "models") : null,
      home ? path.join(home, ".cache", "lm-studio", "models") : null,
    ].filter(Boolean) as string[];

    for (const candidate of candidates) {
      if (existsSync(candidate)) {
        return candidate;
      }
    }

    throw new ManagerError(
      "LM Studio does not appear to be installed or initialized. Open LM Studio once, or set UMR_LMSTUDIO_MODELS_DIR, then try linking again.",
      {
        code: "lmstudio-models-dir",
        exitCode: 2,
      },
    );
  }

  private async resolveLmsCommand(): Promise<string> {
    if (await this.runner.commandExists("lms")) {
      return "lms";
    }

    const home = this.env.HOME ?? process.env.HOME ?? homedir();
    const candidates = [
      home ? path.join(home, ".cache", "lm-studio", "bin", "lms") : null,
      home ? path.join(home, ".lmstudio", "bin", "lms") : null,
    ].filter(Boolean) as string[];

    for (const candidate of candidates) {
      if (await this.runner.commandExists(candidate)) {
        return candidate;
      }
    }

    throw new ManagerError(
      "LM Studio does not appear to be installed. Install LM Studio and make sure the `lms` CLI is available, then try linking again.",
      {
        code: "missing-lms-cli",
        exitCode: 2,
      },
    );
  }

  private getTargetPath(userRepo: string, model: ModelRecord): string {
    return path.join(this.resolveModelsDir(), userRepo, model.entryFilename);
  }

  private deriveBaseRepoName(model: ModelDetails): string {
    const hfSource = model.sources.find((source) => {
      const repo = source.payload.repo;
      return source.kind === "hf" && typeof repo === "string" && repo.trim();
    });
    if (hfSource) {
      const repo = String(hfSource.payload.repo).trim();
      const segment = repo.split("/").at(-1);
      if (segment) {
        return segment;
      }
    }

    const pathSource = model.sources.find((source) => {
      const originalPath = source.payload.originalPath;
      return (
        source.kind === "path" &&
        typeof originalPath === "string" &&
        originalPath.trim()
      );
    });
    if (pathSource) {
      const originalPath = String(pathSource.payload.originalPath).trim();
      const basename = path.basename(originalPath, path.extname(originalPath));
      const slug = slugifySegment(basename);
      if (slug) {
        return slug;
      }
    }

    return slugifySegment(model.name) || model.ref;
  }

  private getUserRepoCandidates(model: ModelDetails): string[] {
    const base = this.deriveBaseRepoName(model);
    return [
      path.posix.join("umr", base),
      path.posix.join("umr", `${base}-${model.ref.slice(2, 10)}`),
    ];
  }

  private getExistingManagedTarget(
    model: ModelDetails,
  ): { userRepo: string; targetPath: string } | null {
    const registration = model.registrations.find(
      (entry) => entry.client === this.client(),
    );
    if (!registration) {
      return null;
    }

    const userRepo = registration.state.userRepo;
    const targetPath = registration.state.targetPath;
    if (
      typeof userRepo !== "string" ||
      typeof targetPath !== "string" ||
      !userRepo.startsWith("umr/")
    ) {
      return null;
    }

    return { userRepo, targetPath };
  }

  private getHistoricalManagedRepos(model: ModelDetails): Set<string> {
    const repos = new Set<string>();

    for (const source of model.sources) {
      if (source.kind !== "lmstudio-repo") {
        continue;
      }

      const userRepo = source.payload.userRepo;
      if (typeof userRepo === "string" && userRepo.startsWith("umr/")) {
        repos.add(userRepo);
      }
    }

    return repos;
  }

  private async chooseFreshManagedRepo(
    model: ModelDetails,
    historicalRepos: Set<string>,
  ): Promise<string> {
    const base = this.deriveBaseRepoName(model);
    const stem = `${base}-${model.ref.slice(2, 10)}`;

    let attempt = 1;
    while (true) {
      const suffix = attempt === 1 ? stem : `${stem}-${attempt}`;
      const userRepo = path.posix.join("umr", suffix);
      const targetPath = this.getTargetPath(userRepo, model);
      if (!historicalRepos.has(userRepo) && !(await pathExists(targetPath))) {
        return userRepo;
      }
      attempt += 1;
    }
  }

  private async chooseImportTarget(
    model: ModelDetails,
  ): Promise<{ userRepo: string; targetPath: string }> {
    const entryMember = getEntryMember(model);
    if (!entryMember) {
      throw new ManagerError(`Missing entry manifest member for ${model.ref}`, {
        code: "missing-entry-member",
        exitCode: 1,
      });
    }

    const existingManagedTarget = this.getExistingManagedTarget(model);
    if (existingManagedTarget) {
      return existingManagedTarget;
    }

    const historicalRepos = this.getHistoricalManagedRepos(model);
    const candidates = this.getUserRepoCandidates(model);
    for (const userRepo of candidates) {
      const targetPath = this.getTargetPath(userRepo, model);
      if (!(await pathExists(targetPath))) {
        if (historicalRepos.has(userRepo)) {
          continue;
        }
        return { userRepo, targetPath };
      }

      const actualSize = await fileSize(targetPath);
      if (actualSize !== entryMember.sizeBytes) {
        continue;
      }

      const actualHash = await sha256File(targetPath);
      if (actualHash === entryMember.sha256) {
        return { userRepo, targetPath };
      }
    }

    const fallbackRepo = await this.chooseFreshManagedRepo(
      model,
      historicalRepos,
    );
    return {
      userRepo: fallbackRepo,
      targetPath: this.getTargetPath(fallbackRepo, model),
    };
  }

  private async prepareImportSource(model: ModelRecord): Promise<string> {
    const importDir = path.join(
      this.dataPaths.adaptersTmpDir,
      "lmstudio",
      model.ref,
    );
    const importPath = path.join(importDir, model.entryFilename);
    await ensureDir(importDir);
    await removeFileIfExists(importPath);

    if (!(await tryLink(model.entryPath, importPath))) {
      await Bun.write(importPath, Bun.file(model.entryPath));
    }

    return importPath;
  }

  async register(
    model: ModelDetails,
    context?: OperationContext,
  ): Promise<RegistrationResult> {
    assertGGUFEntry(model);
    await emitInfo(
      context?.reporter,
      `Locating the LM Studio models directory for ${model.name}`,
    );
    const { userRepo, targetPath } = await this.chooseImportTarget(model);
    const targetDir = path.dirname(targetPath);
    const lms = await this.resolveLmsCommand();
    const importSource = await this.prepareImportSource(model);

    try {
      await ensureDir(targetDir);
      if (await pathExists(targetPath)) {
        await emitInfo(
          context?.reporter,
          `Refreshing existing LM Studio managed artifact at ${targetPath}`,
        );
        await removeFileIfExists(targetPath);
      } else {
        await emitInfo(
          context?.reporter,
          `Importing ${model.name} into LM Studio`,
        );
      }

      const attempts = [
        { flag: "--hard-link", method: "hardlink" },
        { flag: "--symbolic-link", method: "symlink" },
        { flag: "--copy", method: "copy" },
      ] as const;

      let lastError = "";
      for (const attempt of attempts) {
        const result = await this.runner.run(
          lms,
          [
            "import",
            importSource,
            "--user-repo",
            userRepo,
            "--yes",
            attempt.flag,
          ],
          { env: this.env },
        );
        if (result.exitCode === 0) {
          return {
            clientRef: targetPath,
            state: {
              targetPath,
              method: attempt.method,
              userRepo,
            },
          };
        }

        lastError =
          result.stderr || result.stdout || "unknown LM Studio import error";
        await removeFileIfExists(targetPath);
      }

      throw new ManagerError(`LM Studio import failed: ${lastError}`, {
        code: "lmstudio-import-failed",
        exitCode: 1,
      });
    } finally {
      await removeFileIfExists(importSource);
      await removeEmptyParents(
        importSource,
        path.join(this.dataPaths.adaptersTmpDir, "lmstudio"),
      );
    }
  }

  async unregister(
    _model: ModelDetails,
    registration: RegistrationRecord,
    context?: OperationContext,
  ): Promise<void> {
    const targetPath = registration.state.targetPath;
    if (typeof targetPath !== "string") {
      return;
    }

    await emitInfo(
      context?.reporter,
      `Removing LM Studio managed artifact ${targetPath}`,
    );
    await removeFileIfExists(targetPath);
    const modelsDir = this.resolveModelsDir();
    await removeEmptyParents(targetPath, path.join(modelsDir, "umr"));
  }

  async check(
    model: ModelDetails,
    registration: RegistrationRecord,
    _context?: OperationContext,
  ): Promise<RegistrationHealth> {
    const entryMember = getEntryMember(model);
    if (!entryMember) {
      return { ok: false, issues: ["missing-entry-member"] };
    }

    const targetPath = registration.state.targetPath;
    if (typeof targetPath !== "string") {
      return { ok: false, issues: ["missing-target-path"] };
    }

    if (!(await pathExists(targetPath))) {
      return { ok: false, issues: ["missing-target-path"] };
    }

    const actualSize = await fileSize(targetPath);
    if (entryMember.sizeBytes !== actualSize) {
      return { ok: false, issues: ["size-mismatch"] };
    }

    const targetHash = await sha256File(targetPath);
    if (targetHash !== entryMember.sha256) {
      return { ok: false, issues: ["hash-mismatch"] };
    }

    return { ok: true, issues: [] };
  }
}
