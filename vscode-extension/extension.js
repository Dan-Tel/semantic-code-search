const vscode = require("vscode");
const path = require("path");
const fs = require("fs");
const { execFile } = require("child_process");

let running = null;

function runBackend(python, root, args, ask, output, token) {
  return new Promise((resolve, reject) => {
    let cancellation;

    running = execFile(
      python,
      ["-m", "scripts.search_for_vscode", ...args],
      {
        cwd: root,
        timeout: ask ? 360000 : 180000,
        maxBuffer: 16 * 1024 * 1024,
        env: {
          ...process.env,
          PYTHONIOENCODING: "utf-8",
          LANGSMITH_TRACING: "false",
          LANGCHAIN_TRACING_V2: "false",
        },
      },
      (error, stdout, stderr) => {
        running = null;
        cancellation?.dispose();

        if (stderr) {
          output.appendLine(stderr);
        }

        if (token.isCancellationRequested) {
          resolve(null);
          return;
        }

        if (error) {
          reject(new Error(stderr.trim() || error.message));
          return;
        }

        try {
          resolve(JSON.parse(stdout));
        } catch {
          reject(new Error("Backend returned invalid JSON."));
        }
      },
    );

    cancellation = token.onCancellationRequested(() => {
      running?.kill();
    });

    if (token.isCancellationRequested) {
      running?.kill();
    }
  });
}

async function showAnswer(query, payload, repository) {
  const sources = payload.results.map((result, index) => {
    const relativePath = path.relative(repository, result.path);

    return (
      `[${index + 1}] ${result.name} — ` +
      `${relativePath}:${result.start_line}-${result.end_line}`
    );
  });

  const content = [
    "# Repository answer",
    "",
    `Question: ${query}`,
    "",
    `Search phrase: ${payload.search_query}`,
    "",
    payload.answer || "No answer generated.",
    "",
    "## Retrieved sources",
    "",
    ...sources,
    "",
    ...(payload.warnings || []).map((warning) => `Warning: ${warning}`),
  ].join("\n");

  const document = await vscode.workspace.openTextDocument({
    language: "markdown",
    content,
  });

  await vscode.window.showTextDocument(document, {
    preview: false,
  });
}

async function showResults(payload, repository, beside) {
  const items = payload.results.map((result, index) => ({
    label: `$(symbol-method) ${index + 1}. ${result.name}`,
    description: `cosine ${result.score.toFixed(4)}`,
    detail:
      `${path.relative(repository, result.path)}:` +
      `${result.start_line}-${result.end_line}`,
    result,
  }));

  const selected = await vscode.window.showQuickPick(items, {
    title: `Results — ${payload.function_count} indexed functions`,
    placeHolder: "Select a function to open its source",
    matchOnDetail: true,
    ignoreFocusOut: true,
  });

  if (!selected) return;

  const filePath = path.resolve(selected.result.path);
  const relativePath = path.relative(repository, filePath);

  if (
    relativePath === ".." ||
    relativePath.startsWith(".." + path.sep) ||
    path.isAbsolute(relativePath)
  ) {
    throw new Error("Result path is outside the searched folder.");
  }

  const document = await vscode.workspace.openTextDocument(
    vscode.Uri.file(filePath),
  );

  const editor = await vscode.window.showTextDocument(document, {
    preview: false,
    ...(beside ? { viewColumn: vscode.ViewColumn.Beside } : {}),
  });

  const start = Math.max(
    0,
    Math.min(document.lineCount - 1, selected.result.start_line - 1),
  );

  const end = Math.max(
    start,
    Math.min(document.lineCount - 1, selected.result.end_line - 1),
  );

  const range = new vscode.Range(
    start,
    0,
    end,
    document.lineAt(end).text.length,
  );

  editor.selection = new vscode.Selection(range.start, range.end);

  editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
}

function activate(context) {
  const output = vscode.window.createOutputChannel("Semantic Code Search");

  context.subscriptions.push(output);

  async function run(ask) {
    if (!vscode.workspace.isTrusted) {
      vscode.window.showErrorMessage(
        "Trust this workspace before running local Python.",
      );
      return;
    }

    if (running) {
      vscode.window.showInformationMessage("A request is already running.");
      return;
    }

    const folders = vscode.workspace.workspaceFolders;

    if (!folders?.length) {
      vscode.window.showErrorMessage("Open a local TypeScript project first.");
      return;
    }

    const folder =
      folders.length === 1
        ? folders[0]
        : await vscode.window.showWorkspaceFolderPick();

    if (!folder) return;

    if (folder.uri.scheme !== "file") {
      vscode.window.showErrorMessage(
        "This extension supports local folders only.",
      );
      return;
    }

    const query = await vscode.window.showInputBox({
      title: ask ? "Ask Code with AI" : "Search Code with AI",
      prompt: ask
        ? "Ask a question about the repository"
        : "Describe the function you need",
      placeHolder: ask
        ? "Where is the authentication token added?"
        : "add authentication token to request headers",
      ignoreFocusOut: true,
      validateInput: (value) =>
        value.trim() ? null : "Enter a question or query",
    });

    if (!query?.trim()) return;

    if (running) {
      vscode.window.showInformationMessage("A request is already running.");
      return;
    }

    const config = vscode.workspace.getConfiguration("semanticCodeSearch");

    const root =
      config.get("backendPath") || path.resolve(context.extensionPath, "..");

    const python =
      config.get("pythonPath") ||
      path.join(
        root,
        ".venv",
        process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
      );

    const script = path.join(root, "scripts", "search_for_vscode.py");

    if (!fs.existsSync(python) || !fs.existsSync(script)) {
      vscode.window.showErrorMessage(
        "Backend not found. Check Backend Path and Python Path " +
          "in Semantic Code Search settings.",
      );
      return;
    }

    const args = [
      "--repository",
      folder.uri.fsPath,
      "--query",
      query.trim(),
      "--top-k",
      String(config.get("topK", 5)),
    ];

    if (ask) {
      args.push(
        "--ask",
        "--ollama-model",
        config.get("ollamaModel", "qwen3:8b"),
      );

      if (!config.get("rewriteQuery", true)) {
        args.push("--no-rewrite");
      }
    }

    try {
      const payload = await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: ask
            ? "Asking local Qwen and searching your code…"
            : "Searching with your encoder…",
          cancellable: true,
        },
        (_, token) => runBackend(python, root, args, ask, output, token),
      );

      if (!payload) return;

      if (!payload.results.length) {
        vscode.window.showInformationMessage(
          "No supported functions found in .ts files.",
        );
        return;
      }

      output.appendLine(
        `Query: ${query}\n` +
          `Functions: ${payload.function_count} | ` +
          `Device: ${payload.device} | Epoch: ${payload.epoch}`,
      );

      for (const warning of payload.warnings || []) {
        output.appendLine(warning);
      }

      if (ask) {
        await showAnswer(query, payload, folder.uri.fsPath);
      }

      await showResults(payload, folder.uri.fsPath, ask);
    } catch (error) {
      output.appendLine(String(error));
      output.show(true);

      vscode.window.showErrorMessage(
        "Request failed. See Output → Semantic Code Search.",
      );
    }
  }

  context.subscriptions.push(
    vscode.commands.registerCommand("semanticCodeSearch.search", () =>
      run(false),
    ),
    vscode.commands.registerCommand("semanticCodeSearch.ask", () => run(true)),
  );
}

function deactivate() {
  running?.kill();
}

module.exports = { activate, deactivate };
