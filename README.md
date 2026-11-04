# BORROW - Brain On Remote, Run On Workstation

Autonomous terminal coding agent (**Claude Code & agy style**) powered by a self-hosted open coder model (**Qwen2.5-Coder / Qwen3-Coder**).
The model runs for free on Kaggle's dual Tesla T4 GPUs and connects directly to your local workstation through an encrypted Cloudflare quick tunnel.

No local GPU, no API subscription fees, and no closed-source vendor lock-in.

```
 Your workstation (terminal)            Internet                     Kaggle notebook (2x T4 GPUs)
┌────────────────────────┐   HTTPS + API key   ┌────────────────┐    ┌───────────────────────────────┐
│ borrow                 │ ──────────────────► │ Cloudflare     │───►│ llama-server (llama.cpp, CUDA)│
│ (Autonomous Agent CLI) │ ◄────────────────── │ quick tunnel   │◄───│ Qwen2.5-Coder-32B / 30B MoE   │
└────────────────────────┘   streamed tokens   └────────────────┘    └───────────────────────────────┘
```

## Contents

- [What you get](#what-you-get)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Using the CLI](#using-the-cli)
- [Autonomous Agent Workflows](#autonomous-agent-workflows)
- [Agent Skills](#agent-skills)
- [Talking to the model](#talking-to-the-model)
- [Commands](#commands)
- [Attaching files](#attaching-files)
- [One-shot mode](#one-shot-mode)
- [Automated Testing & QA](#automated-testing--qa)
- [Settings and saved data](#settings-and-saved-data)
- [Choosing a model](#choosing-a-model)
- [Limits and honest expectations](#limits-and-honest-expectations)
- [Security and privacy](#security-and-privacy)
- [Troubleshooting](#troubleshooting)
- [Files and log locations](#files-and-log-locations)

## What you get

| File | Runs on | Purpose |
|---|---|---|
| `qwen_serv.py` | Kaggle notebook | Prebuilt CUDA `llama-server`, fast Hugging Face model downloads, starts OpenAI-compatible API server, and opens Cloudflare tunnel |
| `borrow.py` + `pyproject.toml` | Your workstation | The `borrow` CLI: Autonomous agent loop, tool execution (read/edit/write/search/run/load_skill), agent skills system, interactive colored diffs, live streaming spinners, command menu, and chat memory |

Observed performance on Kaggle 2x Tesla T4:
- **Qwen3-Coder-30B-A3B (MoE):** ~58–68 tokens/second (3.3B active parameters per token).
- **Qwen2.5-Coder-32B-Instruct (Dense):** ~14–18 tokens/second (highest accuracy on multi-file coding and subtle bugs).

## Requirements

**Kaggle side**
- A Kaggle account with a **verified phone number** (needed for Internet access).
- In the notebook's right sidebar under **Session options**:
  - **Accelerator: GPU T4 x2**
  - **Internet: On**

**Your computer**
- Python 3.8 or newer.
- `prompt_toolkit` powers the `/` menu (automatically installed by `pipx` or `pip`).
- Optional: `git` (for `/diff`, `/revert`, `/commit`).

## Quick start

### 1. Start the model on Kaggle

1. Create a new Kaggle notebook and set **Accelerator: GPU T4 x2** and **Internet: On**.
2. Paste the **entire** contents of `qwen_serv.py` into a single code cell and click **Run**.
3. Setup fetches prebuilt CUDA-accelerated `llama-server` binaries and runs high-speed downloads, completing in **~3–5 minutes**.
   *(Tip: Mounting the GGUF model as a Kaggle Dataset finishes setup in **~1–2 minutes**).*
4. Once ready, the cell prints a connection code:

   ```
   https://some-random-words.trycloudflare.com YOUR_KEY
   ```

5. **Leave the cell running.** Stopping the cell shuts down the server.

### 2. Install `borrow` on your computer

In the project directory, run:

```bash
pipx install --editable .     # Recommended: isolated, puts `borrow` on your PATH, auto-syncs edits
```

Or with `pip`:

```bash
pip install -e .
```

Verify the installation:

```bash
borrow --version
```

### 3. Connect and code

Launch the interactive CLI:

```bash
borrow
```

Paste the connection code (or type `/connect <url> <key>`):

```
https://some-random-words.trycloudflare.com YOUR_KEY
```

The CLI saves your credentials to `~/.borrow.json`, verifies the connection, and reports `saved. server reachable.`

Now you can chat with the agent or request code edits:

```
> build a simple flask app in app.py and add a test in test_app.py
```

### 4. Next time

Kaggle sessions are ephemeral. When you start a new Kaggle notebook session, run `qwen_serv.py` and run `/connect` with the newly generated URL and key.

## Using the CLI

If no server is reachable, `borrow` first opens a **connect screen**: the BORROW logo and a panel that walks you through it. Press **Connect** (or Enter) to open your Kaggle notebook (asked once, change it later with `/notebook`), run `qwen_serv.py` there, then paste the connection code it prints. It connects on paste, shows **✔ Connected** and moves on. Esc skips the screen.

Once connected, `borrow` opens a full-screen start page in yellow and black: the **BORROW** logo, a prompt box showing the agent mode, the connected **model** and its **context size**, key hints, a tip, and a status bar (connection, folder, version). After your first message the page clears and the conversation scrolls normally, with the same prompt box under each reply showing how full the context window is (for example `9.8K / 32K context (30%)`).

- **Interactive Autonomous Agent:** Simply type instructions in plain English (e.g. `refactor auth.py to use JWT` or `fix the failing pytest`).
- Type **`/`** to open the interactive command dropdown menu. Filter by typing and navigate with arrow keys or Tab.
- Type **`-`** after any command to view supported flags.
- Type **`@`** to attach files with Tab autocomplete (e.g. `@app.py`).
- **Confirmation Prompts:** The agent previews exact unified diffs and command runs, asking `[Y/n/a(ll)]`. Press `a` to auto-approve remaining actions for that session.
- **Shift+Tab** toggles auto-approve (the prompt box shows an `AUTO` badge while it is on).
- **Ctrl+J** or **Alt+Enter** adds a new line to your message; **Enter** sends it (or picks the highlighted menu item).
- **Ctrl+C** stops current generation or running command, or clears the prompt; press it twice on an empty prompt to quit. **Ctrl+D** or `/exit` exits the agent.

## Autonomous Agent Workflows

Like Claude Code and `agy`, `borrow` operates as a true autonomous software engineer:

```
 User Request ──► Agent ReAct Loop ──► Tool Call (read/edit/search/exec/skill)
      ▲                     │                         │
      │                     ▼                         ▼
      └────────── Response / Final Answer ◄── Confirmation & Tool Output
```

### Native Tool Suite
1. **`read_file(path, start_line, end_line)`**: Inspects workspace files with optional line slicing.
2. **`edit_file(path, target, replacement)`**: Applies surgical, exact in-place string replacements.
3. **`write_file(path, content)`**: Creates new files or overwrites existing files atomically.
4. **`list_directory(path)`**: Explores folder structures, skipping ignored directories (`.git`, `__pycache__`, etc.).
5. **`search_files(query, path)`**: Regex-powered workspace search across text files.
6. **`run_command(command)`**: Executes shell commands (e.g. `pytest`, `python app.py`, `npm test`) with output capture and 120s timeout.
7. **`load_skill(name)`**: Loads full instructions, rules, and script manifests from an agent skill by name.

### Safety & Control
- **Interactive Colored Diffs**: Every file modification displays an ANSI unified diff preview (`+added`, `-removed`, `@@hunk@@`).
- **Confirmation Guard**: Interactive prompt `[Y/n/a(ll)]` protects your filesystem and terminal.
- **Auto-Approve Toggle**: Use the `/auto` command or the CLI `-y` / `--yes` flag to run completely autonomously.
- **Git Revert & Diff**: Inspect workspace git state with `/diff` or undo file changes with `/revert <file>`.
- **Greeting & Intent Guard**: Conversational greetings (`hi`, `hello`, `who are you?`) are handled naturally without unprompted file creations or unsolicited directory scans.

### Live Visual Feedback
- **Thinking line**: While the model works, one live line shows a shimmering `Thinking…` label, elapsed time, token count and a peek at the newest words of its reasoning. When it finishes thinking, the line collapses to a one-line summary, and `/think` prints the full reasoning:
  ```
  ⠋ Thinking… 3.4s · 182 tokens  …so the retry loop caps the delay at thirty seconds
  ◇ Thought for 3.4s · 182 tokens
  ```
- **Tool lines**: Each tool call shows as one line, e.g. `● Read borrow.py` or `● Run pytest -q`; while arguments stream in you see `⠋ Preparing write_file… 1.2s · 214 tokens`.
- **Reply footer**: Every reply ends with `◆ Agent · <model> · 4.2s · 61 tok/s`.

## Agent Skills

Borrow supports **Agent Skills** (compatible with Claude Code and Antigravity `agy` specifications). Skills are reusable bundles of instructions, scripts, and workflows.

### Directory Structure & Layout
Skills are placed in `.borrow/skills/<skill-name>/`, `.agents/skills/<skill-name>/`, or globally in `~/.borrow/skills/<skill-name>/`:

```
.borrow/skills/code-audit/
├── SKILL.md            # Required: YAML frontmatter + instructions
├── scripts/            # Optional: verification or automation scripts
└── references/         # Optional: checklists or reference documentation
```

### `SKILL.md` Format
```markdown
---
name: code-audit
description: Rigorous security and performance review with automated static analysis.
---

# Code Audit Workflow
1. Run static checks and inspect dependencies...
2. Check for SQL injection, auth leaks, and memory bottlenecks...
```

### Using Skills
- **/skills**: Lists all available workspace and global skills:
  ```
  > /skills
  Available Skills (2 found):
    qa-review [workspace]
      Run a full independent QA pass on the borrow project...
    qa-tests [workspace]
      Write or extend the automated pytest suite for borrow...
  ```
- **/skill <name> [prompt]**: Run a specific skill or view its documentation:
  ```
  > /skill qa-tests write tests for user authentication
  > /skill qa-review
  ```
- **/skill init <name>**: Scaffolds a new skill directory with starter frontmatter in `.borrow/skills/<name>/SKILL.md`.
- **Tab Autocomplete**: In the REPL, typing `/skill ` and pressing Tab autocompletes skill names with live description hints.
- **Autonomous Tool Calling**: The agent dynamically sees available skills in its workspace summary and can call `load_skill` whenever a user query matches a skill's description.

## Talking to the model

**Natural Coding Prompts:**
```
> inspect index.html and style.css and make the layout responsive for mobile
> run the unit tests and fix whatever fails
> add type annotations to utils.py
```

**Task Modes:**
Prefixing prompts with a command sends specialized system instructions:
```
/code a python script that scrapes hacker news headlines -o hn.py
/review @app.py focus on security vulnerabilities and race conditions
/test @borrow.py -o tests/test_borrow.py
/fix @app.py --run "pytest tests/test_api.py"
```

## Commands

| Command | What it does | Flags |
|---|---|---|
| `/edit <instructions>` | Autonomous agent task: inspects, edits files, and runs verification | `-t TEMP` `-m N` |
| `/skills` | List all discovered workspace and global agent skills | |
| `/skill <name> [prompt]` | Run an agent skill, view its instructions, or scaffold (`init <name>`) | `-t TEMP` `-m N` |
| `/auto` | Toggle auto-approving file edits and shell commands | |
| `/diff` | Display `git diff` of uncommitted changes across workspace | |
| `/revert [file]` | Revert uncommitted changes in a file (`git checkout -- <file>`) | |
| `/code <what to build>` | Generate pure code without conversational filler | `-o FILE` `-t TEMP` `-m N` |
| `/explain [@file ...] [q]` | Explain code, architecture, or an error in plain language | `-o` `-t` `-m` |
| `/review [@file ...] [focus]` | Strict, prioritized code review | `-o` `-t` `-m` |
| `/test [@file ...] [notes]` | Generate unit test suites (pytest by default) | `-o` `-t` `-m` |
| `/fix [@file ...] [error]` | Fix bugs from an error trace or command output | `-o` `-t` `-m` `--run CMD` |
| `/doc [@file ...]` | Add type hints and docstrings without modifying logic | `-o` `-t` `-m` |
| `/commit` | Generate conventional commit messages from git diff | `--staged` `-o` `-t` |
| `/shell <what you want>` | Translate natural language into a shell command with confirmation | `-t` |
| `/connect [<url> <key>]` | Configure or view the active Cloudflare endpoint and API key | |
| `/notebook [<notebook URL>]` | Set or show the Kaggle notebook the **Connect** button opens | |
| `/status` | Verify remote server connectivity and model health | |
| `/think` | Show the model's full reasoning for the last reply | |
| `/clear` | Reset chat history and return to the start screen | |
| `/help [command]` | Display detailed command reference and examples | |
| `/exit` | Exit the CLI (or press `Ctrl+D`) | |

### Flags

| Flag | Description |
|---|---|
| `-y`, `--yes` | Global flag: Auto-approve all file edits and commands without confirmation |
| `-o FILE` | Save generated code or output directly to `FILE` |
| `-t TEMP` | Sampling temperature between `0.0` and `1.0` (default `0.2` for code precision) |
| `-m N` | Maximum tokens to generate (default: `8192`) |
| `--run CMD` | (`/fix`) Execute `CMD` on workstation and pass stdout/stderr to the agent |
| `--staged` | (`/commit`) Inspect only git staged changes (`git diff --cached`) |

## Attaching files

Attach local files to any query using `@`:

```
> why does @utils.py fail when passed None?
/review @src/auth.py @src/models.py focus on SQL injection
/fix @app.py --run "pytest"
```

- Attach multiple files in a single turn.
- Paths are relative to the current working directory.
- Autocomplete file names with Tab. Wrap paths containing spaces in quotes (`"@my file.py"`).
- Attached files are cached in conversation memory for subsequent follow-up questions.

## One-shot mode

Execute autonomous tasks or queries directly from your shell or CI/CD pipelines without entering the interactive REPL:

```bash
# Run a skill directly from the shell
borrow /skill qa-tests "write tests for auth.py"

# Autonomous edit with auto-approval
borrow -y "add docstrings and type annotations to utils.py"

# Review code piped from stdin
git diff | borrow /review -

# Fix bugs using test runner output
borrow /fix @app.py --run "pytest"
```

## Automated Testing & QA

`borrow` contains an independent QA suite and unit tests:

```bash
# Run the test suite
pytest -v
```

Features tested:
- Skills system: YAML frontmatter parsing, workspace/global discovery precedence, `tool_load_skill`, `prompt_toolkit` completers, `/skill init`, `/skills`, and `/skill`.
- Tool execution: `tool_read_file`, `tool_edit_file`, `tool_write_file`, `tool_list_directory`, `tool_search_files`, `tool_run_command`.
- Workspace protection and directory boundary confinement.
- Unified colored diff generation and fallback behavior.
- Argument parsing, flag validation, and non-zero exit codes on errors.
- Mock OpenAI-compatible streaming server and tool-call SSE streaming chunks.

## Settings and saved data

- Configuration is persisted at `~/.borrow.json` (URL and encrypted bearer key, `0600` permissions).
- Prompt history is stored in `~/.borrow_history`.
- Set environment variables `ASK_URL` and `ASK_KEY` to override saved configuration (ideal for CI/CD).

```bash
export ASK_URL=https://your-tunnel.trycloudflare.com
export ASK_KEY=YOUR_SESSION_KEY
```

## Choosing a model

`qwen_serv.py` includes built-in configurations at the top of the file:

```python
MODEL_CHOICE = "auto"   # "auto" | "coder32b_dense" | "coder30b" | "coder14b_dense" | "qwen35_27b"
THINKING     = True     # Enable reasoning tokens when supported
```

| Preset | Model | Context | Required VRAM | Characteristics |
|---|---|---|---|---|
| `coder32b_dense` | Qwen2.5-Coder-32B-Instruct (Q4_K_M) | 32768 | ~26 GB | **Highest coding accuracy**: Best for complex multi-file agentic edits (~14–18 tok/s). |
| `coder30b` | Qwen3-Coder-30B-A3B-Instruct (Q4_K_M) | 32768 | ~26 GB | **Blazing fast MoE**: 3.3B active parameters (~58–68 tok/s). Excellent responsiveness. |
| `coder14b_dense` | Qwen2.5-Coder-14B-Instruct (Q4_K_M) | 32768 | ~12 GB | Balanced reasoning, fits on a single T4 GPU (~25–35 tok/s). |

## Limits and honest expectations

- **Scratchpad, not an oracle**: Always review code changes before committing. While the agent features diff confirmation and verification loops, large multi-repo architecture tasks still benefit from human oversight.
- **Ephemeral Kaggle Sessions**: Sessions are limited by Kaggle weekly quotas and idle timeouts. When a session terminates, restart the Kaggle cell and run `/connect` with the fresh URL.
- **Context Limits**: The default context is 32K tokens. Keep attachments focused on relevant files.

## Security and privacy

- **Bearer Key Protection**: The random bearer key generated on Kaggle prevents unauthorized access to your tunnel. Never share your `/connect` string publicly.
- **Workstation Isolation**: Files are edited only within your active workspace. Diffs require user confirmation unless explicit auto-approval (`-y` or `/auto`) is granted.
- **No Third-Party AI Data Collection**: Prompts flow directly from your workstation through an encrypted Cloudflare quick tunnel to your own private Kaggle instance.

## Troubleshooting

| Symptom | Resolution |
|---|---|
| `Not connected` in CLI | Run `/connect <url> <key>` using the output from your Kaggle notebook |
| `HTTP 401 Unauthorized` | Invalid key. Copy the fresh `/connect` command printed at the end of `qwen_serv.py` |
| `HTTP 530 / 1033` or tunnel drops | Kaggle cell timed out or notebook stopped. Re-run `qwen_serv.py` on Kaggle |
| `No / dropdown menu` | Install prompt toolkit: `pip install prompt_toolkit` and ensure you are in a real terminal |
| Python command blocks in agent | Commands that start daemon/web servers (like `flask run` or `python app.py`) will run until the 120s timeout unless stopped |

## Files and log locations

**On Kaggle:**
- Server log: `/tmp/qwen_server/server.log`
- Tunnel log: `/tmp/qwen_server/tunnel.log`
- Model files: `/tmp/qwen_server/models/`

**On your workstation:**
- Config: `~/.borrow.json`
- REPL history: `~/.borrow_history`
