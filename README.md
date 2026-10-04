# BORROW - Brain On Remote, Run On Workstation

Chat with a strong open coding model (**Qwen3-Coder-30B-A3B**) from your own terminal, in the style of Claude Code:
type `/` to open a command menu, `@file` to attach files, or plain text to chat. The model runs for free on
Kaggle's dual T4 GPUs and reaches your computer through a free Cloudflare tunnel.

No GPU of your own and no paid services needed.

```
 Your computer                          Internet                    Kaggle notebook (2x T4 GPUs)
┌────────────────────┐   HTTPS + API key   ┌────────────────┐   ┌───────────────────────────────┐
│ qwen_cli.py        │ ──────────────────► │ Cloudflare     │──►│ llama-server (llama.cpp, CUDA) │
│ (this CLI)         │ ◄────────────────── │ quick tunnel   │◄──│ Qwen3-Coder-30B-A3B  (Q4_K_M)  │
└────────────────────┘     streamed tokens └────────────────┘   └───────────────────────────────┘
```

## Contents

- [What you get](#what-you-get)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Using the CLI](#using-the-cli)
- [Talking to the model](#talking-to-the-model)
- [Commands](#commands)
- [Attaching files](#attaching-files)
- [One-shot mode](#one-shot-mode)
- [Settings and saved data](#settings-and-saved-data)
- [Choosing a model](#choosing-a-model)
- [Limits and honest expectations](#limits-and-honest-expectations)
- [Security and privacy](#security-and-privacy)
- [Troubleshooting](#troubleshooting)
- [Files and log locations](#files-and-log-locations)

## What you get

| File | Runs on | Purpose |
|---|---|---|
| `serve_qwen_kaggle.py` | Kaggle notebook | Builds llama.cpp with CUDA, downloads the model, starts the server, opens the tunnel and prints your connection details |
| `qwen_cli.py` | Your computer | The terminal you use: `/` command menu, `@file` attachments, streaming replies and chat memory |

Observed in one real run on 2x Tesla T4: about **58-68 tokens/second** with the 30B coder model
(only about 3B parameters are active per token, which is why it is fast).

## Requirements

**Kaggle side**
- A Kaggle account with a **verified phone number** (needed to turn Internet on).
- In the notebook's right sidebar, under **Session options**:
  - **Accelerator: GPU T4 x2**
  - **Internet: On**

**Your computer**
- Python 3.8 or newer. Nothing else is required.
- `qwen_cli.py` installs `prompt_toolkit` automatically on first run (this powers the `/` menu). If that
  fails it falls back to a basic mode with Tab completion and no dropdown. You can install it yourself with
  `pip install prompt_toolkit`.
- Optional: `git` (for `/commit`).

## Quick start

### 1. Start the model on Kaggle

1. Create a new Kaggle notebook and set the accelerator and Internet as described above.
2. Paste the **entire** contents of `serve_qwen_kaggle.py` into **one code cell** and run it.
3. The first run takes roughly **15-25 minutes**: about 10-15 minutes to compile llama.cpp, then a ~19 GB model
   download. You will see progress as `[1/6] ... [6/6]`.
4. When it finishes, the output ends with a ready-to-paste line like:

   ```
   /connect https://some-random-words.trycloudflare.com YOUR_KEY
   ```

5. **Leave the cell running.** It holds the server open. Stopping the cell shuts everything down.

### 2. Open the CLI on your computer

```bash
python qwen_cli.py
```

Paste the `/connect ...` line from the Kaggle output:

```
/connect https://some-random-words.trycloudflare.com YOUR_KEY
```

The CLI saves the endpoint, checks the server and prints `saved. server reachable.` Then try it:

```
/code a python function that merges two sorted lists
```

### 3. Next time

Kaggle clears everything between sessions, so each new session gives a **new URL and key**. Run the Kaggle cell
again (the build and download repeat), then run `/connect` with the new values.

> Tip: to avoid typing `python qwen_cli.py`, add an alias, for example in `~/.bashrc` or `~/.zshrc`:
> `alias qwen='python3 ~/path/to/qwen_cli.py'`. On Windows PowerShell, add
> `function qwen { python C:\path\to\qwen_cli.py $args }` to your profile.

## Using the CLI

- Type **`/`** to open the command menu: every command with a one-line description. Keep typing to filter, and use
  the arrow keys or Tab to choose.
- After a command, type **`-`** to see that command's flags.
- Type **`@`** to attach files. Tab completes paths.
- Type **plain text** to chat.
- **Ctrl+C** stops a reply while it is being generated. **Ctrl+D** or `/exit` quits.
- `/help` lists every command with its flags and what it does. `/help review` shows one command with an example.
- `python qwen_cli.py --help` prints the same help from your normal shell.

## Talking to the model

**Chat.** Anything you type without a leading `/` is sent to the model as a chat message. Replies stream in as they
are generated.

**Memory.** The conversation has memory, so follow-ups work: after `/review @app.py` you can simply type
`fix the second issue` or `show me the corrected function`. Use `/clear` to start fresh. History is trimmed
automatically from the oldest messages to stay inside the model's context window.

**Commands.** A `/command` sends a request with task-specific instructions (for example "reply with code only" for
`/code`). Its question and answer join the same conversation, so you can keep talking about the result.

**Saving output.** Add `-o FILE` to save the model's code to a file, for example `/code a flask hello world -o app.py`.
For `/code`, `/test`, `/doc` and `/fix` it saves just the first code block.

## Commands

| Command | What it does | Flags |
|---|---|---|
| `/code <what to build>` | Generates code only, no explanation | `-o FILE` `-t TEMP` `-m N` |
| `/explain [@file ...] [question]` | Explains code or an error in plain language | `-o` `-t` `-m` |
| `/review [@file ...] [focus]` | Strict code review, most important issues first | `-o` `-t` `-m` |
| `/test [@file ...] [notes]` | Writes unit tests (pytest for Python unless told otherwise) | `-o` `-t` `-m` |
| `/fix [@file ...] [error text]` | Fixes a bug from an error message or a command's output | `-o` `-t` `-m` `--run CMD` |
| `/doc [@file ...]` | Adds docstrings and comments without changing behaviour | `-o` `-t` `-m` |
| `/commit` | Writes a commit message from your git changes | `--staged` `-o` `-t` |
| `/shell <what you want>` | Turns a request into one shell command, then asks before running it | `-t` |
| `/clear` | Forgets the conversation so far | |
| `/connect [<url> <key>]` | Sets or shows the endpoint (saved for next time) | |
| `/status` | Checks that the server is reachable | |
| `/help [command]` | Shows commands, flags and what they do | |
| `/exit` | Quits | |

**Flags**

| Flag | Meaning |
|---|---|
| `-o FILE` | Save the result to FILE (code commands save just the first code block) |
| `-t TEMP` | Creativity from 0.0 to 1.0 (default 0.2, a good value for code) |
| `-m N` | Maximum tokens to generate (default 2048) |
| `--run CMD` | (`/fix`) Run CMD first and send its output to the model as the error |
| `--staged` | (`/commit`) Use only staged changes. The default is all uncommitted changes (`git diff HEAD`) |

**Examples**

```
/code a python function that merges two sorted lists -o merge.py
/review @app.py @utils.py focus on error handling
/test @app.py -o test_app.py
/fix @app.py --run "python app.py"
/commit --staged
/shell find files bigger than 100MB
why is @app.py slow?
```

**Good to know**

- `/shell` always shows the command and asks `Run it? [y/N]`. **Read it before answering yes.** The command was
  written by a language model and can be wrong or destructive.
- `/fix --run` executes the command you give it on your machine (120 second timeout) and sends its output to the
  model.
- `/commit` runs `git diff` for you, so there is nothing to pipe.
- A misspelled command gets a suggestion, for example `/revew` suggests `/review`.

## Attaching files

Put `@` in front of a path, in any command or in plain chat:

```
/review @app.py
/explain @src/utils.py what does the retry loop do?
/review @app.py @utils.py focus on error handling
why is @app.py slow?
```

- Attach as many files as you like in one message.
- Paths are relative to the folder you started the CLI in, so `cd` into your project first.
- Tab after `@` completes file and folder names. Wrap paths containing spaces in quotes: `"@my notes.py"`.
- Attached files stay in the conversation memory, so follow-ups do not need them again.
- **Text files only** (source code, configs, logs, markdown, CSV). Images, PDFs and other binary files are not
  supported.
- Folders and wildcards (`@src/`, `@*.py`) are not supported; attach files one by one.
- Files over 60,000 characters are truncated, with a note to the model.
- Everything you attach plus the chat history must fit in the context window (32K tokens for the default model).
  About 3-4 large files is a practical maximum.

## One-shot mode

Run a single command from your normal shell without opening the interactive prompt:

```bash
python qwen_cli.py /review @app.py
git diff | python qwen_cli.py /review -      # a lone "-" means: read piped input
```

## Settings and saved data

- `/connect <url> <key>` saves the endpoint to `~/.qwen_cli.json`. Prompt history is kept in `~/.qwen_cli_history`.
- The environment variables `ASK_URL` and `ASK_KEY` override the saved values:

  ```bash
  export ASK_URL=https://new-url.trycloudflare.com ASK_KEY=NEW_KEY          # Mac / Linux
  $env:ASK_URL='https://new-url.trycloudflare.com'; $env:ASK_KEY='NEW_KEY'  # PowerShell
  ```

- `/connect` with no arguments shows the current URL and the last 4 characters of the key. `/status` checks that the
  server answers.

## Choosing a model

By default the Kaggle script picks a model from your total GPU memory. Edit the settings at the top of
`serve_qwen_kaggle.py`:

```python
MODEL_CHOICE = "auto"   # "auto" | "coder30b" | "qwen35_27b" | "qwen35_9b"
THINKING     = False    # Qwen3.5 models only: True = better reasoning, much slower
```

| Preset | Model | Context | Needs (total VRAM) | Notes |
|---|---|---|---|---|
| `coder30b` | Qwen3-Coder-30B-A3B-Instruct, Q4_K_M (~18.6 GB) | 32768 | about 26 GB | Default with 2x T4. Coding-specific and fast |
| `qwen35_27b` | Qwen3.5-27B, Q4_K_M | 16384 | about 24 GB | General model, dense, slower |
| `qwen35_9b` | Qwen3.5-9B, Q4_K_M | 16384 | about 10 GB | Fallback when only one GPU is available |

`auto` chooses `coder30b` when at least 26 GB of VRAM is visible, otherwise `qwen35_9b`. Other GGUF repos work if
you add a preset with the repo name and quantization. The model is exposed under the name `qwen`.

## Limits and honest expectations

- **Quality:** strong for functions, scripts, boilerplate, tests, SQL, regex, refactors and explaining errors.
  Weaker than the best frontier models on large multi-file reasoning and subtle bugs. Review everything and run
  tests. Use it as a fast scratchpad, not as an oracle.
- **Not production infrastructure.** Do not put it in a product's request path.
- **Sessions are temporary.** Kaggle limits weekly GPU hours and session length (check Kaggle's documentation for
  current numbers). Every new session needs the 15-25 minute setup again and gives a new URL and key.
- **Context window is 32K tokens** for the default model. Attach only the files you need.
- **One shared GPU pair.** Fine for you and maybe a teammate, not for a whole team.

## Security and privacy

- **Anyone with the URL and the key can use your GPU session.** Treat the key like a password. Do not post
  screenshots or logs that contain it. `~/.qwen_cli.json` stores it in plain text (permissions are restricted where
  the OS supports it); do not commit it to git.
- The key is random per session and the server rejects requests without it. The tunnel URL alone is not enough.
- **Your prompts and attached files travel through Cloudflare to a Kaggle machine.** Before sending proprietary
  company code, get permission from your employer or founder. Never send secrets, API keys, credentials or
  customer data.
- **Terms of service:** this project has not verified that Kaggle's or Cloudflare's terms allow tunnelling from a
  notebook. Check them for your situation.
- **Model licence:** check the Qwen3-Coder licence on its Hugging Face page before commercial use.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `FAILED: No internet access (DNS lookup failed)` (Kaggle) | Internet is off. Sidebar > Session options > Internet > On. If the toggle is greyed out, verify your phone number in Kaggle account settings. Re-run the cell |
| `No GPU found` (Kaggle) | Set Accelerator to **GPU T4 x2** and restart the session |
| `CUDA::cuda_driver ... not found` during configure (Kaggle) | The script finds the driver library itself and falls back to `GGML_CUDA_NO_VMM=ON` automatically. If it still fails, check the last lines of `/tmp/qwen_server/cmake_configure.log` |
| Build fails or the kernel dies while compiling (Kaggle) | Likely out of memory. Re-run the cell, then check `/tmp/qwen_server/cmake_build.log` |
| `llama-server exited early` (Kaggle) | Read the log tail it prints (also `/tmp/qwen_server/server.log`). Try a smaller context by editing `ctx` in the preset, or use `qwen35_9b` |
| Tunnel self-test fails inside Kaggle | That self-test is optional and non-fatal. Run `/connect` and `/status` from your own computer instead; that is the check that matters |
| `Not connected` in the CLI | Run `/connect <url> <key>` with the values from the Kaggle output |
| `HTTP 401` | Wrong or old key. Use the key from the current session's output |
| `HTTP 530` / `1033`, or `/status` says NOT reachable | The session ended or the tunnel dropped. Re-run the Kaggle cell and `/connect` with the new URL and key |
| `Request failed (is the Kaggle session still running?)` | Same as above; check the Kaggle tab |
| No `/` dropdown | `pip install prompt_toolkit`, then restart `qwen_cli.py`. Use a real terminal (some IDE consoles do not support it) |
| `python qwen_cli.py /review -` hangs | The lone `-` waits for piped input. Pipe something in (`git diff \| python qwen_cli.py /review -`) or remove the `-` |

## Files and log locations

On Kaggle (the script chooses the location with the most free disk, usually `/tmp`):

| Path | Contents |
|---|---|
| `/tmp/qwen_server/llama.cpp/` | llama.cpp source and the built `llama-server` |
| `/tmp/qwen_server/models/` | The downloaded GGUF model |
| `/tmp/qwen_server/server.log` | Server log (shows GPU offload info) |
| `/tmp/qwen_server/tunnel.log` | Cloudflare tunnel log |
| `/tmp/qwen_server/cmake_configure.log`, `cmake_build.log` | Build logs |

On your computer: `~/.qwen_cli.json` (saved URL and key) and `~/.qwen_cli_history` (prompt history).

## Status

Verified on a real Kaggle session (2x T4): build, model download, server start, tunnel, and a request through the
tunnel from a separate machine. `qwen_cli.py` was verified with automated tests against a mock server (command
parsing, menu completions, streaming, file attachments, error handling); its behaviour in a real terminal, and on
Windows, has had less testing. If something misbehaves, open an issue with your OS, terminal and the exact error.