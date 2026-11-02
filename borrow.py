#!/usr/bin/env python3
"""borrow - a Claude-Code-style terminal for your self-hosted Qwen coder endpoint.

Start it:      borrow
One-shot:      borrow /review @app.py       (pipe input in with a lone "-": git diff | borrow /review -)
Connect:       set ASK_URL / ASK_KEY (printed by the Kaggle script) or type  /connect <url> <key>

Inside the CLI type  /  to open the command menu, @ to attach files, plain text to chat.
No flags to remember: type /help for every command, its flags and what it does.
"""
import difflib, json, os, re, shlex, shutil, subprocess, sys, threading, time, urllib.error, urllib.request
from pathlib import Path

__version__ = "0.1.1"
CONFIG_PATH = Path.home() / ".borrow.json"
LEGACY_CONFIG_PATH = Path.home() / ".qwen_cli.json"   # older versions saved here
HISTORY_PATH = Path.home() / ".borrow_history"
UA = {"User-Agent": "curl/8.5.0"}
MAX_FILE_CHARS = 60_000
DEFAULT_MAX_TOKENS = 8192   # a full response needs far more than 2048 tokens
MAX_CONTINUES = 2           # auto-continue this many times if a reply hits the token limit
CONTINUE_PROMPT = ("Your previous reply was cut off. Continue exactly where it stopped. Output only the remaining "
                   "text, starting mid-line if needed. Do not repeat anything already written, do not restart, "
                   "and do not add commentary.")
MAX_HISTORY_CHARS = 70_000  # keeps the conversation inside the 32K-token context window

BASE = (
    "You are an expert senior software engineer and technical architect. "
    "Produce correct, robust, production-ready solutions following modern best practices. "
    "Prioritize correctness, edge-case safety, security, and clean idiomatic style. "
    "Write maintainable code with proper error handling and without dummy or omitted logic."
)
MODES = {
    "code": (
        "Write the complete, production-ready implementation inside a single fenced code block. "
        "Ensure the code handles edge cases and runtime failures, adheres to language conventions, "
        "and includes concise inline comments explaining non-trivial logic."
    ),
    "explain": (
        "Explain the given code or error clearly and concisely. Structure your answer with: "
        "1) High-level purpose and architecture, "
        "2) Step-by-step walkthrough of key logic and data flow, "
        "3) Subtle edge cases, performance considerations, or potential pitfalls."
    ),
    "review": (
        "Conduct a rigorous senior engineering code review. Systematically inspect for: "
        "1) Correctness and edge-case bugs, "
        "2) Security vulnerabilities and resource leaks, "
        "3) Performance bottlenecks and code smells. "
        "For each issue, state the severity ([Critical], [Warning], [Suggestion]), explain the exact risk, "
        "and provide a concrete code fix or diff."
    ),
    "test": (
        "Write a thorough, comprehensive unit test suite for the given code (using pytest for Python unless "
        "specified otherwise). Include tests for standard use cases, boundary conditions, edge cases, invalid inputs, "
        "and error handling. Reply with the runnable test code in a single fenced code block."
    ),
    "fix": (
        "Analyze the bug, error message, or failure output thoroughly. "
        "1. First identify and explain the exact root cause in 1-3 concise bullet points. "
        "2. Then provide the complete, corrected code in a single fenced code block that resolves the issue "
        "without introducing regressions."
    ),
    "doc": (
        "Add clear, accurate docstrings and comments following standard conventions (such as PEP 257 / Google style "
        "for Python). Document parameters, return values, exceptions, and non-obvious logic without modifying code behaviour. "
        "Reply with the full documented code in a single fenced code block."
    ),
    "commit": (
        "Write a concise conventional-commit message for the given diff: "
        "subject line under 72 chars (e.g. feat:, fix:, refactor:), followed by optional bullet points highlighting key changes. "
        "Reply with only the message."
    ),
    "shell": (
        "Reply with ONLY a single, safe, effective bash command that accomplishes the request. "
        "No explanation, no markdown code fence."
    ),
}

F_OUT  = ("-o FILE", "save the code (or answer) to FILE")
F_TEMP = ("-t TEMP", "creativity 0.0-1.0 (default 0.2)")
F_MAX  = ("-m N",    "max tokens per reply (default 8192; long replies auto-continue)")
COMMON = [F_OUT, F_TEMP, F_MAX]

COMMANDS = {
    "code":    dict(desc="Generate code only, no explanation", usage="/code <what to build>",
                    mode="code", needs="text", flags=COMMON,
                    example="/code a python function that merges two sorted lists -o merge.py"),
    "explain": dict(desc="Explain code or an error in plain language", usage="/explain [@file ...] [question]",
                    mode="explain", needs="text_or_files", flags=COMMON,
                    example="/explain @app.py what does the retry loop do?"),
    "review":  dict(desc="Strict code review, most important issues first", usage="/review [@file ...] [focus]",
                    mode="review", needs="text_or_files", flags=COMMON,
                    example="/review @app.py @utils.py focus on error handling"),
    "test":    dict(desc="Write unit tests for your code", usage="/test [@file ...] [notes]",
                    mode="test", needs="text_or_files", flags=COMMON,
                    example="/test @app.py -o test_app.py"),
    "fix":     dict(desc="Fix a bug from an error message or a command's output", usage="/fix [@file ...] [error text]",
                    mode="fix", needs="text_or_files_or_run",
                    flags=COMMON + [("--run CMD", "run CMD first and send its output as the error")],
                    example='/fix @app.py --run "python app.py"'),
    "doc":     dict(desc="Add docstrings and comments without changing behaviour", usage="/doc [@file ...]",
                    mode="doc", needs="text_or_files", flags=COMMON,
                    example="/doc @app.py -o app_documented.py"),
    "commit":  dict(desc="Write a commit message from your git changes", usage="/commit",
                    mode="commit", needs=None,
                    flags=[("--staged", "use only staged changes (default: all uncommitted)"), F_OUT, F_TEMP],
                    example="/commit --staged"),
    "shell":   dict(desc="Turn a request into one shell command, then ask before running it", usage="/shell <what you want>",
                    mode="shell", needs="text", flags=[F_TEMP],
                    example="/shell find files bigger than 100MB"),
    "clear":   dict(desc="Forget the conversation so far", usage="/clear", mode=None, needs=None, flags=[],
                    example="/clear"),
    "connect": dict(desc="Set or show the endpoint (saved for next time)", usage="/connect [<url> <key>]", mode=None,
                    needs=None, flags=[], example="/connect https://xyz.trycloudflare.com YOUR_KEY"),
    "status":  dict(desc="Check that the server is reachable", usage="/status", mode=None, needs=None, flags=[],
                    example="/status"),
    "help":    dict(desc="Show all commands, their flags and what they do", usage="/help [command]", mode=None,
                    needs=None, flags=[], example="/help review"),
    "exit":    dict(desc="Quit (Ctrl+D also works)", usage="/exit", mode=None, needs=None, flags=[], example="/exit"),
}
FLAG_DEST = {"-o": ("out", True), "-t": ("temp", True), "-m": ("max_tokens", True),
             "--run": ("run", True), "--staged": ("staged", False)}

STATE = {"url": "", "key": "", "history": [], "piped": ""}
TTY = sys.stdout.isatty()


def dim(s):    return f"\033[2m{s}\033[0m" if TTY else s
def italic(s): return f"\033[3m{s}\033[0m" if TTY else s
def bold(s):   return f"\033[1m{s}\033[0m" if TTY else s
def red(s):    return f"\033[31m{s}\033[0m" if TTY else s
def green(s):  return f"\033[32m{s}\033[0m" if TTY else s


class Spinner:
    FRAMES_BRAILLE = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    FRAMES_ASCII = ["|", "/", "-", "\\"]

    def __init__(self, message="Waiting..."):
        self.message = message
        self.running = False
        self.thread = None
        self.tokens = 0
        use_braille = True
        try:
            "⠋".encode(sys.stderr.encoding or "utf-8")
        except Exception:
            use_braille = False
        self.frames = self.FRAMES_BRAILLE if use_braille else self.FRAMES_ASCII

    def start(self):
        if not TTY or self.running or os.environ.get("BORROW_TEST"):
            return
        self.running = True
        try:
            sys.stderr.write("\033[?25l")
            sys.stderr.flush()
        except Exception:
            pass
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self):
        idx = 0
        while self.running:
            frame = self.frames[idx % len(self.frames)]
            idx += 1
            tok_info = f" ({self.tokens} tokens)" if self.tokens > 0 else ""
            try:
                sys.stderr.write(f"\r\033[K{dim(frame)} {self.message}{dim(tok_info)}")
                sys.stderr.flush()
            except Exception:
                break
            time.sleep(0.08)

    def update(self, message=None, tokens=None):
        if message is not None:
            self.message = message
        if tokens is not None:
            self.tokens = tokens

    def stop(self, final_text=None):
        if not self.running:
            if final_text:
                print(final_text)
            return
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.3)
        if TTY and not os.environ.get("BORROW_TEST"):
            try:
                sys.stderr.write("\r\033[K\033[?25h")
                sys.stderr.flush()
            except Exception:
                pass
        if final_text:
            print(final_text)


class UserError(Exception):
    pass


# ----------------------------------------------------------------------------- config
def load_config():
    cfg = {}
    for path in (CONFIG_PATH, LEGACY_CONFIG_PATH):
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
            break
        except Exception:
            continue
    STATE["url"] = (os.environ.get("ASK_URL") or cfg.get("url", "")).rstrip("/")
    STATE["key"] = os.environ.get("ASK_KEY") or cfg.get("key", "")


def save_config():
    data = json.dumps({"url": STATE["url"], "key": STATE["key"]}, indent=2)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(str(CONFIG_PATH), flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
    except OSError:
        CONFIG_PATH.write_text(data, encoding="utf-8")
        try:
            os.chmod(CONFIG_PATH, 0o600)
        except OSError:
            pass


def health():
    if not STATE["url"]:
        return False
    try:
        req = urllib.request.Request(STATE["url"] + "/health", headers=UA)
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


# ----------------------------------------------------------------------------- model call
def stream_chat(messages, temp, max_tokens, top_p=0.8, top_k=20, repeat_penalty=1.05, silent=False, status_label=None):
    """Stream a reply to the terminal. Returns (text, finish_reason); finish_reason is "length" when the reply
    hit the token limit. Returns (None, None) on error. Text is partial if interrupted with Ctrl+C."""
    if not STATE["url"] or not STATE["key"]:
        print(red("Not connected. Type /connect <url> <key>, or set ASK_URL and ASK_KEY."))
        return None, None
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temp,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "stream": True,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(STATE["url"] + "/v1/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + STATE["key"], **UA})
    text, finish = "", None
    in_reasoning = False
    tokens = 0
    spinner = Spinner(status_label or "Waiting for response...")
    spinner.start()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            if not silent:
                print()
            try:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[5:].strip()
                    if payload_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices")
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    reasoning = delta.get("reasoning_content") or ""
                    content = delta.get("content") or ""
                    finish = choice.get("finish_reason") or finish

                    if reasoning:
                        tokens += 1
                        if not silent:
                            spinner.stop()
                            if not in_reasoning:
                                print(dim(italic("[thinking...] ")), end="", flush=True)
                                in_reasoning = True
                            print(dim(reasoning), end="", flush=True)
                        else:
                            spinner.update("Thinking...", tokens=tokens)

                    if content:
                        tokens += 1
                        text += content
                        if not silent:
                            spinner.stop()
                            if in_reasoning:
                                print()  # end reasoning section
                                in_reasoning = False
                            print(content, end="", flush=True)
                        else:
                            lbl = status_label or "Generating..."
                            spinner.update(lbl, tokens=tokens)
            except KeyboardInterrupt:
                spinner.stop(dim("\n[interrupted]"))
                return text, finish
            finally:
                spinner.stop()
            if not silent:
                print("\n")
        return text, finish
    except urllib.error.HTTPError as e:
        spinner.stop()
        print(red(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"))
    except Exception as e:
        spinner.stop()
        print(red(f"Request failed (is the Kaggle session still running? try /status): {e}"))
    return None, None


def trim_history(extra_chars=0):
    h = STATE["history"]
    while len(h) > 2 and (sum(len(m["content"]) for m in h) + extra_chars) > MAX_HISTORY_CHARS:
        del h[:2]


def merge_continuation(prev, more):
    """Join a continuation onto the earlier text. If we were inside an unclosed code block and the model
    reopened it with a fence such as ```html or ```, drop that repeated fence."""
    if prev.count("```") % 2 == 1:
        m = re.match(r"^\s*```[A-Za-z0-9_+.-]*[ \t]*\r?\n", more)
        if m:
            more = more[m.end():]
    return prev + more


def ask(mode, content, temp=0.2, max_tokens=DEFAULT_MAX_TOKENS, silent=False, status_label=None):
    trim_history(extra_chars=len(content))
    system = BASE + (" " + MODES[mode] if mode else "")
    base = [{"role": "system", "content": system}] + STATE["history"] + [{"role": "user", "content": content}]
    text, finish = stream_chat(base, temp, max_tokens, silent=silent, status_label=status_label)
    if text is None:
        return None
    rounds = 0
    while finish == "length" and rounds < MAX_CONTINUES:
        rounds += 1
        cont_label = f"{status_label} (continuing {rounds}/{MAX_CONTINUES})" if status_label else "Continuing..."
        if not silent:
            print(dim(f"[reached the {max_tokens}-token limit; asking the model to continue ({rounds}/{MAX_CONTINUES})]"))
        more, finish = stream_chat(base + [{"role": "assistant", "content": text},
                                           {"role": "user", "content": CONTINUE_PROMPT}],
                                   temp, max_tokens, silent=silent, status_label=cont_label)
        if more is None:
            break
        text = merge_continuation(text, more)
    if finish == "length":
        print(red("Still cut off. Raise the limit with -m 16384, or ask for one part at a time "
                  "(for example: the HTML first, then the CSS)."))
    STATE["history"] += [{"role": "user", "content": content}, {"role": "assistant", "content": text}]
    trim_history()
    return text


def first_block(text):
    """Return the first fenced code block. Works even if the closing fence is missing (a cut-off reply)."""
    m = re.search(r"```[A-Za-z0-9_+.-]*[ \t]*\r?\n", text)
    if not m:
        return text
    rest = text[m.end():]
    closing = re.search(r"^```", rest, re.M)
    return rest[:closing.start()] if closing else rest


# ----------------------------------------------------------------------------- parsing
def split_line(rest):
    try:
        if os.name == "nt":
            return [t[1:-1] if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'" else t
                    for t in shlex.split(rest, posix=False)]
        return shlex.split(rest)
    except ValueError as e:
        raise UserError(f"Could not parse that line ({e}). Check your quotes.")


def parse_args(name, rest):
    spec = COMMANDS[name]
    allowed = {f.split()[0] for f, _ in spec["flags"]}
    out = {"text": [], "files": []}
    toks, i = split_line(rest), 0
    while i < len(toks):
        t = toks[i]
        if t in ("-h", "--help"):
            out["help"] = True
            return out
        if t in FLAG_DEST and t in allowed:
            dest, takes = FLAG_DEST[t]
            if takes:
                i += 1
                if i >= len(toks):
                    raise UserError(f"{t} needs a value. Try /help {name}")
                out[dest] = toks[i]
            else:
                out[dest] = True
        elif re.match(r"^--?[A-Za-z]", t):
            raise UserError(f"Unknown flag {t} for /{name}. Try /help {name}")
        elif t.startswith("@") and len(t) > 1:
            out["files"].append(t[1:])
        else:
            out["text"].append(t)
        i += 1
    try:
        out["temp"] = float(out.get("temp", 0.2))
        out["max_tokens"] = int(out.get("max_tokens", DEFAULT_MAX_TOKENS))
    except ValueError:
        raise UserError("-t must be a number like 0.2 and -m a whole number like 8192")
    out["text"] = " ".join(out["text"])
    return out


def read_file_block(path):
    p = Path(path).expanduser()
    if p.is_dir():
        raise UserError(f"'{path}' is a directory, not a file. Attach individual files instead.")
    if not p.is_file():
        raise UserError(f"File not found: {path}")
    try:
        data = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise UserError(f"Cannot read {path}: {e}")
    note = ""
    if len(data) > MAX_FILE_CHARS:
        data, note = data[:MAX_FILE_CHARS], f"\n[truncated to {MAX_FILE_CHARS} characters]"
    return f"File: {path}\n```\n{data}\n```{note}"


def run_capture(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + r.stderr).strip()
        return r.returncode, out[-6000:]
    except subprocess.TimeoutExpired:
        return -1, f"(timed out after {timeout}s)"
    except Exception as e:
        return -1, f"(failed to execute: {e})"


def build_content(args, extra=()):
    parts = [args["text"]] if args["text"] else []
    parts += [read_file_block(p) for p in args["files"]]
    parts += list(extra)
    if STATE["piped"]:
        parts.append("Input:\n" + STATE["piped"])
        STATE["piped"] = ""
    return "\n\n".join(parts)


def save_output(path, text, as_code):
    data = first_block(text).strip("\n") if as_code and "```" in text else text.strip()
    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data + "\n", encoding="utf-8")
        line_count = len(data.splitlines())
        print(f"{green('✔')} Saved to {bold(str(path))} {dim(f'({line_count} lines)')}")
    except OSError as e:
        raise UserError(f"Cannot save to {path}: {e}")


# ----------------------------------------------------------------------------- commands
def print_help(only=None):
    names = [only] if only else list(COMMANDS)
    if not only:
        print(bold("Commands") + dim("   (type / to open the menu, Tab or arrows to choose)\n"))
    for n in names:
        c = COMMANDS[n]
        print(f"  {bold(c['usage'])}")
        print(f"      {c['desc']}")
        for flag, d in c["flags"]:
            print(f"      {flag:<10} {d}")
        if only:
            print(f"\n      example: {c['example']}")
        print()
    if not only:
        print(bold("Tips"))
        print("  @path        attach a file to any command or chat message (Tab completes paths)")
        print("  plain text   chat with memory; follow-ups can refer to earlier answers or reviews")
        print("  Ctrl+C       stop a reply while it is generating   |   Ctrl+D or /exit  quit")
        print("  One-shot from your shell:  borrow /review @app.py")
        print("  Pipe input in with a lone -:  git diff | borrow /review -\n")


def cmd_help(rest):
    arg = rest.strip().lstrip("/").lower()
    if arg and arg not in COMMANDS:
        raise UserError(f"No command named /{arg}. Type /help for the list.")
    print_help(arg or None)


def cmd_connect(rest):
    toks = split_line(rest)
    if not toks:
        key = STATE["key"]
        print(f"url: {STATE['url'] or '(not set)'}\nkey: {'…' + key[-4:] if key else '(not set)'}")
        return
    if len(toks) != 2:
        raise UserError("Usage: /connect <url> <key>")
    STATE["url"], STATE["key"] = toks[0].rstrip("/"), toks[1]
    STATE["history"].clear()
    save_config()
    print("saved. " + ("server reachable." if health() else red("saved, but the server did not answer /health yet.")))


def cmd_status(rest):
    if not STATE["url"]:
        raise UserError("Not connected. Use /connect <url> <key>")
    print(f"{STATE['url']}  ->  " + ("reachable" if health() else red("NOT reachable (session ended or URL changed?)")))
    print(dim(f"conversation: {len(STATE['history']) // 2} exchanges in memory"))


def cmd_shell(args):
    if not args["text"]:
        raise UserError("Tell me what you want, e.g. /shell find files bigger than 100MB")
    text = ask("shell", build_content(args), args["temp"])
    if not text:
        return
    cmd = first_block(text).strip() if "```" in text else text.strip()
    print(bold("Command: ") + cmd)
    try:
        ok = input("Run it? [y/N] ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        ok = False
    if ok:
        r = subprocess.run(cmd, shell=True)
        print(dim(f"[exit {r.returncode}]"))


def cmd_generic(name, args):
    if args.get("help"):
        print_help(name)
        return
    spec = COMMANDS[name]
    extra = []
    if name == "commit":
        if not shutil.which("git"):
            raise UserError("git is not installed or not on your PATH.")
        git = "git diff --staged" if args.get("staged") else "git diff HEAD"
        code, out = run_capture(git)
        if code != 0 and not args.get("staged"):
            code, out = run_capture("git diff")
        if code != 0 and args.get("staged"):
            code, out = run_capture("git diff --cached")
        if code != 0 or not out.strip():
            raise UserError("No git changes found to describe." if code == 0 else f"git failed: {out[:200]}")
        extra.append("Diff:\n" + out[-30_000:])
    if name == "fix" and args.get("run"):
        code, out = run_capture(args["run"])
        print(dim(f"ran: {args['run']} (exit {code})"))
        extra.append(f"Command: {args['run']}\nExit code: {code}\nOutput:\n{out}")
    needs = spec["needs"]
    if needs == "text" and not args["text"]:
        raise UserError(f"Usage: {spec['usage']}   (example: {spec['example']})")
    if needs in ("text_or_files", "text_or_files_or_run") and not (args["text"] or args["files"] or extra or STATE["piped"]):
        raise UserError(f"Nothing to work on. Usage: {spec['usage']}   (example: {spec['example']})")
    content = build_content(args, extra)
    out_file = args.get("out")
    silent = bool(out_file)
    status_label = f"Generating {out_file}..." if out_file else None
    text = ask(spec["mode"], content, args["temp"], args["max_tokens"], silent=silent, status_label=status_label)
    if text and out_file:
        save_output(out_file, text, as_code=spec["mode"] in ("code", "test", "doc", "fix"))
        if spec["mode"] == "fix":
            m = re.search(r"```[A-Za-z0-9_+.-]*[ \t]*\r?\n", text)
            if m and m.start() > 0:
                explanation = text[:m.start()].strip()
                if explanation:
                    print(dim(explanation))


def handle(line):
    """Run one input line. Returns False when the user wants to quit."""
    line = line.strip()
    if not line:
        return True
    try:
        if line.startswith("/"):
            name, _, rest = line[1:].partition(" ")
            name = name.lower()
            if name not in COMMANDS:
                close = difflib.get_close_matches(name, COMMANDS, n=1)
                raise UserError(f"Unknown command /{name}." + (f" Did you mean /{close[0]}?" if close else "") + " Type /help.")
            if name == "exit":
                return False
            if name == "help":
                cmd_help(rest)
            elif name == "clear":
                STATE["history"].clear()
                print(dim("conversation cleared"))
            elif name == "connect":
                cmd_connect(rest)
            elif name == "status":
                cmd_status(rest)
            else:
                args = parse_args(name, rest)
                cmd_shell(args) if name == "shell" else cmd_generic(name, args)
        else:
            args = parse_args_chat(line)
            ask(None, build_content(args), args["temp"], args["max_tokens"])
    except UserError as e:
        print(red(str(e)))
    return True


def parse_args_chat(line):
    toks = split_line(line)
    files, text_toks = [], []
    for t in toks:
        if t.startswith("@") and len(t) > 1:
            candidate = t[1:]
            cand_path = Path(candidate).expanduser()
            # Treat as file only if it actually exists as a file/dir on disk or explicitly contains a path separator
            if cand_path.is_file() or "/" in candidate or "\\" in candidate:
                files.append(candidate)
                text_toks.append(candidate)
                continue
        text_toks.append(t)
    text = " ".join(text_toks)
    return {"text": text, "files": files, "temp": 0.2, "max_tokens": DEFAULT_MAX_TOKENS}


# ----------------------------------------------------------------------------- the / menu
def make_completer():
    from prompt_toolkit.completion import Completer, Completion

    class SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if text.startswith("/") and " " not in text:
                word = text[1:].lower()
                for name, c in COMMANDS.items():
                    if name.startswith(word):
                        yield Completion("/" + name + " ", start_position=-len(text),
                                         display="/" + name, display_meta=c["desc"])
                return
            token = re.split(r"\s", text)[-1] if text else ""
            if token.startswith("@"):
                yield from self.paths(token)
            elif token.startswith("-") and text.startswith("/"):
                cmd = text[1:].split(" ")[0].lower()
                for flag, d in COMMANDS.get(cmd, {}).get("flags", []):
                    f = flag.split()[0]
                    if f.startswith(token):
                        yield Completion(f + " ", start_position=-len(token), display=flag, display_meta=d)

        def paths(self, token):
            raw = os.path.expanduser(token[1:])
            folder, base = os.path.split(raw)
            try:
                entries = sorted(os.listdir(folder or "."))
            except OSError:
                return
            for e in entries:
                if e.startswith(base) and (not e.startswith(".") or base.startswith(".")):
                    full = os.path.join(folder, e)
                    isdir = os.path.isdir(os.path.join(folder or ".", e))
                    shown = "@" + full + (os.sep if isdir else "")
                    yield Completion(shown, start_position=-len(token), display=e + (os.sep if isdir else ""))
    return SlashCompleter()


def build_session(inp=None, out=None):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style
    style = Style.from_dict({
        "completion-menu.completion": "bg:#262626 #d0d0d0",
        "completion-menu.completion.current": "bg:#00afaf #000000",
        "completion-menu.meta.completion": "bg:#1c1c1c #808080",
        "completion-menu.meta.completion.current": "bg:#008787 #000000",
        "bottom-toolbar": "bg:#1c1c1c #808080",
    })
    return PromptSession(
        history=FileHistory(str(HISTORY_PATH)), completer=make_completer(), complete_while_typing=True,
        auto_suggest=AutoSuggestFromHistory(), style=style, input=inp, output=out,
        bottom_toolbar="  / commands   @ attach file   Ctrl+C stop reply   Ctrl+D quit")


def get_session():
    try:
        __import__("prompt_toolkit")
    except ImportError:
        print(dim("Installing prompt_toolkit (gives you the / menu)..."))
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "prompt_toolkit"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(red("Could not install prompt_toolkit; using basic mode (no menu). Run: pip install prompt_toolkit"))
            return None
    try:
        return build_session()
    except Exception as e:
        print(red(f"Menu unavailable ({e}); using basic mode."))
        return None


def basic_prompt():
    try:
        import readline
        names = ["/" + n + " " for n in COMMANDS]
        readline.set_completer(lambda t, i: ([n for n in names if n.startswith(t)] + [None])[i])
        readline.parse_and_bind("tab: complete")
    except ImportError:
        pass
    return lambda: input("> ")


def repl(session=None):
    prompt = (lambda: session.prompt("> ")) if session else basic_prompt()
    print(bold("borrow") + dim("  Qwen coder CLI   type / for commands, @ to attach files, /help for details"))
    if not STATE["url"]:
        print(red("Not connected yet. Type /connect <url> <key>  (or set ASK_URL and ASK_KEY)."))
    elif not health():
        print(red("Saved endpoint is not reachable. If you started a new Kaggle session, use /connect with the new URL and key."))
    while True:
        try:
            line = prompt()
        except KeyboardInterrupt:
            print(dim("(Ctrl+D or /exit to quit)"))
            continue
        except EOFError:
            break
        if not handle(line):
            break
    print(dim("bye"))


def main():
    args = sys.argv[1:]
    if args and args[0] in ("-V", "--version"):
        print(f"borrow {__version__}")
        return
    if args and args[0] in ("-h", "--help", "help"):
        print(__doc__)
        print_help()
        return
    load_config()
    if args:                                    # one-shot mode
        if "-" in args:                         # a lone "-" means: read piped input from stdin
            args = [a for a in args if a != "-"]
            STATE["piped"] = sys.stdin.read()
        line = subprocess.list2cmdline(args) if os.name == "nt" else " ".join(shlex.quote(a) for a in args)
        handle(line)
        return
    repl(get_session())


def run():
    try:
        main()
    except BrokenPipeError:                     # e.g. output piped into `head`
        pass


if __name__ == "__main__":
    run()
