#!/usr/bin/env python3
"""borrow - a Claude-Code-style terminal for your self-hosted Qwen coder endpoint.

Start it:      borrow
One-shot:      borrow /review @app.py       (pipe input in with a lone "-": git diff | borrow /review -)
Connect:       set ASK_URL / ASK_KEY (printed by the Kaggle script) or type  /connect <url> <key>

Inside the CLI type  /  to open the command menu, @ to attach files, plain text to chat.
No flags to remember: type /help for every command, its flags and what it does.
"""
import difflib, json, math, os, random, re, secrets, shlex, shutil, subprocess, sys, textwrap, threading, time
import urllib.error, urllib.request
from pathlib import Path

__version__ = "0.1.3"
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
COMPACT_AT = 0.80           # auto-compact once the context window is this full
COMPACT_SYSTEM = (
    "You compress a coding conversation so it can continue in a fresh context window. Write a dense summary "
    "that keeps: the user's goals and requests, decisions made, files and functions touched (exact paths and "
    "names), code that still matters (verbatim if short), errors seen and how they were fixed, and anything "
    "unfinished. Drop greetings, repetition and dead ends. Use short bullet points. Output only the summary.")

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
    "edit":    dict(desc="Ask the agent to edit files or perform tasks in your workspace",
                    usage="/edit <instructions>", mode="edit", needs="text",
                    flags=[F_TEMP, F_MAX], example="/edit fix the bug in utils.py"),
    "auto":    dict(desc="Toggle auto-approving edits and commands without asking",
                    usage="/auto", mode=None, needs=None, flags=[], example="/auto"),
    "diff":    dict(desc="Show git diff of all workspace changes made so far",
                    usage="/diff", mode=None, needs=None, flags=[], example="/diff"),
    "revert":  dict(desc="Revert uncommitted git changes in a file (git checkout -- <file>)",
                    usage="/revert [file]", mode=None, needs=None, flags=[], example="/revert borrow.py"),
    "think":   dict(desc="Show the model's full reasoning for the last reply", usage="/think", mode=None,
                    needs=None, flags=[], example="/think"),
    "compact": dict(desc="Summarize the conversation to free up context (also runs automatically at 80%)",
                    usage="/compact [what to focus on]", mode=None, needs=None, flags=[],
                    example="/compact keep the details of the auth bug"),
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

STATE = {"url": "", "key": "", "history": [], "piped": "", "auto_approve": False,
         "online": None, "model": "", "n_ctx": 0, "ctx_used": 0, "tps": 0, "thought": "",
         "tok_in": 0, "tok_out": 0}
THINK_WORDS = ["Pondering", "Brewing", "Noodling", "Cogitating", "Mulling", "Percolating", "Untangling",
               "Conjuring", "Tinkering", "Ruminating", "Hatching", "Wrangling", "Simmering", "Scheming",
               "Musing", "Unraveling", "Distilling", "Deliberating", "Marinating", "Spelunking"]
TTY = sys.stdout.isatty()


def dim(s):    return f"\033[2m{s}\033[0m" if TTY else s
def italic(s): return f"\033[3m{s}\033[0m" if TTY else s
def bold(s):   return f"\033[1m{s}\033[0m" if TTY else s
def red(s):    return f"\033[31m{s}\033[0m" if TTY else s
def green(s):  return f"\033[32m{s}\033[0m" if TTY else s
def yellow(s): return f"\033[38;5;220m{s}\033[0m" if TTY else s   # the BORROW yellow
def gold(s):   return f"\033[38;5;178m{s}\033[0m" if TTY else s   # muted yellow for secondary info


def shimmer(text, tick):
    """Gold text with a bright highlight sweeping across it, one step per tick."""
    if not TTY:
        return text
    head = tick % (len(text) + 8) - 4
    return "".join(f"\033[38;5;{230 if i == head else 222 if abs(i - head) < 3 else 178}m{ch}"
                   for i, ch in enumerate(text)) + "\033[0m"


def fmt_secs(s):
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m {int(s % 60):02d}s"


def fmt_k(n):
    """Token counts the way people say them: 512, 4.9K, 32K."""
    if n < 1024:
        return str(n)
    return f"{n / 1024:.1f}K" if n < 10 * 1024 else f"{round(n / 1024)}K"


def ctx_meter(used, n, width=10):
    """━━━─────── how full the context window is."""
    filled = min(width, round(used * width / n)) if n else 0
    return "━" * filled + "─" * (width - filled)


def mode_label():
    return "Auto" if STATE.get("auto_approve") else "Agent"


class Spinner:
    """One live status line on stderr: spinner, shimmering label, elapsed time, tokens and a peek at the model's thoughts."""
    FRAMES_BRAILLE = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    FRAMES_ASCII = ["|", "/", "-", "\\"]

    def __init__(self, message="Working…"):
        self.message, self.detail, self.tokens = message, "", 0
        self.running, self.thread, self.t0 = False, None, time.time()
        self.words = ["Thinking"] + random.sample(THINK_WORDS, len(THINK_WORDS))   # "Thinking" rotates through these
        try:
            "⠋…".encode(sys.stderr.encoding or "utf-8")
            self.frames = self.FRAMES_BRAILLE
        except Exception:
            self.frames = self.FRAMES_ASCII

    def start(self):
        if not TTY or self.running or os.environ.get("BORROW_TEST"):
            return
        self.running, self.t0 = True, time.time()
        try:
            sys.stderr.write("\033[?25l")
            sys.stderr.flush()
        except Exception:
            pass
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def render(self, tick):
        cols = shutil.get_terminal_size((80, 20)).columns - 1
        meta = " " + fmt_secs(time.time() - self.t0) + (f" · {self.tokens} tokens" if self.tokens else "")
        word = self.words[int((time.time() - self.t0) / 2.5) % len(self.words)]
        label = self.message.replace("Thinking", word, 1)[:max(8, cols - len(meta) - 3)]
        room = cols - len(label) - len(meta) - 6
        peek = " ".join(self.detail.split()) if room > 12 else ""
        if len(peek) > room:                    # keep the newest words, starting on a word boundary
            peek = "…" + peek[-room + 1:].partition(" ")[2]
        return (f"\r\033[K{yellow(self.frames[tick % len(self.frames)])} {shimmer(label, tick)}{dim(meta)}"
                + (f"  {dim(italic(peek))}" if peek else ""))

    def _spin(self):
        tick = 0
        while self.running:
            try:
                sys.stderr.write(self.render(tick))
                sys.stderr.flush()
            except Exception:
                break
            tick += 1
            time.sleep(0.08)

    def update(self, message=None, tokens=None, detail=None):
        if message is not None:
            self.message = message
        if tokens is not None:
            self.tokens = tokens
        if detail is not None:
            self.detail = detail

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


def pretty_model(name):
    """'/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf' -> 'Qwen3-Coder-30B-A3B-Instruct'."""
    name = re.sub(r"\.gguf$", "", re.split(r"[\\/]", name)[-1], flags=re.I)
    name = re.sub(r"-\d{5}-of-\d{5}$", "", name)
    return re.sub(r"[-_.](UD-)?(I?Q\d\w*|BF16|F16|F32)$", "", name, flags=re.I) or name


def fetch_server_info():
    """Fill STATE with reachability, model name and context size. Runs in a background thread at startup."""
    STATE.update(online=None, model="", n_ctx=0)
    STATE["online"] = health()
    if not STATE["online"]:
        return
    headers = {"Authorization": "Bearer " + STATE["key"], **UA}
    for path in ("/props", "/v1/models"):     # /props is llama-server's own; /v1/models is the OpenAI-style fallback
        try:
            with urllib.request.urlopen(urllib.request.Request(STATE["url"] + path, headers=headers), timeout=10) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            if path == "/props":
                STATE["n_ctx"] = (data.get("default_generation_settings") or {}).get("n_ctx") or data.get("n_ctx") or 0
                name = data.get("model_path") or ""
            else:
                name = (data.get("data") or [{}])[0].get("id") or ""
            if name and not STATE["model"]:
                STATE["model"] = pretty_model(name)
        except Exception:
            continue


# ----------------------------------------------------------------------------- model call
def stream_chat(messages, temp, max_tokens, top_p=0.8, top_k=20, repeat_penalty=1.05, silent=False, status_label=None, tools=None):
    """Stream a reply to the terminal. Returns (text, finish_reason) or (text, finish_reason, tool_calls) if tools is provided.
    finish_reason is 'length' when token limit is reached, or 'tool_calls' when tools are invoked."""
    if not STATE["url"] or not STATE["key"]:
        print(red("Not connected. Type /connect <url> <key>, or set ASK_URL and ASK_KEY."))
        return (None, None, []) if tools is not None else (None, None)
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temp,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "stream": True,
        "stream_options": {"include_usage": True},   # lets us show how full the context window is
    }
    if tools:
        payload["tools"] = tools
    body = json.dumps(payload).encode()
    req = urllib.request.Request(STATE["url"] + "/v1/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + STATE["key"], **UA})
    text, finish, thought = "", None, ""
    tool_calls_acc = []
    in_tool_call = False
    tokens = think_tokens = 0
    t_start, t_think, usage, timings = time.time(), None, None, None
    spinner = Spinner(status_label or "Thinking…")

    def close_thought():
        """Collapse a finished reasoning phase into one summary line (/think shows the full text)."""
        nonlocal t_think
        if t_think is None:
            return
        spinner.stop(None if silent else
                     gold("◇ ") + dim(f"Thought for {fmt_secs(time.time() - t_think)} · {think_tokens} tokens"))
        t_think = None

    spinner.start()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
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
                    usage = chunk.get("usage") or usage
                    timings = chunk.get("timings") or timings
                    choices = chunk.get("choices")
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    reasoning = delta.get("reasoning_content") or ""
                    content = delta.get("content") or ""
                    finish = choice.get("finish_reason") or finish

                    if reasoning:
                        if t_think is None:              # a new thinking phase starts
                            t_think, think_tokens = time.time(), 0
                            if STATE["thought"] and not STATE["thought"].endswith("\n"):
                                STATE["thought"] += "\n\n"
                        tokens += 1
                        think_tokens += 1
                        thought += reasoning
                        STATE["thought"] += reasoning
                        spinner.update(tokens=tokens, detail=thought[-400:])
                        spinner.start()

                    if content or delta.get("tool_calls"):
                        close_thought()

                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            while len(tool_calls_acc) <= idx:
                                tool_calls_acc.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                            if tc.get("id"):
                                tool_calls_acc[idx]["id"] += tc["id"]
                            fn = tc.get("function", {})
                            if fn.get("name"):
                                tool_calls_acc[idx]["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                tool_calls_acc[idx]["function"]["arguments"] += fn["arguments"]

                        tokens += 1
                        if not in_tool_call:
                            in_tool_call = True
                            if not silent and text and not text.endswith("\n"):
                                sys.stdout.write("\n")
                                sys.stdout.flush()

                        fn_name = next((tc["function"]["name"] for tc in tool_calls_acc if tc["function"]["name"]), "")
                        spinner.update(f"Preparing {fn_name}…" if fn_name else "Preparing tool call…",
                                       tokens=tokens, detail="")
                        spinner.start()

                    if content:
                        tokens += 1
                        text += content
                        if silent:
                            spinner.update(tokens=tokens)
                        else:
                            spinner.stop()
                            print(content, end="", flush=True)
            except KeyboardInterrupt:
                spinner.stop(dim("\n[interrupted]"))
                valid_tcs = [tc for tc in tool_calls_acc if tc["function"]["name"]]
                return (text, finish, valid_tcs) if tools is not None else (text, finish)
            finally:
                spinner.stop()
            close_thought()                     # the stream ended while the model was still thinking
            if not silent and text and not in_tool_call and not text.endswith("\n"):
                print()
        STATE["tok_in"] = (usage or {}).get("prompt_tokens") or 0
        STATE["tok_out"] += (usage or {}).get("completion_tokens") or tokens   # summed over agent steps
        STATE["ctx_used"] = (usage or {}).get("total_tokens") or (
            sum(len(str(m.get("content") or "")) for m in messages) // 4 + tokens)   # rough guess without usage
        secs = time.time() - t_start
        STATE["tps"] = (timings or {}).get("predicted_per_second") or (tokens / secs if secs > 0 else 0)
        valid_tcs = [tc for tc in tool_calls_acc if tc["function"]["name"]]
        if tools is not None:
            return text, finish, valid_tcs
        return text, finish
    except urllib.error.HTTPError as e:
        spinner.stop()
        print(red(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"))
    except Exception as e:
        spinner.stop()
        print(red(f"Request failed (is the Kaggle session still running? try /status): {e}"))
    return (None, None, []) if tools is not None else (None, None)


def trim_history(extra_chars=0):
    h = STATE["history"]
    while len(h) > 2 and (sum(len(m["content"]) for m in h) + extra_chars) > MAX_HISTORY_CHARS:
        del h[:2]


def compact(focus="", reason=""):
    """Replace the conversation with a model-written summary, keeping the latest exchange word for word.
    Returns True when the history was compacted."""
    h = STATE["history"]
    if not h:
        print(dim("Nothing to compact yet."))
        return False
    old, keep = (h[:-2], h[-2:]) if len(h) > 2 else (h, [])
    before = STATE.get("ctx_used") or sum(len(m["content"]) for m in h) // 4
    transcript = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in old)[-MAX_HISTORY_CHARS:]
    ask_for = "Summarize this conversation." + (f" Pay special attention to: {focus}" if focus else "")
    res = stream_chat([{"role": "system", "content": COMPACT_SYSTEM},
                       {"role": "user", "content": f"{transcript}\n\n---\n{ask_for}"}],
                      temp=0.2, max_tokens=2048, silent=True, status_label="Compacting conversation…")
    summary = (res[0] or "").strip()
    if not summary:
        print(red("Compaction failed; the conversation was left as it was."))
        return False
    STATE["history"] = [{"role": "user", "content": "Summary of our conversation so far:\n" + summary},
                        {"role": "assistant", "content": "Understood. I'll continue from this summary."}] + keep
    STATE["ctx_used"] = sum(len(m["content"]) for m in STATE["history"]) // 4    # rough until the next reply
    print(gold("◇ ") + dim(f"Compacted {len(old)} messages into a summary{reason} · "
                           f"~{fmt_k(before)} → ~{fmt_k(STATE['ctx_used'])} tokens"))
    return True


def maybe_compact(extra_chars=0):
    """Auto-compact before a call when the context window (or our character budget) is nearly full."""
    n, used = STATE.get("n_ctx") or 0, STATE.get("ctx_used") or 0
    chars = sum(len(m["content"]) for m in STATE["history"]) + extra_chars
    # > 4 messages: right after compacting there are 4 (summary pair + last exchange); don't re-summarize those
    if len(STATE["history"]) > 4 and ((n and used >= n * COMPACT_AT) or chars > MAX_HISTORY_CHARS):
        pct = f" ({used * 100 // n}% of context used)" if n and used else ""
        compact(reason=pct)
    trim_history(extra_chars)            # safety net if compaction failed or the summary is still too big


def merge_continuation(prev, more):
    """Join a continuation onto the earlier text. If we were inside an unclosed code block and the model
    reopened it with a fence such as ```html or ```, drop that repeated fence."""
    if prev.count("```") % 2 == 1:
        m = re.match(r"^\s*```[A-Za-z0-9_+.-]*[ \t]*\r?\n", more)
        if m:
            more = more[m.end():]
    return prev + more


def print_footer(t0, label):
    """Dim one-line summary under each reply:
    ◆ Agent · Qwen3-Coder-30B · 4.2s · 61 tok/s · 3.1K in / 412 out · ━━──────── 3.5K/32K context (11%)"""
    if not TTY:
        return
    tin, tout = STATE.get("tok_in"), STATE.get("tok_out")
    bits = [label, STATE.get("model"), fmt_secs(time.time() - t0),
            f"{STATE['tps']:.0f} tok/s" if STATE.get("tps") else "",
            (f"{fmt_k(tin)} in / " if tin else "") + (f"{fmt_k(tout)} out" if tout else "")]
    line = yellow("◆ ") + dim(" · ".join(b for b in bits if b))
    n, used = STATE.get("n_ctx") or 0, STATE.get("ctx_used") or 0
    if n and used:
        pct = used * 100 // n
        meter = f"{ctx_meter(used, n)} {fmt_k(used)}/{fmt_k(n)} context ({pct}%)"
        line += dim(" · ") + (red(meter) if pct >= 80 else gold(meter))
    print("\n" + line)


def ask(mode, content, temp=0.2, max_tokens=DEFAULT_MAX_TOKENS, silent=False, status_label=None):
    maybe_compact(extra_chars=len(content))
    t0 = time.time()
    STATE.update(thought="", tok_in=0, tok_out=0)
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
    if not silent:
        print_footer(t0, (mode or "chat").capitalize())
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


def format_diff(old_text, new_text, path):
    diff = list(difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3
    ))
    if not diff:
        return "(no changes)"
    out = []
    for line in diff:
        line_s = line.rstrip("\r\n")
        if line.startswith("+++") or line.startswith("---"):
            out.append(bold(line_s))
        elif line.startswith("@@"):
            out.append(gold(line_s))
        elif line.startswith("+"):
            out.append(green(line_s))
        elif line.startswith("-"):
            out.append(red(line_s))
        else:
            out.append(" " + line_s if not line_s.startswith(" ") else line_s)
    return "\n".join(out)


def prompt_confirm(prompt_msg, default=True):
    if STATE.get("auto_approve"):
        return True
    prompt_str = f"{yellow('?')} {bold(prompt_msg)} " + dim("[Y/n/a(ll)] " if default else "[y/N/a(ll)] ")
    try:
        ans = input(prompt_str).strip().lower()
        if not ans:
            return default
        if ans in ("a", "all"):
            STATE["auto_approve"] = True
            print(dim("(Auto-approve enabled for this session)"))
            return True
        return ans in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# ----------------------------------------------------------------------------- agent tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file in the workspace. Returns line-numbered text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative or absolute file path to read"},
                    "start_line": {"type": "integer", "description": "1-based start line (optional)"},
                    "end_line": {"type": "integer", "description": "1-based end line (optional)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit an existing file by replacing an exact string target with replacement. Displays a diff and requests user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to modify"},
                    "target": {"type": "string", "description": "Exact text sequence to be replaced, including whitespace and indentation"},
                    "replacement": {"type": "string", "description": "The replacement text"}
                },
                "required": ["path", "target", "replacement"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file or completely overwrite an existing file with the provided content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Full file content"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories in a given workspace path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list (default: .)", "default": "."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for a string or regex pattern across workspace files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term or regex pattern"},
                    "path": {"type": "string", "description": "Directory to search within (default: .)", "default": "."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the workspace (e.g. pytest, git status, python) and return its output and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command line to execute"}
                },
                "required": ["command"]
            }
        }
    }
]


TOOL_VERBS = {"read_file": ("Read", "path"), "edit_file": ("Edit", "path"), "write_file": ("Write", "path"),
              "list_directory": ("List", "path"), "search_files": ("Search", "query"), "run_command": ("Run", "command")}


def tool_read_file(path, start_line=1, end_line=None):
    p = Path(path).expanduser()
    if p.is_dir():
        return f"Error: {path} is a directory, not a file. Use list_directory instead."
    if not p.is_file():
        return f"Error: File not found: {path}"
    try:
        with p.open("rb") as f:
            chunk = f.read(4096)
            if b"\x00" in chunk:
                return f"Error: {path} is a binary file; cannot read."
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return f"Error reading {path}: {e}"

    total_lines = len(lines)
    start = max(1, start_line or 1)
    end = min(total_lines, end_line or total_lines)
    if start > total_lines:
        return f"File {path} only has {total_lines} lines (requested start line {start})."

    if end - start + 1 > 400:
        end = start + 399
        note = f"\n[truncated: showing lines {start}-{end} of {total_lines}. Use start_line={end+1} to read further]"
    else:
        note = f"\n[lines {start}-{end} of {total_lines}]"

    formatted = [f"{i:4d} | {lines[i-1]}" for i in range(start, end + 1)]
    return f"File: {path}\n" + "\n".join(formatted) + note


def tool_edit_file(path, target, replacement, auto_approve=False):
    p = Path(path).expanduser()
    if not p.is_file():
        return f"Error: File not found: {path}"
    try:
        content = p.read_text(encoding="utf-8")
    except OSError as e:
        return f"Error reading {path}: {e}"

    c_norm = content.replace("\r\n", "\n")
    t_norm = target.replace("\r\n", "\n")
    r_norm = replacement.replace("\r\n", "\n")

    count = c_norm.count(t_norm)
    if count == 0:
        return f"Error: target string not found in {path}. Make sure whitespace and indentation match exactly."
    if count > 1:
        return f"Error: target string occurs {count} times in {path}. Provide more surrounding context lines so the target is unique."

    new_content = c_norm.replace(t_norm, r_norm, 1)
    if "\r\n" in content:
        new_content = new_content.replace("\n", "\r\n")

    diff_text = format_diff(content, new_content, path)
    print("\n" + bold(f"Proposed edit for {path}:"))
    print("─" * 60)
    print(diff_text)
    print("─" * 60)

    if not auto_approve and not STATE.get("auto_approve"):
        if not prompt_confirm(f"Apply changes to {path}?"):
            return f"User declined the changes to {path}."

    try:
        tmp_path = p.with_suffix(p.suffix + ".borrow_tmp")
        tmp_path.write_text(new_content, encoding="utf-8")
        tmp_path.replace(p)
        print(green(f"✔ Successfully edited {path}"))
        return f"Successfully updated {path}."
    except OSError as e:
        return f"Error writing to {path}: {e}"


def tool_write_file(path, content, auto_approve=False):
    p = Path(path).expanduser()
    if p.is_dir():
        return f"Error: {path} is a directory, not a file."

    exists = p.is_file()
    lines_count = len(content.splitlines())
    action_label = f"Overwrite {path}" if exists else f"Create {path}"

    if exists:
        try:
            old_text = p.read_text(encoding="utf-8", errors="replace")
            print("\n" + bold(f"Proposed overwrite for {path}:"))
            print("─" * 60)
            print(format_diff(old_text, content, path))
            print("─" * 60)
        except OSError:
            pass

    if not auto_approve and not STATE.get("auto_approve"):
        if not prompt_confirm(f"{action_label} ({lines_count} lines)?"):
            return f"User declined to write {path}."

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = p.with_suffix(p.suffix + ".borrow_tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(p)
        print(green(f"✔ {action_label} ({lines_count} lines)"))
        return f"Successfully wrote {path} ({lines_count} lines)."
    except OSError as e:
        return f"Error writing {path}: {e}"


def tool_list_directory(path="."):
    p = Path(path).expanduser()
    if not p.is_dir():
        return f"Error: Directory not found: {path}"
    ignored = {".git", "__pycache__", ".venv", ".pytest_cache", "node_modules", "build", "dist", ".qa-venv"}
    try:
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except OSError as e:
        return f"Error listing {path}: {e}"

    results = []
    for e in entries:
        if e.name in ignored:
            continue
        if e.is_dir():
            results.append(f"  [dir]  {e.name}/")
        else:
            try:
                size = e.stat().st_size
                results.append(f"  {size:>8} B  {e.name}")
            except OSError:
                results.append(f"  {e.name}")
    return f"Directory {path}:\n" + ("\n".join(results) if results else "  (empty)")


def tool_search_files(query, path="."):
    p = Path(path).expanduser()
    if not p.is_dir():
        return f"Error: {path} is not a directory"
    ignored = {".git", "__pycache__", ".venv", ".pytest_cache", "node_modules", "build", "dist", ".qa-venv"}
    try:
        regex = re.compile(query, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex pattern '{query}': {e}"

    spinner = Spinner(f"Searching for '{query}'...")
    spinner.start()
    try:
        matches = []
        cwd = Path.cwd()
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if d not in ignored]
            for f in sorted(files):
                file_path = Path(root) / f
                try:
                    with file_path.open("rb") as fp:
                        if b"\x00" in fp.read(1024):
                            continue
                    with file_path.open("r", encoding="utf-8", errors="ignore") as fp:
                        for lnum, line in enumerate(fp, 1):
                            if regex.search(line):
                                try:
                                    rel = file_path.relative_to(cwd)
                                except ValueError:
                                    rel = file_path
                                matches.append(f"{rel}:{lnum}: {line.strip()[:150]}")
                                if len(matches) >= 50:
                                    return "Search results (capped at 50):\n" + "\n".join(matches)
                except OSError:
                    continue
        return "Search results:\n" + ("\n".join(matches) if matches else "No matches found.")
    finally:
        spinner.stop()


def tool_run_command(command, auto_approve=False):
    if not auto_approve and not STATE.get("auto_approve"):
        if not prompt_confirm(f"Execute `{command}`?"):
            return f"User declined to run command: {command}"
    spinner = Spinner(f"Running `{command}`...")
    spinner.start()
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        truncated = out[-4000:] if len(out) > 4000 else out
        return f"Command: {command}\nExit code: {r.returncode}\nOutput:\n{truncated}"
    except subprocess.TimeoutExpired:
        return f"Command `{command}` timed out after 120 seconds."
    except Exception as e:
        return f"Execution error: {e}"
    finally:
        spinner.stop()


def execute_tool(name, args, auto_approve=False):
    if name == "read_file":
        return tool_read_file(args.get("path", ""), args.get("start_line", 1), args.get("end_line"))
    elif name == "edit_file":
        return tool_edit_file(args.get("path", ""), args.get("target", ""), args.get("replacement", ""), auto_approve)
    elif name == "write_file":
        return tool_write_file(args.get("path", ""), args.get("content", ""), auto_approve)
    elif name == "list_directory":
        return tool_list_directory(args.get("path", "."))
    elif name == "search_files":
        return tool_search_files(args.get("query", ""), args.get("path", "."))
    elif name == "run_command":
        return tool_run_command(args.get("command", ""), auto_approve)
    return f"Unknown tool: {name}"


def get_workspace_summary(max_files=60):
    ignored = {".git", "__pycache__", ".venv", ".pytest_cache", "node_modules", "build", "dist", ".qa-venv"}
    files_list = []
    root = Path.cwd()
    for p in root.rglob("*"):
        if any(part in ignored for part in p.parts):
            continue
        if p.is_file():
            try:
                rel = p.relative_to(root)
                files_list.append(str(rel))
                if len(files_list) >= max_files:
                    break
            except ValueError:
                pass
    if not files_list:
        return "(empty or outside repository)"
    return "\n".join(f"- {f}" for f in sorted(files_list))


def is_pure_greeting(text):
    cleaned = re.sub(r"[^\w\s]", "", text.strip().lower())
    greetings = {
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
        "good evening", "howdy", "sup", "yo", "hi there", "hello there", "hey there",
        "who are you", "what can you do", "help", "thanks", "thank you"
    }
    return cleaned in greetings


def run_agent(prompt, auto_approve=False, max_turns=12):
    """Autonomous agent loop supporting tool calling, file reading, editing, and commands."""
    maybe_compact(extra_chars=len(prompt))
    ws_tree = get_workspace_summary()
    system_prompt = (
        BASE + "\n\n"
        "You are an autonomous pair-programming agent running in the user's terminal workspace (like Claude Code or agy).\n"
        "You have tools to inspect files, make direct in-place edits, write new files, search, and run commands.\n\n"
        "CRITICAL BEHAVIOR RULES:\n"
        "1. ONLY use tools when the user's request requires interacting with files, code, or the terminal.\n"
        "2. NEVER proactively call tools (such as list_directory, read_file, or write_file) for greetings (e.g. 'hi', 'hello'), general conversation, or conceptual questions. Just respond politely and ask how you can help.\n"
        "3. NEVER create files or edit code unless the user explicitly requested you to build, change, or create something.\n"
        "4. When the user DOES ask to modify, fix, or inspect code:\n"
        "   - First read the file with `read_file` or search with `search_files`.\n"
        "   - To modify existing code, use `edit_file` with the exact target text and replacement.\n"
        "   - Only use `write_file` for brand new files or full rewrites.\n"
        "   - Verify your work with `run_command` when appropriate.\n"
        "   - Be concise and explain what you did when done.\n\n"
        f"Workspace Directory: {Path.cwd()}\n"
        f"Workspace Files:\n{ws_tree}\n"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        *STATE["history"],
        {"role": "user", "content": prompt}
    ]

    turns = 0
    final_text = ""
    t0, failed = time.time(), False
    STATE.update(thought="", tok_in=0, tok_out=0)
    use_tools = None if is_pure_greeting(prompt) else TOOLS
    while turns < max_turns:
        turns += 1
        step_lbl = f"Thinking · step {turns}…" if turns > 1 else "Thinking…"
        res = stream_chat(messages, temp=0.2, max_tokens=DEFAULT_MAX_TOKENS, tools=use_tools, status_label=step_lbl)
        text, finish, tool_calls = res[0], res[1], res[2] if len(res) > 2 else []
        failed = text is None                   # the error is already printed
        use_tools = TOOLS

        if text:
            final_text += text

        if not tool_calls:
            # Model finished without calling any tools
            break

        # Record assistant tool call turn
        messages.append({
            "role": "assistant",
            "content": text or "",
            "tool_calls": tool_calls
        })

        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            call_id = tc.get("id") or f"call_{secrets.token_hex(4)}"
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {}
            if not isinstance(args, dict):
                args = {}

            verb, key = TOOL_VERBS.get(fn_name, (fn_name, None))
            arg = str(args.get(key, "")) if key else ", ".join(f"{k}={v!r}" for k, v in args.items())
            print(yellow("● ") + bold(verb) + " " + dim((arg.splitlines() or [""])[0][:100]))
            result = execute_tool(fn_name, args, auto_approve=auto_approve)
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(result)
            })

    if final_text:
        STATE["history"].append({"role": "user", "content": prompt})
        STATE["history"].append({"role": "assistant", "content": final_text})
        trim_history()
    if not failed:
        print_footer(t0, mode_label())
    return final_text


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
    if out["temp"] < 0:
        raise UserError("-t (temperature) must be 0.0 or greater")
    if out["max_tokens"] < 1:
        raise UserError("-m (max tokens) must be a positive integer (at least 1)")
    out["text"] = " ".join(out["text"])
    return out


def read_file_block(path):
    p = Path(path).expanduser()
    if p.is_dir():
        raise UserError(f"'{path}' is a directory, not a file. Attach individual files instead.")
    if not p.is_file():
        raise UserError(f"File not found: {path}")
    try:
        with p.open("rb") as f:
            chunk = f.read(4096)
            if b"\x00" in chunk:
                raise UserError(f"Cannot attach binary file: {path}")
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
    if p.is_dir():
        raise UserError(f"Cannot save to {path}: it is a directory, not a file.")
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
        print(yellow(bold("Commands")) + dim("   (type / to open the menu, Tab or arrows to choose)\n"))
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
        print(yellow(bold("Tips")))
        print("  @path        attach a file to any command or chat message (Tab completes paths)")
        print("  plain text   chat with memory; follow-ups can refer to earlier answers or reviews")
        print("  Shift+Tab    toggle auto-approve   |   Ctrl+J or Alt+Enter  new line in the prompt")
        print("  Ctrl+C       stop a reply, or clear the prompt (twice on an empty prompt quits)   |   Ctrl+D or /exit  quit")
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
    STATE["ctx_used"] = 0
    save_config()
    fetch_server_info()
    print("saved. " + ("server reachable." if STATE["online"] else red("saved, but the server did not answer /health yet.")))


def cmd_status(rest):
    if not STATE["url"]:
        raise UserError("Not connected. Use /connect <url> <key>")
    fetch_server_info()
    print(f"{STATE['url']}  ->  " + ("reachable" if STATE["online"] else red("NOT reachable (session ended or URL changed?)")))
    if STATE["model"]:
        print(dim(f"model: {STATE['model']}" + (f" · {fmt_k(STATE['n_ctx'])} context" if STATE["n_ctx"] else "")))
    print(dim(f"conversation: {len(STATE['history']) // 2} exchanges in memory"))


def cmd_think():
    thought = STATE.get("thought", "").strip()
    if not thought:
        print(dim("No reasoning recorded for the last reply (this model may not think out loud)."))
        return
    print(gold("◇ Reasoning behind the last reply"))
    for line in thought.splitlines():
        print(gold("│ ") + dim(line))


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


def cmd_diff():
    if not shutil.which("git"):
        raise UserError("git is not installed.")
    code, out = run_capture("git diff")
    if not out.strip():
        print(dim("No unstaged git changes."))
        return
    for l in out.splitlines():
        if l.startswith("+"):
            print(green(l))
        elif l.startswith("-"):
            print(red(l))
        elif l.startswith("@@"):
            print(gold(l))
        else:
            print(l)


def cmd_revert(rest):
    target = rest.strip() if rest else "."
    if not shutil.which("git"):
        raise UserError("git is not installed.")
    if prompt_confirm(f"Revert git changes in {target}?"):
        code, out = run_capture(f"git checkout -- {shlex.quote(target)}")
        print(green(f"✔ Reverted {target}") if code == 0 else red(f"Revert failed: {out}"))


def handle(line, raise_errors=False):
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
            elif name == "compact":
                compact(focus=rest.strip())
            elif name == "clear":
                STATE["history"].clear()
                STATE.update(ctx_used=0, thought="", show_splash=True)
                print(dim("conversation cleared"))
            elif name == "think":
                cmd_think()
            elif name == "connect":
                cmd_connect(rest)
            elif name == "status":
                cmd_status(rest)
            elif name == "auto":
                STATE["auto_approve"] = not STATE.get("auto_approve", False)
                status = green("ENABLED (no confirmation prompts)") if STATE["auto_approve"] else dim("DISABLED (will ask confirmation)")
                print(f"Auto-approval {status}")
            elif name == "diff":
                cmd_diff()
            elif name == "revert":
                cmd_revert(rest)
            elif name == "edit":
                args = parse_args(name, rest)
                run_agent(args["text"], auto_approve=STATE.get("auto_approve"))
            else:
                args = parse_args(name, rest)
                cmd_shell(args) if name == "shell" else cmd_generic(name, args)
        else:
            args = parse_args_chat(line)
            run_agent(build_content(args), auto_approve=STATE.get("auto_approve"))
    except UserError as e:
        if raise_errors:
            raise
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


def input_fragments(line, first=True):
    """Colour one prompt line as you type: /command, -flags and @files each get their own style."""
    is_cmd = first and line.startswith("/")
    out = []
    for i, tok in enumerate(re.split(r"(\s+)", line)):
        if i == 0 and is_cmd:
            name = tok[1:].lower()
            known = any(n.startswith(name) for n in COMMANDS)    # a half-typed /con is still fine
            out.append(("class:cmd" if known else "class:cmd-bad", tok))
        elif tok.startswith("@") and len(tok) > 1:
            out.append(("class:file", tok))
        elif is_cmd and re.match(r"^--?[A-Za-z]", tok):
            out.append(("class:flag", tok))
        else:
            out.append(("", tok))
    return out


def make_lexer():
    from prompt_toolkit.lexers import Lexer

    class InputLexer(Lexer):
        def lex_document(self, document):
            return lambda i: input_fragments(document.lines[i], first=i == 0)
    return InputLexer()


def caret_on(now, typed_at, period=0.53):
    """Blink phase of the caret: solid while you type, then on/off every `period` seconds."""
    return int((now - typed_at) / period) % 2 == 0


def make_caret():
    """Draws our own blinking caret. The terminal's cursor can't be trusted to blink: prompt_toolkit sends
    'stop blinking' every time it redraws, and legacy Windows consoles ignore cursor shapes entirely."""
    from prompt_toolkit.layout.processors import Processor, Transformation
    from prompt_toolkit.layout.utils import explode_text_fragments

    class BlinkingCaret(Processor):
        def apply_transformation(self, ti):
            doc = ti.document
            if ti.lineno != doc.cursor_position_row:
                return Transformation(ti.fragments)
            col = ti.source_to_display(doc.cursor_position_col)
            frags = list(explode_text_fragments(ti.fragments))
            on = caret_on(time.time(), INPUT.get("typed", 0))
            frags[col:col] = [("class:caret", "│") if on else ("", " ")]   # same width either way: no jitter
            return Transformation(frags, source_to_display=lambda i: i if i <= col else i + 1,
                                  display_to_source=lambda i: i if i <= col else i - 1)
    return BlinkingCaret()


def pulse_color(t, lo=(0x7a, 0x5c, 0x00), hi=(0xff, 0xd0, 0x00)):
    """A slow yellow 'breathing' colour so the prompt box never looks frozen."""
    k = (math.sin(t * 2.2) + 1) / 2
    return "#%02x%02x%02x" % tuple(round(a + (b - a) * k) for a, b in zip(lo, hi))


# ----------------------------------------------------------------------------- the screen (yellow on black)
Y, Y_DIM, Y_SHADOW = "#ffd000", "#a37c00", "#5c4000"
BG, PANEL, TEXT, MUTED = "#0a0a0a", "#1a1a1a", "#e8e8e8", "#7c7c7c"
LOGO_GRADIENT = ["#ffee55", "#ffdd22", "#ffd000", "#ffc400", "#ffb400", "#ffa500"]    # one colour per pixel row
LOGO_FONT = {                                                                        # 6 pixels tall
    "B": ["#####.", "##..##", "#####.", "##..##", "##..##", "#####."],
    "O": [".####.", "##..##", "##..##", "##..##", "##..##", ".####."],
    "R": ["#####.", "##..##", "#####.", "##.##.", "##..##", "##..##"],
    "W": ["##...##", "##...##", "##.#.##", "##.#.##", "#######", ".##.##."],
}
PLACEHOLDERS = ['Ask anything…  "what does this project do?"', 'Ask anything…  "fix the failing test in tests/"',
                'Ask anything…  "add type hints to utils.py"', 'Ask anything…  "review @app.py for security bugs"']
TIPS = [("Type ", "@", " to attach a file, Tab completes the path"),
        ("Press ", "shift+tab", " to let the agent edit and run commands without asking"),
        ("Run ", "/think", " to read the model's full reasoning for the last reply"),
        ("Use ", "/review @file", " for a strict code review, most important issues first"),
        ("Use ", "/clear", " to start a fresh conversation and come back here"),
        ("Pipe input in from your shell: ", "git diff | borrow /review -", "")]
CONNECT_TIP = ("Paste the ", "/connect <url> <key>", " line that qwen_serv.py prints on Kaggle")
STYLE_RULES = {
    "screen": f"bg:{BG}",
    "box": f"bg:{PANEL} {TEXT}",
    "bar": Y,
    "placeholder": "#5a5a5a",
    "auto-suggestion": "#5a5a5a",
    "mode": f"{Y} bold",
    "auto": f"bg:{Y} #000000 bold",
    "model": f"{TEXT} bold",
    "muted": MUTED,
    "key": f"{TEXT} bold",
    "tip": Y,
    "tagline": Y_DIM,
    "caret": f"{Y} bold",
    "cmd": f"{Y} bold",
    "cmd-bad": "#ff5f5f bold",
    "flag": "#d7a7ff",
    "file": "#5fafff underline",
    "good": "#5fd75f",
    "warn": "#ffaf00",
    "bad": "#ff5f5f",
    "completion-menu.completion": "bg:#1c1c1c #d0d0d0",
    "completion-menu.completion.current": f"bg:{Y} #000000",
    "completion-menu.meta.completion": "bg:#141414 #808080",
    "completion-menu.meta.completion.current": f"bg:{Y_DIM} #000000",
    "scrollbar.background": "bg:#262626",
    "scrollbar.button": f"bg:{Y}",
}
INPUT = {}   # the prompt's buffer (history, completer) lives here between prompts


def logo_fragments():
    """BORROW in block letters with a drop shadow, drawn with half blocks: two square pixels per cell."""
    if shutil.get_terminal_size().columns < 50:
        return [("class:mode", "B O R R O W")]
    rows = [".".join(LOGO_FONT[c][y] for c in "BORROW") for y in range(6)]
    on = lambda x, y: 0 <= y < 6 and 0 <= x < len(rows[0]) and rows[y][x] == "#"
    head = int(time.time() * 30) % (len(rows[0]) + 60) - 10        # a glint sweeps across, then pauses
    lit = lambda x, y: "#fff6c0" if abs(x - y - head) < 2 else LOGO_GRADIENT[y]
    color = lambda x, y: lit(x, y) if on(x, y) else Y_SHADOW if on(x - 1, y - 1) else BG
    out = []
    for y in range(0, 8, 2):
        out += [(f"fg:{color(x, y)} bg:{color(x, y + 1)}", "▀") for x in range(len(rows[0]) + 1)]
        out.append(("", "\n"))
    return out[:-1]


def info_fragments():
    """The line under the input: mode · model · context."""
    out = [("class:auto", " AUTO ") if STATE.get("auto_approve") else ("class:mode", "Agent"), ("class:muted", "  ·  ")]
    if not STATE["url"]:
        return out + [("class:bad", "not connected"), ("class:muted", "   /connect <url> <key>")]
    if STATE.get("online") is None:
        return out + [("class:muted", "connecting…")]
    if not STATE["online"]:
        return out + [("class:bad", "server not reachable"), ("class:muted", "   /status to retry")]
    n, used = STATE.get("n_ctx") or 0, STATE.get("ctx_used") or 0
    out.append(("class:model", STATE.get("model") or "connected"))
    if n and used:
        pct = used * 100 // n
        out += [("class:bad" if pct >= 80 else "class:tip", f"   {ctx_meter(used, n)}"),
                ("class:warn" if pct >= 80 else "class:muted", f" {fmt_k(used)} / {fmt_k(n)} context ({pct}%)")]
    elif n:
        out.append(("class:muted", f"   {fmt_k(n)} context"))
    return out


def hint_fragments():
    if time.time() - INPUT.get("ctrl_c", 0) < 2:
        return [("class:warn", "press ctrl+c again to quit ")]
    out = []
    for key, label in (("/", "commands"), ("@", "files"), ("shift+tab", "auto-approve"), ("ctrl+d", "quit")):
        out += [("class:key", "   " + key), ("class:muted", " " + label)]
    return out + [("", " ")]


def status_fragments():
    """Bottom bar of the start screen: folder and connection."""
    cwd, home = str(Path.cwd()), str(Path.home())
    cwd = "~" + cwd[len(home):] if cwd.startswith(home) else cwd
    host = re.sub(r"^https?://", "", STATE["url"]).split("/")[0]
    if not STATE["url"]:
        dot, msg = "class:bad", "not connected"
    elif STATE.get("online") is None:
        dot, msg = "class:warn", f"connecting to {host}"
    elif STATE["online"]:
        dot, msg = "class:good", host
    else:
        dot, msg = "class:bad", f"{host} is not reachable"
    room = shutil.get_terminal_size().columns - len(msg) - 28       # leave space for the version on the right
    cwd = cwd if len(cwd) <= room else "…" + cwd[-room + 1:] if room > 10 else ""
    return [("", "  "), (dot, "● "), ("class:muted", f"{msg}    {cwd}")]


def read_input(splash=False):
    """Draw the prompt box and return what the user submitted. splash=True draws the full-screen start screen.
    Raises EOFError when the user quits."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.filters import Condition, has_completions
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import (ConditionalContainer, Float, FloatContainer, HSplit, Layout, VSplit, Window,
                                       WindowAlign)
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension as D
    from prompt_toolkit.layout.menus import CompletionsMenu
    from prompt_toolkit.layout.processors import AfterInput, AppendAutoSuggestion, ConditionalProcessor
    from prompt_toolkit.output import ColorDepth
    from prompt_toolkit.styles import Style

    buf = INPUT.get("buf")
    if buf is None:
        buf = INPUT["buf"] = Buffer(multiline=True, history=FileHistory(str(HISTORY_PATH)), completer=make_completer(),
                                    complete_while_typing=True, auto_suggest=AutoSuggestFromHistory())
        buf.on_cursor_position_changed += lambda _: INPUT.__setitem__("typed", time.time())   # caret solid while typing
    empty = Condition(lambda: not buf.text)
    INPUT["ctrl_c"] = 0
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(e):
        state = buf.complete_state
        if state and state.current_completion:       # Enter picks the highlighted menu item
            buf.apply_completion(state.current_completion)
        elif buf.text.strip():
            buf.append_to_history()
            e.app.exit(result=buf.text)

    @kb.add("escape", "enter")
    @kb.add("c-j")
    def _newline(e):
        buf.insert_text("\n")

    @kb.add("s-tab", filter=~has_completions)
    def _toggle_auto(e):
        STATE["auto_approve"] = not STATE.get("auto_approve")

    @kb.add("c-c")
    def _ctrl_c(e):
        if buf.text:
            buf.reset()
        elif time.time() - INPUT.get("ctrl_c", 0) < 2:
            e.app.exit(exception=EOFError)
        else:
            INPUT["ctrl_c"] = time.time()

    @kb.add("c-d", filter=empty)
    def _quit(e):
        e.app.exit(exception=EOFError)

    off = random.randrange(len(PLACEHOLDERS))
    placeholder = lambda: PLACEHOLDERS[(off + int(time.time() / 4)) % len(PLACEHOLDERS)]   # rotates every 4s
    input_win = Window(BufferControl(buf, lexer=make_lexer(), input_processors=[
        AppendAutoSuggestion(),
        ConditionalProcessor(AfterInput(placeholder, style="class:placeholder"), empty),
        make_caret()]),
        height=D(min=1, max=10), dont_extend_height=True, wrap_lines=True, always_hide_cursor=True)
    box = VSplit([
        Window(width=1, char="┃", style=lambda: "fg:" + pulse_color(time.time())),
        Window(width=2),
        HSplit([Window(height=1), input_win, Window(height=1),
                Window(FormattedTextControl(info_fragments), height=1), Window(height=1)]),
        Window(width=1),
    ], style="class:box")
    hints = Window(FormattedTextControl(hint_fragments), height=1, align=WindowAlign.RIGHT)

    if splash:
        width = lambda: D.exact(max(30, min(78, shutil.get_terminal_size().columns - 4)))
        tips = TIPS if STATE["url"] else [CONNECT_TIP]

        def tip():                              # a new tip every 8s
            a, b, c = tips[(off + int(time.time() / 8)) % len(tips)]
            return [("fg:" + pulse_color(time.time()), "● Tip  "), ("class:muted", a), ("class:key", b), ("class:muted", c)]
        body = HSplit([
            Window(),
            Window(FormattedTextControl(logo_fragments), height=4, align=WindowAlign.CENTER),
            Window(height=1),
            Window(FormattedTextControl([("class:tagline", "brain on remote  ·  run on workstation")]),
                   height=1, align=WindowAlign.CENTER),
            Window(height=2),
            VSplit([Window(), HSplit([box, hints], width=width), Window()]),
            Window(height=2),
            Window(FormattedTextControl(tip), height=1, align=WindowAlign.CENTER),
            Window(),
            VSplit([Window(FormattedTextControl(status_fragments)),
                    Window(FormattedTextControl([("class:muted", f"borrow {__version__}  ")]), dont_extend_width=True)],
                   height=1),
        ], style="class:screen")
    else:
        body = HSplit([box, hints, ConditionalContainer(Window(height=8), filter=has_completions)])  # room for the menu

    truecolor = os.environ.get("COLORTERM") in ("truecolor", "24bit") or os.environ.get("WT_SESSION")
    app = Application(
        layout=Layout(FloatContainer(body, floats=[Float(xcursor=True, ycursor=True,
                                                         content=CompletionsMenu(max_height=8, scroll_offset=1))]),
                      focused_element=input_win),
        key_bindings=kb, style=Style.from_dict(STYLE_RULES), full_screen=splash, erase_when_done=True,
        refresh_interval=0.1, color_depth=ColorDepth.TRUE_COLOR if truecolor else None)   # 0.1s drives the animations
    try:
        return app.run()
    finally:
        buf.reset()


def print_user_block(text):
    """Echo what the user sent as a panel with a yellow bar, matching the prompt box it came from."""
    if not TTY:
        return
    cols = shutil.get_terminal_size((80, 20)).columns - 1
    lines = []
    for para in text.replace("\t", "    ").splitlines():
        lines += textwrap.wrap(para, max(10, cols - 4)) or [""]
    if len(lines) > 12:
        lines = lines[:12] + [f"… {len(lines) - 12} more lines"]
    for i, line in enumerate(["", *lines, ""]):
        line = line.ljust(cols - 3)
        if i == 1 and text.startswith("/"):     # the /command keeps its yellow after you send it
            name = line.split(" ", 1)[0]
            line = f"\033[38;5;220m\033[1m{name}\033[22m\033[38;5;255m{line[len(name):]}"
        print(f"\033[48;5;234m\033[38;5;220m┃\033[38;5;255m  {line}\033[0m")


def ensure_prompt_toolkit():
    try:
        __import__("prompt_toolkit")
        return True
    except ImportError:
        print(dim("Installing prompt_toolkit (gives you the / menu)..."))
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "prompt_toolkit"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(red("Could not install prompt_toolkit; using basic mode (no menu). Run: pip install prompt_toolkit"))
            return False
        return True


def basic_prompt():
    try:
        import readline
        names = ["/" + n + " " for n in COMMANDS]
        readline.set_completer(lambda t, i: ([n for n in names if n.startswith(t)] + [None])[i])
        readline.parse_and_bind("tab: complete")
    except ImportError:
        pass
    return lambda: input("> ")


def repl():
    fancy = TTY and sys.stdin.isatty() and ensure_prompt_toolkit()
    prompt = None if fancy else basic_prompt()
    if fancy:
        threading.Thread(target=fetch_server_info, daemon=True).start()
    else:
        print(bold("borrow") + dim("  Qwen coder CLI   type / for commands, @ to attach files, /help for details"))
        if not STATE["url"]:
            print(red("Not connected yet. Type /connect <url> <key>  (or set ASK_URL and ASK_KEY)."))
        elif not health():
            print(red("Saved endpoint is not reachable. If you started a new Kaggle session, use /connect with the new URL and key."))
    splash = fancy
    while True:
        try:
            line = read_input(splash) if fancy else prompt()
        except KeyboardInterrupt:
            if not fancy:
                print(dim("(Ctrl+D or /exit to quit)"))
            continue
        except EOFError:
            break
        except Exception as e:                  # a console prompt_toolkit cannot drive
            print(red(f"Prompt box unavailable ({e}); using basic mode."))
            fancy, splash, prompt = False, False, basic_prompt()
            continue
        if fancy:
            if splash:
                sys.stdout.write("\033[2J\033[H")   # leave the start screen onto a blank page
                splash = False
            print_user_block(line.strip())
            print()
        try:
            if not handle(line):
                break
        except KeyboardInterrupt:
            print(dim("\n[interrupted]"))
        splash = STATE.pop("show_splash", False) and fancy
        if fancy and not splash:
            print()
    print(dim("bye"))


def enable_ansi():
    """Windows consoles only understand colour codes once VT processing is switched on."""
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        for std in (-11, -12):                  # stdout, stderr
            handle, mode = k32.GetStdHandle(std), ctypes.c_uint32()
            if k32.GetConsoleMode(handle, ctypes.byref(mode)):
                k32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


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
    if "-y" in args or "--yes" in args:
        STATE["auto_approve"] = True
        args = [a for a in args if a not in ("-y", "--yes")]
    if args:                                    # one-shot mode
        if "-" in args:                         # a lone "-" means: read piped input from stdin
            args = [a for a in args if a != "-"]
            STATE["piped"] = sys.stdin.read()
        line = subprocess.list2cmdline(args) if os.name == "nt" else " ".join(shlex.quote(a) for a in args)
        try:
            handle(line, raise_errors=True)
        except UserError as e:
            print(red(str(e)))
            sys.exit(1)
        return
    repl()


def run():
    enable_ansi()
    try:
        main()
    except BrokenPipeError:                     # e.g. output piped into `head`
        pass
    except SystemExit:
        raise


if __name__ == "__main__":
    run()
