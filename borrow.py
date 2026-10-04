#!/usr/bin/env python3
"""borrow - a Claude-Code-style terminal for your self-hosted Qwen coder endpoint.

Start it:      borrow
One-shot:      borrow /review @app.py       (pipe input in with a lone "-": git diff | borrow /review -)
Connect:       set ASK_URL / ASK_KEY (printed by the Kaggle script) or type  /connect <url> <key>

Inside the CLI type  /  to open the command menu, @ to attach files, plain text to chat.
No flags to remember: type /help for every command, its flags and what it does.
"""
import difflib, json, os, re, shlex, subprocess, sys, urllib.error, urllib.request
from pathlib import Path

__version__ = "0.1.0"
CONFIG_PATH = Path.home() / ".borrow.json"
LEGACY_CONFIG_PATH = Path.home() / ".qwen_cli.json"   # older versions saved here
HISTORY_PATH = Path.home() / ".borrow_history"
UA = {"User-Agent": "curl/8.5.0"}
MAX_FILE_CHARS = 60_000
MAX_HISTORY_CHARS = 70_000      # keeps the conversation inside the 32K-token context window

BASE = "You are a precise senior software engineer. Be concise and correct."
MODES = {
    "code":    "Reply with ONLY the code in a single fenced code block. No explanation.",
    "explain": "Explain what the given code or error does in plain language, concisely, with bullet points.",
    "review":  "Review the given code like a strict senior reviewer. List concrete bugs, risks and improvements, most important first.",
    "test":    "Write thorough unit tests for the given code (pytest for Python unless told otherwise). Reply with only the test code in one fenced block.",
    "fix":     "Find and fix the bug or error described. Reply with the corrected code in one fenced block, then 1-3 lines on the cause.",
    "doc":     "Add clear docstrings and comments to the given code without changing behaviour. Reply with the full updated code in one fenced block.",
    "commit":  "Write a concise conventional-commit message for the given diff: subject line under 72 chars, then optional bullets. Reply with only the message.",
    "shell":   "Reply with ONLY a single bash command that does what is asked. No explanation, no code fence.",
}

F_OUT  = ("-o FILE", "save the code (or answer) to FILE")
F_TEMP = ("-t TEMP", "creativity 0.0-1.0 (default 0.2)")
F_MAX  = ("-m N",    "max tokens to generate (default 2048)")
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


def dim(s):  return f"\033[2m{s}\033[0m" if TTY else s
def bold(s): return f"\033[1m{s}\033[0m" if TTY else s
def red(s):  return f"\033[31m{s}\033[0m" if TTY else s


class UserError(Exception):
    pass


# ----------------------------------------------------------------------------- config
def load_config():
    cfg = {}
    for path in (CONFIG_PATH, LEGACY_CONFIG_PATH):
        try:
            cfg = json.loads(path.read_text())
            break
        except Exception:
            continue
    STATE["url"] = (os.environ.get("ASK_URL") or cfg.get("url", "")).rstrip("/")
    STATE["key"] = os.environ.get("ASK_KEY") or cfg.get("key", "")


def save_config():
    CONFIG_PATH.write_text(json.dumps({"url": STATE["url"], "key": STATE["key"]}))
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass


def health():
    try:
        req = urllib.request.Request(STATE["url"] + "/health", headers=UA)
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


# ----------------------------------------------------------------------------- model call
def stream_chat(messages, temp, max_tokens):
    """Stream a reply to the terminal. Returns the text (partial if interrupted with Ctrl+C), or None on error."""
    if not STATE["url"] or not STATE["key"]:
        print(red("Not connected. Type /connect <url> <key>, or set ASK_URL and ASK_KEY."))
        return None
    body = json.dumps({"messages": messages, "max_tokens": max_tokens, "temperature": temp, "stream": True}).encode()
    req = urllib.request.Request(STATE["url"] + "/v1/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + STATE["key"], **UA})
    text = ""
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            print()
            try:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    delta = json.loads(payload)["choices"][0].get("delta", {}).get("content") or ""
                    text += delta
                    print(delta, end="", flush=True)
            except KeyboardInterrupt:
                print(dim("\n[interrupted]"), end="")
            print("\n")
        return text
    except urllib.error.HTTPError as e:
        print(red(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"))
    except Exception as e:
        print(red(f"Request failed (is the Kaggle session still running? try /status): {e}"))
    return None


def trim_history():
    h = STATE["history"]
    while len(h) > 2 and sum(len(m["content"]) for m in h) > MAX_HISTORY_CHARS:
        del h[:2]


def ask(mode, content, temp=0.2, max_tokens=2048):
    system = BASE + (" " + MODES[mode] if mode else "")
    messages = [{"role": "system", "content": system}] + STATE["history"] + [{"role": "user", "content": content}]
    text = stream_chat(messages, temp, max_tokens)
    if text is not None:
        STATE["history"] += [{"role": "user", "content": content}, {"role": "assistant", "content": text}]
        trim_history()
    return text


def first_block(text):
    m = re.search(r"```[a-zA-Z0-9_+-]*\n(.*?)```", text, re.S)
    return m.group(1) if m else text


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
        out["max_tokens"] = int(out.get("max_tokens", 2048))
    except ValueError:
        raise UserError("-t must be a number like 0.2 and -m a whole number like 2048")
    out["text"] = " ".join(out["text"])
    return out


def read_file_block(path):
    p = Path(path).expanduser()
    if not p.is_file():
        raise UserError(f"File not found: {path}")
    data = p.read_text(encoding="utf-8", errors="replace")
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
    Path(path).expanduser().write_text(data + "\n", encoding="utf-8")
    print(dim(f"saved to {path}"))


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
    spec = COMMANDS[name]
    extra = []
    if name == "commit":
        git = "git diff --staged" if args.get("staged") else "git diff HEAD"
        code, out = run_capture(git)
        if code != 0 and not args.get("staged"):
            code, out = run_capture("git diff")
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
    text = ask(spec["mode"], content, args["temp"], args["max_tokens"])
    if text and args.get("out"):
        save_output(args["out"], text, as_code=spec["mode"] in ("code", "test", "doc", "fix"))


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
    files = [t[1:] for t in toks if t.startswith("@") and len(t) > 1]
    text = " ".join(t if not (t.startswith("@") and len(t) > 1) else t[1:] for t in toks)
    return {"text": text, "files": files, "temp": 0.2, "max_tokens": 2048}


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