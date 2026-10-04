#!/usr/bin/env python3
"""
Qwen coder on Kaggle (free 2x T4 GPUs) -> OpenAI-style endpoint -> public tunnel -> your terminal.

HOW TO USE
  1. Kaggle notebook > right sidebar > Session options:
       - Accelerator: GPU T4 x2   (needed for the 30B coder model; one GPU runs the small model)
       - Internet: On             (needs a phone-verified Kaggle account)
  2. Paste this ENTIRE file into ONE code cell and run it.
  3. First run takes ~15-25 min (build + ~19 GB download). When ready it prints an `ask.py`
     script. Save it on your own computer and use it from your terminal.
  4. The cell keeps running on purpose (it holds the server open). Stop the cell to shut down.

Every step stops with a clear error and a log tail if it fails.
"""
import glob, json, os, re, secrets, shutil, socket, subprocess, time
import requests

# ----------------------------- settings you may edit -----------------------------
MODEL_CHOICE = "auto"   # "auto" | "coder30b" | "qwen35_27b" | "qwen35_9b"
THINKING     = False    # Qwen3.5 only: True = better reasoning but much slower
PORT         = 8080
UA           = {"User-Agent": "curl/8.5.0"}   # Cloudflare can reject python's default user agent

PRESETS = {   # min_vram_gb = total VRAM across all visible GPUs needed for this preset
    "coder30b":   dict(repo="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", quant="Q4_K_M", ctx=32768, min_vram_gb=26),
    "qwen35_27b": dict(repo="unsloth/Qwen3.5-27B-GGUF",                  quant="Q4_K_M", ctx=16384, min_vram_gb=24),
    "qwen35_9b":  dict(repo="unsloth/Qwen3.5-9B-GGUF",                   quant="Q4_K_M", ctx=16384, min_vram_gb=10),
}
# ---------------------------------------------------------------------------------

WORK = None          # chosen at runtime (needs lots of free disk)
SERVER_BIN = None
API_KEY = secrets.token_urlsafe(24)
SERVER_PROC = None
TUNNEL_PROC = None


def sh(cmd, cwd=None, log=None, show_progress=False, quiet=False):
    """Run a shell command. Full output goes to `log`. On failure print the last 60 lines and raise."""
    log = log or f"{WORK or '/tmp'}/last_cmd.log"
    out, last_pct = [], -10
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, shell=True, cwd=cwd, executable="/bin/bash",
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            f.write(line)
            out.append(line)
            if show_progress:
                m = re.match(r"\[\s*(\d+)%\]", line)
                if m and int(m.group(1)) >= last_pct + 10:
                    last_pct = int(m.group(1))
                    print(f"  build progress: {last_pct}%", flush=True)
        p.wait()
    if p.returncode != 0:
        if not quiet:
            print("".join(out[-60:]))
        raise RuntimeError(f"Command failed (exit {p.returncode}): {cmd}\nFull log: {log}")
    return "".join(out)


def check_internet():
    try:
        requests.get("https://huggingface.co", headers=UA, timeout=10)
    except requests.RequestException:
        raise RuntimeError("No internet access (DNS lookup failed). Fix: Kaggle right sidebar > Session options "
                           "> Internet > On. If the toggle is greyed out, verify your phone number in your "
                           "Kaggle account settings first. Then run this cell again.") from None


def check_gpus():
    """Return (CUDA arch list like '75', total VRAM in GB). Raise a clear error if there is no GPU."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()] if r.returncode == 0 else []
    except FileNotFoundError:
        lines = []
    if not lines:
        raise RuntimeError("No GPU found. Sidebar > Session options > Accelerator > GPU T4 x2, then restart.")
    caps, total_mb = set(), 0
    for i, l in enumerate(lines):
        name, mem, cap = [x.strip() for x in l.split(",")]
        print(f"GPU {i}: {name} | {int(float(mem))/1024:.1f} GB | compute capability {cap}")
        caps.add(cap.replace(".", ""))
        total_mb += int(float(mem))
    return ";".join(sorted(caps)), total_mb / 1024


def pick_workdir(need_gb=35):
    """Pick the writable location with the most free disk (the 30B model alone is ~19 GB)."""
    best, best_free = None, -1
    for d in ("/kaggle/temp", "/tmp", "/kaggle/working"):
        if os.path.isdir(d) and os.access(d, os.W_OK):
            free = shutil.disk_usage(d).free / 1e9
            print(f"  disk {d}: {free:.0f} GB free")
            if free > best_free:
                best, best_free = d, free
    if best is None:
        raise RuntimeError("No writable directory found.")
    if best_free < need_gb:
        print(f"WARNING: only {best_free:.0f} GB free in {best}; the download may fail.")
    path = os.path.join(best, "qwen_server")
    os.makedirs(path, exist_ok=True)
    return path


def choose_preset(total_vram):
    if MODEL_CHOICE != "auto":
        return MODEL_CHOICE, PRESETS[MODEL_CHOICE]
    for key in ("coder30b", "qwen35_9b"):
        if total_vram >= PRESETS[key]["min_vram_gb"]:
            return key, PRESETS[key]
    raise RuntimeError(f"Only {total_vram:.0f} GB VRAM visible; need at least 10 GB.")


def find_nvcc():
    path = shutil.which("nvcc")
    if path:
        return path
    for c in ["/usr/local/cuda/bin/nvcc"] + sorted(glob.glob("/usr/local/cuda-*/bin/nvcc")):
        if os.path.exists(c):
            return c
    return None


def find_libcuda():
    """Locate libcuda (the NVIDIA driver library) so CMake can link against it.
    Prefers the real driver library; falls back to the CUDA toolkit's stub."""
    found = []
    try:
        out = subprocess.run("ldconfig -p", shell=True, capture_output=True, text=True).stdout
        found += [l.split("=>")[-1].strip() for l in out.splitlines() if "libcuda.so" in l and "=>" in l]
    except Exception:
        pass
    for pat in ("/usr/lib/x86_64-linux-gnu/libcuda.so*", "/usr/lib64/libcuda.so*",
                "/usr/local/nvidia/lib64/libcuda.so*", "/usr/local/cuda*/targets/x86_64-linux/lib/stubs/libcuda.so",
                "/usr/local/cuda*/lib64/stubs/libcuda.so"):
        found += glob.glob(pat)
    found = [f for f in dict.fromkeys(found) if os.path.exists(f)]
    real = [f for f in found if "/stubs/" not in f]
    stubs = [f for f in found if "/stubs/" in f]
    return (real or stubs or [None])[0]


def build_llama_cpp(arch):
    if os.path.exists(SERVER_BIN):
        print("llama-server already built, skipping.")
        return
    nvcc = find_nvcc()
    if not nvcc:
        raise RuntimeError("CUDA compiler (nvcc) not found. Make sure a GPU accelerator is selected.")
    os.environ["CUDACXX"] = nvcc
    os.environ["PATH"] = os.path.dirname(nvcc) + ":" + os.environ["PATH"]
    if not shutil.which("cmake"):
        sh("pip install -q cmake")
    src = f"{WORK}/llama.cpp"
    if not os.path.isdir(src):
        sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {src}")
    sh("rm -rf build", cwd=src)

    base_flags = (f"-DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER={nvcc} -DCMAKE_CUDA_ARCHITECTURES='{arch}' "
                  "-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_OPENSSL=OFF "
                  "-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF")
    cfg_log = f"{WORK}/cmake_configure.log"
    libcuda = find_libcuda()
    print("CUDA driver library:", libcuda or "not found")
    attempts = []
    if libcuda:
        attempts.append((f"linking driver lib {libcuda}", f"-DCUDA_cuda_driver_LIBRARY={libcuda}"))
    # Fallback: skip CUDA VMM so the driver library is not needed at link time (tiny perf cost)
    attempts.append(("without CUDA VMM", "-DGGML_CUDA_NO_VMM=ON"))
    for desc, extra in attempts:
        print(f"Configuring ({desc})...")
        sh("rm -rf build", cwd=src)
        try:
            sh(f"cmake -B build {base_flags} {extra}", cwd=src, log=cfg_log, quiet=True)
            break
        except RuntimeError:
            print("  configure attempt failed, trying next option...")
    else:
        print("".join(open(cfg_log).readlines()[-60:]))
        raise RuntimeError(f"cmake configure failed with every option. Full log: {cfg_log}")

    jobs = max(1, min(4, os.cpu_count() or 2))
    print(f"Compiling with {jobs} jobs (about 10-15 min, this is normal)...")
    sh(f"cmake --build build --config Release -j{jobs} --target llama-server",
       cwd=src, log=f"{WORK}/cmake_build.log", show_progress=True)
    if not os.path.exists(SERVER_BIN):
        raise RuntimeError(f"Build finished but {SERVER_BIN} is missing. See {WORK}/cmake_build.log")
    print("Built:", SERVER_BIN)


def download_model(repo, quant):
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        sh("pip install -q huggingface_hub")
        from huggingface_hub import list_repo_files, hf_hub_download
    files = [f for f in list_repo_files(repo)
             if f.lower().endswith(".gguf") and "mmproj" not in f.lower()]
    matches = sorted([f for f in files if quant.lower() in f.lower()], key=len)
    if not matches:
        raise RuntimeError(f"No '{quant}' file in {repo}. Files available:\n" + "\n".join(files))
    print(f"Downloading {len(matches)} file(s) from {repo} (large, be patient)...")
    paths = [hf_hub_download(repo, f, local_dir=f"{WORK}/models") for f in matches]
    first = next((p for p in paths if "00001-of" in p), paths[0])   # split models: point at shard 1
    total = sum(os.path.getsize(p) for p in paths) / 1e9
    print(f"Model ready: {first} ({total:.1f} GB total)")
    return first


def stop_server():
    global SERVER_PROC
    if SERVER_PROC and SERVER_PROC.poll() is None:
        SERVER_PROC.terminate()
        try:
            SERVER_PROC.wait(10)
        except subprocess.TimeoutExpired:
            SERVER_PROC.kill()
    SERVER_PROC = None


def start_server(model_path, ctx, repo):
    global SERVER_PROC
    stop_server()
    log_path = f"{WORK}/server.log"
    cmd = [SERVER_BIN, "-m", model_path, "-ngl", "99", "-c", str(ctx),
           "--host", "127.0.0.1", "--port", str(PORT),
           "--api-key", API_KEY, "--alias", "qwen"]
    if "Qwen3.5" in repo and not THINKING:      # Qwen3-Coder has no thinking mode, so no flag needed
        cmd += ["--chat-template-kwargs", json.dumps({"enable_thinking": False})]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.path.dirname(SERVER_BIN) + ":" + env.get("LD_LIBRARY_PATH", "")
    SERVER_PROC = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
                                   env=env, cwd=os.path.dirname(SERVER_BIN))
    deadline = time.time() + 600       # loading ~19 GB from disk can take a few minutes
    while time.time() < deadline:
        if SERVER_PROC.poll() is not None:
            print(open(log_path).read()[-4000:])
            raise RuntimeError("llama-server exited early. Log tail is printed above.")
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    else:
        print(open(log_path).read()[-4000:])
        raise TimeoutError("Server did not become ready within 10 minutes.")
    gpu_lines = [l.strip() for l in open(log_path) if "offloaded" in l]
    print("Server ready.", gpu_lines[-1] if gpu_lines else f"(see {log_path} for GPU offload info)")


def test_chat(base, prompt="Write a Python one-liner that reverses a string."):
    t0 = time.time()
    r = requests.post(base + "/v1/chat/completions",
                      headers={"Authorization": f"Bearer {API_KEY}", **UA},
                      json={"messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 200, "temperature": 0.2}, timeout=300)
    r.raise_for_status()
    j = r.json()
    print(j["choices"][0]["message"]["content"])
    n, dt = j.get("usage", {}).get("completion_tokens"), time.time() - t0
    if n:
        print(f"\n[{n} tokens in {dt:.1f}s = {n/dt:.1f} tokens/sec]")


def selftest_tunnel(url):
    """Optional check from inside Kaggle. Never fatal: Kaggle's own network can fail to reach the
    public tunnel URL even though your computer reaches it fine."""
    import urllib3.util.connection as u3c
    original = u3c.allowed_gai_family
    u3c.allowed_gai_family = lambda: socket.AF_INET      # Kaggle has no IPv6 route; use IPv4 only
    try:
        for attempt in range(1, 6):
            try:
                test_chat(url)
                print("Tunnel self-test passed.")
                return True
            except Exception as e:
                print(f"  self-test attempt {attempt}/5 failed: {type(e).__name__}")
                time.sleep(4)
        try:
            host = url.split("//")[1]
            print("  DNS for tunnel host:", sorted({r[4][0] for r in socket.getaddrinfo(host, 443)}))
        except Exception as e:
            print("  DNS lookup failed:", e)
        print("Self-test from inside Kaggle did not pass. This check is optional: the server is still "
              "running. Test from your own computer with ask.py (printed below).")
        return False
    finally:
        u3c.allowed_gai_family = original


ASK_TEMPLATE = r'''#!/usr/bin/env python3
"""ask.py - talk to your Qwen coder endpoint from the terminal.

  ask.py "question"                          plain answer (streams as it is generated)
  ask.py --code "fizzbuzz in python" > f.py  only the code block
  cat app.py | ask.py --review               review a file
  cat app.py | ask.py --test --code > test_app.py
  python x.py 2>&1 | ask.py --fix            paste an error and get a fix
  git diff | ask.py --commit                 commit message from a diff
  ask.py --shell "find files over 100MB"     one shell command (READ it before running!)
  ask.py -f a.py -f b.py "why does a call b wrong?"
  ask.py --chat                              interactive chat with memory (/clear, /exit)
Set ASK_URL / ASK_KEY in your environment to point at a new session without editing this file.
"""
import argparse, json, os, re, sys, urllib.request, urllib.error

URL = os.environ.get("ASK_URL", "__URL__").rstrip("/")
KEY = os.environ.get("ASK_KEY", "__KEY__")

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

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("prompt", nargs="*", help="your instruction")
for name in MODES:
    if name != "code":
        ap.add_argument("--" + name, dest="mode", action="store_const", const=name,
                        help=MODES[name].split(". ")[0].rstrip("."))
ap.add_argument("-c", "--code", action="store_true", help="output only the first code block")
ap.add_argument("-f", "--file", action="append", default=[], help="include a file (repeatable)")
ap.add_argument("-o", "--out", help="write the result to this file")
ap.add_argument("-t", "--temp", type=float, default=0.2)
ap.add_argument("-m", "--max-tokens", type=int, default=2048)
ap.add_argument("--chat", action="store_true", help="interactive chat")
ap.add_argument("-n", "--no-stdin", action="store_true", help="do not read piped input (use in scripts/IDEs)")
a = ap.parse_args()

mode = a.mode or ("code" if a.code else None)
system = BASE + (" " + MODES[mode] if mode else "")


def call(messages, stream):
    body = json.dumps({"messages": messages, "max_tokens": a.max_tokens,
                       "temperature": a.temp, "stream": stream}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + KEY,
        "User-Agent": "curl/8.5.0"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            if not stream:
                return json.load(r)["choices"][0]["message"]["content"]
            text = ""
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
            return text
    except urllib.error.HTTPError as e:
        sys.exit("HTTP %s: %s" % (e.code, e.read().decode(errors="replace")[:500]))
    except Exception as e:
        sys.exit("Request failed (is the Kaggle session still running?): %s" % e)


def first_block(text):
    m = re.search(r"```[a-zA-Z0-9_+-]*\n(.*?)```", text, re.S)
    return m.group(1) if m else text


if a.chat:
    messages = [{"role": "system", "content": system}]
    print("Chat started. /clear resets, /exit quits.")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q in ("/exit", "/quit"):
            break
        if q == "/clear":
            messages = messages[:1]
            print("(history cleared)")
            continue
        if not q:
            continue
        messages.append({"role": "user", "content": q})
        print("qwen> ", end="", flush=True)
        reply = call(messages, True)
        print()
        messages.append({"role": "assistant", "content": reply})
        messages = messages[:1] + messages[1:][-12:]        # keep within the 32K context
    sys.exit(0)

parts = []
if a.prompt:
    parts.append(" ".join(a.prompt))
for path in a.file:
    try:
        parts.append("File: %s\n```\n%s\n```" % (path, open(path, encoding="utf-8", errors="replace").read()))
    except OSError as e:
        sys.exit("Cannot read %s: %s" % (path, e))
if not a.no_stdin and not sys.stdin.isatty():
    piped = sys.stdin.read()
    if piped.strip():
        parts.append(("Input:\n" + piped) if parts else piped)
if not parts:
    ap.print_help()
    sys.exit(1)

messages = [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]
extract = a.code or mode == "shell"
text = call(messages, stream=not (extract or a.out))
if extract:
    text = first_block(text).strip("\n") if "```" in text else text.strip()
if a.out:
    open(a.out, "w", encoding="utf-8").write(text + "\n")
    print("wrote " + a.out, file=sys.stderr)
elif extract:
    print(text)
else:
    print()
'''


def stop_tunnel():
    global TUNNEL_PROC
    if TUNNEL_PROC and TUNNEL_PROC.poll() is None:
        TUNNEL_PROC.terminate()
    TUNNEL_PROC = None


def start_tunnel():
    global TUNNEL_PROC
    stop_tunnel()
    cf = f"{WORK}/cloudflared"
    if not os.path.exists(cf):
        sh(f"wget -q -O {cf} https://github.com/cloudflare/cloudflared/releases/latest/download/"
           f"cloudflared-linux-amd64 && chmod +x {cf}")
    log_path = f"{WORK}/tunnel.log"
    TUNNEL_PROC = subprocess.Popen(
        [cf, "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate", "--protocol", "http2"],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    url = None
    for _ in range(90):
        if TUNNEL_PROC.poll() is not None:
            break
        # (?!api\.) skips "api.trycloudflare.com", which cloudflared also prints in its logs
        m = re.search(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com", open(log_path).read())
        if m:
            url = m.group(0)
            break
        time.sleep(1)
    if not url:
        print(open(log_path).read()[-3000:])
        raise RuntimeError("Could not get a tunnel URL. Log tail printed above; re-run the cell.")
    for _ in range(30):            # wait until the public URL actually answers
        try:
            if requests.get(url + "/health", headers=UA, timeout=5).status_code == 200:
                return url
        except requests.RequestException:
            pass
        time.sleep(2)
    print("Warning: tunnel URL created but not answering yet; it may need a few more seconds.")
    return url


def main():
    global WORK, SERVER_BIN
    print("[1/6] Checking internet, GPUs and disk")
    check_internet()
    arch, total_vram = check_gpus()
    WORK = pick_workdir()
    SERVER_BIN = f"{WORK}/llama.cpp/build/bin/llama-server"
    key, preset = choose_preset(total_vram)
    print(f"Model preset: {key} -> {preset['repo']} ({preset['quant']}), context {preset['ctx']}")

    print("[2/6] Building llama.cpp")
    build_llama_cpp(arch)
    print("[3/6] Downloading model")
    model_path = download_model(preset["repo"], preset["quant"])
    print("[4/6] Starting server")
    start_server(model_path, preset["ctx"], preset["repo"])
    test_chat(f"http://127.0.0.1:{PORT}")
    print("[5/6] Opening tunnel")
    public_url = start_tunnel()
    print("\nPublic URL:", public_url)
    print("\n--- optional self-test through the tunnel (from inside Kaggle) ---")
    selftest_tunnel(public_url)

    print("[6/6] Your terminal client")
    client = ASK_TEMPLATE.replace("__URL__", public_url).replace("__KEY__", API_KEY)
    open(f"{WORK}/ask.py", "w").write(client)
    print("\n" + "=" * 70)
    print("Save everything between the lines as ask.py on YOUR computer:")
    print("=" * 70)
    print(client)
    print("=" * 70)
    print('Then run, in your terminal:   python3 ask.py --code "write fizzbuzz in python" > fizz.py')
    print("More modes: python3 ask.py --help")
    print("\nNEXT SESSION: no need to re-save ask.py, just set these two variables to the new values:")
    print(f"  Mac/Linux:   export ASK_URL={public_url} ASK_KEY={API_KEY}")
    print(f"  PowerShell:  $env:ASK_URL='{public_url}'; $env:ASK_KEY='{API_KEY}'")
    print(f"  Windows cmd: set ASK_URL={public_url}& set ASK_KEY={API_KEY}")
    print("\nUsing qwen_cli.py (the / menu)? Start it and type:")
    print(f"  /connect {public_url} {API_KEY}")
    print("Keep this cell running. A new session means a new URL and key: run the cell again.")

    try:
        while True:                    # hold the session open; notice if a process dies
            time.sleep(30)
            if SERVER_PROC is None or SERVER_PROC.poll() is not None:
                print("llama-server stopped. Log tail:\n" + open(f"{WORK}/server.log").read()[-2000:])
                break
            if TUNNEL_PROC is None or TUNNEL_PROC.poll() is not None:
                print("Tunnel stopped. Run the cell again to get a new URL.")
                break
    except KeyboardInterrupt:
        print("Interrupted, shutting down.")
    finally:
        stop_tunnel()
        stop_server()


try:
    main()
except Exception as e:
    stop_tunnel()
    stop_server()
    # print instead of re-raising: Jupyter's traceback printer can crash on chained errors
    print(f"\nFAILED: {e}")