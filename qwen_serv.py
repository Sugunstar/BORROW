#!/usr/bin/env python3
"""
Qwen coder on Kaggle (free 2x T4 GPUs) -> OpenAI-style endpoint -> public tunnel -> your terminal.

HOW TO USE
  1. Kaggle notebook > right sidebar > Session options:
       - Accelerator: GPU T4 x2   (needed for the 30B coder model; one GPU runs the small model)
       - Internet: On             (needs a phone-verified Kaggle account)
  2. Paste this ENTIRE file into ONE code cell and run it.
  3. First run takes ~15-25 min (build + ~19 GB download). When ready it prints the URL and
     key. Paste the connection code it prints into `borrow` on your own computer.
  4. The cell keeps running on purpose (it holds the server open). Stop the cell to shut down.

Every step stops with a clear error and a log tail if it fails.
"""
import glob, json, os, re, secrets, shutil, socket, subprocess, time
import requests

# ----------------------------- settings you may edit -----------------------------
MODEL_CHOICE = "auto"   # "auto" or a preset name below. Either way, if a model fails the next one that fits is tried
THINKING     = True     # Enable reasoning/thinking when supported by the model (better accuracy)
PORT         = 8080
UA           = {"User-Agent": "curl/8.5.0"}   # Cloudflare can reject python's default user agent

PRESETS = {   # tried in this order (best first); min_vram_gb = total VRAM across all visible GPUs
    # Best quality: dense 27B, top open-model coding scores at this size (19.8 GB file)
    "qwen38_27b":     dict(repo="unsloth/Qwen3.8-27B-GGUF",                  quant="UD-Q5_K_M", ctx=32768, min_vram_gb=26),
    # Fast MoE (3B active per token): several times quicker on T4s, slightly lower quality (22.1 GB)
    "qwen36_35b_a3b": dict(repo="unsloth/Qwen3.6-35B-A3B-GGUF",              quant="UD-Q4_K_M", ctx=32768, min_vram_gb=26),
    # The same 27B at 4-bit, in case the 5-bit build runs out of memory (16.5 GB)
    "qwen38_27b_q4":  dict(repo="unsloth/Qwen3.8-27B-GGUF",                  quant="UD-Q4_K_M", ctx=32768, min_vram_gb=22),
    # Older models, known to work with this script
    "coder30b":       dict(repo="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", quant="Q4_K_M", ctx=32768, min_vram_gb=26),
    "qwen35_27b":     dict(repo="unsloth/Qwen3.5-27B-GGUF",                  quant="Q4_K_M", ctx=16384, min_vram_gb=24),
    "coder32b_dense": dict(repo="unsloth/Qwen2.5-Coder-32B-Instruct-GGUF",   quant="Q4_K_M", ctx=32768, min_vram_gb=26),
    "coder14b_dense": dict(repo="unsloth/Qwen2.5-Coder-14B-Instruct-GGUF",   quant="Q4_K_M", ctx=32768, min_vram_gb=12),
    "qwen35_9b":      dict(repo="unsloth/Qwen3.5-9B-GGUF",                   quant="Q4_K_M", ctx=16384, min_vram_gb=10),
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


def preset_order(total_vram):
    """Presets to try, best first: MODEL_CHOICE (if set), then every preset that fits in the VRAM."""
    order = [k for k, p in PRESETS.items() if total_vram >= p["min_vram_gb"]]
    if MODEL_CHOICE != "auto":
        order = [MODEL_CHOICE] + [k for k in order if k != MODEL_CHOICE]
    if not order:
        raise RuntimeError(f"Only {total_vram:.0f} GB VRAM visible; need at least 10 GB.")
    return order


class GpuError(RuntimeError):
    """The server runs on CPU. A problem with llama-server itself, so trying other models won't help."""


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


def server_env(binary):
    """Env for llama-server: its own dir (libggml-cuda.so) plus the CUDA runtime libs on the library path."""
    env = dict(os.environ)
    dirs = [os.path.dirname(binary)] + glob.glob("/usr/local/cuda/lib64") + glob.glob("/usr/local/nvidia/lib64")
    env["LD_LIBRARY_PATH"] = ":".join(dirs + [env.get("LD_LIBRARY_PATH", "")])
    return env


def try_download_prebuilt_llama():
    """Attempt to fetch an official prebuilt CUDA-enabled llama-server binary from GitHub releases.
    Saves 10-15 minutes of compilation time."""
    global SERVER_BIN
    try:
        url = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5"
        req = requests.get(url, headers=UA, timeout=10)
        if req.status_code != 200:
            return False
        releases = req.json()
        asset_url = None
        for rel in releases:
            for a in rel.get("assets", []):
                name = a["name"].lower()
                if "ubuntu" in name and "cuda" in name and "x64" in name and name.endswith(".tar.gz") and not name.startswith("cudart-"):
                    asset_url = a["browser_download_url"]
                    break
            if asset_url:
                break
        if not asset_url:
            return False

        print(f"  Downloading prebuilt llama-server ({asset_url.split('/')[-1]})...")
        tar_path = f"{WORK}/llama_prebuilt.tar.gz"
        sh(f"wget -q -O {tar_path} '{asset_url}'")
        sh(f"tar -xzf {tar_path} -C {WORK}/")

        # Find the extracted llama-server executable
        for root, _, files in os.walk(WORK):
            if "llama-server" in files:
                cand = os.path.join(root, "llama-server")
                os.chmod(cand, 0o755)
                ok, out = cuda_check(cand)
                if ok:
                    SERVER_BIN = cand
                    print(f"  Prebuilt llama-server verified on GPU: {SERVER_BIN}")
                    return True
                print("  Prebuilt llama-server cannot see the GPU, compiling instead. It said:\n" + out[-1500:])
                return False
    except Exception as e:
        print(f"  Prebuilt binary attempt skipped ({e}), falling back to compilation.")
    return False


def cuda_check(binary):
    """(True, output) if this llama-server can see a CUDA GPU. `--version` is no proof: a build whose CUDA
    backend fails to load still runs, silently on CPU. A usable GPU shows up as a 'CUDA0: Tesla T4' line."""
    try:
        r = subprocess.run([binary, "--list-devices"], capture_output=True, text=True,
                           env=server_env(binary), timeout=120)
        out = (r.stdout + r.stderr).strip()
        return r.returncode == 0 and re.search(r"\bCUDA\d+:", out) is not None, out
    except Exception as e:
        return False, str(e)


def gpu_mem_used_gb():
    """VRAM in use across all GPUs (GB), or -1 if nvidia-smi fails."""
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return sum(float(x) for x in r.stdout.split()) / 1024 if r.returncode == 0 else -1


def setup_llama_cpp(arch):
    global SERVER_BIN
    if SERVER_BIN and os.path.exists(SERVER_BIN):
        ok, out = cuda_check(SERVER_BIN)
        if ok:
            print("llama-server already available and sees the GPU, skipping.")
            return
        print("Existing llama-server cannot see the GPU, rebuilding it. It said:\n" + out[-1500:])
        shutil.rmtree(os.path.dirname(os.path.dirname(SERVER_BIN)), ignore_errors=True)   # its build/ folder
    print("Checking for prebuilt CUDA-accelerated llama-server (fast setup)...")
    if try_download_prebuilt_llama():
        return
    print("Compiling llama.cpp from source (fallback)...")
    build_llama_cpp(arch)


def build_llama_cpp(arch):
    global SERVER_BIN
    if SERVER_BIN and os.path.exists(SERVER_BIN):
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
    SERVER_BIN = f"{src}/build/bin/llama-server"
    if not os.path.exists(SERVER_BIN):
        raise RuntimeError(f"Build finished but {SERVER_BIN} is missing. See {WORK}/cmake_build.log")
    print("Built:", SERVER_BIN)


def download_model(repo, quant):
    # 1. First check if model is already available on disk (e.g. attached as a Kaggle Dataset)
    for input_dir in ("/kaggle/input", "/kaggle/working"):
        if os.path.isdir(input_dir):
            for root, _, files in os.walk(input_dir):
                for f in files:
                    if f.lower().endswith(".gguf") and quant.lower() in f.lower() and "mmproj" not in f.lower():
                        found = os.path.join(root, f)
                        size_gb = os.path.getsize(found) / 1e9
                        print(f"Using pre-mounted model from dataset: {found} ({size_gb:.1f} GB) [0s wait!]")
                        return found

    # 2. Enable ultra-fast multi-threaded download via hf-transfer
    try:
        import hf_transfer
    except ImportError:
        sh("pip install -q hf-transfer")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

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
    print(f"Downloading {len(matches)} file(s) from {repo} (accelerated with hf-transfer)...")
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
    if re.search(r"Qwen3\.\d", repo) and not THINKING:   # Qwen3.5/3.6/3.8 can think; Qwen3-Coder can't
        cmd += ["--chat-template-kwargs", json.dumps({"enable_thinking": False})]
    env = server_env(SERVER_BIN)
    env["LLAMA_ARG_FLASH_ATTN"] = "on"
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
    log = open(log_path, errors="replace").read()
    gpu_lines = [l.strip() for l in log.splitlines() if "offloaded" in l]
    vram = gpu_mem_used_gb()          # ground truth: the model's weights sit in VRAM, many GB of it
    if 0 <= vram < 1:
        print("\n".join(l for l in log.splitlines() if re.search(r"cuda|gpu|offload|error|fail", l, re.I))[-3000:])
        stop_server()
        raise GpuError(f"llama-server loaded the model on CPU (only {vram:.1f} GB VRAM in use). The GPU lines "
                           f"from its log are printed above; paste them to get this fixed. Full log: {log_path}")
    print(f"Server ready. {vram:.1f} GB VRAM in use.", gpu_lines[-1] if gpu_lines else "")


def test_chat(base, prompt="Write a Python one-liner that reverses a string."):
    t0 = time.time()
    r = requests.post(base + "/v1/chat/completions",
                      headers={"Authorization": f"Bearer {API_KEY}", **UA},
                      json={"messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 200, "temperature": 0.2,
                            "top_p": 0.8, "top_k": 20}, timeout=300)
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
              "running. Test from your own computer using the URL and key printed below.")
        return False
    finally:
        u3c.allowed_gai_family = original


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
    order = preset_order(total_vram)
    print("Models to try, best first:", " -> ".join(order))

    print("[2/6] Setting up llama.cpp")
    setup_llama_cpp(arch)
    for i, key in enumerate(order):             # a model that fails to download, load or answer -> next one
        preset = PRESETS[key]
        print(f"[3/6] Downloading model: {key} -> {preset['repo']} ({preset['quant']}), context {preset['ctx']}")
        try:
            model_path = download_model(preset["repo"], preset["quant"])
            print("[4/6] Starting server")
            start_server(model_path, preset["ctx"], preset["repo"])
            test_chat(f"http://127.0.0.1:{PORT}")
            break
        except GpuError:
            raise
        except Exception as e:
            stop_server()
            if i == len(order) - 1:
                raise
            print(f"\nModel {key} failed: {e}\nFalling back to {order[i + 1]}...\n")
    print("[5/6] Opening tunnel")
    public_url = start_tunnel()
    print("\nPublic URL:", public_url)
    print("\n--- optional self-test through the tunnel (from inside Kaggle) ---")
    selftest_tunnel(public_url)

    print("[6/6] Connection details")
    print("\n" + "=" * 70)
    print(f"URL: {public_url}")
    print(f"KEY: {API_KEY}")
    print("=" * 70)
    print("\nConnection code (copy this line and paste it into borrow):")
    print(f"  {public_url} {API_KEY}")
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