"""Read-only machine/model probe for deployers (and their agents): what this box has, which sub-plugins can be on, and a
suggested environment.  Prints one JSON object; touches no GPU context, starts nothing.

  python -m vllm_video_ttft.probe [--model PATH_OR_DIR] [--tp N]
"""
import argparse, json, os, shutil, subprocess

from . import REQUIRED, OPENCV, _version, is_qwen3vl, version_problems



def _read(p):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return None


def _cpulist(s):
    out = set()
    for part in (s or "").split(","):
        if part:
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


def cpus():
    allowed = sorted(os.sched_getaffinity(0))
    cores = set()                                   # (package, core) pairs = physical cores among the allowed CPUs
    for c in allowed:
        pkg = _read(f"/sys/devices/system/cpu/cpu{c}/topology/physical_package_id")
        core = _read(f"/sys/devices/system/cpu/cpu{c}/topology/core_id")
        cores.add((pkg, core))
    nodes = {}
    base = "/sys/devices/system/node"
    if os.path.isdir(base):
        for d in sorted(os.listdir(base)):
            if d.startswith("node") and d[4:].isdigit():
                nodes[int(d[4:])] = [c for c in _cpulist(_read(f"{base}/{d}/cpulist")) if c in set(allowed)]
    load = _read("/proc/loadavg")
    return {"logical_allowed": len(allowed), "physical_allowed": len(cores), "numa_nodes": nodes,
            "loadavg_1m": float(load.split()[0]) if load else None}


def memory():
    kv = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        kv[k] = int(v.split()[0]) if v.split() else 0
    shm = shutil.disk_usage("/dev/shm") if os.path.isdir("/dev/shm") else None
    gib = lambda kb: round(kb / 2**20, 1)
    return {"total_gib": gib(kv.get("MemTotal", 0)), "available_gib": gib(kv.get("MemAvailable", 0)),
            "dev_shm_total_gib": round(shm.total / 2**30, 1) if shm else None,
            "dev_shm_free_gib": round(shm.free / 2**30, 1) if shm else None}


def gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        i, name, tot, used, drv, cc = [x.strip() for x in line.split(",")]
        rows.append({"index": int(i), "name": name, "mem_total_mib": int(tot), "mem_used_mib": int(used),
                     "mem_free_mib": int(tot) - int(used), "driver": drv, "compute_capability": cc})
    return rows


def model_info(path):
    if not path:
        return None
    cfg = os.path.join(path, "config.json")
    vp = os.path.join(path, "video_preprocessor_config.json")
    info = {"path": path, "architectures": None, "video_processor": None, "qwen3_vl_family": is_qwen3vl(path)}
    try:
        with open(cfg) as f:
            info["architectures"] = json.load(f).get("architectures")
    except (OSError, ValueError):
        info["error"] = f"no readable {cfg} (pass a local model directory)"
    for p in (vp, os.path.join(path, "preprocessor_config.json")):
        try:
            with open(p) as f:
                d = json.load(f)
            info["video_processor"] = d.get("video_processor_type") or info["video_processor"]
        except (OSError, ValueError):
            pass
    return info


def decide(c, m, g, model, tp):
    """Per sub-plugin: can it be on here, and why."""
    qwen = bool(model and model.get("qwen3_vl_family"))
    unknown = model is None or not model.get("architectures")
    free_phys = max(1, c["physical_allowed"] - int(round(c["loadavg_1m"] or 0)))
    vc_gpu = [x for x in g if x["mem_free_mib"] >= 3500]
    seg_ok = free_phys >= 8
    out = {
        "m2_seg_decode": {"ok": seg_ok, "why": ("works with any model on vLLM's default OpenCV video path; "
                                                f"{free_phys} idle physical cores")
                          + ("" if seg_ok else " — fewer than 8 idle cores, little to gain")},
        "m2_copyfree_resize": {"ok": qwen or unknown,
                               "why": "Qwen3-VL video processor only" + ("" if qwen else " (model is not Qwen3-VL)" if not unknown else " (model unknown: pass --model)")},
        "m2_pipe_resize": {"ok": qwen or unknown,
                           "why": "Qwen3-VL video processor only; serves only when vLLM's processor cache is off "
                                  "(--mm-processor-cache-gb 0)"},
        "m2_vision_chunk": {"ok": (qwen or unknown) and bool(vc_gpu),
                            "why": "Qwen3-VL vision tower only; needs m2_pipe_resize and ~3 GiB free on one GPU "
                                   f"({len(vc_gpu)} GPU(s) have >= 3.5 GiB free right now; a vLLM server started with "
                                   "--gpu-memory-utilization leaves room only if you lower it); changes the tower "
                                   "output: not bit-identical vs. the per-GPU tower (34 of 100 test videos identical; median relative "
                                   "diff 2.9% on the rest)"},
    }
    return out, free_phys


def suggest(c, decisions, free_phys, g, tp):
    """The formula in README "怎么按机器选参数".  Slice scan (offline, one NUMA node = 24 physical / 48 logical cores):
    16 slices x 4 threads = 64 decode threads was best, 32 slices no faster, so decode threads = 4/3 x logical cores
    (capped at 64).  Nodes 0+1 (96 logical) only ran 16 x 4.  OMP 12 was measured against 1 only.  Other core counts
    are extrapolation; confirm with the selfcheck timings."""
    env = {}
    exact = all(decisions[k]["ok"] for k in ("m2_seg_decode", "m2_copyfree_resize", "m2_pipe_resize"))
    env["VLLM_VIDEO_TTFT_PRESET"] = "all" if exact and decisions["m2_vision_chunk"]["ok"] else "exact" if exact else "off"
    if env["VLLM_VIDEO_TTFT_PRESET"] == "off" and decisions["m2_seg_decode"]["ok"]:
        env["M2_SEG_DECODE"] = "1"; env["M2_SEG_SKIP_NONREF"] = "1"
    nodes = [n for n, cl in c["numa_nodes"].items() if cl]
    load = int(round(c["loadavg_1m"] or 0))
    if len(nodes) > 1:                          # API process, decode pool and frame memory on one socket
        use = nodes[:2] if len(nodes) >= 4 else nodes[:1]
        env["M2_SEG_WNODES"] = ",".join(map(str, use))
        L = sum(len(c["numa_nodes"][n]) for n in use)
        P = max(1, c["physical_allowed"] * len(use) // len(nodes))
    else:
        L, P = c["logical_allowed"], c["physical_allowed"]
    L, P = max(2, L - load), max(1, P - load)
    omp = max(1, min(12, P // 4))               # resize/patchify threads in the API process
    env["OMP_NUM_THREADS"] = str(omp)
    if decisions["m2_seg_decode"]["ok"]:
        d = min(64, (4 * L) // 3)               # decode threads: 64 on 48 logical cores was the measured best
        t = 4 if d >= 16 else 1
        env["M2_SEG_THREADS"] = str(t)
        env["M2_SEG_N"] = str(max(2, min(16, d // t)))
    if env.get("VLLM_VIDEO_TTFT_PRESET") == "all":
        best = max((x for x in g if x["mem_free_mib"] >= 3500), key=lambda x: x["mem_free_mib"], default=None)
        env["M2_VC_DEVICE"] = f"cuda:{best['index']}" if best and tp > 1 else "cuda:0"
    if tp > 1:
        env["VLLM_MQ_MAX_CHUNK_BYTES_MB"] = "256"   # single GPU: vLLM does not use this broadcast, leave it unset
    args = ["--mm-processor-cache-gb 0"]
    if tp > 1:
        args.append("--mm-encoder-tp-mode data")
    if len(nodes) > 1:
        args.insert(0, f"numactl --cpunodebind={nodes[0]} --membind={nodes[0]} (prefix of the vllm serve command)")
    return env, args


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="local model directory (reads config.json only)")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size you plan to serve with")
    a = ap.parse_args()
    c, m, g = cpus(), memory(), gpus()
    model = model_info(a.model)
    decisions, free_phys = decide(c, m, g, model, a.tp)
    env, args = suggest(c, decisions, free_phys, g, a.tp)
    versions = {d: _version(d) for d in (*REQUIRED, OPENCV[0], "torch")}
    print(json.dumps({
        "versions": versions, "version_problems": version_problems(),
        "cpu": c, "memory": m, "gpus": g, "model": model, "plugins": decisions,
        "suggested_env": env, "suggested_vllm_args": args,
        "notes": [
            "Only time-to-first-token of single video requests was measured (answers cut at 16 tokens).",
            "m2_seg_decode: a segmented request uses ~31 CPU core-seconds on average (12-57 by length); the worker pool keeps a median 0.85-1.0 GiB resident, at most 2.6 GiB.",
            "Run `python -m vllm_video_ttft.selfcheck VIDEO...` before serving to confirm bit-identical inputs.",
        ],
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
