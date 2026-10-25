"""Offline self-check: run vLLM's own video path (VideoMediaIO -> processor.apply) on your videos with the plugins off
and on, and compare what would be sent to the vision tower.

  python -m vllm_video_ttft.selfcheck --model MODEL_DIR VIDEO [VIDEO ...] [--out report.json] [--no-gpu]

Configurations (each video, in this order):
  off    no plugin (reference)
  exact  m2_seg_decode + skip non-reference + m2_copyfree_resize + m2_pipe_resize   (Qwen3-VL models)
         m2_seg_decode + skip non-reference only                                    (other models)
  all    exact + m2_vision_chunk, only for Qwen3-VL and only if a GPU is visible (--no-gpu skips it)
Pass = pixel_values_videos and video_grid_thw bit-identical to `off` (for `all`: the pixel_values the tower copy was fed),
and every plugin that should act on the video reports served.  Runs with M2_STRICT=1, so a plugin that silently stepped
aside fails the check.  Also prints decode+preprocess seconds per configuration, as a hint only (one run, no warm-up
repetition): use it to see whether slicing pays off on this machine for videos near M2_SEG_MIN_SRC_FRAMES.
"""
import argparse, hashlib, json, os, sys, time

from . import _visible_gpus, is_qwen3vl, register_all

SWITCHES = ("M2_SEG_DECODE", "M2_SEG_SKIP_NONREF", "M2_COPYFREE_RESIZE", "M2_PIPE_RESIZE", "M2_VISION_CHUNK")


def _h(x):
    import numpy as np, torch
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.blake2b(np.ascontiguousarray(x).reshape(-1).view(np.uint8).tobytes(), digest_size=16).hexdigest()


def _set(on):
    for k in SWITCHES:
        if k in on:
            os.environ[k] = "1"
        else:
            os.environ.pop(k, None)
    if on:
        os.environ["M2_STRICT"] = "1"
        os.environ["M2_VC_KEEP_PV"] = "1"
    else:
        os.environ.pop("M2_STRICT", None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local model directory")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", help="write the full report as JSON here")
    ap.add_argument("--no-gpu", action="store_true", help="skip the m2_vision_chunk check")
    a = ap.parse_args()

    qwen = is_qwen3vl(a.model)
    os.environ.setdefault("M2_SEG_WCPUS", ",".join(map(str, sorted(os.sched_getaffinity(0)))))
    gpu = False
    if qwen and not a.no_gpu:
        n = _visible_gpus()
        gpu = bool(n)
        if gpu and not os.environ.get("M2_VC_DEVICE"):
            os.environ["M2_VC_DEVICE"] = "cuda:0"
    register_all()

    from pathlib import Path
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.media.image import ImageMediaIO
    from vllm.multimodal.media.video import VideoMediaIO
    from vllm.multimodal.processing import ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.video import get_video_loader_backend_for_processor
    from vllm.utils.torch_utils import set_default_torch_num_threads
    import m2_copyfree_resize as cf, m2_pipe_resize as pipe, m2_vision_chunk as vc

    mc = ModelConfig(model=a.model, allowed_local_media_path="/", mm_processor_cache_gb=0)
    proc = MULTIMODAL_REGISTRY.create_processor(mc)
    try:
        vp_name = type(proc.info.get_hf_processor().video_processor).__name__
    except Exception:
        vp_name = None
    backend = get_video_loader_backend_for_processor(vp_name)
    text = proc.dummy_inputs.get_dummy_text({"video": 1}) + "Describe this video in one sentence."
    ids = proc.info.get_tokenizer().encode(text)
    io_kw = {"video_backend": backend} if backend else {}

    configs = [("off", set())]
    exact = {"M2_SEG_DECODE", "M2_SEG_SKIP_NONREF"} | ({"M2_COPYFREE_RESIZE", "M2_PIPE_RESIZE"} if qwen else set())
    configs.append(("exact", exact))
    if gpu:
        configs.append(("all", exact | {"M2_VISION_CHUNK"}))

    def run(video):
        io = VideoMediaIO(ImageMediaIO(image_mode="RGB"), backend="opencv", **io_kw)
        pipe._tls.status = None; cf._tls.status = None; vc._tls.status = None; vc.last_pv[0] = None
        t0 = time.perf_counter()
        with pipe.request_context(mc, {}):
            media = io.load_file(Path(video))
            items = proc.info.parse_mm_data({"video": [media.media]})
            with set_default_torch_num_threads():
                o = proc.apply(ProcessorInputs(ids, items, hf_processor_mm_kwargs={}), TimingContext())
        dt = time.perf_counter() - t0
        it = o["mm_kwargs"]["video"][0]
        pv = it["pixel_values_videos"].data if "pixel_values_videos" in it else vc.last_pv[0]
        seg = getattr(media, "m2_seg_status", None) or {}
        st = lambda m: {k: v for k, v in ((getattr(m._tls, "status", None) or {}).items()) if k in ("served", "why", "path")}
        return {"pv": _h(pv) if pv is not None else None, "grid": it["video_grid_thw"].data.tolist(),
                "embeds": "video_embeds" in it, "seconds": round(dt, 3),
                "seg": {k: seg.get(k) for k in ("served", "why", "path", "gated", "slices", "decode_path")},
                "copyfree": st(cf), "pipe": st(pipe), "vision_chunk": st(vc)}

    report, ok_all = [], True
    for v in a.videos:
        ref = None
        for name, on in configs:
            _set(on)
            try:
                r = run(v); r["error"] = None
            except Exception as e:
                r = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"}
            if name == "off":
                ref = r
                r["pass"] = r["error"] is None
            else:
                same = r.get("error") is None and ref.get("error") is None and r["pv"] == ref["pv"] and r["grid"] == ref["grid"]
                gated = bool(r.get("seg", {}).get("gated"))
                served = {
                    "seg_decode": bool(r.get("seg", {}).get("served")) or gated,
                    "pipe_or_copyfree": (not qwen) or bool(r.get("pipe", {}).get("served") or r.get("copyfree", {}).get("served")),
                    "vision_chunk": "M2_VISION_CHUNK" not in on or (bool(r.get("vision_chunk", {}).get("served")) and r.get("embeds")),
                }
                r["identical"], r["served"] = same, served
                r["pass"] = same and all(served.values())
            ok_all &= r["pass"]
            r.update(video=os.path.basename(v), config=name)
            report.append(r)
            seg = r.get("seg") or {}
            print(f"{os.path.basename(v):28s} {name:6s} {'PASS' if r['pass'] else 'FAIL'}  {r.get('seconds', '-'):>7}s  "
                  f"identical={r.get('identical', '-')}  seg={'gated' if seg.get('gated') else seg.get('served')}"
                  f"{'/' + str(seg.get('decode_path')) if seg.get('decode_path') else ''}  "
                  f"pipe={(r.get('pipe') or {}).get('served')}  vc={(r.get('vision_chunk') or {}).get('served')}"
                  f"{'  ' + r['error'] if r.get('error') else ''}", flush=True)
    _set(set())
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"model": a.model, "qwen3_vl": qwen, "gpu_checked": gpu, "pass": ok_all, "rows": report}, f,
                      ensure_ascii=False, indent=1)
    print("ALL PASS" if ok_all else "SOME FAILED", "(gpu check skipped)" if not gpu and qwen else "")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
