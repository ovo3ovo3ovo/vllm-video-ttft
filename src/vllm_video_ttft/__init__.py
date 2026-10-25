"""vllm-video-ttft: vLLM 0.31.0 general plugin that shortens time-to-first-token for video requests.

One entry point (vllm.general_plugins -> register) for four sub-plugins kept as their own top-level modules, byte for
byte the versions validated on 8x RTX 5090 (VENDORED.sha256), except m2_vision_chunk, which later gained
_tower_weight_files (d50d776: read a single unsharded model.safetensors when there is no shard index) and builds
its tower from the ModelConfig each part carries, retrying a failed build (10-25: long video as first request):
  m2_seg_decode       slice-parallel OpenCV decode (+ skip non-reference frames, background buffer release)   any model on vLLM's OpenCV path
  m2_copyfree_resize  resize/patchify without the extra copies                                                 Qwen3-VL video processor only
  m2_pipe_resize      resize/patchify while the decoder is still running                                       Qwen3-VL video processor only
  m2_vision_chunk     vision tower on finished blocks, overlapped with decode (output not bit-identical)       Qwen3-VL vision tower only

Every sub-plugin reads its switch on every call (M2_* = "1"), so they can all be registered and stay inert.
This wrapper only adds: a preset, machine-dependent defaults, a version gate and one log line saying what is on.
  VLLM_VIDEO_TTFT_PRESET = off (default) | exact | all
    exact: M2_SEG_DECODE M2_SEG_SKIP_NONREF M2_COPYFREE_RESIZE M2_PIPE_RESIZE   (pixel_values bit-identical to vLLM)
    all:   exact + M2_VISION_CHUNK                                             (tower output not bit-identical, median relative diff 2.9%)
  A switch set explicitly (e.g. M2_PIPE_RESIZE=0) wins over the preset.
"""
import logging, os, sys

__version__ = "0.1.0"
_log = logging.getLogger("vllm_video_ttft")
_SWITCHES = ("M2_SEG_DECODE", "M2_SEG_SKIP_NONREF", "M2_COPYFREE_RESIZE", "M2_PIPE_RESIZE", "M2_VISION_CHUNK")
PRESETS = {
    "off": {},
    "exact": {"M2_SEG_DECODE": "1", "M2_SEG_SKIP_NONREF": "1", "M2_COPYFREE_RESIZE": "1", "M2_PIPE_RESIZE": "1"},
}
PRESETS["all"] = {**PRESETS["exact"], "M2_VISION_CHUNK": "1"}
REQUIRED = {"vllm": "0.31.0", "transformers": "5.17.0"}
OPENCV = ("opencv-python-headless", "5.0.0.93")   # m2_seg_decode's packet index checks the libavformat it ships
_done = False

QWEN3VL_VIDEO_PROCESSOR = "Qwen3VLVideoProcessor"   # what m2_copyfree_resize / m2_pipe_resize / m2_vision_chunk hook


def is_qwen3vl(model_dir):
    """True when the model's video processor is Qwen3-VL's (Qwen3-VL itself, and models built on it such as
    Qwen3.8-Flash-Next, whose architecture name is Qwen4ExpForConditionalGeneration)."""
    import json
    for name in ("video_preprocessor_config.json", "preprocessor_config.json"):
        try:
            with open(os.path.join(model_dir, name)) as f:
                if json.load(f).get("video_processor_type") == QWEN3VL_VIDEO_PROCESSOR:
                    return True
        except (OSError, ValueError):
            pass
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            return bool({"Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration"}
                        & set(json.load(f).get("architectures") or []))
    except (OSError, ValueError):
        return False


def _version(dist):
    from importlib import metadata
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def version_problems():
    """List of human-readable version mismatches (empty = fine)."""
    out = []
    for dist, want in REQUIRED.items():
        have = _version(dist)
        if have is None or not have.startswith(want):
            out.append(f"{dist} {have} (need {want})")
    have = _version(OPENCV[0])
    if have != OPENCV[1]:
        out.append(f"{OPENCV[0]} {have} (need {OPENCV[1]}; the skip-non-reference path turns itself off otherwise)")
    return out


def _visible_gpus():
    """Number of GPUs this process can see, without creating a CUDA context."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return len([x for x in cvd.split(",") if x.strip() not in ("", "-1")])
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            return pynvml.nvmlDeviceGetCount()
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def apply_defaults():
    """Preset + machine defaults into os.environ (setdefault only: explicit settings win).  Returns the switch state."""
    preset = os.environ.get("VLLM_VIDEO_TTFT_PRESET", "off").strip().lower() or "off"
    if preset not in PRESETS:
        _log.warning("vllm_video_ttft: unknown VLLM_VIDEO_TTFT_PRESET=%r (use off / exact / all), treated as off", preset)
        preset = "off"
    for k, v in PRESETS[preset].items():
        os.environ.setdefault(k, v)
    on = {k: os.environ.get(k) == "1" for k in _SWITCHES}
    if on["M2_SEG_DECODE"] and not os.environ.get("M2_SEG_WCPUS") and not os.environ.get("M2_SEG_WNODES"):
        # m2_seg_decode alone would look for NUMA nodes 0 and 1; use the CPUs this process may run on instead
        os.environ["M2_SEG_WCPUS"] = ",".join(map(str, sorted(os.sched_getaffinity(0))))
    if on["M2_VISION_CHUNK"] and not os.environ.get("M2_VC_DEVICE"):
        n = _visible_gpus()
        if n == 1:
            os.environ["M2_VC_DEVICE"] = "cuda:0"
        else:
            _log.warning("vllm_video_ttft: M2_VISION_CHUNK=1 but M2_VC_DEVICE is not set and %s GPUs are visible; "
                         "set M2_VC_DEVICE=cuda:N (a GPU with ~3 GiB free).  m2_vision_chunk stays inert until then.", n)
    return preset, on


def _guard_pipe_non_qwen():
    """m2_pipe_resize finds a request's processor kwargs with a dummy apply and caches only successes; on a model whose
    processor is not Qwen3-VL that dummy run fails on every request.  Remember the failure per ModelConfig instead."""
    import m2_pipe_resize as pipe
    orig = pipe._params_for
    if getattr(orig, "_vvt_guard", False):
        return
    failed = {}

    def _params_for(model_config, mm_kwargs):
        e = failed.get(id(model_config))
        if e is not None and e[0] is model_config:
            raise e[1]
        _params_for.dummy_runs += 1             # how often the dummy apply really ran (selfcheck reads it)
        try:
            return orig(model_config, mm_kwargs)
        except RuntimeError as x:
            if "never reached the video processor" in str(x):
                failed[id(model_config)] = (model_config, x)
            raise
    _params_for._vvt_guard = True
    _params_for.dummy_runs = 0
    pipe._params_for = _params_for


def register_all():
    """Register all four sub-plugins (inert unless their switch is "1").  Used by register() and the self-check."""
    import m2_copyfree_resize, m2_seg_decode, m2_pipe_resize, m2_vision_chunk
    for m in (m2_copyfree_resize, m2_seg_decode, m2_pipe_resize, m2_vision_chunk):
        m.register()                       # each is idempotent; m2_vision_chunk also registers m2_pipe_resize first
    _guard_pipe_non_qwen()


_LOGGERS = ("vllm_video_ttft", "m2_copyfree_resize", "m2_seg_decode", "m2_pipe_resize", "m2_vision_chunk")


def _route_logs():
    """Our loggers are not under vLLM's "vllm" logger, so vLLM's logging config gave them no handler: INFO lines were
    dropped and warnings went through Python's last-resort handler.  Make "vllm" their parent: records then use vLLM's
    handler and level, whether vLLM configures its logging before or after this plugin is loaded."""
    vl = logging.getLogger("vllm")
    for name in _LOGGERS:
        lg = logging.getLogger(name)
        if lg.parent is not vl and not lg.handlers:
            lg.parent = vl


def register():
    """vLLM general-plugin entry point; runs in every vLLM process, possibly more than once."""
    global _done
    if _done:
        return
    _done = True
    _route_logs()
    preset, on = apply_defaults()
    if not any(on.values()):
        _log.info("vllm_video_ttft %s: preset=%s, on: none", __version__, preset)
        return                             # installed but not enabled: touch nothing else
    bad = version_problems()
    if bad and any(b.startswith(("vllm", "transformers")) for b in bad):
        msg = "vllm_video_ttft: not registered, version mismatch: " + "; ".join(bad)
        if os.environ.get("M2_STRICT") == "1":
            raise RuntimeError(msg)
        _log.warning(msg)
        return
    if bad:
        _log.warning("vllm_video_ttft: %s", "; ".join(bad))
    register_all()
    _log.info("vllm_video_ttft %s: preset=%s, on: %s", __version__, preset,
              ", ".join(k for k, v in on.items() if v) or "none")
