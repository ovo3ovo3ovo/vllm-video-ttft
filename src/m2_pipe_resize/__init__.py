"""vLLM general plugin (mission_2 unit 5, 10-24): "边解码边缩放" for Qwen3-VL video inputs (vLLM 0.31.0, transformers 5.17.0).

Today the API server decodes the whole video (VideoMediaIO.load_bytes), and only then resizes + patchifies + normalizes it
(processor.apply -> Qwen3VLVideoProcessor.preprocess).  The target size depends only on the frame count, the source size
and the request's processor kwargs, all known as soon as the frame buffer is allocated, and resize / normalize are per
frame while patchify groups frames in pairs.  So the frames are cut into fixed blocks of M2_PIPE_CHUNK frames (even), each
owning a fixed row range of the final pixel_values buffer; while the decoder fills the frame buffer, a background thread
resizes + patchifies + normalizes every block whose frames have all landed, in whatever order they complete (so
m2_seg_decode's slices, which finish almost together, are taken as each one lands).  preprocess then returns that buffer.
CPU side only: the vision tower is untouched.  Same math as plugins/m2_copyfree_resize (bit-identical to HF there), per
block; only the last block can be odd, and patchify pads it exactly as the original pads the end of the video.

Processor kwargs are this request's own (main 17:23, review P2/P5): the chat parser builds one MediaConnector per request
and knows the request's mm_processor_kwargs and the ModelConfig (vllm/entrypoints/chat_utils.py:1036-1052, 1386-1395); the
plugin hands them to the VideoMediaIO of that request (MediaConnector.load_from_url[_async]).  Before decoding, the
standardized video kwargs are obtained by running vLLM's real _apply_hf_processor_main on a 4-frame dummy video with the
same kwargs and stopping inside Qwen3VLVideoProcessor.preprocess once they are known (HF processes videos before it
tokenizes, so the tokenizer is never touched).  Computed once per (ModelConfig, kwargs) per process.  Offline scripts that
call VideoMediaIO directly pass the same pair with `with m2_pipe_resize.request_context(model_config, mm_kwargs): ...`.

Replaced (all no-ops unless M2_PIPE_RESIZE=1, read on every call):
  VideoMediaIO.load_bytes                     opens a "sink" for the decode running in this thread, registers the result
  OpenCVVideoBackendMixin._read_frames_no_recovery   vLLM's own read loop (opencv.py:254-300), plus sink notifications
  Qwen3VLVideoProcessor.preprocess            BaseVideoProcessor.preprocess (video_processing_utils.py:258-300), but hands
                                              back the precomputed buffer when frames and kwargs match exactly
  MultiModalContentParser.parse_video, AsyncMultiModalContentParser._video_with_uuid_async,
  MediaConnector.load_from_url[_async]        only carry (ModelConfig, request mm_processor_kwargs) to the VideoMediaIO
plugins/m2_seg_decode feeds the same sink (ready_range per finished part of a slice; invalidate_range when it re-decodes
a slice; abort() when it gives up on its buffer).
The precomputed result is used only if: one video; it is the very array the sink saw (data pointer, shape, strides) and
still alive; every block was filled; HF did not re-sample it; and every preprocess kwarg equals the ones it was built
with.  The background thread ends when the decoder calls end() and every block it can fill is done (or a block can never
fill); there is no timeout (main 17:23).  With vLLM's processor cache on (mm_processor_cache_gb > 0) nothing is started:
a cache hit skips preprocess, and the cache key includes the decoded frame indices, so a hit cannot be known before
decoding; the plugin requires the cache off (our server runs --mm-processor-cache-gb 0).
M2_STRICT=1: whenever the plugin is on and does not serve a video that vLLM itself can process, it raises M2StrictError
with the reason (no request context, cache on, kwargs mismatch, multi-video, ...).  A video that vLLM cannot process
(0 decoded frames, unreadable file) keeps vLLM's own error.
A video with no more sampled frames than one block (T <= M2_PIPE_CHUNK) has nothing to overlap with its decode, so
no thread is started for it and preprocess computes it inline with the same per-block math (status path "pipe-inline";
review R2).  Env: M2_PIPE_CHUNK=32 frames per block (even); M2_PIPE_MIN_FRAMES=0 do not pipeline videos with fewer sampled frames
(performance gate, not a fallback); M2_PIPE_DTYPE=bfloat16 (vLLM casts to the model dtype right after,
vllm/multimodal/processing/context.py:283).  The thread sets its own torch thread count (OMP_NUM_THREADS) via
set_default_torch_num_threads; torch keeps that per thread, the request thread is not affected.
M2_PIPE_RESIZE_THREADS=N (10-25, record 4): the resize thread uses N torch threads instead of OMP_NUM_THREADS, and so
does a one-block video computed inline (in the request thread, restored right after); nothing else changes (the decode
workers, vLLM's own processor threads).  Unset: exactly as before.  Read on every video; the count actually used is in
the status as resize_threads.
Status per request: _tls.status in the thread that runs preprocess (inside apply, what m2_mm_trace reads), and
media.m2_pipe_status on the MediaWithBytes from load_bytes (decode side).  When the precomputed result is served,
m2_copyfree_resize's code does not run; its _tls.status is then set to why "skipped: served by m2_pipe_resize" (review C4).
`last` is a process-wide copy of the latest values, for single-threaded offline scripts only."""
import contextlib, json, logging, math, os, sys, threading, time, weakref
_MARK = "_m2_pipe"
_CHUNKROWS = 4096
_log = logging.getLogger("m2_pipe_resize")
_said = set()
_tls = threading.local()
_pre_lock = threading.Lock()
_pre = {}            # array key -> (weakref to frames, kwargs key, pixel_values, grid, T, token, load stats)
_params_lock = threading.Lock()
_params = {}         # (id(ModelConfig), request kwargs key) -> (ModelConfig, video processor, std kwargs, kwargs key)
_procs = {}          # id(ModelConfig) -> (ModelConfig, BaseMultiModalProcessor) used only to compute kwargs
last = {}

_INLINE, _GATED = object(), object()     # markers in _pre: compute in preprocess (R2) / gated on purpose (R1)

class M2StrictError(RuntimeError):
    """M2_STRICT=1: the plugin was on but did not serve this video (no silent fallback)."""

class _Captured(Exception):
    """Stops the dummy run inside preprocess once the standardized kwargs are known."""

def _on():
    return os.environ.get("M2_PIPE_RESIZE") == "1"

def _strict():
    return os.environ.get("M2_STRICT") == "1"

def _resize_threads():
    """M2_PIPE_RESIZE_THREADS as a positive int, or None (unset / 0 / not a number: vLLM's OMP_NUM_THREADS as before)."""
    n = _env_num("M2_PIPE_RESIZE_THREADS", 0)
    return n if n > 0 else None

def _env_num(name, default, cast=int):
    try:
        return cast(os.environ.get(name, default))
    except ValueError:
        return default

def _note(reason):
    if reason not in _said:
        _said.add(reason); _log.warning("m2_pipe_resize: not using the pipelined result (%s)", reason)

def current_sink():
    """The sink of the decode running in this thread, or None (called by m2_seg_decode)."""
    return getattr(_tls, "sink", None) if _on() else None

@contextlib.contextmanager
def request_context(model_config, mm_processor_kwargs=None):
    """For callers that use VideoMediaIO directly (offline scripts): the ModelConfig and the hf_processor_mm_kwargs the
    following processor.apply will get."""
    old = getattr(_tls, "req", None)
    _tls.req = (model_config, dict(mm_processor_kwargs or {}))
    try:
        yield
    finally:
        _tls.req = old

def _arr_key(a):
    return (a.__array_interface__["data"][0], a.shape, a.strides, a.dtype.str)

def _kw_key(kw):
    return repr(sorted((k, repr(v)) for k, v in kw.items()))

def _req_key(mm_kwargs):
    return json.dumps(mm_kwargs or {}, sort_keys=True, default=repr)

def _drop(key, token):
    """weakref.finalize callback: remove the entry only if it is still the one this array registered (review P3)."""
    with _pre_lock:
        e = _pre.get(key)
        if e is not None and e[5] is token:
            del _pre[key]

def _plain(d):
    return {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool, type(None)))}

def _params_for(model_config, mm_kwargs):
    """(video processor, standardized kwargs, kwargs key) that processor.apply will use for a video of this request:
    vLLM's real _apply_hf_processor_main on a 4-frame dummy video, stopped inside preprocess.  Raises on failure."""
    key = (id(model_config), _req_key(mm_kwargs))
    with _params_lock:
        hit = _params.get(key)
        if hit is not None and hit[0] is model_config:
            return hit[1:]
        import numpy as np
        from vllm.multimodal import MULTIMODAL_REGISTRY
        pr = _procs.get(id(model_config))
        if pr is None or pr[0] is not model_config:
            pr = (model_config, MULTIMODAL_REGISTRY.create_processor(model_config))
            _procs[id(model_config)] = pr
        proc = pr[1]
        frames = np.zeros((4, 64, 64, 3), dtype=np.uint8)
        meta = {"fps": 2.0, "duration": 2.0, "total_num_frames": 4, "frames_indices": [0, 1, 2, 3],
                "video_backend": "opencv", "do_sample_frames": False}
        items = proc.info.parse_mm_data({"video": [(frames, meta)]})
        _tls.capture = None
        try:
            proc._apply_hf_processor_main(items, dict(mm_kwargs or {}))
        except BaseException as e:
            x = e
            while x is not None and not isinstance(x, _Captured):
                x = x.__cause__ or x.__context__
            if x is None:
                raise
        finally:
            cap, _tls.capture = getattr(_tls, "capture", None), False
        if not cap:
            raise RuntimeError("the dummy run never reached the video processor")
        _params[key] = (model_config,) + cap
        return cap

class _Sink:
    """Consumes the frames of a (T, H, W, 3) uint8 array in fixed blocks as a decoder reports them ready, any order."""

    def __init__(self, proc, kw, kwkey, chunk, torch, Q, SizeDict, smart_resize, threads_ctx):
        self.proc, self.kw, self.kwkey, self.chunk = proc, kw, kwkey, chunk
        self.torch, self.Q, self.SizeDict, self.smart_resize, self.threads_ctx = torch, Q, SizeDict, smart_resize, threads_ctx
        self.cv = threading.Condition()
        self.gen, self.th, self.arr, self.key, self.result, self.err = 0, None, None, None, None, None
        self.mask, self.closed, self.stats, self.ver, self.gated, self.inline = None, False, {}, None, False, False
        self.nthreads, self.resize_n = None, None

    def _target(self, T, H, W):
        kw = self.kw; size = kw["size"]
        factor = kw["patch_size"] * kw["merge_size"]
        max_pixels = size.longest_edge
        if kw.get("cap_pixels_per_frame"):          # same as Qwen3VLVideoProcessor.resize
            frame_cap = self.proc.max_video_tokens * factor * factor
            ppf = max(min(frame_cap, size.longest_edge // T), int(size.shortest_edge * 1.05))
            max_pixels = ppf * T
        return self.smart_resize(height=H, width=W, num_frames=T, factor=factor,
                                 temporal_factor=kw["temporal_patch_size"], min_pixels=size.shortest_edge,
                                 max_pixels=max_pixels)

    def begin(self, arr):
        self._stop()
        torch, kw = self.torch, self.kw
        with self.cv:
            self.gen += 1; gen = self.gen
            self.closed, self.result, self.err, self.arr, self.key, self.mask, self.stats = False, None, None, None, None, None, {}
            self.gated = self.inline = False
        try:
            if arr.dtype.str != "|u1" or arr.ndim != 4 or arr.shape[3] != 3:
                raise ValueError(f"unsupported frames {arr.dtype} {arr.shape}")
            if arr.shape[0] < 1:
                raise ValueError("no frames")
            T, H, W = arr.shape[:3]
            if T < _env_num("M2_PIPE_MIN_FRAMES", 0):
                self.gated = True
                raise ValueError(f"below threshold ({T} frames)")
            P, TP = kw["patch_size"], kw["temporal_patch_size"]
            if self.chunk % TP:
                raise ValueError(f"block size {self.chunk} not a multiple of temporal_patch_size {TP}")
            if T <= self.chunk:                  # R2: one block, nothing to overlap -> preprocess computes it inline
                self.inline, self.arr, self.key, self.arr_shape = True, arr, _arr_key(arr), arr.shape
                return
            h, w = self._target(T, H, W)
            rows = math.ceil(T / TP) * (h // P) * (w // P)
            out = torch.empty((rows, 3 * TP * P * P), dtype=getattr(torch, os.environ.get("M2_PIPE_DTYPE", "bfloat16")))
        except Exception as e:
            self.err = str(e) if isinstance(e, ValueError) else repr(e); return
        import numpy as np
        self.mask = np.zeros(T, dtype=bool)
        self.ver = [0] * math.ceil(T / self.chunk)     # bumped when a block's frames are rewritten (invalidate_range)
        self.todo = set(range(len(self.ver)))
        self.arr, self.key, self.arr_shape, self.t_begin = arr, _arr_key(arr), arr.shape, time.perf_counter()
        self.th = threading.Thread(target=self._run, args=(gen, arr, out, h, w), name="m2_pipe_resize", daemon=True)
        self.th.start()

    def ready_range(self, lo, hi):
        """Frames [lo, hi) are in the buffer."""
        with self.cv:
            if self.mask is not None:
                self.mask[lo:hi] = True; self.cv.notify_all()

    def invalidate_range(self, lo, hi):
        """Frames [lo, hi) will be written again (m2_seg_decode re-decodes a slice): any block touching them is redone."""
        with self.cv:
            if self.mask is None:
                return
            self.mask[lo:hi] = False
            for b in range(lo // self.chunk, (max(hi, lo + 1) - 1) // self.chunk + 1):
                if b < len(self.ver):
                    self.ver[b] += 1; self.todo.add(b)
            self.cv.notify_all()

    def rekey(self, arr):
        """The decoder hands out a copy of the buffer the pipeline read (same bytes): match the result to the copy."""
        if (self.result is not None or self.inline) and arr.shape == self.arr_shape:
            self.key = _arr_key(arr)

    def ready(self, n):
        """Frames [0, n) are in the buffer (sequential decoder)."""
        self.ready_range(0, n)

    def end(self):
        with self.cv:
            self.closed = True; self.cv.notify_all()

    def wait(self):
        """Wait for the worker (it ends after end(): every fillable block done, or one can never fill); True if every
        block was processed.  Drops the reference to the frame buffer."""
        if self.th is not None:
            self.th.join(); self.th = None
        self.arr = None
        return self.result is not None

    def abort(self):
        """The decoder is giving up on this buffer (m2_seg_decode falls back): drop everything computed from it."""
        self._stop()
        self.result, self.key, self.err = None, None, "aborted by the decoder"

    def _stop(self):
        if self.th is not None:
            with self.cv:
                self.gen += 1; self.closed = True; self.cv.notify_all()
            self.th.join()                          # returns after at most the block being computed
            self.th = None
        self.arr = None

    def _norm(self):
        torch, kw = self.torch, self.kw
        mean = kw["image_mean"] if isinstance(kw["image_mean"], (list, tuple)) else [kw["image_mean"]] * 3
        std = kw["image_std"] if isinstance(kw["image_std"], (list, tuple)) else [kw["image_std"]] * 3
        # fused rescale+normalize exactly as m2_copyfree_resize (image_processing_backends.py:308-311 + tvF.normalize)
        return (float(torch.tensor(mean[0]) * (1.0 / kw["rescale_factor"])),
                float(torch.tensor(std[0]) * (1.0 / kw["rescale_factor"])))

    def _block(self, arr, lo, hi, out, h, w, m, s):
        """Frames [lo, hi) of arr -> their rows of out (resize, patchify, normalize; same math as m2_copyfree_resize)."""
        torch, kw, proc = self.torch, self.kw, self.proc
        P, M, TP = kw["patch_size"], kw["merge_size"], kw["temporal_patch_size"]
        gh, gw = h // P, w // P
        x = torch.from_numpy(arr[lo:hi]).permute(0, 3, 1, 2)          # channels_last view, no copy
        if kw["do_convert_rgb"]:
            x = proc.convert_to_rgb(x)
        x = super(self.Q, proc).resize(image=x.unsqueeze(0), size=self.SizeDict(height=h, width=w), resample=kw["resample"])
        patches, _, ph, pw = proc.patchify(x, patch_size=P, merge_size=M, temporal_patch_size=TP)
        if (ph, pw) != (gh, gw):
            raise RuntimeError(f"patch grid {ph}x{pw}, expected {gh}x{gw}")
        patches = patches[0]
        r0 = (lo // TP) * gh * gw
        for i in range(0, patches.shape[0], _CHUNKROWS):
            piece = patches[i:i + _CHUNKROWS]            # the last piece of a block can be shorter
            out[r0 + i:r0 + i + piece.shape[0]] = piece.to(torch.float32).sub_(m).div_(s)

    def compute_inline(self, arr):
        """R2: the whole video in the calling thread (T <= one block: nothing to overlap).  Returns (pixel_values, grid)."""
        torch, kw = self.torch, self.kw
        T, H, W = arr.shape[:3]
        h, w = self._target(T, H, W)
        P, TP = kw["patch_size"], kw["temporal_patch_size"]
        out = torch.empty((math.ceil(T / TP) * (h // P) * (w // P), 3 * TP * P * P),
                          dtype=getattr(torch, os.environ.get("M2_PIPE_DTYPE", "bfloat16")))
        m, s = self._norm()
        if self.resize_n is not None:           # M2_PIPE_RESIZE_THREADS: same count as the thread would use
            with self.threads_ctx():
                self.nthreads = torch.get_num_threads()
                for lo in range(0, T, self.chunk):
                    self._block(arr, lo, min(lo + self.chunk, T), out, h, w, m, s)
        else:
            self.nthreads = torch.get_num_threads()
            for lo in range(0, T, self.chunk):
                self._block(arr, lo, min(lo + self.chunk, T), out, h, w, m, s)
        return out, torch.tensor([[math.ceil(T / TP), h // P, w // P]])

    def _run(self, gen, arr, out, h, w):
        torch, kw = self.torch, self.kw
        P, TP = kw["patch_size"], kw["temporal_patch_size"]
        m, s = self._norm()
        T, C = arr.shape[0], self.chunk
        nblk, busy, first, order, redone = len(self.ver), 0.0, None, [], 0
        gh, gw = h // P, w // P
        try:
            with self.threads_ctx():
                self.nthreads = torch.get_num_threads()
                while True:
                    with self.cv:
                        while True:                 # finish only at end(): a block may still be invalidated
                            if gen != self.gen:
                                return
                            b = next((b for b in sorted(self.todo) if self.mask[b * C:min(b * C + C, T)].all()), None)
                            if b is not None or self.closed:
                                break
                            self.cv.wait()
                        if b is None:               # closed: all done, or the rest can never arrive (review P8)
                            break
                        v = self.ver[b]
                    lo, hi = b * C, min(b * C + C, T)
                    t0 = time.perf_counter()
                    self._block(arr, lo, hi, out, h, w, m, s)
                    busy += time.perf_counter() - t0
                    if first is None:
                        first = time.perf_counter() - self.t_begin
                    with self.cv:
                        if self.ver[b] == v:        # not rewritten while it was being computed
                            self.todo.discard(b)
                        else:
                            redone += 1
                    order.append(b)
            if not self.todo:
                self.result = (out, torch.tensor([[math.ceil(T / TP), gh, gw]]))
            else:
                self.err = f"decoder closed with {len(self.todo)} of {nblk} blocks unfilled"
        except Exception as e:
            self.err = repr(e)[:200]
        finally:
            ooo = sum(1 for a, b in zip(order, order[1:]) if b < a)
            self.stats = {"busy_s": round(busy, 4), "first_s": round(first, 4) if first is not None else None,
                          "done_s": round(time.perf_counter() - self.t_begin, 4), "blocks": nblk,
                          "blocks_out_of_order": ooo, "blocks_redone": redone, "resize_threads": self.nthreads}

def register():
    try:
        import numpy as np, torch, transformers, vllm, cv2
        from functools import partial
        import transformers.video_processing_utils as vpu
        from transformers.video_processing_utils import BaseVideoProcessor
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor, smart_resize
        from transformers.image_processing_utils import BatchFeature
        from transformers.image_utils import SizeDict
        from vllm.multimodal.media.video import VideoMediaIO
        from vllm.multimodal.media.connector import MediaConnector
        from vllm.multimodal.video_decoders.opencv import OpenCVVideoBackendMixin
        from vllm.utils.torch_utils import set_default_torch_num_threads
        from vllm.entrypoints import chat_utils
        if transformers.__version__ != "5.17.0" or not vllm.__version__.startswith("0.31.0"):
            raise RuntimeError(f"written for transformers 5.17.0 / vLLM 0.31.0, found {transformers.__version__} / {vllm.__version__}")
        if "preprocess" in Qwen3VLVideoProcessor.__dict__ and not getattr(Qwen3VLVideoProcessor.preprocess, _MARK, False):
            raise RuntimeError("Qwen3VLVideoProcessor overrides preprocess")
    except Exception as e:  # layout changed: leave everything untouched
        _log.warning("m2_pipe_resize: not registered (%r)", e); return
    if getattr(VideoMediaIO.load_bytes, _MARK, False):
        return
    orig_load_bytes = VideoMediaIO.load_bytes
    orig_rfnr = OpenCVVideoBackendMixin._read_frames_no_recovery
    orig_lfu, orig_lfu_async = MediaConnector.load_from_url, MediaConnector.load_from_url_async
    orig_parse_video = chat_utils.MultiModalContentParser.parse_video
    orig_vid_async = chat_utils.AsyncMultiModalContentParser._video_with_uuid_async
    vlog = logging.getLogger("vllm.multimodal.video_decoders.opencv")

    # ---- carry (ModelConfig, request mm_processor_kwargs) from the chat parser to this request's VideoMediaIO
    def parse_video(self, video_url, uuid=None):
        self._connector._m2_req = (self.model_config, self._mm_processor_kwargs or {})
        return orig_parse_video(self, video_url, uuid)

    async def _video_with_uuid_async(self, video_url, uuid):
        self._connector._m2_req = (self.model_config, self._mm_processor_kwargs or {})
        return await orig_vid_async(self, video_url, uuid)

    def _tag(conn, media_io):
        req = getattr(conn, "_m2_req", None)
        if req is not None and isinstance(media_io, VideoMediaIO):
            media_io._m2_req = req

    def load_from_url(self, url, media_io, *a, **k):
        _tag(self, media_io)
        return orig_lfu(self, url, media_io, *a, **k)

    async def load_from_url_async(self, url, media_io, *a, **k):
        _tag(self, media_io)
        return await orig_lfu_async(self, url, media_io, *a, **k)

    def _attach(media, st):
        try: media.m2_pipe_status = st
        except Exception: pass

    def _refuse(why):
        """Not pipelining a video vLLM can process: strict raises, otherwise vLLM's path runs unchanged."""
        _note(why)
        if _strict():
            raise M2StrictError(f"m2_pipe_resize: {why}")

    def load_bytes(self, data):
        if not _on() or getattr(_tls, "sink", None) is not None:
            return orig_load_bytes(self, data)
        req = getattr(self, "_m2_req", None) or getattr(_tls, "req", None)
        why, params = None, None
        if req is None:
            why = "no request context (VideoMediaIO used outside the chat parser and request_context)"
        else:
            mc, mm_kwargs = req
            mmc = mc.get_multimodal_config() if hasattr(mc, "get_multimodal_config") else None
            if mmc is not None and getattr(mmc, "mm_processor_cache_gb", 0) > 0:
                why = f"vLLM's processor cache is on (mm_processor_cache_gb={mmc.mm_processor_cache_gb})"
            else:
                try:
                    params = _params_for(mc, mm_kwargs)
                except Exception as e:
                    why = f"could not compute this request's processor kwargs: {e!r}"[:200]
        if why is not None:
            _refuse(why)
            media = orig_load_bytes(self, data)
            _attach(media, {"pipelined": False, "why": why})
            return media
        proc, kw, kwkey = params
        chunk = max(2, _env_num("M2_PIPE_CHUNK", 32))
        chunk -= chunk % kw["temporal_patch_size"]
        rn = _resize_threads()                    # fixed for this video, used by the thread and by compute_inline
        sink = _Sink(proc, kw, kwkey, chunk, torch, Qwen3VLVideoProcessor, SizeDict, smart_resize,
                     set_default_torch_num_threads if rn is None else (lambda: set_default_torch_num_threads(rn)))
        sink.resize_n = rn
        t0 = time.perf_counter()
        _tls.sink = sink
        try:
            media = orig_load_bytes(self, data)
        except BaseException:
            sink.abort(); raise
        finally:
            _tls.sink = None
            sink.end()
        t1 = time.perf_counter()
        ok = sink.wait() or (sink.inline and sink.err is None)
        t2 = time.perf_counter()
        frames = media.media[0] if isinstance(media.media, tuple) else None
        why = (sink.err or "the decoder never reported frames" if not ok else
               "decoder returned something other than an ndarray" if not isinstance(frames, np.ndarray) else
               "returned frames are not the buffer the pipeline read" if _arr_key(frames) != sink.key else None)
        st = {"pipelined": why is None, "why": why, "gated": sink.gated, "inline": sink.inline and why is None,
              "decode_s": round(t1 - t0, 4), "tail_s": round(t2 - t1, 4), **_plain(sink.stats)}
        if why is None:
            pv, grid = (_INLINE, sink) if sink.inline else sink.result
            key, token = _arr_key(frames), object()
            with _pre_lock:
                _pre[key] = (weakref.ref(frames), kwkey, pv, grid, frames.shape[0], token, dict(st))
            weakref.finalize(frames, _drop, key, token)
        elif isinstance(frames, np.ndarray) and frames.ndim == 4 and frames.shape[0] == 0:
            pass                                  # vLLM decoded no frames: its processor raises, nothing to serve
        elif sink.gated and isinstance(frames, np.ndarray):
            key, token = _arr_key(frames), object()   # R1: tell preprocess this array was gated on purpose
            with _pre_lock:
                _pre[key] = (weakref.ref(frames), kwkey, _GATED, None, frames.shape[0], token, dict(st))
            weakref.finalize(frames, _drop, key, token)
        else:
            _attach(media, st)
            _refuse(why)
        _attach(media, st)
        last.clear(); last.update(st, path="pipelined" if why is None else "not pipelined: " + why)
        return media

    def _read_frames_no_recovery(cls, cap, frame_indices, max_frame_idx):
        sink = current_sink()
        if sink is None:
            return orig_rfnr.__func__(cls, cap, frame_indices, max_frame_idx)
        # vllm/multimodal/video_decoders/opencv.py:254-300 (0.31.0), plus sink.begin / sink.ready
        num_expected_frames = len(frame_indices)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((num_expected_frames, height, width, 3), dtype=np.uint8)
        sink.begin(frames)
        step = sink.chunk
        i = 0
        valid_frame_indices = []
        for idx in range(max_frame_idx + 1):
            ok = cap.grab()
            if not ok:
                if idx in frame_indices:
                    vlog.debug("Failed to grab frame %d during video loading. This frame will be skipped.", idx)
                continue
            if idx in frame_indices:
                ret, frame = cap.retrieve()
                if ret:
                    frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    valid_frame_indices.append(idx)
                    i += 1
                    if i % step == 0:
                        sink.ready(i)
                else:
                    vlog.debug("Failed to retrieve frame %d during video loading. This frame will be skipped.", idx)
        if i < num_expected_frames and i > 0:
            sink.begin(frames[:i]); sink.ready(i)     # fewer frames: the target size changes, redo on what is returned
        else:
            sink.ready(i)
        valid_num_frames = len(valid_frame_indices)
        if valid_num_frames < num_expected_frames:
            vlog.warning("Video loading completed with %d broken/unreadable frames. "
                         "Expected %d frames but only loaded %d frames.",
                         num_expected_frames - valid_num_frames, num_expected_frames, valid_num_frames)
        return frames[:valid_num_frames], valid_frame_indices

    def _first_array(videos):
        while isinstance(videos, (list, tuple)) and len(videos) == 1:
            videos = videos[0]
        return videos if isinstance(videos, np.ndarray) else None

    orig_preprocess = BaseVideoProcessor.preprocess

    def _set(st):
        _tls.status = st
        last["served"], last["served_why"] = st["served"], st["why"]

    def preprocess(self, videos, **kwargs):
        capturing = getattr(_tls, "capture", False) is None
        if not _on() and not capturing:
            _tls.status = {"served": False, "why": "off"}
            return orig_preprocess(self, videos, **kwargs)
        src = None if capturing else _first_array(videos)
        entry = None
        if src is not None:
            with _pre_lock:
                entry = _pre.pop(_arr_key(src), None)
        # transformers 5.17.0 BaseVideoProcessor.preprocess (video_processing_utils.py:258-300); only change: the
        # kwargs are standardised before _prepare_input_videos (neither depends on the other), so the precomputed
        # result can skip _prepare_input_videos and _preprocess
        vpu.validate_kwargs(captured_kwargs=kwargs.keys(),
                            valid_processor_keys=list(self.valid_kwargs.__annotations__.keys()) + ["return_tensors"])
        vpu.validate_typed_dict(self.valid_kwargs, kwargs)
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))
        input_data_format = kwargs.pop("input_data_format")
        do_sample_frames = kwargs.pop("do_sample_frames")
        device = kwargs.pop("device")
        video_metadata = kwargs.pop("video_metadata")
        sample_indices_fn = partial(self.sample_frames, **kwargs) if do_sample_frames else None
        videos, video_metadata = self._decode_and_sample_videos(
            videos, video_metadata=video_metadata, do_sample_frames=do_sample_frames, sample_indices_fn=sample_indices_fn)
        kwargs = self._standardize_kwargs(**kwargs)
        self._validate_preprocess_kwargs(**kwargs)
        kwargs.pop("data_format")
        return_metadata = kwargs.pop("return_metadata")
        kwkey = _kw_key(kwargs)
        if capturing:
            _tls.capture = (self, dict(kwargs), kwkey)
            raise _Captured()
        out, why, load_st, gated, path = None, None, {}, False, "pipe"
        if entry is None:
            why = "no precomputed result for this array"
        else:
            ref, ekey, pv, grid, T, _, load_st = entry
        if entry is not None and pv is _GATED:    # R1/R3: gated in load_bytes (M2_PIPE_MIN_FRAMES), a performance choice
            gated, why = True, "gated: " + str(load_st.get("why"))
        elif entry is not None:
            mean, std = kwargs.get("image_mean"), kwargs.get("image_std")
            mean = mean if isinstance(mean, (list, tuple)) else [mean] * 3
            std = std if isinstance(std, (list, tuple)) else [std] * 3
            why = ("frames array was released" if ref() is None else
                   "kwargs differ from the ones it was built with" if ekey != kwkey else
                   "more than one video" if len(videos) != 1 else
                   "HF re-sampled the frames" if do_sample_frames and list(video_metadata[0].frames_indices) != list(range(T)) else
                   "resize/rescale/normalize not all on" if not (kwargs["do_resize"] and kwargs["do_rescale"] and kwargs["do_normalize"]) else
                   "per-channel mean/std differ" if len(set(mean)) != 1 or len(set(std)) != 1 else
                   "device set" if device is not None else None)
            if why is None and pv is _INLINE:     # R2: one block, computed here with the same per-block math
                sk = grid; pv, grid = sk.compute_inline(src); path = "pipe-inline"
                load_st = {**load_st, "resize_threads": sk.nthreads}
            if why is None:
                out = BatchFeature(data={"pixel_values_videos": pv, "video_grid_thw": grid},
                                   tensor_type=kwargs.get("return_tensors"))
        extra = {k: load_st[k] for k in ("busy_s", "tail_s", "first_s", "blocks_out_of_order", "resize_threads") if k in load_st}
        if why is None:
            _set({"served": True, "why": None, "path": path, **extra})
            cf = sys.modules.get("m2_copyfree_resize")
            if cf is not None and hasattr(cf, "_tls"):
                cf._tls.status = {"served": False, "why": "skipped: served by m2_pipe_resize"}
        else:
            _set({"served": False, "why": why, "gated": gated,
                  **({"wasted_busy_s": extra["busy_s"]} if "busy_s" in extra else {})})
            has_frames = src is not None and src.ndim == 4 and src.shape[0] > 0
            if has_frames and not gated:
                _refuse(why)                       # vLLM can process this video but it was not served
            videos = self._prepare_input_videos(videos=videos, input_data_format=input_data_format, device=device)
            out = self._preprocess(videos=videos, **kwargs)
        if return_metadata:
            out["video_metadata"] = video_metadata
        return out

    for f in (load_bytes, preprocess, parse_video, _video_with_uuid_async, load_from_url, load_from_url_async):
        setattr(f, _MARK, True)
    _read_frames_no_recovery = classmethod(_read_frames_no_recovery)
    VideoMediaIO.load_bytes = load_bytes
    OpenCVVideoBackendMixin._read_frames_no_recovery = _read_frames_no_recovery
    Qwen3VLVideoProcessor.preprocess = preprocess
    MediaConnector.load_from_url, MediaConnector.load_from_url_async = load_from_url, load_from_url_async
    chat_utils.MultiModalContentParser.parse_video = parse_video
    chat_utils.AsyncMultiModalContentParser._video_with_uuid_async = _video_with_uuid_async
