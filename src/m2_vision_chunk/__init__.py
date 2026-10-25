"""vLLM general plugin (mission_2 unit 5 route A, a5, 10-25): "连视觉塔也分块叠进流水线" for Qwen3-VL video inputs
(vLLM 0.31.0, transformers 5.17.0; on top of plugins/m2_pipe_resize).

Today the GPU worker runs the vision tower on the whole video once the request is scheduled
(v1/worker/gpu_model_runner.py:2958 _execute_mm_encoder -> :3157 embed_multimodal), after the API server has decoded,
resized and patchified all of it.  The tower handles every time group (2 frames) as its own attention sequence and its
position embeddings depend on (h, w) only, so cutting pixel_values on time-group boundaries and running the parts
separately is mathematically the whole-video result (runs/u5plan/plugin_vision_chunk.txt step 1, with file:line; fp32
chunk vs whole 1e-6, bf16 differences at bf16 rounding level, step 2).  This plugin runs a copy of vLLM's own
Qwen3_VisionTransformer in the API server: each block m2_pipe_resize finishes (M2_PIPE_CHUNK frames, even) goes to the
tower while the decoder and the resize thread are still busy with the rest; the processor output then carries
video_embeds (+ video_grid_thw, timestamps) instead of pixel_values_videos, which the worker uses as they are
(model_executor/models/qwen3_vl.py:2276-2296, 2329-2330).  The prompt is token-identical (timestamps stay float64, step 2
of plugin_vision_chunk.txt); --enable-mm-embeds is not needed (the request still carries frames; the swap happens inside
the processor, so no embeddings input is opened to clients).

Hooks (no-ops unless M2_VISION_CHUNK=1, read on every call; m2_pipe_resize must be on):
  m2_pipe_resize._Sink._block           after a block's rows of pixel_values are written, queue that block for the tower
                                        (latest computation of a block wins: m2_seg_decode can re-decode a slice and
                                        m2_pipe_resize then redoes the block)
  Qwen3VLVideoProcessor.preprocess      (outermost) remembers which tower state belongs to the pixel_values it returns;
                                        a video m2_pipe_resize computed inline (one block) is run here as one part
  Qwen3VLMultiModalProcessor._apply_hf_processor_main   waits for the tower parts, concatenates them in block order and
                                        replaces pixel_values_videos with video_embeds
Tower: built on first use on M2_VC_DEVICE (required, e.g. cuda:7) in a one-process tensor-parallel group (gloo, file://
rendezvous in a fresh temp dir; vLLM's parameters read the TP rank at construction even with disable_tp, step 2 of
plugin_vision_chunk.txt), mm_encoder_tp_mode="data" so every linear layer is unsplit; weights model.visual.* from the
model's own safetensors; one thread runs the parts in arrival order.  M2_VC_ATTN: attention backend override (offline
tests on CPU).  Test hooks, never in production: M2_VC_FAKE=1 a row-local stand-in tower; M2_VC_KEEP_PV=1 keeps the
pixel_values the embeds were built from in `last_pv` (to check a damaged stream, whose pixels differ run to run).
Not served (M2_STRICT=1 raises M2StrictError): m2_pipe_resize did not serve the video, more than one video, video pruning
on (EVS / VidCom2 keep tokens by looking at the whole video), M2_VC_DEVICE unset, the tower failed to build or to run.
Status per request: _tls.status in the thread that runs the processor: served, why, tower ("api" = this copy on
`device`, "worker" = vLLM's own tower in the GPU worker), blocks (tower parts the video was cut into), parts (tower calls
made, redone blocks included), path vc-blocks / vc-inline, rows, tower_s, wait_s; `last` is a process-wide copy for
offline scripts.  Hash (10-25): while the file named by $M2_HASH_FLAG exists, the pixel_values_videos replaced by
video_embeds are kept in _tls.swapped_pv for m2_mm_trace to hash with the same algorithm as every other phase (it drops
them right after apply)."""
import logging, math, os, queue, sys, tempfile, threading, time, weakref
_MARK = "_m2_vc"
_log = logging.getLogger("m2_vision_chunk")
_said = set()
_tls = threading.local()
_lock = threading.Lock()
_states = {}            # data_ptr of the pixel_values buffer -> _State
last = {}

class M2StrictError(RuntimeError):
    """M2_STRICT=1: the plugin was on but did not serve this video (no silent fallback to the worker's tower)."""

def _on():
    return os.environ.get("M2_VISION_CHUNK") == "1"

def _strict():
    return os.environ.get("M2_STRICT") == "1"

def _note(why):
    if why not in _said:
        _said.add(why); _log.warning("m2_vision_chunk: not served (%s)", why)

class _State:
    """The tower parts of one pixel_values buffer: block b -> (seq, embeds) of its latest computation."""
    def __init__(self, nblk):
        self.cv, self.nblk, self.want, self.done, self.err = threading.Condition(), nblk, {}, {}, None
        self.busy, self.parts = 0.0, 0

    def submit(self, b, job):
        with self.cv:
            seq = self.want.get(b, 0) + 1
            self.want[b] = seq
        _Runner.get().put((self, b, seq, job))

    def finish(self, b, seq, emb, dt, err=None):
        with self.cv:
            self.busy += dt
            if err is not None:
                self.err = self.err or err
            elif b not in self.done or self.done[b][0] < seq:
                # never let an older computation replace a newer one (the runner is FIFO today, so an older part
                # always lands first; this keeps wait() correct if that ever changes)
                self.done[b] = (seq, emb); self.parts += 1
            self.cv.notify_all()

    def _complete(self):
        return all(b in self.want and b in self.done and self.done[b][0] == self.want[b] for b in range(self.nblk))

    def wait(self):
        """Block until every block's latest part is in (no timeout, as m2_pipe_resize).  Returns the embeds in order;
        a part computed from rows m2_pipe_resize later rewrote (seq behind want) is never used."""
        with self.cv:
            while self.err is None and not self._complete():
                self.cv.wait()
            if self.err is not None:
                raise RuntimeError(self.err)
            return [self.done[b][1] for b in range(self.nblk)]

def _worker_backend(own):
    """The ViT attention backend the worker's tower gets: the same selection (model_executor/models/vision.py:100-113 ->
    platforms/cuda.py:556-590) with the dtype the worker builds under (the model dtype, model_loader/base_loader.py:66)
    and the mm_encoder_attn_backend override `own` carries (copied from the server's config in _Runner._build)."""
    from vllm.model_executor.models.vision import get_vit_attn_backend
    vcfg = own.hf_config.vision_config
    return get_vit_attn_backend(head_size=vcfg.hidden_size // vcfg.num_heads, dtype=own.dtype)

def _tower_weight_files(model_dir):
    """The safetensors files holding `model.visual.*`: from the shard index when the checkpoint is sharded, else the
    single unsharded model.safetensors (small models such as Qwen3-VL-2B ship one file and no index)."""
    import json
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        files = sorted({fn for k, fn in weight_map.items() if k.startswith("model.visual.")})
    elif os.path.exists(os.path.join(model_dir, "model.safetensors")):
        files = ["model.safetensors"]
    else:
        raise RuntimeError(f"no model.safetensors.index.json or model.safetensors in {model_dir}")
    if not files:
        raise RuntimeError(f"no model.visual.* weights listed in {index}")
    return files

class _Runner:
    """One tower and one thread per process; parts run in arrival order (requests share it)."""
    _inst, _ilock = None, threading.Lock()

    @classmethod
    def get(cls):
        with cls._ilock:
            if cls._inst is None:
                cls._inst = cls()
            return cls._inst

    def __init__(self):
        self.q, self.tower, self.err, self.device = queue.Queue(), None, None, None
        self.ready = threading.Event()
        threading.Thread(target=self._loop, name="m2_vision_chunk", daemon=True).start()

    def put(self, item):
        self.q.put(item)

    def _build(self):
        import torch
        if os.environ.get("M2_VC_FAKE") == "1":
            self.device = "cpu"
            def fake(pv, grid):
                return pv.reshape(-1, 4 * pv.shape[1])[:, :2560].to(torch.bfloat16).clone()
            self.tower = fake; return
        dev = os.environ.get("M2_VC_DEVICE")
        if not dev:
            raise RuntimeError("M2_VC_DEVICE is not set (the GPU the API server's tower copy runs on)")
        mc = _MC[0]
        if mc is None:
            raise RuntimeError("no model config seen yet")
        from safetensors import safe_open
        from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
        from vllm.distributed import parallel_state as ps
        from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
        import json
        kw = dict(model=mc.model, dtype=mc.dtype, mm_processor_cache_gb=0, mm_encoder_tp_mode="data")
        # review A5-1 (10-25): the worker's tower takes the server's --mm-encoder-attn-backend, so this copy must too,
        # or both checks below would agree on a backend the server never chose.  FP8 encoder attention (scales, amax
        # history) is not replicated here: refused rather than computed differently.
        smm = mc.get_multimodal_config() if hasattr(mc, "get_multimodal_config") else None
        if smm is not None and getattr(smm, "mm_encoder_attn_dtype", None) is not None:
            raise RuntimeError(f"server uses mm_encoder_attn_dtype={smm.mm_encoder_attn_dtype}; not supported")
        if smm is not None and getattr(smm, "mm_encoder_attn_backend", None) is not None:
            kw["mm_encoder_attn_backend"] = smm.mm_encoder_attn_backend
        if os.environ.get("M2_VC_ATTN"):            # offline tests only (CPU): overrides the server's choice
            kw["mm_encoder_attn_backend"] = os.environ["M2_VC_ATTN"]
        own = ModelConfig(**kw)                    # a fresh config: the server's own objects are never touched
        vc = VllmConfig(model_config=own)
        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        from vllm.utils.torch_utils import set_default_torch_dtype
        if dev.startswith("cuda"):
            # 10-25 window 2: the caching allocator kept ~3 GiB more reserved than allocated across videos of different
            # sizes (peak reserved 5404 MiB vs allocated 2358 MiB, plugin_vision_chunk.txt step 4).  Expandable segments
            # let it grow and reuse one mapping instead.  This process does no other CUDA work; numerics unchanged.
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
            torch.cuda.set_device(torch.device(dev))
        target = torch.device(dev)
        with set_current_vllm_config(vc):
            if not ps.model_parallel_is_initialized():
                rdv = tempfile.mkdtemp(prefix="m2_vc_rdv_")
                ps.init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                                distributed_init_method=f"file://{rdv}/init", backend="gloo")
                ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
            hf = own.hf_config
            # built the way the worker builds it (model_loader/base_loader.py:66-72): default dtype = model dtype and
            # parameters created on the target device.  The tower picks its attention backend from
            # torch.get_default_dtype() at construction (qwen3_vl.py:651-654); built under the fp32 default, FLASH_ATTN
            # (fp16/bf16 only) was skipped for TRITON_ATTN while the worker uses FLASH_ATTN (window 2).
            # norm_eps as qwen3_vl.py:1864 reads it (top-level config, default 1e-6).
            with set_default_torch_dtype(own.dtype), target:
                tower = Qwen3_VisionTransformer(hf.vision_config, norm_eps=getattr(hf, "rms_norm_eps", 1e-6),
                                                prefix="visual")
            files = _tower_weight_files(mc.model)
            w = []
            for fn in files:
                with safe_open(os.path.join(mc.model, fn), "pt") as f:
                    w += [(k[len("model.visual."):], f.get_tensor(k)) for k in f.keys() if k.startswith("model.visual.")]
            loaded = tower.load_weights(w)
            names = {n for n, _ in tower.named_parameters()}
            if names - loaded:
                raise RuntimeError(f"tower weights missing: {sorted(names - loaded)[:5]}")
            # the worker's post-load pass (model_loader/utils.py:133-195: quant methods' and MMEncoderAttention's
            # process_weights_after_loading), so nothing the worker does after loading is skipped here
            process_weights_after_loading(tower, own, target)
            tower = tower.eval()
            if str(tower.attn_backend) != str(_worker_backend(own)):
                raise RuntimeError(f"tower attention backend {tower.attn_backend} differs from the worker's "
                                   f"{_worker_backend(own)}")
        self.vc, self.device, self.backend = vc, dev, str(tower.attn_backend)

        def run(pv, grid):
            from vllm.config import set_current_vllm_config
            with set_current_vllm_config(self.vc), torch.inference_mode():
                return tower(pv.to(dev, non_blocking=False), grid_thw=[list(grid)]).to("cpu")
        self.tower = run

    def _loop(self):
        while True:
            st, b, seq, (pv, grid) = self.q.get()
            t0 = time.perf_counter()
            try:
                if self.tower is None and self.err is None:
                    try:
                        self._build()
                    except Exception as e:
                        self.err = f"tower build failed: {e!r}"[:300]
                if self.err is not None:
                    raise RuntimeError(self.err)
                emb = self.tower(pv, grid)
                st.finish(b, seq, emb, time.perf_counter() - t0)
            except Exception as e:
                st.finish(b, seq, None, time.perf_counter() - t0, err=f"{e!r}"[:300])

last_pv = [None]
_MC = [None]            # the ModelConfig of the first served request (the tower is built from its model path / dtype)

def _set(st):
    _tls.status = st
    last.clear(); last.update(st)

def _refuse(why):
    _note(why)
    if _strict():
        raise M2StrictError(f"m2_vision_chunk: {why}")

def register():
    try:
        import torch, transformers, vllm
        import m2_pipe_resize
        m2_pipe_resize.register()                  # its hooks must be in place before ours wrap them
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
        from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
        if transformers.__version__ != "5.17.0" or not vllm.__version__.startswith("0.31.0"):
            raise RuntimeError(f"written for transformers 5.17.0 / vLLM 0.31.0, found {transformers.__version__} / {vllm.__version__}")
        if not getattr(Qwen3VLVideoProcessor.preprocess, "_m2_pipe", False):
            raise RuntimeError("m2_pipe_resize did not register")
    except Exception as e:
        _log.warning("m2_vision_chunk: not registered (%r)", e); return
    if getattr(Qwen3VLMultiModalProcessor._apply_hf_processor_main, _MARK, False):
        return
    Sink = m2_pipe_resize._Sink
    orig_block = Sink._block
    pipe_preprocess = Qwen3VLVideoProcessor.preprocess
    orig_main = Qwen3VLMultiModalProcessor._apply_hf_processor_main

    def _state_for(out, T, TP, chunk):
        key = out.data_ptr()
        with _lock:
            st = _states.get(key)
            if st is None:
                st = _states[key] = _State(math.ceil(T / chunk))
                weakref.finalize(out, _states.pop, key, None)
            return st

    def _block(self, arr, lo, hi, out, h, w, m, s):
        orig_block(self, arr, lo, hi, out, h, w, m, s)
        if not _on() or self.inline:
            return
        kw = self.kw
        P, TP = kw["patch_size"], kw["temporal_patch_size"]
        gh, gw = h // P, w // P
        T = arr.shape[0]
        r0, tg = (lo // TP) * gh * gw, math.ceil((hi - lo) / TP)
        st = _state_for(out, T, TP, self.chunk)
        st.submit(lo // self.chunk, (out[r0:r0 + tg * gh * gw], (tg, gh, gw)))

    def preprocess(self, videos, **kwargs):
        res = pipe_preprocess(self, videos, **kwargs)
        if not _on():
            return res
        pst = getattr(m2_pipe_resize._tls, "status", None) or {}
        pv = res.get("pixel_values_videos") if hasattr(res, "get") else None
        if not pst.get("served") or pv is None:
            _tls.pending = ("refused", f"m2_pipe_resize did not serve this video ({pst.get('why')})")
            return res
        if pst.get("path") == "pipe-inline":       # one block: nothing overlapped, run it here as one part
            g = res["video_grid_thw"][0].tolist()
            st = _State(1); st.submit(0, (pv, tuple(g)))
            _tls.pending = ("inline", st)
            return res
        with _lock:
            st = _states.get(pv.data_ptr())
        _tls.pending = ("blocks", st) if st is not None else ("refused", "no tower parts for this pixel_values buffer")
        return res

    def _apply_hf_processor_main(self, mm_items, hf_kwargs):
        if not _on():
            return orig_main(self, mm_items, hf_kwargs)
        _MC[0] = _MC[0] or self.info.ctx.model_config
        _tls.pending = None
        out = orig_main(self, mm_items, hf_kwargs)
        pend, _tls.pending = getattr(_tls, "pending", None), None
        if "pixel_values_videos" not in out:
            return out                             # no video in this request
        t0 = time.perf_counter()
        why = None
        nvid = len(out["video_grid_thw"])
        if self.info.ctx.get_mm_config().get_video_pruning_spec() is not None:
            why = "video pruning is on (it keeps tokens by looking at the whole video)"
        elif nvid != 1:
            why = f"{nvid} videos in one request"
        elif pend is None:
            why = "preprocess did not run through the plugin"
        elif pend[0] == "refused":
            why = pend[1]
        if why is None:
            st = pend[1]
            try:
                parts = st.wait()
            except Exception as e:
                why = f"tower failed: {e}"[:300]
        blocks = pend[1].nblk if pend is not None and pend[0] != "refused" else None
        if why is not None:
            _set({"served": False, "why": why, "tower": "worker", "device": None, "blocks": blocks})
            vllm_ok = out["pixel_values_videos"].shape[0] > 0
            if vllm_ok:
                _refuse(why)
            return out
        import torch
        emb = torch.cat(parts) if len(parts) > 1 else parts[0]
        grid = out["video_grid_thw"]
        if emb.shape[0] != int(grid.prod()) // 4:
            why = f"tower output has {emb.shape[0]} rows, grid {grid.tolist()} needs {int(grid.prod()) // 4}"
            _set({"served": False, "why": why, "tower": "worker", "device": None, "blocks": blocks}); _refuse(why)
            return out
        if os.environ.get("M2_VC_KEEP_PV") == "1":
            last_pv[0] = out["pixel_values_videos"]
        flag, tr = os.environ.get("M2_HASH_FLAG"), sys.modules.get("m2_mm_trace")
        if flag and getattr(tr, "_F", None) and os.path.exists(flag):    # m2_mm_trace hashes and drops it after apply
            _tls.swapped_pv = out["pixel_values_videos"]
        del out["pixel_values_videos"]
        out["video_embeds"] = emb
        _set({"served": True, "why": None, "tower": "api", "device": _Runner.get().device, "blocks": st.nblk,
              "path": "vc-" + pend[0], "parts": st.parts, "tower_s": round(st.busy, 4),
              "wait_s": round(time.perf_counter() - t0, 4), "rows": int(emb.shape[0])})
        return out

    for f in (preprocess, _apply_hf_processor_main, _block):
        setattr(f, _MARK, True)
    preprocess._m2_pipe = True                     # m2_pipe_resize's own register() checks preprocess carries its mark
    Sink._block = _block
    Qwen3VLVideoProcessor.preprocess = preprocess
    Qwen3VLMultiModalProcessor._apply_hf_processor_main = _apply_hf_processor_main
