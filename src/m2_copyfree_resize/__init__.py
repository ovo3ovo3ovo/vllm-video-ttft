"""vLLM general plugin (mission_2 unit 5, 10-24): "去掉缩放切块里多余的拷贝" for Qwen3-VL video inputs.

Same math as the HF Qwen3VLVideoProcessor (BICUBIC antialiased resize, fused rescale+normalize, patchify), only the
order and memory layout change, so pixel_values_videos / video_grid_thw stay bit-identical (offline check:
runs/u5plan/plugin_copyfree.txt).  Copies removed (transformers 5.17.0):
  video_processing_utils.py:242 / :249  _prepare_input_videos: from_numpy().contiguous() and permute().contiguous()
  video_utils.group_videos_by_shape     torch.stack of a single video
  image_processing_backends.py:337      full float32 tensor for normalize
  video_processing_qwen3_vl.py:338      torch.cat of a single video
New path: channels_last view -> resize (uint8) -> patchify in uint8 -> normalize in row chunks straight into a
preallocated bf16 buffer.

Off by default; on when M2_COPYFREE_RESIZE=1 (read on every call, so one process can A/B).  Only one video per call,
uint8 on CPU, 3-channel, resize+rescale+normalize all on, equal mean/std per channel; anything else falls back to the
original code and logs once per reason; with M2_STRICT=1 such a fallback raises M2StrictError instead.  Output dtype is bfloat16 (M2_COPYFREE_DTYPE, default bfloat16) because vLLM
casts pixel values to the model dtype right after (vllm/multimodal/processing/context.py:283) and this model runs
in bf16; set M2_COPYFREE_DTYPE=float32 for other models.  register() is idempotent."""
import logging, os, threading
_MARK = "_m2_copyfree"
_CHUNK = 4096
_log = logging.getLogger("m2_copyfree_resize")
_said = set()
class M2StrictError(RuntimeError):
    """M2_STRICT=1: the plugin was on but could not serve this call (no silent fallback)."""

def _strict():
    return os.environ.get("M2_STRICT") == "1"

_tls = threading.local()   # _tls.status = {"served": bool, "why": str|None}, read by m2_mm_trace in the same thread

def _on():
    return os.environ.get("M2_COPYFREE_RESIZE") == "1"

def _fallback(reason):
    if reason not in _said:
        _said.add(reason); _log.warning("m2_copyfree_resize: falling back to the original path (%s)", reason)

def register():
    try:
        import torch
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
        from transformers.image_processing_utils import BatchFeature
        from transformers.image_utils import ChannelDimension, infer_channel_dimension_format
    except Exception as e:  # transformers layout changed: leave everything untouched
        _log.warning("m2_copyfree_resize: not registered (%r)", e); return
    if getattr(Qwen3VLVideoProcessor._preprocess, _MARK, False):
        return
    orig_prep = Qwen3VLVideoProcessor._prepare_input_videos
    orig_pre = Qwen3VLVideoProcessor._preprocess

    def _prepare_input_videos(self, videos, input_data_format=None, device=None):
        if not _on() or device is not None or len(videos) != 1:
            why = "off" if not _on() else "device set" if device is not None else "more than one video"
            _tls.status = {"served": False, "why": why}
            if why != "off" and _strict():
                raise M2StrictError(f"m2_copyfree_resize: {why}")
            return orig_prep(self, videos, input_data_format=input_data_format, device=device)
        out = []
        for video in videos:
            import numpy as np
            if isinstance(video, np.ndarray):
                video = torch.from_numpy(video)            # view, no .contiguous()
            fmt = input_data_format or infer_channel_dimension_format(video)
            if fmt == ChannelDimension.LAST:
                video = video.permute(0, 3, 1, 2)          # channels_last view, no .contiguous()
            out.append(video)
        return out

    def _preprocess(self, videos, do_convert_rgb, do_resize, size, resample, do_rescale, rescale_factor,
                    do_normalize, image_mean, image_std, patch_size=None, temporal_patch_size=None,
                    merge_size=None, cap_pixels_per_frame=None, return_tensors=None, **kwargs):
        args = dict(do_convert_rgb=do_convert_rgb, do_resize=do_resize, size=size, resample=resample,
                    do_rescale=do_rescale, rescale_factor=rescale_factor, do_normalize=do_normalize,
                    image_mean=image_mean, image_std=image_std, patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size, merge_size=merge_size,
                    cap_pixels_per_frame=cap_pixels_per_frame, return_tensors=return_tensors, **kwargs)
        if not _on():
            _tls.status = {"served": False, "why": "off"}
            return orig_pre(self, videos, **args)
        v = videos[0] if len(videos) == 1 else None
        mean = image_mean if isinstance(image_mean, (list, tuple)) else [image_mean] * 3
        std = image_std if isinstance(image_std, (list, tuple)) else [image_std] * 3
        why = ("more than one video" if v is None else
               "not uint8" if v.dtype != torch.uint8 else
               "not on CPU" if v.device.type != "cpu" else
               "not 4-D / 3-channel" if v.dim() != 4 or v.shape[1] != 3 else
               "resize/rescale/normalize not all on" if not (do_resize and do_rescale and do_normalize) else
               "per-channel mean/std differ" if len(set(mean)) != 1 or len(set(std)) != 1 else None)
        if why:
            _fallback(why); _tls.status = {"served": False, "why": why}
            if _strict():
                raise M2StrictError(f"m2_copyfree_resize: {why}")
            return orig_pre(self, videos, **args)
        if cap_pixels_per_frame is None:
            cap_pixels_per_frame = False                   # same default as the original (it only adds a warning)
        if do_convert_rgb:
            v = self.convert_to_rgb(v)                     # 3 channels: returns a view
        x = self.resize(videos=v.unsqueeze(0), size=size, resample=resample, factor=patch_size * merge_size,
                        temporal_factor=temporal_patch_size, cap_pixels_per_frame=cap_pixels_per_frame)
        patches, gt, gh, gw = self.patchify(x, patch_size=patch_size, merge_size=merge_size,
                                            temporal_patch_size=temporal_patch_size)
        patches = patches[0]                               # (seq, dim) uint8, the one real copy
        # fused rescale+normalize exactly as image_processing_backends.py:308-311 + tvF.normalize: (x - m) / s
        m = float(torch.tensor(mean[0]) * (1.0 / rescale_factor)); s = float(torch.tensor(std[0]) * (1.0 / rescale_factor))
        dt = getattr(torch, os.environ.get("M2_COPYFREE_DTYPE", "bfloat16"))
        out = torch.empty(patches.shape, dtype=dt)
        for i in range(0, patches.shape[0], _CHUNK):
            out[i:i + _CHUNK] = patches[i:i + _CHUNK].to(torch.float32).sub_(m).div_(s)
        _tls.status = {"served": True, "why": None}
        return BatchFeature(data={"pixel_values_videos": out, "video_grid_thw": torch.tensor([[gt, gh, gw]])},
                            tensor_type=return_tensors)

    for f in (_prepare_input_videos, _preprocess):
        setattr(f, _MARK, True)
    Qwen3VLVideoProcessor._prepare_input_videos = _prepare_input_videos
    Qwen3VLVideoProcessor._preprocess = _preprocess
