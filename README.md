English | [简体中文](README.zh-CN.md)

# vllm-video-ttft

Plugins for vLLM 0.31.0 that cut the **time to first token (TTFT) of video requests**. They change the decode, resize-and-patchify and vision-tower steps. For the four options that don't use the GPU (the `exact` preset), the input sent to the vision tower is bit-identical to stock vLLM.

```bash
pip install -e .                                                   # 1. install (inside a vLLM 0.31.0 environment)
python -m vllm_video_ttft.probe --model /path/to/model --tp 1       # 2. see what this machine can enable, and the recommended settings
VLLM_VIDEO_TTFT_PRESET=exact vllm serve /path/to/model --mm-processor-cache-gb 0   # 3. start the server
```

---

## For the deploying agent: follow these four steps

1. **Run probe**: `python -m vllm_video_ttft.probe --model <model_dir> --tp <planned_tp>`. Read-only, finishes in a few seconds, prints JSON: core count, NUMA layout, memory, `/dev/shm`, each GPU's model / memory / compute capability, software versions, which model family the model belongs to, whether each plugin can be enabled and why, plus a recommended set of environment variables and launch arguments.
2. **Pick a setup from the three tables below**: [Pick plugins by model](#pick-plugins-by-model), [Choose parameters by machine](#choose-parameters-by-machine), [Plugins at a glance](#plugins-at-a-glance). probe's recommendations are computed from the rules in these sections. Wherever a table says "Inferred", trust the measurement from step 3.
3. **Run selfcheck**: `python -m vllm_video_ttft.selfcheck --model <model_dir> video1 video2 video3`. Offline, it compares running with and without the plugins: is what goes into the vision tower bit-identical, and did each plugin actually take effect. Go live only when everything PASSes. Suggested videos: 1 short clip and 2 long videos over 5 minutes.
4. **Start the server**: add the environment variables and arguments probe gave you. The log will contain a line `vllm_video_ttft 0.1.0: preset=..., on: ...` that says what is actually on.

When in doubt, use the `exact` preset: it doesn't change outputs, so the worst case is no speedup.

---

## Plugins at a glance

| Plugin | Switch | Step it shortens | Models | Changes output? | Cost |
|---|---|---|---|---|---|
| Segmented decode | `M2_SEG_DECODE=1` | Decode: split a video into segments at keyframes and decode them in parallel in a process pool | Generic: anything that goes through vLLM's default OpenCV decoder | No (bit-identical) | Process pool resident memory median 0.85–1.0 GiB, up to 2.6 GiB; a segmented request averages about 31 CPU core-seconds (warm-up included), 12–57 by video length |
| Frame skipping (skip non-reference frames) | `M2_SEG_SKIP_NONREF=1` | Decode: frames that are neither sampled nor referenced by another frame are not decoded | Same as above; stacks on segmented decode | No | Total time unchanged (on long videos with only I/P frames, the frame-table pass adds 0.03–0.05 s, lost in the noise); only helps videos with B-frames or other non-reference frames |
| Copy-free resize (removes redundant copies in resize-and-patchify) | `M2_COPYFREE_RESIZE=1` | Resize and patchify | Qwen3-VL family only | No | None |
| Pipelined resize (resize while decoding) | `M2_PIPE_RESIZE=1` | Overlaps resize-and-patchify with decode | Qwen3-VL family only; requires `--mm-processor-cache-gb 0` | No | One extra resize thread |
| Chunked vision tower (vision tower runs in chunks inside the pipeline) | `M2_VISION_CHUNK=1` | Overlaps the vision tower with decode and resize; computed in the API process | Qwen3-VL family only; depends on pipelined resize | **Yes**: vision-tower output is not bit-identical (vs. a full vision tower per GPU, 34 of 100 videos are bit-identical; the other 66 have a median relative difference of 2.9%, max 6.25%) | One extra copy of the vision tower on one GPU (measured up to 3.0 GiB for Qwen3.8) |

**Presets**: `VLLM_VIDEO_TTFT_PRESET`
- `off` (default): installed but does nothing.
- `exact`: the first four (segmented decode, frame skipping, copy-free resize, pipelined resize); output is bit-identical to vLLM.
- `all`: adds the chunked vision tower.

An explicit per-plugin switch overrides the preset, e.g. `VLLM_VIDEO_TTFT_PRESET=exact M2_PIPE_RESIZE=0`.

`M2_STRICT=1` is for testing and self-checks: when a plugin should take effect but doesn't, it raises an error instead of silently falling back to vLLM's original path. Don't turn it on in production.

---

## Pick plugins by model

### Where each plugin hooks in (Code-read, vLLM 0.31.0 / transformers 5.17.0)

- **Segmented decode, frame skipping**: replace `vllm.multimodal.video_decoders.opencv.decode_opencv` and `VideoMediaIO.load_file`. Only video backends that end up calling `decode_video(backend="opencv")` → `decode_opencv` are taken over.
- **Copy-free resize**: replaces `Qwen3VLVideoProcessor._prepare_input_videos / _preprocess` in `transformers`.
- **Pipelined resize**: replaces `Qwen3VLVideoProcessor.preprocess`, `VideoMediaIO.load_bytes`, `MediaConnector.load_from_url(_async)`, OpenCV frame reading, and `parse_video` in chat parsing.
- **Chunked vision tower**: builds one copy of vLLM's own `Qwen3_VisionTransformer` in the API process, replaces `Qwen3VLMultiModalProcessor._apply_hf_processor_main`, and swaps `pixel_values_videos` for `video_embeds`.

### By model family

Legend for the Basis column: **Measured** = run in this repo's validation; **Code-read** = confirmed in the vLLM 0.31.0 source that the code path reaches the hooks above; **Inferred** = not verified, confirm with selfcheck.

| Model family | Architecture (config.json) | Example | Segmented decode + frame skipping | Copy-free / pipelined resize | Chunked vision tower | Basis |
|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | `Qwen4ExpForConditionalGeneration` | Qwen/Qwen3.8-Flash-Next | Yes | Yes | Yes | **Measured**: 8×5090, TEP8 (TP8 + expert parallel), 908 videos bit-identical, speed measured in a running server; accuracy: see [Accuracy (the two options that change outputs)](#accuracy-the-two-options-that-change-outputs) |
| Qwen3-VL (incl. MoE) | `Qwen3VLForConditionalGeneration`, `Qwen3VLMoeForConditionalGeneration` | Qwen/Qwen3-VL-2B-Instruct | Yes | Yes | Yes | **Measured**: RTX 3080 (WSL2), in a server, `all` preset; selfcheck 15/15 PASS, vision-tower input bit-identical; see [Results on another machine](#results-on-another-machine-rtx-3080-20-gb-wsl2-2b-class-models) |
| Qwen3.5, Cosmos3-Edge, InternS1-Pro / InternS2 and other models that use the Qwen3-VL vision tower | Varies | — | Yes | Depends on whether the video processor is `Qwen3VLVideoProcessor` | Inferred: possibly | Inferred: the source reuses Qwen3-VL classes, but the processor and template may differ; confirm with selfcheck |
| Qwen2.5-VL / Qwen2-VL | `Qwen2_5_VLForConditionalGeneration`, `Qwen2VLForConditionalGeneration` | Qwen/Qwen2.5-VL-7B-Instruct | Yes | No effect | No effect | Code-read: video backend `qwen2_vl` inherits the default OpenCV path; the processor is `Qwen2VLVideoProcessor`, which is not a hook point |
| LLaVA-NeXT-Video | `LlavaNextVideoForConditionalGeneration` | llava-hf/LLaVA-NeXT-Video-7B-hf | Yes | No effect | No effect | Code-read: no dedicated video backend registered, uses the default OpenCV path; not verified, confirm with selfcheck |
| LLaVA-OneVision | `LlavaOnevisionForConditionalGeneration` | llava-hf/llava-onevision-qwen2-0.5b-ov-hf | Yes | No effect | No effect | **Measured (processor level, CPU)**: 3 videos; under the `all` preset, pixels are bit-identical to running without the plugins; segmented decode + frame skipping took effect on the two long videos; copy-free resize, pipelined resize and the chunked vision tower all stepped aside |
| InternVL | `InternVLChatModel` etc. | OpenGVLab/InternVL3-8B | Yes | No effect | No effect | Same as above |
| Gemma family (with video) | `Gemma4ForConditionalGeneration` etc. | google/gemma-4-E2B-it | Yes | No effect | No effect | **Measured**: RTX 3080 (WSL2), in a server, segmented decode + frame skipping; selfcheck 15/15 PASS, bit-identical; see [Results on another machine](#results-on-another-machine-rtx-3080-20-gb-wsl2-2b-class-models) |
| GLM-4.1V / GLM-4.5V / GLM-4.6V | `Glm4vForConditionalGeneration` etc. | — | Yes | No effect | No effect | Code-read: backends `opencv_dynamic` / `glm46v` inherit the default OpenCV path |
| GLM-5-Next video | — | — | **No effect** | No effect | No effect | Code-read: its backend overrides `read_frames`, so segmented decode steps aside automatically (log `loader overrides read_frames`) |
| Molmo2, OpenPangu-VL | — | — | Inferred: possibly | No effect | No effect | Inferred: they call `decode_video` directly; not verified whether it ends up in `decode_opencv` |
| Image-only models (no video input) | — | — | No effect | No effect | No effect | The plugins only touch the video path; image requests are unaffected |

"No effect" means the hook is never called on that model, or the plugin decides it shouldn't take over and steps aside. The server runs normally and you don't need to turn anything off; only `M2_STRICT=1` makes it raise an error.

### When not to enable

| Situation | Don't enable | Why |
|---|---|---|
| Output must be bit-identical to vLLM | Chunked vision tower | It makes the vision-tower output no longer bit-identical (vs. a full vision tower per GPU, 34 of 100 videos are bit-identical; the other 66 have a median relative difference of 2.9%, max 6.25%, so this is more than last-bit rounding). No accuracy drop was seen on the test questions, see [Accuracy (the two options that change outputs)](#accuracy-the-two-options-that-change-outputs) |
| Video pruning is on (`--video-pruning-rate > 0`, EVS / VidCom2) | Chunked vision tower | Pruning needs the whole video to decide which tokens to keep, so chunking is no longer equivalent; the plugin refuses to chunk and hands the work back to the worker's own vision tower |
| More than one video in a request | Chunked vision tower | Not supported; handed back to the worker automatically |
| Processor cache is on (`--mm-processor-cache-gb > 0`) | Pipelined resize, chunked vision tower | On a cache hit nothing is decoded again, so the pipeline plugins step aside. All our measurements use `--mm-processor-cache-gb 0` |
| A different decode backend (`--media-io-kwargs '{"video":{"backend":"pynvvideocodec" or "torchcodec"}}'`) | Segmented decode, frame skipping, and the pipeline that depends on them | Segmented decode only takes over the OpenCV path; with another backend there is nothing to take over |
| `frame_recovery=True` | Segmented decode | vLLM's bad-frame recovery scan is not reimplemented; it falls back to the original path automatically |
| High concurrency, CPU is tight | Be careful with segmented decode | A segmented request averages about 31 CPU core-seconds (12–57 by video length), resident memory up to 2.6 GiB; we only verified bit-identical results at 4/8 concurrency and never measured high-concurrency throughput, see [High concurrency](#high-concurrency-inferred-not-measured) |
| A GPU with no free memory | Chunked vision tower | It needs one extra copy of the vision tower; when it doesn't fit, the plugin reports OOM and hands the work back to the worker (raises in strict mode) |
| All videos are short (raw frame count < `M2_SEG_MIN_SRC_FRAMES`, default 2642) | Segmented decode does nothing even when on | Short videos are not segmented; most of a short video's TTFT is prefill, so the plugins save little |

### What happens when a plugin doesn't take effect (Code-read)

- Each reason for not taking effect is logged once as a warning (e.g. `m2_seg_decode: using vLLM's decoder (...)`, `m2_pipe_resize: not using the pipelined result (...)`, `m2_vision_chunk: not served (...)`), then vLLM's original path runs and the result is the same as without the plugin. One pair of these, `m2_pipe_resize: not using the pipelined result (no precomputed result for this array)` followed by `m2_vision_chunk: not served (...)`, appears once at startup before `Application startup complete`: vLLM runs its processor on a dummy video for warm-up and memory profiling, and that video never goes through the decoder, so there is nothing to serve. It is not a real request.
- Wrong vLLM or transformers version: the plugin logs one warning and does not register (in strict mode, startup fails).
- Not a Qwen3-VL family model: the hooks of copy-free resize and the chunked vision tower are never called. Pipelined resize does one trial run on the first request, finds it can't get Qwen3-VL's processor parameters, remembers that failure, steps aside from then on, and logs only one line `not using the pipelined result (...)`. Measured (Gemma-4-E2B, LLaVA-OneVision, CPU, 3 videos × 2 runs each): the trial run happens once, on the first request, and 0 times after that; no exceptions, no hangs.
- On a non-Qwen3-VL model with `M2_STRICT=1` and pipelined resize on (e.g. the `all` preset), every request raises `M2StrictError: m2_pipe_resize: could not compute this request's processor kwargs ...` instead of silently computing something wrong. The reason is that strict mode treats every "on but not in effect" as an error and doesn't distinguish "not applicable to this model" from "something went wrong". On non-Qwen models the hooks of copy-free resize and the chunked vision tower are never called, so they don't raise; pipelined resize only finds out it doesn't apply after a trial run, so it raises. Strict mode is therefore only for self-checks on Qwen3-VL. On non-Qwen models, enable only segmented decode + frame skipping (this is what probe recommends); without strict mode, pipelined resize steps aside by itself and the result is the same as without the plugins (measured 6/6).

---

## Choose parameters by machine

This section gives general rules; work out your own values from your machine's core count and NUMA layout. All numbers come from one machine: 2-socket Xeon Platinum 8558, 96 physical cores, 192 logical cores, 4 NUMA nodes; 8×RTX 5090, TEP8. Anything marked "Inferred" is extrapolated from this machine; confirm it with the timings selfcheck prints.

### CPU: where the threads go

On the API-process side, one video request uses three groups of threads at the same time:

| Who | Threads | When |
|---|---|---|
| Segmented decode process pool | `M2_SEG_N × M2_SEG_THREADS` (default 16 × 4 = 64) | Only on long videos that get segmented |
| Resize and patchify | `OMP_NUM_THREADS`; with pipelined resize on, it runs at the same time as decode | Every video |
| vLLM scheduler process + API main thread | About 1–2 cores | Always |

Decode threads can be oversubscribed: measured, 64 decode threads on 48 logical cores was fastest (32 segments was no faster). Take the value from the formula below and don't go higher. The served setup used 16 × 4 decode threads plus `OMP_NUM_THREADS=12`, 76 threads in total (about 78 including the scheduler process and the API main thread), which is also more than the 48 physical cores of the two nodes it ran on; this combination was measured in the server. The segment-count scan was done on one NUMA node (24 physical cores, 48 logical cores), where 64 decode threads (16 segments × 4 threads) was the fastest in that scan. The server used two nodes (48 physical cores, 96 logical cores) and only ran 16 × 4; no new scan was done there.

### CPU: how to allocate

1. **First map cores to NUMA nodes and sockets**: `lscpu`, or `numa_nodes` in probe's output.
2. **Put the API process, the segmented-decode pool and the frame buffers on the same socket**:
   - API process: `numactl --cpunodebind=N --membind=N vllm serve ...`.
   - Process pool: `M2_SEG_WNODES=N[,other node on the same socket]`, or list core IDs directly with `M2_SEG_WCPUS=...`.
   - Frame buffers follow the server's `--membind`.
   - Measured: offline, pinning the main process to the other socket raised the frame copy-back for a 35-minute raw video from 1.06 s to 1.72 s, and vLLM's own decode from 7.10 s to 8.27 s. In the server, with the zero-copy path, pinning to the wrong socket costs much less (only resize-and-patchify gets 6–10% slower), but pinning to the same socket is still recommended.
   - When neither `M2_SEG_WNODES` nor `M2_SEG_WCPUS` is set, this plugin uses all cores the process is allowed to run on.
3. **Segment count vs. threads per segment** (measured offline on the same machine, one NUMA node only: 24 physical cores, 48 logical cores):

   | Video (raw frames) | vLLM's own decode | 1 thread per segment: 8 / 16 / 32 segments | 4 threads per segment: 8 / 16 / 32 segments |
   |---|---|---|---|
   | 2642 frames | 0.637 s | 0.758 / 0.490 / 0.497 | 0.350 / **0.295** / 0.362 |
   | 9156 frames | 1.316 s | 1.169 / 0.704 / 0.684 | 0.643 / **0.506** / 0.612 |
   | 49715 frames | 7.411 s | 7.656 / 4.432 / 3.443 | 3.881 / **2.881** / 2.897 |

   - 4 threads per segment with 16 segments is best; 32 segments is no faster.
   - With 1 thread per segment, one video is already faster than vLLM's own decode at 8 segments, while the other two need 16 segments. On machines with few cores, 1 thread per segment with more segments may beat 4 threads per segment (inferred).
   - FFmpeg's own multithreading is already on: vLLM's own decode is multithreaded by default, and forcing 1 thread makes it 6–8 times slower. So on machines with few cores, segmented decode gains noticeably less than the numbers in the results tables below (inferred).
4. **Resize-and-patchify threads (`OMP_NUM_THREADS`)**: going from 1 thread to 12 threads cuts resize-and-patchify to 0.36–0.46 of its original time, almost all of the saving is in the "convert to tensor" step. When unset, vLLM drops the API process to 1 thread, so always set it. We did not measure whether going above 12 threads helps.
5. **Resize threads only (`M2_PIPE_RESIZE_THREADS`, only matters with pipelined resize on)**: falls back to `OMP_NUM_THREADS` when unset. Measured (offline, 15 videos, alternated with "unset" 3 times): at 4 threads and above, raw videos differ by only 1–3% and short clips by 5–8% (only a few milliseconds); 2 threads is 2–6% slower, 1 thread 5–20% slower. What slows down is the decode segment, which includes the resizing that overlaps with decode; decode itself does not get slower. With this switch set, serial runs (five levels, the full 908 videos twice) and 4/8 concurrency are all bit-identical to unset, and the thread count only applies to that request.

### CPU: rule-of-thumb formula (this is what probe uses)

```text
C_node = usable nodes on the API process's socket (4-node machine: the first two; 2-node: 1; single socket: all)
L = logical cores on these nodes − current 1-minute load
P = physical cores on these nodes − current 1-minute load

OMP_NUM_THREADS = clamp(P / 4, 1, 12)
D               = min(64, 4·L / 3)                # total segmented-decode threads
M2_SEG_THREADS  = 4 if D ≥ 16 else 1
M2_SEG_N        = clamp(D / M2_SEG_THREADS, 2, 16)
M2_SEG_WNODES   = C_node
```

- Measured basis: `D = 4·L / 3` comes from the fastest setting, 64 threads (16 segments × 4 threads), under the segment-scan conditions (one NUMA node, L = 48 logical cores). 32 segments was no faster, so the cap is 64. `OMP_NUM_THREADS` was only measured at 1 vs. 12.
- Extrapolated: on two nodes (L = 96) the formula also gives 16 × 4 (it hits the cap), which is what the server used, but no other combination was scanned on 96 cores. Values computed for machines with fewer than 48 cores, and the ratios 1/4 and 4/3 themselves, were not measured at other core counts. On machines with few cores, compare two or three settings using the timings selfcheck prints.

### CPU: segmenting threshold

- `M2_SEG_MIN_SRC_FRAMES` (default 2642): a video is segmented only when the number of raw frames decode has to grab reaches this value. Raw frames = duration × source frame rate.
- This value was set on 48 physical cores. Among our 100 speed-test videos, every video above 2642 frames was faster with segmenting; below it some were faster and some slower (the 5 raw videos below the threshold: 1479 frames 0.015 s slower, 2006 frames 0.085 s slower, 1153 frames 0.045 s faster, 1500 frames 0.008 s faster, 1838 frames 0.175 s faster), and all 70 short clips were slower. 2642 is the smallest value "above which every video is faster", not the crossover point.
- On machines with fewer cores, the fixed cost of segmenting is a larger share, so the crossover moves up (inferred). How to check: run selfcheck on two or three videos whose raw frame counts are near the threshold and compare the seconds in the `off` and `exact` rows; if segmenting is slower, raise the threshold. selfcheck runs each setting only once, so it shows the direction, not a precise value.

### GPU: single GPU vs. multiple GPUs

| Setting | Single GPU (TP=1) | Multiple GPUs (TP>1) |
|---|---|---|
| `--mm-encoder-tp-mode data` | Makes no difference | Recommended: each GPU holds a full copy of the vision tower, no cross-GPU sharding. The cost is the memory of one vision-tower copy on every GPU; it changes the vision-tower output; for accuracy, see [Accuracy (the two options that change outputs)](#accuracy-the-two-options-that-change-outputs) |
| `VLLM_MQ_MAX_CHUNK_BYTES_MB` | **Don't set it**: with only one worker, vLLM doesn't use the multiprocess executor (`vllm/config/parallel.py:1033` picks the uni executor), and this broadcast ring is only built with this value inside the multiprocess executor (`vllm/v1/executor/multiproc_executor.py:151`) | Recommended: set it larger than the largest single message of a request. Without the chunked vision tower the message is pixels (our raw videos: 106–142 MiB); with it, the message is vision features (at most 60.5 MiB across our 100 videos). The default is 16 MiB; larger messages go over zmq instead, which is noticeably slower. We use 256: the handoff time on raw videos drops to 0.59–0.65 of what it was, and TTFT gets 2–4% faster. The cost is about 10 × this value in shared memory: at 256 the broadcast ring measured 2.5 GiB, about 2.34 GiB more than the default (by formula 10 × 16 MiB = 160 MiB, not measured) |
| `M2_VC_DEVICE` of the chunked vision tower | Use this GPU (with only one visible GPU, this plugin sets it to `cuda:0` automatically) | Must be set explicitly to a GPU with free memory |
| Leaving GPU memory for the chunked vision tower | `--gpu-memory-utilization` has to leave room for it, otherwise it competes with the KV cache. Qwen3.8's vision tower measured up to 3.0 GiB; small models have much smaller vision towers, so estimate vision-tower parameter count × 2 bytes, plus activations | Same as on the left, only for the GPU that holds the vision tower |

### GPU: why GPU decoding is not the default

vLLM ships two kinds of GPU decoding: pynvvideocodec (`--media-io-kwargs '{"video":{"backend":"pynvvideocodec"}}'`, NVDEC decodes and then copies frames back to the host) and torchcodec cuda (`{"backend":"torchcodec","device":"cuda"}`, frames stay in GPU memory). We tested both and don't recommend them by default:

- **pynvvideocodec is faster than stock OpenCV but slower than segmented decode**: measured in the server, the decode step drops to 0.26–0.80 of stock OpenCV (short clips 0.26, raw-video groups 0.69–0.80), and TTFT of the >30-minute group drops to 0.74 of stock; the plugins are still 1.10–3.80× faster than it (see results table 2). It can't be combined with segmented decode (segmented decode only takes over the OpenCV path). The decoder context takes about 1.25 GiB of GPU memory.
- **pynvvideocodec changes pixels**: decoded pixels differ from OpenCV, by up to 17 gray levels on our videos. Its `hw_decoders` only spreads concurrent requests across decoders; it doesn't split one video for parallel decoding.
- **torchcodec cuda runs out of memory when the GPU is full**: frames stay in GPU memory. Under `--gpu-memory-utilization 0.85`, a 305 s video needed 1.6 GiB and failed outright. It also changes pixels.
- **When it's worth a try**: if you don't need bit-identical output, CPU is tight and the GPU has headroom (torchcodec cuda also needs enough headroom to hold every frame of a video), pynvvideocodec is worth trying.

### GPU: hardware limits

- NVIDIA driver ≥ 580: torch 2.13.0+cu130 needs CUDA 13.
- Vision-tower FP8 attention (`--mm-encoder-attn-dtype fp8`) does not start on SM120 (RTX 50 series): cuDNN 9.20's "FP8 input, bf16 output" attention kernel is only provided for SM100, but vLLM's check only looks at compute capability ≥ 10.0, so it lets it through and then fails while building the graph. It may work on SM100; we haven't tested it. The chunked vision tower doesn't support FP8 attention either.
- probe prints each GPU's model, memory and compute capability.

---

## Using it together with vLLM's own options

The plugins only cover the API-process side. The options below are vLLM's own; without them the plugins gain less:

```bash
export OMP_NUM_THREADS=12                 # resize-and-patchify threads; pick it with the formula above
export VLLM_MQ_MAX_CHUNK_BYTES_MB=256     # only set it when TP>1
numactl --cpunodebind=0 --membind=0 \
vllm serve <model_dir> --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-gb 0
```

`--enable-expert-parallel` only matters for MoE models. Our measurements used it (TEP8); drop that line for non-MoE models.

---

## Results (8×RTX 5090, Qwen3.8-Flash-Next, online FP8, TEP8)

**What is measured**: TTFT only (from sending the request to receiving the first token), one request at a time, answers capped at 16 tokens. The plugins only touch steps before the first token, so the TTFT numbers don't depend on answer length. How much faster a whole request gets depends on how long the answer is, because generation itself is not faster.

### 1. Stacking up from the most basic vLLM (100 videos)

The most basic baseline = vLLM 0.31.0 + online FP8, default settings. Each column is the server with everything to its left stacked on. Speedups are computed per video first, then the median is taken within each group.

| Video type | Videos | Baseline TTFT | + 12 threads | + copy-free resize | + segmented decode | + pipelined resize | + frame skipping | + full vision tower per GPU | + chunked vision tower | + message threshold 64 MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| Short clips (MV-Bench) | 70 | 0.86 s | 1.25× | 1.21× | 1.35× | 1.35× | 1.35× | 1.58× | 1.72× | 1.76× |
| Raw video ≤2 min | 9 | 3.55 s | 1.28× | 1.42× | 1.48× | 1.54× | 1.59× | 1.83× | 2.22× | 2.27× |
| Raw video 2–10 min | 7 | 5.79 s | 1.25× | 1.43× | 2.07× | 2.18× | 2.44× | 2.73× | 3.15× | 3.30× |
| Raw video 10–30 min | 4 | 9.06 s | 1.25× | 1.43× | 2.27× | 2.53× | 2.94× | 3.29× | 3.54× | 3.81× |
| Raw video >30 min | 10 | 16.95 s | 1.11× | 1.15× | 2.79× | 2.96× | 4.14× | 4.54× | 4.83× | 4.92× |

These columns were measured in several sessions. For each step's own speedup, trust the control that was measured right next to it; this table is for the cumulative picture. "12 threads", "full vision tower per GPU" and "message threshold" are vLLM's own options, not plugins, so the next table compares the plugins' contribution on its own. Background release of frame buffers was added afterwards (it is included in this repo's segmented decode); on 15 videos it lowered TTFT of the 2–10 / 10–30 / >30-minute groups further, to 0.986 / 0.936 / 0.950.

### 2. How much faster than "vLLM tuned to its best" (15 videos)

- 3 videos per length group: the shortest, the median and the longest by duration.
- The three configurations were alternated, two passes each; two passes of the same configuration differ by only 0.98–1.01.
- **Best native (input unchanged)**: stock vLLM without the plugins ("native" below means the same), with `OMP_NUM_THREADS=12` + `--mm-encoder-tp-mode data` + `VLLM_MQ_MAX_CHUNK_BYTES_MB=256`.
- **Best native (input may change)**: the previous one plus pynvvideocodec (NVDEC decoding), which changes pixels.
- **Plugins**: the same native options as the first one, plus all plugins (`all` preset). The input to the vision tower is unchanged.

| Video type | Best native (input unchanged) | Best native (input may change) | Plugins | Speedup vs. the best native per group | Saved per video vs. best native (input unchanged) |
|---|---|---|---|---|---|
| Short clips | 0.383 s | 0.369 s | 0.335 s | 1.10× | 0.05 s |
| Raw video ≤2 min | 2.380 s | 2.103 s | 1.453 s | 1.37× | 0.79 s |
| Raw video 2–10 min | 3.831 s | 4.092 s | 1.740 s | 2.20× | 2.09 s |
| Raw video 10–30 min | 6.378 s | 5.703 s | 2.091 s | 2.69× | 4.29 s |
| Raw video >30 min | 10.064 s | 9.806 s | 2.549 s | 3.80× | 7.52 s |

"Speedup vs. the best native per group": for each length group, the plugins are compared with whichever native configuration was fastest for that group, which is the most generous choice for native. Seconds are group medians and speedups are medians of per-video ratios, so one can't be converted into the other directly. With only 3 videos per group, this shows the direction and the rough size, nothing finer.

### 3. Correctness

- The options that don't change output: all 908 videos (MV-Bench-mini 878 + Video-MME raw 30) bit-identical; a 500-run long test showed no leaks, and handle and thread counts don't grow; 4 and 8 concurrency are bit-identical to serial; edge-case videos (odd width/height, rotation, variable frame rate, corrupt streams, etc.) 1155 cases tested.
- On corrupt streams, pixels of frames filled in by error concealment can differ from vLLM; vLLM itself isn't stable across thread counts there. The structure (frame count, indices, shapes) stays the same, and frames outside the broken GOP are bit-identical.
- The two options that change output (full vision tower per GPU, chunked vision tower) were tested for accuracy, see the next section.

#### Accuracy (the two options that change outputs)

**Which ones need testing, and why**
- The four options that don't change output (`exact` preset: segmented decode, frame skipping, copy-free resize, pipelined resize) send bit-identical input to the vision tower. The model sees exactly the same thing, so no accuracy test is needed.
- Only two options change output, and both change the output of the vision tower:
  - vLLM's own "full vision tower per GPU" (`--mm-encoder-tp-mode data`): vs. the default cross-GPU sharding, the vision-tower output has a median relative difference of 6.2%.
  - This repo's "chunked vision tower" (`M2_VISION_CHUNK=1`): vs. a full vision tower per GPU, 34 of 100 videos are bit-identical; the other 66 have a median relative difference of 2.9%, max 6.25%.
- Neither difference is just last-bit rounding, so we used an exam to see whether they change answers.

**How we tested**
- Question bank: 990 video multiple-choice questions, 900 from MV-Bench-mini plus a local subset of 90 from Video-MME. Split with fixed seed 20261025 into 10 groups of 50 (45 MV-Bench + 5 Video-MME), with no overlap between groups. "Full vision tower per GPU" took all 10 groups, 500 questions; "chunked vision tower" took the first 5 groups, 250 questions.
- Generation: EvalScope 1.14.0 `general_vmcq`; the prompt asks the model to think step by step and finish with a line `ANSWER: <letter>`. Temperature 0, top_p 1, seed 0, `max_tokens` 1024, thinking off, one question at a time.
- Answer extraction: the letter after the last `ANSWER:` in the response, case-insensitive. Without that marker, the last uppercase letter, which must be one of the question's options; otherwise the question counts as unanswered (parsed at `evalscope/utils/multi_choices.py:255`, fallback at `:190`).
- Scoring: correct if the extracted letter equals the reference answer, otherwise wrong. Unanswered counts as wrong.
- Test: only the questions where the two sides differ in correctness count; two-sided exact McNemar test.
- Compared against: the baseline is the most basic server (vLLM 0.31.0 + online FP8, default settings, no plugins). Each option is compared with a baseline taken in the same time window on exactly the same questions, and we confirmed that the input to the vision tower is bit-identical on both sides, so the only difference is the vision tower. The "chunked vision tower" run also had the other three plugins, 12 threads and a full vision tower per GPU on, with strict mode enabled.
- How much the baseline varies on its own: even at temperature 0 with a fixed seed, answers are not identical every time. Taking group 1 twice on the same server instance, 3 questions differed in correctness (31 vs. 32 correct). Retaking the first 5 groups on a new server instance, 23 of 250 questions flipped (right→wrong 12, wrong→right 11), 171 vs. 170 correct. Use this variation as the yardstick when comparing the two options with the baseline.
- Bad outputs: three kinds were checked: empty answers; garbled text or repeated characters; hitting 1024 tokens without writing an answer.

**Results**

| Option | Questions | Baseline correct | Option correct | Fewer correct | More correct | McNemar p | Empty answers | Garbled / repeated | Hit 1024 with no answer (baseline) |
|---|---|---|---|---|---|---|---|---|---|
| Full vision tower per GPU (built into vLLM) | 500 | 343 (68.6%) | 344 (68.8%) | 30 | 31 | 1.0 | 0 | 0 | 44 (49) |
| Chunked vision tower (this repo, stacked on full vision tower per GPU) | 250 | 170 (68.0%) | 176 (70.4%) | 14 | 20 | 0.39 | 0 | 0 | 28 (21) |

- Every unanswered question on both sides hit 1024 tokens before reaching `ANSWER:`; none of them was a formatting failure.
- The chunked vision tower has 7 more of these (28 vs. 21). By group: 0, 0, +4, +2, +1, and no group is below the baseline. Paired by question, 18 were unanswered only by it and 11 only by the baseline, McNemar p=0.26. The difference points toward the chunked vision tower leaving more questions unanswered, but it is within the noise.

<details><summary>Per-group numbers</summary>

Full vision tower per GPU (the group 1 baseline uses the second pass):

| Group | Baseline correct | Option correct | Fewer correct | More correct | p | Unanswered: baseline / option |
|---|---|---|---|---|---|---|
| 1 | 32 | 34 | 2 | 4 | 0.69 | 9 / 8 |
| 2 | 34 | 33 | 5 | 4 | 1.0 | 5 / 4 |
| 3 | 36 | 34 | 7 | 5 | 0.77 | 5 / 6 |
| 4 | 31 | 33 | 1 | 3 | 0.63 | 3 / 2 |
| 5 | 38 | 37 | 2 | 1 | 1.0 | 2 / 6 |
| 6 | 35 | 37 | 2 | 4 | 0.69 | 4 / 3 |
| 7 | 37 | 37 | 3 | 3 | 1.0 | 4 / 2 |
| 8 | 30 | 28 | 3 | 1 | 0.63 | 6 / 7 |
| 9 | 35 | 35 | 2 | 2 | 1.0 | 5 / 3 |
| 10 | 35 | 36 | 3 | 4 | 1.0 | 6 / 3 |
| Total | 343 | 344 | 30 | 31 | 1.0 | 49 / 44 |

Chunked vision tower (stacked on full vision tower per GPU; the baseline was retaken in the same time window):

| Group | Baseline correct | Option correct | Fewer correct | More correct | p | Unanswered: baseline / option |
|---|---|---|---|---|---|---|
| 1 | 35 | 37 | 2 | 4 | 0.69 | 7 / 7 |
| 2 | 34 | 36 | 3 | 5 | 0.73 | 6 / 6 |
| 3 | 34 | 35 | 5 | 6 | 1.0 | 3 / 7 |
| 4 | 33 | 31 | 2 | 0 | 0.50 | 0 / 2 |
| 5 | 34 | 37 | 2 | 5 | 0.45 | 5 / 6 |
| Total | 170 | 176 | 14 | 20 | 0.39 | 21 / 28 |

</details>

**Conclusion**: neither option showed an accuracy drop on these questions. Note that the chunked vision tower row compares "full vision tower per GPU + chunked vision tower" stacked together against the default cross-GPU sharding; it is not the effect of the chunked vision tower alone. The differences in correct answers are within the baseline's own variation, and there were no bad outputs such as empty answers or garbled text. The number of questions is limited (500 and 250), so this only rules out a clear drop; it does not show that the two options are fully equivalent to vLLM's default. The chunked vision tower's 7 extra "hit 1024 with no answer" deserve a specific look when retesting with a larger token limit.

---

## Results on another machine (RTX 3080 20 GB, WSL2, 2B-class models)

Same 5 length groups × 3 videos, one request at a time, 16-token answers, TTFT measured at the client.

- **Machine**: RTX 3080 20 GB (compute capability 8.6), Ryzen 7 9800X3D (8 physical cores / 16 threads, one NUMA node), WSL2 (Ubuntu 24.04) on Windows; vLLM 0.31.0, transformers 5.17.0, torch 2.13.0+cu130.
- **Baseline**: stock vLLM, only the two caches off (`--no-enable-prefix-caching --mm-processor-cache-gb 0`), `OMP_NUM_THREADS` not set.
- **Plugins**: what probe recommended, other arguments identical. Qwen3-VL-2B: `all` preset (all five switches), `OMP_NUM_THREADS=2 M2_SEG_N=5 M2_SEG_THREADS=4 M2_VC_DEVICE=cuda:0`. Gemma-4-E2B: only segmented decode + frame skipping (the other three are Qwen3-VL only), same thread settings.
- Each configuration started twice; two starts of the same configuration differ by 0.96–1.09. 330 requests per model, no errors, prompt tokens identical everywhere; selfcheck 15/15 PASS on both models.
- Speedup here = baseline group median ÷ plugin group median (not the median of per-video ratios used in the tables above).

| Video type | Videos | Qwen3-VL-2B baseline | Qwen3-VL-2B plugins | Speedup | Saved per video | Gemma-4-E2B baseline | Gemma-4-E2B plugins | Speedup | Saved per video |
|---|---|---|---|---|---|---|---|---|---|
| Short clips (≤75 s) | 3 | 0.125 s | 0.130 s | 0.96× | −0.01 s | 0.706 s | 0.696 s | 1.01× | 0.01 s |
| Raw video ≤2 min | 3 | 3.678 s | 2.051 s | 1.79× | 1.63 s | 1.225 s | 1.059 s | 1.16× | 0.17 s |
| Raw video 2–10 min | 3 | 8.153 s | 2.815 s | 2.90× | 5.34 s | 3.586 s | 1.653 s | 2.17× | 1.93 s |
| Raw video 10–30 min | 3 | 8.640 s | 3.012 s | 2.87× | 5.63 s | 3.515 s | 1.565 s | 2.25× | 1.95 s |
| Raw video >30 min | 3 | 12.307 s | 5.911 s | 2.08× | 6.40 s | 9.264 s | 4.966 s | 1.87× | 4.30 s |

- Short clips are never segmented, so they gain nothing.
- Engine-side time (`vllm:request_prefill_time_seconds`, from scheduling to the first token) is 9–23% of baseline TTFT on videos over 2 minutes and 45–83% on the shorter groups; the plugins only shorten what comes before it.
- On Qwen3-VL-2B every plugin request had a shorter engine-side time than the baseline for the same video (median 0.59×): the vision tower ran in the API server on every request.
- Outputs: on Qwen3-VL-2B, 2 of 15 videos gave a different 16-token answer with the plugins (the same in both plugin starts), as expected from the chunked vision tower, which is not bit-identical. On Gemma-4-E2B the plugins change no input; its answers vary between server starts with or without the plugins.

## How to verify it yourself

```bash
python tools/pick_videos15.py videos15 --dry-run   # randomly pick 15 videos by length group from public datasets (fixed seed); check the list first
python tools/pick_videos15.py videos15             # download
python -m vllm_video_ttft.selfcheck --model <model_dir> videos15/*.mp4 --out selfcheck.json
```

For each video, selfcheck runs `off`, `exact` and `all` in turn (`all` only on the Qwen3-VL family and only when a GPU is visible), compares `pixel_values_videos` and `video_grid_thw`, and checks whether each plugin was served. The exit code is 0 when everything passes.

Before release, this repo was run on the 8×5090 machine (Qwen3.8-Flash-Next, 3 videos: a 3.5 s short clip, a 120 s raw video and a 305 s raw video). On the first `all` run, building the vision tower ran out of memory on the two long videos (the chosen GPU still had a GPU-holding process on it, leaving no headroom), so the `all` cells FAILed. After freeing a GPU, the second run PASSed everywhere; the table below is the second run. It also shows the "chunked vision tower runs out of memory without headroom" case: in strict mode it raises an error, without strict mode it hands the work back to the worker's own vision tower.

| Video | off | exact | all |
|---|---|---|---|
| 180257.mp4 (3.5 s, not segmented) | 0.20 s | PASS, 0.04 s | PASS, 1.11 s (the first request builds the vision tower) |
| kePBvNotYy4.mp4 (120 s) | 2.78 s | PASS, 1.50 s, frame skipping used | PASS, 0.87 s |
| -gnfUTPmnNU.mp4 (305 s) | 4.84 s | PASS, 1.52 s, frame skipping used | PASS, 1.49 s |

These seconds are offline, single-run decode + preprocessing times, not TTFT in a server.

Non-Qwen model (`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`, CPU, the same 3 videos): on non-Qwen models selfcheck only turns on segmented decode + frame skipping; 3/3 PASS, the two long videos used frame skipping, and the short clip was not segmented because of the threshold.

---

## High concurrency (inferred, not measured)

This section is inferred from the code and from single-request measurements. We only verified that concurrent results are bit-identical to serial and that nothing hangs: the four options that don't change output at 4 and 8 concurrency, the chunked vision tower only at 4 concurrency (24 requests). Concurrent throughput was not measured. Load-test it yourself before a high-concurrency deployment.

### Effect of the plugins on throughput

- **Segmented decode doesn't reduce the total amount of decoding**: it spreads one request's decode across many cores. Under high concurrency, stock vLLM also decodes several requests at once and fills the CPU, so both sides end up limited by total CPU. The extra work on the plugin side is re-decoding from the previous keyframe for each segment, plus inter-process communication; the savings are the non-reference frames that frame skipping doesn't decode. So full-load throughput is estimated to be about the same as stock, and the TTFT advantage shrinks as concurrency rises. A segmented request averages about 31 CPU core-seconds (12–57 by video length); combine that with your core count to estimate the CPU-side ceiling.
- **The vision tower queues on both sides**: without plugins, the vision tower was measured to run one request after another (4 requests arriving together started their vision tower at 2.68, 4.62, 5.96 and 7.31 s). With the chunked vision tower, that copy is fixed in the API process on the `M2_VC_DEVICE` GPU, and it also competes for compute with the language model on that GPU. Under high concurrency, whether the plugin side or the stock side clogs first was not measured.
- **A single API process contends for the GIL**: decode scheduling, the resize threads and the vision tower all live in the same Python process. The extra ~11 ms of decode on short clips is inferred to come from this (not confirmed with a profiler). It gets more visible as concurrency rises.

### More API processes (`--api-server-count`)

- **Measured: on its own it barely helps TTFT**. The test ran without plugins: 4 requests arriving together, all for the same 26-second, 1080p short video sampled at 52 frames, median of 3 rounds. With 4 API processes, resize-and-patchify barely queues anymore: 11 of 12 requests start it about 0.4 s after being sent. But TTFT hardly moves: the 4th request only went from 8.84 s to 8.67 s, and the 1st actually got slower, from 3.65 s to 4.62 s.
- **Why**: behind them there is still only one engine doing the scheduling, so the vision tower and prefill still queue; the bottleneck moves from the CPU to the GPU.
- **Combined with the plugins: not measured**. Inferred from the code: each API process has its own GIL, which should reduce the GIL contention described above, but it can't fix the queueing on the GPU side either.
- **If you combine them**:
  - Each API process has its own segmented-decode pool. Work out `M2_SEG_N × M2_SEG_THREADS × number of processes` together, using the rules in "Choose parameters by machine" above; don't let every process use the default 16 × 4.
  - Each pool's resident memory is a median of about 0.85–1.0 GiB, up to about 2.6 GiB; multiply by the number of processes.
  - `M2_VC_DEVICE` takes only one GPU, so the vision towers of all API processes land on that same GPU, about 3 GiB of GPU memory each.

### Recommendations

- Low concurrency, TTFT matters: turn everything on (`all` preset).
- High concurrency, throughput matters: turn on segmented decode, frame skipping and copy-free resize; leave the chunked vision tower off and use `--mm-encoder-tp-mode data`, so that each GPU holds a full vision tower.
- Pipelined resize: it doesn't change output and can stay on. But it starts one extra resize-and-patchify thread per request in the API process, which under high concurrency competes with other requests for CPU and the GIL, so the gain may shrink (inferred, not measured).
- More API processes: measured on its own, it barely helps TTFT; combined with the plugins, not measured. If you do turn it on, divide the pool's segment count among the processes.
- The real ceiling is prefill on the GPU, which the plugins can't touch.

---

## Versions and limitations

- Currently supports only vLLM 0.31.0, transformers 5.17.0 and opencv-python-headless 5.0.0.93. The plugins replace internal functions of these libraries: with a different vLLM or transformers version they log one warning and don't load; with a different opencv version, the frame-skipping path turns itself off.
- torch 2.13.0+cu130 is not listed as a dependency: install the cu130 build following the PyTorch website first, then install this package.
- Tested on only one 8×RTX 5090 machine. On a different machine, especially one with fewer cores, the gains will be different.
- Only single-request TTFT was measured, not high-concurrency throughput (4/8 concurrency only verified bit-identical results and no hangs). For high-concurrency settings see [High concurrency](#high-concurrency-inferred-not-measured) above; that section is inferred.
- Most of a short clip's TTFT is prefill (the language model computing on the GPU), so the plugins save little there.
- Frame skipping only helps videos with B-frames or other non-reference frames; videos with only I and P frames take the original path automatically.
- The pipeline is not supported with `--mm-processor-cache-gb > 0` (see "When not to enable" above).

## Layout

```text
src/vllm_video_ttft/     entry point (presets, defaults, version check), probe, selfcheck
src/m2_seg_decode/       segmented decode + frame skipping + background release of frame buffers
src/m2_copyfree_resize/  copy-free resize
src/m2_pipe_resize/      pipelined resize (incl. M2_PIPE_RESIZE_THREADS: set the resize thread count on its own)
src/m2_vision_chunk/     chunked vision tower
tools/pick_videos15.py   picks test videos from public datasets
VENDORED.sha256          file hashes of the four plugins (which validated version each one matches: see "Plugin versions" below)
```

## Plugin versions

- `m2_seg_decode` (including background release of frame buffers), `m2_copyfree_resize`: byte-identical to the versions used for the in-server speed measurements.
- `m2_vision_chunk`: the version used for the in-server speed measurements (hash in `VENDORED.sha256`) plus one later change, `_tower_weight_files` (commit d50d776): when the model directory has no `model.safetensors.index.json`, the tower weights are read from the single `model.safetensors` (small models such as Qwen3-VL-2B). With a shard index it reads the same files as the validated version. Checked with selfcheck on Qwen3-VL-2B: `off` / `exact` / `all` all PASS, vision-tower input bit-identical. It differs from the version used in the accuracy test only in status fields and in the few lines that keep pixels for timing instrumentation; the vision-tower computation is untouched. Also checked in the server: for 100 videos, the input to the vision tower matched the previous row (without the chunked vision tower) video by video. Second later change (10-25): the tower copy is built from the ModelConfig carried with each tower part (set when the request's decode starts) instead of a process-wide one recorded later in the request; before, a server whose first video request was a long (multi-block) video never built the tower and served no video with it for its whole lifetime. A failed build is now retried after `M2_VC_RETRY_S` (default 60 s) instead of being kept forever. Checked with selfcheck, long video first, on Qwen3-VL-2B and Qwen3.8: `exact` / `all` all PASS, vision-tower input bit-identical (the previous code fails all three in `all`).
- `m2_pipe_resize`: adds the `M2_PIPE_RESIZE_THREADS` switch. The results tables measured the version without this switch. With it unset, the compute path is identical to that version, with only a read-only thread-count record added (an independent reviewer read the diff line by line). With it set, acceptance passed: serial (each of the five levels, 36 requests bit-identical to unset; the full 908 videos twice, matching video by video; timing 15 videos × 5 levels, 75/75 bit-identical, with the thread count equal to the set value at every level); 4 and 8 concurrency, 3 rounds each, two groups (8 raw videos, and single-chunk inline mixed with the pipeline), each 36/36 bit-identical, with every request's resize thread count equal to the set value and no leakage into other requests.

## License

MIT, author ovo3ovo3ovo.
