# vllm-video-ttft

vLLM 0.31.0 的插件：缩短**视频请求的首字等待**（time to first token）。在解码、缩放切块、视觉塔这几步上做改动；不占 GPU 的那三项，送进视觉塔的输入和 vLLM 原生逐位相同。

```bash
pip install -e .                                                   # 1. 装（vLLM 0.31.0 环境里）
python -m vllm_video_ttft.probe --model /path/to/model --tp 1       # 2. 看本机能开什么、推荐参数
VLLM_VIDEO_TTFT_PRESET=exact vllm serve /path/to/model --mm-processor-cache-gb 0   # 3. 起服务
```

---

## 给部署 agent 看：按这四步做

1. **跑 probe**：`python -m vllm_video_ttft.probe --model 模型目录 --tp 计划的TP大小`。只读，几秒跑完，输出 JSON，里面有本机核数、NUMA 分布、内存、`/dev/shm`、每张 GPU 的型号/显存/计算能力、各软件版本、模型属于哪个系列，以及每个插件能不能开、为什么，还有一套推荐的环境变量和启动参数。
2. **对照下面三张表定方案**：[按模型选插件](#按模型选插件)、[按机器选参数](#按机器选参数)、[插件一览](#插件一览)。probe 的推荐值就是按这几节的规则算出来的；表里写着"推断"的地方，要以第 3 步的实测为准。
3. **跑 selfcheck**：`python -m vllm_video_ttft.selfcheck --model 模型目录 视频1 视频2 视频3`。离线比较开插件和不开插件时，送进视觉塔的东西是否逐位相同、每个插件是否真的生效。全部 PASS 再上线。视频建议 1 段短片、2 段 5 分钟以上的长片。
4. **起服务**：把 probe 给的环境变量和参数加上去。日志里会有一行 `vllm_video_ttft 0.1.0: preset=..., on: ...`，写着实际开了哪些。

不确定的时候，用 `exact` 预设：它不改输出，最坏情况是没提速。

---

## 插件一览

| 插件 | 开关 | 省的是哪一步 | 适用模型 | 改不改输出 | 代价 |
|---|---|---|---|---|---|
| 切段并行解码 | `M2_SEG_DECODE=1` | 解码：按关键帧把一段视频切成多段，用进程池并行解 | 通用：凡是走 vLLM 默认 OpenCV 解码的都行 | 不改（逐位相同） | 进程池常驻内存中位 0.85–1.0 GiB、最多 2.6 GiB；每个切段的长片请求约 31 CPU 核秒 |
| 跳过非参考帧 | `M2_SEG_SKIP_NONREF=1` | 解码：没被抽中、也不被别的帧参考的帧不解 | 同上，叠在切段解码上 | 不改 | 无额外代价；只对含 B 帧或非参考帧的视频有用 |
| 去掉缩放切块里多余的拷贝 | `M2_COPYFREE_RESIZE=1` | 缩放切块 | 只限 Qwen3-VL 系 | 不改 | 无 |
| 边解码边缩放 | `M2_PIPE_RESIZE=1` | 缩放切块和解码重叠 | 只限 Qwen3-VL 系；要求 `--mm-processor-cache-gb 0` | 不改 | 多一个缩放线程 |
| 视觉塔分块叠进流水线 | `M2_VISION_CHUNK=1` | 视觉塔和解码、缩放重叠，在 API 进程里算 | 只限 Qwen3-VL 系；依赖边解码边缩放 | **改**：视觉塔输出 bf16 末位不同（相对差中位 2.9%） | 一张卡上多放一份视觉塔（Qwen3.8 实测最多 3.0 GiB） |

**预设**：`VLLM_VIDEO_TTFT_PRESET`
- `off`（默认）：装了也什么都不做。
- `exact`：前四项（切段解码、跳帧、去拷贝、边解码边缩放），输出和 vLLM 逐位相同。
- `all`：再加视觉塔分块。

单个开关的显式设置优先于预设，比如 `VLLM_VIDEO_TTFT_PRESET=exact M2_PIPE_RESIZE=0`。

`M2_STRICT=1` 是测试和自检用的：插件该生效却没生效时直接报错，不悄悄退回 vLLM 原路径。生产环境不要开。

---

## 按模型选插件

### 每个插件挂在哪（读代码确认，vLLM 0.31.0 / transformers 5.17.0）

- **切段解码、跳帧**：替换 `vllm.multimodal.video_decoders.opencv.decode_opencv` 和 `VideoMediaIO.load_file`。只有最后走到 `decode_video(backend="opencv")` → `decode_opencv` 的视频后端才会被接管。
- **去拷贝**：替换 `transformers` 的 `Qwen3VLVideoProcessor._prepare_input_videos / _preprocess`。
- **边解码边缩放**：替换 `Qwen3VLVideoProcessor.preprocess`、`VideoMediaIO.load_bytes`、`MediaConnector.load_from_url(_async)`、OpenCV 读帧，以及聊天解析里的 `parse_video`。
- **视觉塔分块**：在 API 进程里建一份 vLLM 自己的 `Qwen3_VisionTransformer`，替换 `Qwen3VLMultiModalProcessor._apply_hf_processor_main`，把 `pixel_values_videos` 换成 `video_embeds`。

### 各模型系列

依据一栏：**实测** = 在本仓库的验证里跑过；**读代码** = 按上面的挂点在 vLLM 0.31.0 源码里确认过路径；**推断** = 没验证，用 selfcheck 确认。

| 模型系列 | 架构名（config.json） | 例子 | 切段解码 + 跳帧 | 去拷贝 / 边解码边缩放 | 视觉塔分块 | 依据 |
|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | `Qwen4ExpForConditionalGeneration` | Qwen/Qwen3.8-Flash-Next | 能开 | 能开 | 能开 | **实测**：8×5090、TP8，908 段逐位相同，服务内测速、精度 750 题 |
| Qwen3-VL（含 MoE） | `Qwen3VLForConditionalGeneration`、`Qwen3VLMoeForConditionalGeneration` | Qwen/Qwen3-VL-2B-Instruct | 能开 | 能开 | 能开 | 读代码：同一个视频处理器（`Qwen3VLVideoProcessor`）和视觉塔类；未在服务里测，用 selfcheck 确认 |
| Qwen3.5、Cosmos3-Edge、InternS1-Pro / InternS2 等用 Qwen3-VL 视觉塔的 | 各自不同 | — | 能开 | 看视频处理器是不是 `Qwen3VLVideoProcessor` | 推断：可能能开 | 推断：源码里复用了 Qwen3-VL 的类，但处理器和模板可能不同；用 selfcheck 确认 |
| Qwen2.5-VL / Qwen2-VL | `Qwen2_5_VLForConditionalGeneration`、`Qwen2VLForConditionalGeneration` | Qwen/Qwen2.5-VL-7B-Instruct | 能开 | 不生效 | 不生效 | 读代码：视频后端 `qwen2_vl` 继承默认 OpenCV 路径；处理器是 `Qwen2VLVideoProcessor`，不是挂点 |
| LLaVA-NeXT-Video | `LlavaNextVideoForConditionalGeneration` | llava-hf/LLaVA-NeXT-Video-7B-hf | 能开 | 不生效 | 不生效 | 读代码：没有注册专门的视频后端，用默认 OpenCV；未验证，用 selfcheck 确认 |
| LLaVA-OneVision | `LlavaOnevisionForConditionalGeneration` | llava-hf/llava-onevision-qwen2-7b-ov-hf | 能开 | 不生效 | 不生效 | 同上 |
| InternVL | `InternVLChatModel` 等 | OpenGVLab/InternVL3-8B | 能开 | 不生效 | 不生效 | 同上 |
| Gemma 系（带视频的） | `Gemma4ForConditionalGeneration` 等 | google/gemma-4-E2B-it | 能开 | 不生效 | 不生效 | 同上 |
| GLM-4.1V / GLM-4.5V / GLM-4.6V | `Glm4vForConditionalGeneration` 等 | — | 能开 | 不生效 | 不生效 | 读代码：后端 `opencv_dynamic` / `glm46v` 继承默认 OpenCV 路径 |
| GLM-5-Next 视频 | — | — | **不生效** | 不生效 | 不生效 | 读代码：它的后端重写了 `read_frames`，切段解码会自动让路（日志 `loader overrides read_frames`） |
| Molmo2、OpenPangu-VL | — | — | 推断：可能能开 | 不生效 | 不生效 | 推断：它们直接调用 `decode_video`，没核实最后是不是走 `decode_opencv` |
| 只看图片、不收视频的模型 | — | — | 不生效 | 不生效 | 不生效 | 插件只碰视频路径，图片请求不受影响 |

"不生效"的意思是：挂点在这个模型上根本不会被调用，或者插件判断不该接管、自己让路。服务照常跑，不需要特意关掉；只有加了 `M2_STRICT=1` 才会报错。

### 什么时候别开

| 情况 | 别开哪个 | 为什么 |
|---|---|---|
| 要求输出和 vLLM 逐位相同 | 视觉塔分块 | 它改视觉塔输出的 bf16 末位。我们在 Qwen3.8 上考过 250 题，没有精度下降，但不是逐位相同 |
| 开了视频剪枝（`--video-pruning-rate > 0`，EVS / VidCom2） | 视觉塔分块 | 剪枝要看完整段视频才决定保留哪些 token，分块就不等价了；插件会拒绝分块，交回 worker 自己的视觉塔 |
| 一个请求里有多段视频 | 视觉塔分块 | 不支持，自动交回 worker |
| 开着处理器缓存（`--mm-processor-cache-gb > 0`） | 边解码边缩放、视觉塔分块 | 缓存命中时不会重新解码，流水线插件会让路。我们的测量都是 `--mm-processor-cache-gb 0` |
| 换了解码后端（`--media-io-kwargs '{"video":{"backend":"pynvvideocodec"或"torchcodec"}}'`） | 切段解码、跳帧，以及依赖它的流水线 | 切段解码只接管 OpenCV 路径；换了后端就接管不到 |
| `frame_recovery=True` | 切段解码 | vLLM 的坏帧恢复扫描没有重新实现，自动走原路径 |
| 高并发、CPU 吃紧 | 切段解码要谨慎 | 每个切段的长片请求约 31 CPU 核秒，常驻内存最多 2.6 GiB；我们只测了单请求，没测过高并发下的吞吐 |
| 显存没有余量的卡 | 视觉塔分块 | 要多放一份视觉塔；放不下时插件会报 OOM 并交回 worker（严格模式下报错） |
| 视频都很短（原始帧数 < `M2_SEG_MIN_SRC_FRAMES`，默认 2642） | 切段解码开着也没用 | 短片不切段；短片首字等待的大头在预填充，插件省得少 |

### 插件不生效时会怎样（读代码确认）

- 每种不生效的原因，在日志里只打一次 warning（例如 `m2_seg_decode: falling back ...`、`m2_pipe_resize: not served (...)`），然后走 vLLM 原来的路径，结果和没装插件一样。
- vLLM 或 transformers 版本不对：插件打一行 warning，不注册（严格模式下启动报错）。
- 模型不是 Qwen3-VL 系：去拷贝和视觉塔分块的挂点根本不会被调用；边解码边缩放在第一次请求时发现拿不到 Qwen3-VL 的处理参数，记住这次失败，以后直接让路。

---

## 按机器选参数

这一节写的是通用规律，部署方按自己机器的核数和 NUMA 布局自己算。数字都来自一台机器：2 路 Xeon Platinum 8558，96 个物理核、192 个逻辑核，4 个 NUMA 节点；8×RTX 5090，TP8。凡是写"推断"的，都是从这台机器外推的，要用 selfcheck 的耗时确认。

### CPU：线程花在哪

一个视频请求在 API 进程这一侧同时用到三份线程：

| 谁 | 线程数 | 什么时候 |
|---|---|---|
| 切段解码的进程池 | `M2_SEG_N × M2_SEG_THREADS`（默认 16 × 4 = 64） | 只在切段的长视频上 |
| 缩放切块 | `OMP_NUM_THREADS`；开了边解码边缩放以后，和解码同时跑 | 每个视频 |
| vLLM 调度进程 + API 主线程 | 约 1–2 个核 | 一直在 |

三项加起来，不要超过 API 所在 socket 上能用的核数。超线程的逻辑核不能算满额：我们的 64 个解码线程跑在 48 个物理核、96 个逻辑核上，是测下来最快的配置。

### CPU：怎么分配

1. **先把核按 NUMA 节点和 socket 划清楚**：`lscpu`，或者看 probe 输出里的 `numa_nodes`。
2. **API 进程、切段进程池、帧缓冲放在同一个 socket 上**：
   - API 进程：`numactl --cpunodebind=N --membind=N vllm serve ...`。
   - 进程池：`M2_SEG_WNODES=N[,同 socket 的另一个节点]`，或者直接写核号 `M2_SEG_WCPUS=...`。
   - 帧缓冲跟着服务的 `--membind` 走。
   - 实测：离线把主进程钉到另一个 socket，35 分钟原片的帧拷回从 1.06 s 涨到 1.72 s，vLLM 原解码从 7.10 s 涨到 8.27 s。服务里用不拷贝的方式时，钉错的代价小得多，只有缩放切块慢 6–10%，但仍然建议钉在同一个 socket。
   - 不设 `M2_SEG_WNODES` / `M2_SEG_WCPUS` 时，本插件默认用本进程允许用的全部核。
3. **段数和每段线程数怎么取舍**（实测，离线，同一台机器）：

   | 视频（原始帧数） | vLLM 原解码 | 每段 1 线程：8 段 / 16 段 / 32 段 | 每段 4 线程：8 段 / 16 段 / 32 段 |
   |---|---|---|---|
   | 2642 帧 | 0.637 s | 0.758 / 0.490 / 0.497 | 0.350 / **0.295** / 0.362 |
   | 9156 帧 | 1.316 s | 1.169 / 0.704 / 0.684 | 0.643 / **0.506** / 0.612 |
   | 49715 帧 | 7.411 s | 7.656 / 4.432 / 3.443 | 3.881 / **2.881** / 2.897 |

   - 每段 4 线程、16 段最好；32 段不再变快。
   - 每段 1 线程时，切 8 段有一段已经比原解码快，另两段要 16 段才快。核少的机器上，每段 1 线程、段数多一点，可能比每段 4 线程划算（推断）。
   - FFmpeg 自己的多线程已经开着：vLLM 原解码默认就是多线程，强制 1 线程会慢 6–8 倍。所以核少的机器上，切段解码的收益会明显小于下面结果表里的数（推断）。
4. **缩放切块线程数（`OMP_NUM_THREADS`）**：从 1 线程改成 12 线程，缩放切块降到原来的 0.36–0.46，省的几乎都在"转张量"那一步。不设时 vLLM 会把 API 进程降到 1 线程，所以一定要设。超过 12 线程有没有收益，我们没测。

### CPU：经验公式（probe 用的就是这个）

```text
C_node = API 所在 socket 上能用的节点（4 节点的机器取前两个，2 节点取 1 个，单路取全部）
L = 这些节点的逻辑核数 − 当前 1 分钟负载
P = 这些节点的物理核数 − 当前 1 分钟负载

OMP_NUM_THREADS = clamp(P / 4, 1, 12)
D               = min(64, 2·L / 3)                # 切段解码的线程总数
M2_SEG_THREADS  = 4 if D ≥ 16 else 1
M2_SEG_N        = clamp(D / M2_SEG_THREADS, 2, 16)
M2_SEG_WNODES   = C_node
```

- 实测依据：机器空闲时，这个公式在我们这台机器上（P = 48，L = 96）算出 `OMP_NUM_THREADS=12`、16 段 × 4 线程，正好是实测最好的那组。
- 推断：公式里的比例（1/4、2/3）是从这一个点外推的，没在别的核数上测过。核少的机器请用 selfcheck 打印的耗时比较两三组设置。

### CPU：切段门槛

- `M2_SEG_MIN_SRC_FRAMES`（默认 2642）：解码要抓的原始帧数达到这个值才切段。原始帧数 = 时长 × 原视频帧率。
- 这个值是在 48 个物理核上定的：我们 100 段测速视频里，2642 帧以上的每段都是切段更快，以下的切段反而慢。
- 核少的机器上，切段的固定开销占比更大，交叉点会往上移（推断）。判断方法：拿两三段原始帧数在门槛附近的视频跑 selfcheck，比较 `off` 和 `exact` 两行的秒数；切了反而慢，就把门槛往上调。selfcheck 每种设置只跑一次，看的是方向，不是精确值。

### GPU：单卡和多卡分开看

| 设置 | 单卡（TP=1） | 多卡（TP>1） |
|---|---|---|
| `--mm-encoder-tp-mode data` | 开不开都一样 | 建议开：每张卡放一整份视觉塔，不用跨卡切分。代价是每张卡多占一份视觉塔的显存 |
| `VLLM_MQ_MAX_CHUNK_BYTES_MB` | **别设**：只有一个 worker 时，vLLM 不走这条广播通道（`vllm/config/parallel.py:1033`） | 建议设成大于单条请求里最大的那份消息：不开视觉塔分块时发的是像素（我们的原片 106–145 MiB），开了以后发的是视觉特征（最大约 70 MiB）。默认 16 MiB，超过的消息改走 zmq，慢一截。我们用 256：原片的交接时间降到 0.59–0.65 倍，首字等待快 2–4%。代价是共享内存约 10 × 这个值，256 时实测多占 2.34 GiB |
| 视觉塔分块的 `M2_VC_DEVICE` | 就用这张卡（只有一张可见卡时，本插件自动设成 `cuda:0`） | 必须显式设一张有余量的卡 |
| 给视觉塔分块留显存 | `--gpu-memory-utilization` 要给它留出空间，否则会和 KV cache 抢。Qwen3.8 的视觉塔实测最多 3.0 GiB；小模型的视觉塔小得多，按视觉塔参数量 × 2 字节估，再加激活 | 同左，只算放视觉塔的那一张 |

### GPU：为什么默认不用 GPU 解码

vLLM 自带 NVDEC / torchcodec 解码（`--media-io-kwargs '{"video":{"backend":"pynvvideocodec"}}'` 或 `{"backend":"torchcodec","device":"cuda"}`），我们测过，默认不推荐：

- **帧落在显存里**：卡被语言模型占满时，长片会 OOM。我们在 `--gpu-memory-utilization 0.85` 下用 torchcodec cuda，一段 305 s 的视频要 1.6 GiB，直接失败。
- **会改像素**：NVDEC 解出的像素和 OpenCV 不同，我们的视频里最多差 17 个灰度。
- **单路 NVDEC 解长片不比 CPU 快**：768 帧 720p，NVDEC 6.81 s，OpenCV 6.93 s。pynvvideocodec 的 `hw_decoders` 只是给并发请求分路，不会把一段视频拆开并行解。
- **可以试的情况**：GPU 有富余显存、不要求逐位相同、视频以 1080p 的短片为主时，NVDEC 能省掉一部分 CPU。注意它和切段解码互斥。

### GPU：硬件限制

- NVIDIA 驱动 ≥ 580：torch 2.13.0+cu130 需要 CUDA 13。
- 视觉塔 FP8 注意力（`--mm-encoder-attn-dtype fp8`）在 SM120（RTX 50 系）上起不来：cuDNN 9.20 的"FP8 输入、bf16 输出"注意力内核只给 SM100 提供，vLLM 的检查只看计算能力 ≥ 10.0，所以会放行，然后在建图时失败。SM100 上可能可以，我们没测过。视觉塔分块也不支持 FP8 注意力。
- probe 会输出每张卡的型号、显存和计算能力。

---

## 和 vLLM 自带的选项一起配

插件只管 API 进程这一侧。下面几个是 vLLM 自带的选项，不开的话插件的收益会打折扣：

```bash
export OMP_NUM_THREADS=12                 # 缩放切块的线程数，按上面的公式定
export VLLM_MQ_MAX_CHUNK_BYTES_MB=256     # 只在 TP>1 时设
numactl --cpunodebind=0 --membind=0 \
vllm serve 模型目录 --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-gb 0
```

---

## 结果（8×RTX 5090，Qwen3.8-Flash-Next，在线 FP8，TP8）

**口径**：只量首字等待（从发出请求到收到第一个 token），单请求，回答截在 16 个 token。插件只动第一个 token 之前的步骤，所以首字等待的数字和回答长短无关；整个请求快几倍，要看回答有多长，生成那一段没变快。

### 1. 从最朴素的 vLLM 一路叠上去（100 段视频）

最朴素的基线 = vLLM 0.31.0 + 在线 FP8，默认设置。每一列都是叠上前面所有行以后的服务。倍数是先算每段视频各自的比值，再取组内中位数。

| 视频类型 | 段数 | 基线首字等待 | + 12 线程 | + 去拷贝 | + 切段解码 | + 边解码边缩放 | + 跳帧 | + 视觉塔每卡一整份 | + 视觉塔分块 | + 消息门槛 64 MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| 短片（MV-Bench） | 70 | 0.86 s | 1.25× | 1.21× | 1.35× | 1.35× | 1.35× | 1.58× | 1.72× | 1.76× |
| 原片 ≤2 分钟 | 9 | 3.55 s | 1.28× | 1.42× | 1.48× | 1.54× | 1.59× | 1.83× | 2.22× | 2.27× |
| 原片 2–10 分钟 | 7 | 5.79 s | 1.25× | 1.43× | 2.07× | 2.18× | 2.44× | 2.73× | 3.15× | 3.30× |
| 原片 10–30 分钟 | 4 | 9.06 s | 1.25× | 1.43× | 2.27× | 2.53× | 2.94× | 3.29× | 3.54× | 3.82× |
| 原片 >30 分钟 | 10 | 16.95 s | 1.11× | 1.15× | 2.79× | 2.96× | 4.14× | 4.54× | 4.83× | 4.93× |

这几列是分几次测的；每一项自己的提速，以和它紧挨着测的对照为准，这张表用来看累计。"12 线程""视觉塔每卡一整份""消息门槛"是 vLLM 自带的选项，不是插件，所以下一张表要单独比插件的贡献。之后又加了帧缓冲后台释放（已包含在本仓库的切段解码里），在 15 段视频上让 2–10 / 10–30 / >30 分钟组的首字等待再降到 0.986 / 0.936 / 0.950。

### 2. 插件比"vLLM 自己调到最好"快多少（15 段视频）

- 每个长度组 3 段，按时长取最短、中位、最长。
- 三种配置交替各测两遍，同一配置两遍之间只差 0.98–1.01。
- **原生最佳（不改输入）**：`OMP_NUM_THREADS=12` + `--mm-encoder-tp-mode data` + `VLLM_MQ_MAX_CHUNK_BYTES_MB=256`。
- **原生最佳（允许改输入）**：上一种再加 pynvvideocodec（NVDEC 解码），会改像素。
- **插件**：和第一种同样的原生选项，再加全部插件（`all` 预设）。送进视觉塔的输入不变。

| 视频类型 | 原生最佳（不改输入） | 原生最佳（允许改输入） | 插件 | 比原生各组最好快 | 比原生（不改输入）每段省 |
|---|---|---|---|---|---|
| 短片 | 0.383 s | 0.369 s | 0.335 s | 1.10× | 0.05 s |
| 原片 ≤2 分钟 | 2.380 s | 2.103 s | 1.453 s | 1.37× | 0.79 s |
| 原片 2–10 分钟 | 3.831 s | 4.092 s | 1.740 s | 2.20× | 2.09 s |
| 原片 10–30 分钟 | 6.378 s | 5.703 s | 2.091 s | 2.69× | 4.29 s |
| 原片 >30 分钟 | 10.064 s | 9.806 s | 2.549 s | 3.80× | 7.52 s |

"比原生各组最好快"：每个长度组从所有原生配置里挑最快的那个来比，对原生最宽松。秒数是组内中位数，倍数是每段比值的中位数，所以两者不能直接互相换算。每组只有 3 段，只能看方向和大致幅度。

### 3. 正确性

- 不改输出的那几项：全量 908 段视频（MV-Bench-mini 878 段 + Video-MME 原片 30 段）逐位相同；500 次长跑没有泄漏，句柄、线程数不涨；4 并发、8 并发和串行逐位相同；边界视频（奇数宽高、旋转、可变帧率、坏码流等）1155 条测过。
- 坏码流上，被错误隐藏补出来的帧，像素可能和 vLLM 不同：vLLM 自己在不同线程数下也不固定。结构（帧数、下标、形状）保持一致，坏 GOP 以外的帧逐位相同。
- 视觉塔分块：精度 250 题，基线 170 对、插件 176 对，p=0.39，没有下降。

---

## 怎么自己验证

```bash
python tools/pick_videos15.py videos15 --dry-run   # 从公开数据集按长度组随机抽 15 段（固定种子），先看清单
python tools/pick_videos15.py videos15             # 下载
python -m vllm_video_ttft.selfcheck --model 模型目录 videos15/*.mp4 --out selfcheck.json
```

selfcheck 对每段视频依次跑 `off`、`exact`、`all`（`all` 只在 Qwen3-VL 系、并且有可见 GPU 时跑），比较 `pixel_values_videos` 和 `video_grid_thw`，并检查每个插件是否 served。全部通过时退出码为 0。

本仓库发布前在 8×5090 那台机器上跑过一遍（Qwen3.8-Flash-Next，3 段视频：一段 3.5 s 的短片、一段 120 s 的原片、一段 305 s 的原片；`all` 用一张空卡）：

| 视频 | off | exact | all |
|---|---|---|---|
| 180257.mp4（3.5 s，不切段） | 0.20 s | PASS，0.04 s | PASS，1.11 s（第一次请求要建视觉塔） |
| kePBvNotYy4.mp4（120 s） | 2.78 s | PASS，1.50 s，走了跳帧 | PASS，0.87 s |
| -gnfUTPmnNU.mp4（305 s） | 4.84 s | PASS，1.52 s，走了跳帧 | PASS，1.49 s |

这些秒数是离线单次的解码 + 预处理耗时，不是服务里的首字等待。

---

## 版本和限制

- **锁死** vLLM 0.31.0、transformers 5.17.0、opencv-python-headless 5.0.0.93。插件替换的是 vLLM 和 transformers 的内部函数，版本不对时插件会拒绝加载。opencv 版本不对时，跳帧那条路会自动关掉。
- torch 2.13.0+cu130 不放在依赖里：请按 PyTorch 官网装 cu130 版，再装本包。
- 只在一台 8×RTX 5090 上测过。换一台机器，尤其是核少的机器，收益会不一样。
- 只测了单请求的首字等待，没测高并发吞吐。
- 短片首字等待的大头在预填充（语言模型自己在 GPU 上算），插件省得少。
- 跳帧只对含 B 帧或非参考帧的视频有用；只有 I、P 帧的视频，自动走原来的路径。
- 不支持 `--mm-processor-cache-gb > 0` 时的流水线（见上文"什么时候别开"）。

## 目录

```text
src/vllm_video_ttft/     入口（预设、默认值、版本检查）、probe、selfcheck
src/m2_seg_decode/       切段并行解码 + 跳过非参考帧 + 帧缓冲后台释放
src/m2_copyfree_resize/  去掉缩放切块里多余的拷贝
src/m2_pipe_resize/      边解码边缩放（含 M2_PIPE_RESIZE_THREADS：单独设缩放线程数，不设时行为不变）
src/m2_vision_chunk/     视觉塔分块叠进流水线
tools/pick_videos15.py   从公开数据集抽测试视频
VENDORED.sha256          四个插件的文件哈希（和验证时的版本逐字节一致）
```

## 许可证

MIT，作者 ovo3ovo3ovo。
