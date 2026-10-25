"""从公开数据集里按长度组随机抽测试视频，每组 3 段，一共 5 组 15 段（分组和 5090 上测的 speed_set15 一样）。

  原片 4 组：VLM2Vec/videomme-rawvideo（Video-MME 原片，898 段，公开，不用申请）
  短片 1 组：OpenGVLab/MVBench 里的 Moments_in_Time_Raw.zip（64 MiB，200 段 mp4，每段约 3 s）

先远程 ffprobe 读时长和分辨率（不下载整段），抽够了才下载选中的文件。种子固定，同一个种子每次抽到的一样。
用法：python3 pick_videos15.py videos15            # 下载到 ./videos15，清单写进 videos15/videos15.json
      python3 pick_videos15.py videos15 --dry-run  # 只抽、只打印，不下载
需要：ffprobe（apt install ffmpeg）、huggingface_hub（装 vllm 时已带）。
"""
import json, os, random, subprocess, sys, zipfile
from huggingface_hub import HfApi, hf_hub_download

SEED = 20261025
RAW = "VLM2Vec/videomme-rawvideo"
MVB = "OpenGVLab/MVBench"
MVB_ZIPS = ["video/Moments_in_Time_Raw.zip"]   # 64 MiB，200 段 mp4；ssv2_video.zip 里全是 webm，不用
# 组名、时长下限（含）、上限（不含），单位秒；边界和 5090 上的分组一致
GROUPS = [("le2min", 0, 120), ("2to10min", 120, 600), ("10to30min", 600, 1800), ("gt30min", 1800, 1e9)]
MAX_MIB = 400            # 单个文件上限，免得下载太久
PER_GROUP = 3


def probe(src):
    """ffprobe 读时长、宽高、帧率；src 可以是 URL 也可以是本地路径。读不出返回 None。"""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json",
                              "-show_entries", "format=duration:stream=width,height,r_frame_rate", src],
                             capture_output=True, text=True, timeout=120)
        d = json.loads(out.stdout)
        s = d["streams"][0]
        num, den = s["r_frame_rate"].split("/")
        fps = float(num) / float(den)
        dur = float(d["format"]["duration"])
        return {"dur_s": round(dur, 2), "w": s["width"], "h": s["height"], "fps": round(fps, 3),
                "src_frames": round(dur * fps)}
    except Exception:
        return None


def group_of(dur):
    for name, lo, hi in GROUPS:
        if lo <= dur < hi:
            return name
    return None


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "videos15"
    dry = "--dry-run" in sys.argv
    rng = random.Random(SEED)
    picked = []

    # 原片 4 组：打乱文件列表，逐个远程 ffprobe，分辨率在 640–1280 宽之间（和 5090 上那批一样），每组凑够 3 段
    files = [f for f in HfApi().list_repo_tree(RAW, repo_type="dataset")
             if f.path.endswith(".mp4") and (getattr(f, "size", 0) or 0) < MAX_MIB * 2**20]
    files.sort(key=lambda f: f.path)
    rng.shuffle(files)
    need = {g: PER_GROUP for g, _, _ in GROUPS}
    base = f"https://huggingface.co/datasets/{RAW}/resolve/main/"
    for f in files:
        if not any(need.values()):
            break
        m = probe(base + f.path)
        if not m or not (640 <= m["w"] <= 1280):
            continue
        g = group_of(m["dur_s"])
        if g and need[g] > 0:
            need[g] -= 1
            picked.append({"group": g, "repo": RAW, "file": f.path, "size_mib": round(f.size / 2**20, 1), **m})
            print(f"{g:10s} {f.path:24s} {m['dur_s']:8.1f} s  {m['w']}x{m['h']}  {m['fps']} fps  {m['src_frames']} 帧")
    for g, n in need.items():
        if n:
            print(f"警告：{g} 组只抽到 {PER_GROUP - n} 段")

    if dry:
        print(json.dumps(picked, ensure_ascii=False, indent=1))
        return

    os.makedirs(out_dir, exist_ok=True)
    for p in picked:
        p["path"] = hf_hub_download(RAW, p["file"], repo_type="dataset", local_dir=out_dir)

    # 短片 1 组：下两个小压缩包，解出 mp4，选高 <= 480、时长 <= 75 s 的，随机取 3 段
    short = []
    zdir = os.path.join(out_dir, "mvbench")
    for z in MVB_ZIPS:
        zp = hf_hub_download(MVB, z, repo_type="dataset", local_dir=zdir)
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if name.endswith(".mp4"):
                    zf.extract(name, zdir)
                    short.append(os.path.join(zdir, name))
    short.sort()
    rng.shuffle(short)
    n = 0
    for path in short:
        if n == PER_GROUP:
            break
        m = probe(path)
        if m and m["h"] <= 480 and m["dur_s"] <= 75:
            n += 1
            picked.append({"group": "short", "repo": MVB, "file": os.path.relpath(path, zdir), "path": path, **m})
            print(f"{'short':10s} {os.path.basename(path):24s} {m['dur_s']:8.1f} s  {m['w']}x{m['h']}  {m['fps']} fps")
    if n < PER_GROUP:
        print(f"警告：short 组只抽到 {n} 段")

    with open(os.path.join(out_dir, "videos15.json"), "w") as f:
        json.dump({"seed": SEED, "videos": picked}, f, ensure_ascii=False, indent=1)
    print(f"清单写进 {out_dir}/videos15.json，一共 {len(picked)} 段")


if __name__ == "__main__":
    main()
