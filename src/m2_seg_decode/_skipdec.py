"""Skip-nonref decode loop of m2_seg_decode (b5, M2_SEG_SKIP_NONREF=1), run inside a pool worker (_worker.py).

H.264 pictures whose slices all have nal_ref_idc == 0 are never used as a reference, so leaving one out changes no other
picture.  A slice job here decodes with the libavformat / libavcodec / libswscale that OpenCV itself ships (the same
build OpenCV's VideoCapture runs, version-checked), sends every packet except those "non-reference and not wanted" ones,
and converts each wanted picture exactly as OpenCV's retrieve() + vLLM's cvtColor do (cap_ffmpeg_impl.hpp:1918-2036:
sws_flags=SWS_BICUBIC, threads, sws_scale_frame into a BGR24 frame of the stream's size, av_frame_get_buffer(32); then
the metadata rotation, then BGR->RGB), straight into the request's shared frame buffer (no copy afterwards).
Offline prototype: 908/908 videos frame-identical to vLLM (runs/u5plan/plugin_segdecode.txt, 10-25 01:40).

Frame numbering is vLLM's: the parent's frame table (_index.frame_table_ex) gives frame i's pts; a decoded picture is
frame i by its pts.  The job checks that, from the anchor on, pictures come out exactly in table order with nothing
missing (dropped frames excepted); any deviation, a start that does not lead to the anchor, or the end of the file before
the slice's last frame makes decode() return False with out["skip_fail"] set, and the worker decodes the job again with
OpenCV (the parent decides whether that was the stream or a bug: M2_STRICT=1 raises on the latter).
Start points mirror _worker._locate: the anchor's own GOP keyframe if it is an IDR picture (nothing before an IDR can be
referenced), then the keyframes before it, then frame 0; a start is accepted only if the lead-in printed nothing but
_worker._NOOP_MSGS.  Seeks are exact: av_seek_frame to the start keyframe's dts (AVSEEK_FLAG_BACKWARD), checked by the
first packet read."""
import ctypes, glob, os, time
import numpy as np
import cv2
import _index as X

_VERSIONS = {"avcodec": (62, 28, 101), "avutil": (60, 26, 101), "swscale": (9, 5, 101)}   # + _index._CHECKED_VERSION
_NOPTS, _DISCARD = -0x8000000000000000, 0x0004
_AV_PIX_FMT_BGR24, _SWS_BICUBIC, _AVSEEK_FLAG_BACKWARD = 3, 4, 1
_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

class _Frame(ctypes.Structure):        # AVFrame, leading fields (libavutil/frame.h, FFmpeg 8.1.1)
    _fields_ = [("data", ctypes.POINTER(ctypes.c_uint8) * 8), ("linesize", ctypes.c_int * 8),
                ("extended_data", ctypes.c_void_p), ("width", ctypes.c_int), ("height", ctypes.c_int),
                ("nb_samples", ctypes.c_int), ("format", ctypes.c_int), ("pict_type", ctypes.c_int),
                ("sample_aspect_ratio", X._Rat), ("pts", ctypes.c_int64), ("pkt_dts", ctypes.c_int64)]

_L = None

def _opencv_log_level(u):
    """Review D5-1 (10-25): put av_log at the level OpenCV puts it, before this path decodes anything.  libavutil
    starts at AV_LOG_INFO = 32 (libavutil/log.c:59); OpenCV sets it once per process when it first opens a capture
    (cap_ffmpeg_impl.hpp 5.0.0: CvCapture_FFMPEG::open :1069 -> InternalFFMpegRegister::init, a static instance :968
    -> initLogger_ :970-992): OPENCV_FFMPEG_DEBUG or OPENCV_FFMPEG_LOGLEVEL set -> that level (default AV_LOG_VERBOSE
    40), else AV_LOG_ERROR = 16 (:990).  Until then a worker that had not opened a capture yet printed WARNING-level
    lines here that the OpenCV path never prints (libswscale/utils.c:1920 "deprecated pixel format used" on yuvj420p),
    so what a slice logged depended on the worker's history.  Every message the damage rule relies on is AV_LOG_ERROR
    (h2645_parse.h:140, h264_parser.c:583, h264_refs.c:397/400, h264dec.c:613/741/1064, h264_slice.c:2650/2709,
    mov.c:11184/11436), so this hides none of them; vLLM itself and the OpenCV path run at 16 anyway.
    OpenCV's debug log callback (printing to stdout) is not installed here."""
    dbg = os.environ.get("OPENCV_FFMPEG_DEBUG", "").strip().lower() in ("1", "true", "on", "yes")
    lvl = os.environ.get("OPENCV_FFMPEG_LOGLEVEL", "")
    if dbg or lvl:
        try:
            u.av_log_set_level(int(lvl) if lvl else 40)
        except ValueError:
            u.av_log_set_level(0)                   # OpenCV: atoi() of a non-number is 0
    else:
        u.av_log_set_level(16)

def libs():
    """(avutil, avcodec, avformat, swscale) with prototypes set; None if any library is not the checked build."""
    global _L
    if _L is None:
        _L = False
        base = X._load()
        if base is not None:
            u, c, f = base
            d = os.path.join(os.path.dirname(cv2.__file__), "..", "opencv_python_headless.libs")
            hits = sorted(glob.glob(os.path.join(d, "libswscale-*.so*")))
            s = ctypes.CDLL(hits[0]) if hits else None
            ok = s is not None
            for name, lib in (("avcodec", c), ("avutil", u), ("swscale", s)):
                if ok:
                    v = getattr(lib, f"{name}_version")()
                    ok = (v >> 16, (v >> 8) & 255, v & 255) == _VERSIONS[name]
            if ok:
                P, VP, I, I64, CP = ctypes.POINTER, ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_char_p
                f.av_seek_frame.argtypes = [P(X._Fmt), I, I64, I]
                c.avcodec_find_decoder.argtypes = [I]; c.avcodec_find_decoder.restype = VP
                c.avcodec_alloc_context3.argtypes = [VP]; c.avcodec_alloc_context3.restype = VP
                c.avcodec_parameters_to_context.argtypes = [VP, VP]
                c.avcodec_open2.argtypes = [VP, VP, P(VP)]
                c.avcodec_send_packet.argtypes = [VP, VP]
                c.avcodec_receive_frame.argtypes = [VP, P(_Frame)]
                c.avcodec_flush_buffers.argtypes = [VP]; c.avcodec_flush_buffers.restype = None
                c.avcodec_free_context.argtypes = [P(VP)]; c.avcodec_free_context.restype = None
                u.av_dict_set.argtypes = [P(VP), CP, CP, I]
                u.av_dict_free.argtypes = [P(VP)]; u.av_dict_free.restype = None
                u.av_frame_alloc.restype = P(_Frame)
                u.av_frame_free.argtypes = [P(P(_Frame))]; u.av_frame_free.restype = None
                u.av_frame_unref.argtypes = [P(_Frame)]; u.av_frame_unref.restype = None
                u.av_frame_get_buffer.argtypes = [P(_Frame), I]
                u.av_opt_set_int.argtypes = [VP, CP, I64, I]
                s.sws_alloc_context.restype = VP
                s.sws_scale_frame.argtypes = [VP, P(_Frame), P(_Frame)]
                s.sws_freeContext.argtypes = [VP]; s.sws_freeContext.restype = None
                u.av_log_set_level.argtypes = [I]; u.av_log_set_level.restype = None
                u.av_log_get_level.restype = I
                _opencv_log_level(u)
                _L = (u, c, f, s)
    return _L or None

_tabs = {}

def table(path, token):
    """The parent's frame table (written once per video into a memfd, see __init__._Tab), cached per token."""
    t = _tabs.get(token)
    if t is None:
        with open(path, "rb") as fh:
            pts, dts, times, drop, kidx, kidr = (np.load(fh, allow_pickle=False) for _ in range(6))
        t = {"pts": pts, "dts": dts, "times": times, "drop": drop, "kidx": kidx, "kidr": kidr,
             "p2i": dict(zip(pts.tolist(), range(len(pts))))}
        if len(_tabs) >= 8:
            _tabs.clear()
        _tabs[token] = t
    return t

class _Dec:
    """One demuxer + decoder + converter, set up the way OpenCV's capture is (cap_ffmpeg_impl.hpp:1415-1462: thread
    count, avcodec_parameters_to_context, avcodec_open2; other streams discarded, which does not change the video packets
    read: scripts/u5plan/b5/probe_tabcost.py)."""
    def __init__(self, path, threads):
        self.L = u, c, f, s = libs()
        self.ic, self.ctx, self.sws, self.pkt, self.fr, self.bgr = ctypes.POINTER(X._Fmt)(), ctypes.c_void_p(), None, None, None, None
        if f.avformat_open_input(ctypes.byref(self.ic), os.fsencode(path), None, None) < 0:
            self.ic = None
            raise RuntimeError("open failed")
        try:
            if f.avformat_find_stream_info(self.ic, None) < 0:
                raise RuntimeError("find_stream_info failed")
            self.vs = vs = f.av_find_best_stream(self.ic, 0, -1, -1, None, 0)
            if vs < 0:
                raise RuntimeError("no video stream")
            X.discard_other_streams(self.ic, vs)
            st = self.ic.contents.streams[vs].contents
            par = ctypes.cast(st.codecpar, ctypes.POINTER(X._Par)).contents
            self.W, self.H = par.width, par.height
            codec = c.avcodec_find_decoder(par.codec_id)
            if not codec:
                raise RuntimeError(f"no decoder for codec_id {par.codec_id}")
            self.ctx = ctypes.c_void_p(c.avcodec_alloc_context3(codec))
            c.avcodec_parameters_to_context(self.ctx, st.codecpar)
            opts = ctypes.c_void_p()
            u.av_dict_set(ctypes.byref(opts), b"threads", str(threads).encode(), 0)
            r = c.avcodec_open2(self.ctx, codec, ctypes.byref(opts))
            u.av_dict_free(ctypes.byref(opts))
            if r < 0:
                raise RuntimeError(f"avcodec_open2 failed ({r})")
            self.sws = s.sws_alloc_context()
            u.av_opt_set_int(self.sws, b"sws_flags", _SWS_BICUBIC, 0)
            u.av_opt_set_int(self.sws, b"threads", threads, 0)
            self.pkt, self.fr, self.bgr = c.av_packet_alloc(), u.av_frame_alloc(), u.av_frame_alloc()
            b = self.bgr.contents
            b.format, b.width, b.height = _AV_PIX_FMT_BGR24, self.W, self.H
            if u.av_frame_get_buffer(self.bgr, 32) < 0:
                raise RuntimeError("av_frame_get_buffer failed")
            self.view = np.ctypeslib.as_array(b.data[0], shape=(self.H * b.linesize[0],)).reshape(
                self.H, b.linesize[0])[:, :self.W * 3].reshape(self.H, self.W, 3)
        except Exception:
            self.close(); raise

    def seek(self, dts):
        u, c, f, s = self.L
        r = f.av_seek_frame(self.ic, self.vs, int(dts), _AVSEEK_FLAG_BACKWARD)
        c.avcodec_flush_buffers(self.ctx)
        return r >= 0

    def close(self):
        u, c, f, s = self.L
        if self.sws:
            s.sws_freeContext(self.sws); self.sws = None
        for name in ("fr", "bgr"):
            if getattr(self, name):
                u.av_frame_free(ctypes.byref(getattr(self, name))); setattr(self, name, None)
        if self.pkt:
            c.av_packet_free(ctypes.byref(self.pkt)); self.pkt = None
        if self.ctx:
            c.avcodec_free_context(ctypes.byref(self.ctx)); self.ctx = None
        if self.ic:
            f.avformat_close_input(ctypes.byref(self.ic)); self.ic = None

def decode(j, arr, out, h):
    """Decode job `j` into arr[j["offset"]:] with non-reference pictures left out.  h: the worker's helpers (beat,
    written, prog, log_take, log_reset, lead_ok, hash).  True when the job is done (out filled as a normal slice job),
    False with out["skip_fail"] when the worker has to decode it with OpenCV instead."""
    if libs() is None:
        out["skip_fail"] = "library build is not the checked one"; return False
    t0 = time.perf_counter()
    t = table(j["tab"], j["tab_token"])
    t_table = time.perf_counter() - t0
    p2i, drop, times, kidx, kidr = t["p2i"], t["drop"], t["times"], t["kidx"], t["kidr"]
    want, last, offset, a = j["want"], j["last"], j["offset"], j["anchor"]
    chain_idx = j.get("chain_idx")
    A = a["idx"] if a is not None else -1
    keep = set(want) | {A}
    if chain_idx is not None:
        keep.add(chain_idx)
    wpos = {i: offset + n for n, i in enumerate(want)}
    rot = _ROT.get(j.get("rot") or 0)
    if a is None or a.get("from_start"):
        starts = ["zero"]
    else:
        kb = kidx[kidx <= A]
        starts = []
        if len(kb) and kb[-1] > 0 and kidr[len(kb) - 1] and a.get("idr_ok", True):
            starts.append(("idr", int(kb[-1])))
        starts += [("kf", int(k)) for k in kb[-2::-1] if k > 0]
        starts.append("zero")
    rejected, overshoots, dec, tries, logs = 0, 0, None, [], []
    u = c = None
    tm = {"open_s": 0.0, "loop_s": 0.0}
    tm["table_s"] = t_table
    out["skip_t"] = tm
    try:
        for how in starts:
            S = 0 if how == "zero" else how[1]
            if dec is None or how == "zero":
                if dec is not None:
                    dec.close()
                t0 = time.perf_counter()
                dec = _Dec(j["path"], j["threads"])
                tm["open_s"] += time.perf_counter() - t0
                u, c, f, s = dec.L
            name = "zero" if how == "zero" else f"{how[0]}@{S}"
            if how != "zero" and not dec.seek(t["dts"][S]):
                overshoots += 1; tries.append(f"{name}:seek-error"); continue
            logs.append(h.log_take()); h.log_reset()
            t0 = time.perf_counter()
            res = _run(dec, S, how == "zero", A, a, last, keep, wpos, chain_idx, p2i, drop, times, arr, rot, out, h)
            tm["loop_s"] += time.perf_counter() - t0
            if res == "next":
                rejected += 1; tries.append(f"{name}:{out.pop('_why', '?')}"); continue
            if res == "seek":
                overshoots += 1; tries.append(f"{name}:{out.pop('_why', 'seek')}"); continue
            out["start_rejected"], out["seek_overshoots"] = rejected, overshoots
            out["start_chosen"] = how == "zero" and a is not None and not a.get("from_start") and len(starts) == 1
            out["start_fallback"] = how == "zero" and a is not None and not a.get("from_start") and len(starts) > 1
            out["how"] = None if a is None else ("start-restart" if a.get("from_start") else
                                                 "start-fallback" if how == "zero" else how[0])
            return res
        out["skip_fail"] = ("no start point reached the anchor: " + "; ".join(tries))[:300]
        return False
    finally:
        # what FFmpeg printed during the attempts: on a damaged stream it says why the skip path could not finish
        # (frames dropped or concealed there make vLLM's count follow the decoder, not the packets); the parent uses
        # it to tell the stream's damage from a bug (M2_STRICT=1)
        logs.append(h.log_take())
        logs += out.pop("_logs", [])
        bad = [l for l in "\n".join(logs).splitlines() if l.strip() and not h.lead_ok(l)]
        out["skip_log"] = "\n".join(bad)[:200] or None
        out["skip_tries"] = "; ".join(tries)[:300] or None
        if dec is not None:
            dec.close()

def _run(dec, S, zero, A, a, last, keep, wpos, chain_idx, p2i, drop, times, arr, rot, out, h):
    """One attempt from start frame S.  Returns True (done), False (skip_fail set), "next" (start rejected: lead-in
    printed errors, or the anchor's pixels differ) or "seek" (the seek did not land on S)."""
    u, c, f, s = dec.L
    pkt, fr, ctx, ic, vs = dec.pkt, dec.fr, dec.ctx, dec.ic, dec.vs
    N = len(drop)
    def nxt(e):                                    # next frame the decoder must output after frame e
        e += 1
        while e < N and drop[e] and e not in keep:
            e += 1
        return e

    st = {"sent": 0, "skipped": 0, "lead": 0, "e": nxt(-1) if A < 0 else A, "found": A < 0, "bad": None,
          "first": True, "k": 0, "at_start": zero, "pre_start": 0}

    def put(dst):
        s.sws_scale_frame(dec.sws, dec.bgr, fr)
        if rot is None:
            cv2.cvtColor(dec.view, cv2.COLOR_BGR2RGB, dst=dst)
        else:
            cv2.cvtColor(cv2.rotate(dec.view, rot), cv2.COLOR_BGR2RGB, dst=dst)

    def frames():
        """Take every picture the decoder has ready.  Returns None, or a result for _run."""
        while c.avcodec_receive_frame(ctx, fr) >= 0:
            try:
                fp = fr.contents.pts if fr.contents.pts != _NOPTS else fr.contents.pkt_dts
                i = p2i.get(fp)
                if i is None:
                    st["bad"] = f"picture with pts {fp} is not in the frame table"; return False
                if not st["found"] and i < A:
                    continue                       # lead-in output before the anchor
                if not st["found"] and i > A:
                    out["_why"] = f"anchor {A} not output, frame {i} came first"
                    return "next"                  # the anchor never came out (not recovered yet): start earlier
                if i != st["e"]:
                    st["bad"] = f"frame {i} came out where frame {st['e']} was due"; return False
                if not st["found"]:                # i == A: the anchor
                    rgb = np.empty(arr.shape[1:], dtype=np.uint8); put(rgb)
                    txt = h.log_take().strip(); h.log_reset()
                    out.setdefault("_logs", []).append(txt)      # b5 04:50: also evidence for skip_log
                    if not zero and not h.lead_ok(txt):
                        out["_why"] = "lead-in log: " + " | ".join(txt.splitlines()[:2])[:120]
                        return "next"
                    if a.get("hash") is not None and h.hash(rgb) != a["hash"]:
                        out["_why"] = "anchor pixels differ"
                        return "next"
                    st["found"] = True
                    out["anchor_found"], out["anchor_ms"], out["anchor_hash"] = True, times[i], h.hash(rgb)
                    out["lead_frames"], out["lead_log"] = st["lead"], (txt[:200] or None)
                    out["lead_damaged"] = not h.lead_ok(txt)   # full text (review D5-1); only a frame-0 start gets here
                elif i == chain_idx:
                    rgb = np.empty(arr.shape[1:], dtype=np.uint8); put(rgb)
                    out["chain_ms"], out["chain_hash"] = times[i], h.hash(rgb)
                elif i in wpos:
                    put(arr[wpos[i]])
                    st["k"] += 1; out["last_ms"] = times[i]
                    h.written[0] = st["k"]
                    h.prog(st["k"])
                if i == last:
                    return True
                st["e"] = nxt(i)
            finally:
                u.av_frame_unref(fr)
        return None

    r = None
    while r is None:
        h.beat()
        if f.av_read_frame(ic, pkt) < 0:
            c.avcodec_send_packet(ctx, None)       # end of file: drain
            r = frames()
            if r is None:
                st["bad"] = f"end of file before frame {last}"; r = False
            break
        p = pkt.contents
        if p.stream_index != vs:
            c.av_packet_unref(pkt); continue
        pp = p.pts if p.pts != _NOPTS else p.dts
        i = p2i.get(pp) if not (p.flags & _DISCARD) else None
        if st["first"]:
            st["first"] = False
            if not zero and i is not None and i > S:   # landed past the start keyframe: an earlier one
                # (i None = a packet the demuxer marks discard, e.g. an edit list's hidden lead-in: that is at the
                # start of the file, before S; editlist_cut, b5 04:55)
                out["_why"] = f"seek landed at {i}"
                c.av_packet_unref(pkt); return "seek"
        if not st["at_start"]:
            # b5 04:50: av_seek_frame(dts, BACKWARD) can land one keyframe early (mov: editlist_cut 72 -> 34, 112 -> 72;
            # scripts/u5plan/b5 probe); keyframes come in display order, so every packet before S's packet is from an
            # earlier GOP and is not sent: decoding starts exactly at S
            if i != S:
                st["pre_start"] += 1
                c.av_packet_unref(pkt); continue
            st["at_start"] = True
        if i is not None and drop[i] and i not in keep:
            st["skipped"] += 1
            c.av_packet_unref(pkt); continue
        st["sent"] += 1
        if i is not None and i < A:
            st["lead"] += 1
        c.avcodec_send_packet(ctx, ctypes.cast(pkt, ctypes.c_void_p))
        c.av_packet_unref(pkt)
        r = frames()
    if r == "next":
        return "next"
    out["skipped"], out["sent"] = st["skipped"], st["sent"]
    if r is True:
        # b5 05:00: FFmpeg reporting anything inside the slice (from the anchor on; from the start for a frame-0 start)
        # means the stream is damaged there.  The decoder may then output fewer pictures than there are packets, and
        # vLLM's numbering follows the decoder, while this loop numbers by the table and leaves pictures out, so its
        # count cannot be trusted (c30_1pct, c55_2pct: "exact anchor not found" on the next slice).  The slice goes to
        # the OpenCV path, which counts exactly as vLLM does.  Read without clearing: on success the worker keeps it
        # as the slice's decode_log.
        txt = h.log_take().strip()
        if any(l.strip() and not h.lead_ok(l) for l in txt.splitlines()):
            out.setdefault("_logs", []).append(txt)
            out["skip_fail"] = f"FFmpeg reported damage inside the slice (start frame {S})"
            return False
        out["n"] = st["k"]; out["path"] = "skip"
        return True
    out["skip_fail"] = (st["bad"] or "decode stopped") + f" (start frame {S})"
    return False
