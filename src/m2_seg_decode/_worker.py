"""Pool worker of m2_seg_decode, run as `python -I _worker.py` (stdlib + numpy + cv2 only, so it starts in well under a
second and never imports vLLM).  Connects back to the API server over an abstract unix socket (M2_SEG_ADDR without the
leading NUL, M2_SEG_AUTH), pins itself to M2_SEG_WCPUS, and decodes one job per message until the socket closes.

A job decodes the wanted frame indices of one slice into the request's frame buffer, with vLLM's own grab / retrieve /
BGR->RGB loop (vllm/multimodal/video_decoders/opencv.py:254-300).  Where it starts:
  anchor None   a fresh capture from frame 0, i.e. exactly what vLLM does (slice 0, or a tail from the start)
  anchor {idx, ms, tol}   find the frame vLLM counts as `idx`: seek near it, then grab forward until the frame whose
                CAP_PROP_POS_MSEC is within tol of `ms` (OpenCV's POS_MSEC is the frame's own timestamp, so this does not
                trust the seek's frame arithmetic, which is wrong on variable-frame-rate streams); if the seek overshoots,
                seek further back (1 s, 2 s, 4 s, ...) and finally decode from frame 0.  The anchor frame is hashed and
                reported, never written; the parent chains it against the previous slice's last frame (timestamp + hash),
                which is what proves the slice is aligned (runs/u5plan/plugin_segdecode.txt).
After the anchor the loop is sequential, so frame numbering is the same as vLLM's from there on.  A normal job stops at the
first grab or retrieve failure and reports where ("irregular").  What FFmpeg writes to stderr is collected per interval:
the vLLM metadata probe ("probe_log"), the lead-in from the start point to the anchor ("lead_log", "lead_frames"), and
the slice's own frames ("decode_log", the only one that marks a slice as damaged); an `emulate` job (the parent's tail pass) keeps going the
way vLLM does: a failed grab still consumes an index, a failed retrieve skips the frame, and it reports the indices it got.
b5 (10-25): a job with "skip" set (M2_SEG_SKIP_NONREF=1, and the parent's frame table shows pictures nobody references
in the slice's range) is decoded by _skipdec.decode, which leaves those pictures out; if that path cannot finish the
job it says why (out["skip_fail"]), sends ("prog", seq, -1) so the parent forgets the frames it reported, and the job is
decoded here with OpenCV as usual.  A keyframe that is an IDR picture (anchor["kf0_idr"]) is a start point by itself:
nothing before an IDR can be referenced (main 00:59: "IDR 起点").
Replies: ("prog", seq, n) every prog_every frames and at least every second (the parent's stall timer); ("done", seq,
dict) or ("err", seq, repr).  `seq` is the parent's request number, echoed so a stale reply is never taken for a later
request (review S3)."""
import hashlib, mmap, os, sys, time
from multiprocessing.connection import Client
import numpy as np
import cv2

_LOG = {"fd": None, "orig": None}   # FFmpeg's av_log goes to fd 2; main() points fd 2 at a memfd to read it per job
_cached = [None, None]  # "reuse" mode: [buffer path, mmap] kept open across requests (one slot: a new buffer replaces it)
_BEAT_S = 1.0

def _map(path, keep):
    if keep and _cached[0] == path and _cached[1].size() == os.path.getsize(path):
        return _cached[1]
    fd = os.open(path, os.O_RDWR)
    try:
        mm = mmap.mmap(fd, 0)
    finally:
        os.close(fd)
    if keep:
        if _cached[1] is not None:
            _cached[1].close()
        _cached[0], _cached[1] = path, mm
    return mm

def _log_reset():
    if _LOG["fd"] is not None:
        os.ftruncate(2, 0); os.lseek(2, 0, os.SEEK_SET)

def _log_take():
    """FFmpeg / OpenCV messages written to stderr since _log_reset(); copied to the real stderr so nothing is lost."""
    if _LOG["fd"] is None:
        return ""
    n = os.fstat(2).st_size
    txt = os.pread(_LOG["fd"], n, 0).decode("utf-8", "replace") if n else ""
    if txt and _LOG["orig"] is not None:
        try: os.write(_LOG["orig"], txt.encode())
        except OSError: pass
    return txt

# The one libavcodec message that a decode started after frames it never saw prints and that changes no picture:
# h264_refs.c:640-648 (FFmpeg 8.1.1) — the MMCO names a short-term reference that is not in the DPB, the decoder logs
# and `continue`s.  On an open-GOP stream it comes right after every non-IDR start point (review round 5, opengop_bf3).
# Any other message during the lead-in (missing references, concealment) means a picture was built from substitutes,
# so the start point is moved one keyframe earlier.
_NOOP_MSGS = ("mmco: unref short failure",)

def _damage(txt):
    """The one rule for "FFmpeg reported damage" (review D5-1, 10-25): the lines of `txt` that are not _NOOP_MSGS.
    Every decision uses it on the full text it was taken from: start points (_lead_ok), the slice's decode_log, the
    skip path (_skipdec, through h.lead_ok).  Stored copies (lead_log, decode_log, skip_log) are cut for display and
    are never judged again (dev5: lead_log cut at 200 characters ended in half a line, which this rule did not
    recognise, so a harmless open-GOP lead-in became a "damaged" slice and was decoded again from frame 0)."""
    return [l for l in txt.splitlines() if l.strip() and not any(m in l for m in _NOOP_MSGS)]

def _lead_ok(txt):
    return not _damage(txt)

_kf_cache = {}

def _keyframes_before(path, ms):
    """Times (CAP_PROP_POS_MSEC, ms) of the keyframes at or before time `ms`, latest first, from a packet-only scan
    (OpenCV raw mode, CAP_PROP_FORMAT=-1: demuxing, no decoding) of a window before `ms`, widened until it holds three
    keyframes or reaches the start of the file.  Keyframes are kept as times, never as frame numbers: in raw mode
    CAP_PROP_PTS is the stream pts (a start offset or a variable frame rate makes it differ from vLLM's frame count;
    review round 5c), while CAP_PROP_POS_MSEC there is on the same clock as the decoded frames' POS_MSEC, start offset
    already removed (checked on start_3p5, vfr_select, editlist_cut, opengop_bf3; runs/u5plan/plugin_segdecode.txt).
    53-minute HvKpnaXYUPU: ~12 ms for a 10 s window.  Cached per (file, ms).  None if the scan fails."""
    try:
        st = os.stat(path)
        key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, round(ms, 3))
    except OSError:
        return None
    if key in _kf_cache:
        return _kf_cache[key]
    res = None
    try:
        win = 10000.0
        while True:
            start_ms = max(0.0, ms - win)
            c = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_FORMAT, -1])
            if not c.isOpened():
                break
            if start_ms > 0:
                c.set(cv2.CAP_PROP_POS_MSEC, start_ms)
            kf = set()
            while c.grab():
                t = c.get(cv2.CAP_PROP_POS_MSEC)
                if t > ms + 5000.0:                # decode order: every keyframe at or before ms has been seen
                    break
                if t <= ms + 1e-3 and c.get(cv2.CAP_PROP_LRF_HAS_KEY_FRAME):
                    kf.add(round(t, 3))
            c.release()
            if len(kf) >= 3 or start_ms == 0:
                res = sorted(kf, reverse=True)
                break
            win *= 2
    except Exception:
        res = None
    if len(_kf_cache) > 4096:
        _kf_cache.clear()
    _kf_cache[key] = res
    return res

def _hash(a):
    return hashlib.blake2b(memoryview(np.ascontiguousarray(a)).cast("B"), digest_size=16).hexdigest()

def _open(path, threads):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, threads])
    return cap if cap.isOpened() else None

def _vllm_probe(cap, out=None):
    """The capture calls vLLM's get_video_metadata makes before it reads (opencv.py:83-101): seek to the end, grab, seek
    back to frame 0.  Replayed on every capture that reads from frame 0, so it is left in the same state as vLLM's (on a
    one-frame file this leaves grab() failing, and vLLM then reads 0 frames; we must do the same).  What FFmpeg prints
    during it belongs to the probe, not to the frames read afterwards (vLLM prints the same), so it is kept apart."""
    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0 and cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0):
        cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.grab()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    txt = _log_take().strip()
    _log_reset()
    if txt and out is not None:
        out["probe_log"] = txt[:200]

def _locate(cap, path, threads, a, beat, out):
    """Leave `cap` just after grabbing the frame with timestamp a["ms"] (+-tol) and, when a["hash"] is given, those exact
    pixels.  Start points, in order: a["kf"] (keyframe display indices from the parent's packet scan: the keyframe before
    the GOP that holds the anchor, then earlier ones), else a frame seek near a["idx"] and time seeks 1, 2, 4, ... s
    earlier; finally frame 0 as vLLM does.  A start is accepted only if the lead-in (start .. anchor) made FFmpeg print
    nothing but _NOOP_MSGS; the lead-in log and frame count are recorded apart from the slice's own decode log.
    Returns (cap, found, how, frame_rgb or None)."""
    ms, tol, want_hash = a["ms"], a["tol"], a.get("hash")
    lead = [0]

    def grab_ms():
        beat()
        lead[0] += 1
        return cap.get(cv2.CAP_PROP_POS_MSEC) if cap.grab() else None

    def forward(m):
        while m is not None and m < ms - tol:
            m = grab_ms()
        if m is None or abs(m - ms) > tol:
            return None
        ok, fr = cap.retrieve()
        if not ok:
            return None
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        return rgb if want_hash is None or _hash(rgb) == want_hash else None

    kf = None
    if not a.get("from_start"):
        # keyframe times from the parent's frame table (exact, vLLM's clock); a packet scan only without a table
        kf = a["kf"] if a.get("kf") is not None else _keyframes_before(path, ms)
    if a.get("from_start"):
        starts = ["restart"]
    elif kf is not None:
        # review round 5 / main 20:09 and 20:47: by time, kf[0] is the last keyframe at or before the anchor (the
        # keyframe of the anchor's GOP; on an open-GOP stream a non-IDR I picture whose references are not there);
        # start at the one before it, then earlier ones.  If there is none before it, frame 0 is the chosen start
        # ("chosen"); frame 0 after every keyframe failed is a fallback (main 21:12: counted apart)
        prior = [t for t in kf[1:] if t > 0]
        starts = [("kf", t) for t in prior] + (["fallback"] if prior else ["chosen"])
        if a.get("kf0_idr") and kf and kf[0] > 0:
            # b5: the anchor's own GOP starts with an IDR picture, so decoding can start right there
            starts = [("idr", kf[0])] + starts[:-1] + ["fallback"]
    else:
        starts = ["frames"] + [1000.0 * 2 ** i for i in range(40) if ms - 1000.0 * 2 ** i > 0] + ["fallback"]
    rejected = 0
    for how in starts:
        lead[0] = 0
        if how in ("restart", "chosen", "fallback"):
            cap.release()
            cap = _open(path, threads)
            if cap is None:
                return None, False, "reopen failed", None
            _vllm_probe(cap, out)
            m = grab_ms()
        else:
            _log_reset()
            if isinstance(how, tuple):
                # OpenCV 5.0.0 turns POS_MSEC into a frame count with the average frame rate, seeks back to a keyframe
                # before that and counts grabs (cap_ffmpeg_impl.hpp:2304-2378), so on a variable-frame-rate stream it
                # stops early or late.  The decoder still starts at a keyframe before the requested one, so early is
                # only a longer lead-in; only a stop past the anchor needs an earlier keyframe.  The actual landing
                # time decides.
                cap.set(cv2.CAP_PROP_POS_MSEC, how[1])
            elif how == "frames":
                cap.set(cv2.CAP_PROP_POS_FRAMES, a["idx"])
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, ms - how)
            m = grab_ms()
            if m is None or m > ms + tol:
                out["seek_overshoots"] = out.get("seek_overshoots", 0) + 1
                continue                           # landed past the anchor: start further back
        rgb = forward(m)
        txt = _log_take().strip()
        _log_reset()                               # from here on: the slice's own frames
        if rgb is None:
            continue
        from_zero = how in ("restart", "chosen", "fallback")
        if not from_zero and not _lead_ok(txt):
            rejected += 1                          # the lead-in used substitute pictures: start one keyframe earlier
            continue
        # A start at frame 0 is how vLLM itself decodes, so it is never rejected for what the lead-in printed: on a
        # damaged stream the messages are the stream's own (with several FFmpeg threads they can even come from
        # packets past the anchor, read ahead), and decode() merges them into this slice's decode_log, which marks
        # the slice damaged (review K3: the dev3 rename of "start" had turned this exemption off).
        out["lead_frames"], out["lead_log"], out["start_rejected"] = lead[0], (txt[:200] or None), rejected
        out["lead_damaged"] = bool(_damage(txt))   # judged here on the full text (only a frame-0 start can be damaged)
        out["start_chosen"], out["start_fallback"] = how == "chosen", how == "fallback"
        name = (how[0] if isinstance(how, tuple) else f"start-{how}" if how in ("restart", "chosen", "fallback")
                else how if isinstance(how, str) else f"msec-{how:.0f}")
        return cap, True, name, rgb
    return cap, False, "not found", None

def decode(conn, seq, j):
    fault = j.get("fault")
    if fault == "hang":                        # test hooks (M2_SEG_TEST_FAULT in the parent), never set in production
        time.sleep(3600)
    elif fault == "die":
        os._exit(3)
    elif fault == "exc":
        raise RuntimeError("injected test fault")
    out = {"n": 0, "valid": None, "anchor_found": None, "anchor_ms": None, "anchor_hash": None, "how": None,
           "last_ms": None, "last_hash": None, "irregular": None, "error": None, "decode_log": None,
           "chain_ms": None, "chain_hash": None, "probe_log": None, "lead_log": None, "lead_frames": 0,
           "start_rejected": 0, "seek_overshoots": 0, "start_chosen": False, "start_fallback": False,
           "path": "opencv", "skipped": 0, "sent": None, "skip_fail": None, "lead_damaged": False, "noop_lines": 0}
    _log_reset()
    try:
        return _decode(conn, seq, j, out)
    finally:
        txt = _log_take().strip()
        bad = _damage(txt)
        out["noop_lines"] = sum(1 for l in txt.splitlines() if l.strip()) - len(bad)
        if out.get("lead_damaged"):                # a frame-0 restart through damage: frames after it may carry it
            txt = ((out.get("lead_log") or "") + "\n" + txt).strip()
        if bad or out.get("lead_damaged"):         # decode_log is set only for damage (the parent reads it as such)
            out["decode_log"] = txt[:300]

def _decode(conn, seq, j, out):
    nxt = [time.monotonic() + _BEAT_S]
    offset, prog_every = j["offset"], j["prog_every"]
    written = [0]

    def beat():
        if time.monotonic() >= nxt[0]:
            conn.send(("prog", seq, written[0])); nxt[0] = time.monotonic() + _BEAT_S

    mm = _map(j["buf"], j["keep"])
    shape = tuple(j["shape"])
    arr = np.ndarray(shape, dtype=np.uint8, buffer=mm)
    cap = None
    try:
        if j.get("skip") and not j["emulate"]:
            if _skip(conn, seq, j, arr, out, written, nxt, beat):
                return out
        cap = _open(j["path"], j["threads"])
        if cap is None:
            out["error"] = "could not open the video"; return out
        a, pos = j["anchor"], -1
        if a is None:
            _vllm_probe(cap, out)
        else:
            cap, found, out["how"], rgb = _locate(cap, j["path"], j["threads"], a, beat, out)
            out["anchor_found"] = found
            if not found:
                return out
            out["anchor_ms"] = cap.get(cv2.CAP_PROP_POS_MSEC)
            out["anchor_hash"] = _hash(rgb)
            pos = a["idx"]
        want, last, emulate = set(j["want"]), j["last"], j["emulate"]
        chain_idx = j.get("chain_idx")            # a frame before the slice to hash on the way (restart from earlier)
        valid = [] if emulate else None
        k = offset
        while pos < last:
            pos += 1
            beat()
            if not cap.grab():
                if emulate:
                    continue                           # vLLM: a failed grab still consumes the index
                out["irregular"] = f"grab failed at frame {pos}"; break
            if pos == chain_idx:
                ok, fr = cap.retrieve()
                if ok:
                    out["chain_ms"] = cap.get(cv2.CAP_PROP_POS_MSEC)
                    out["chain_hash"] = _hash(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                continue
            if pos in want:
                ok, fr = cap.retrieve()
                if not ok:
                    if emulate:
                        continue                       # vLLM: skip the frame
                    out["irregular"] = f"retrieve failed at frame {pos}"; break
                if fr.shape[:2] != shape[1:3]:
                    out["error"] = f"frame {pos} is {fr.shape[:2]}, header says {shape[1:3]}"   # vLLM raises here too
                    return out
                cv2.cvtColor(fr, cv2.COLOR_BGR2RGB, dst=arr[k])
                out["last_ms"] = cap.get(cv2.CAP_PROP_POS_MSEC)
                if emulate:
                    valid.append(pos)
                k += 1
                written[0] = k - offset
                if prog_every and written[0] % prog_every == 0:
                    conn.send(("prog", seq, written[0])); nxt[0] = time.monotonic() + _BEAT_S
        out["n"], out["valid"] = k - offset, valid
        if k > offset and (j["want_last"] or out["irregular"]):
            out["last_hash"] = _hash(arr[k - 1])
        return out
    finally:
        del arr
        if cap is not None:
            cap.release()
        if not j["keep"]:
            mm.close()

_SKIP_RESET = ("n", "anchor_found", "anchor_ms", "anchor_hash", "how", "last_ms", "last_hash", "chain_ms", "chain_hash",
               "lead_frames", "lead_log", "lead_damaged", "start_rejected", "seek_overshoots", "start_chosen",
               "start_fallback")

def _skip(conn, seq, j, arr, out, written, nxt, beat):
    """The skip-nonref path for this job (_skipdec.py).  True: done, `out` filled.  False: the OpenCV path takes the job
    (out["skip_fail"] says why); the frames it reported are withdrawn first with ("prog", seq, -1)."""
    import types
    fresh = dict(out)
    prog_every = j["prog_every"]

    def prog(n):
        if prog_every and n % prog_every == 0:
            conn.send(("prog", seq, n)); nxt[0] = time.monotonic() + _BEAT_S
    h = types.SimpleNamespace(beat=beat, written=written, prog=prog, log_take=_log_take, log_reset=_log_reset,
                              lead_ok=_lead_ok, hash=_hash)
    try:
        import _skipdec
        ok = _skipdec.decode(j, arr, out, h)
    except Exception as e:
        out["skip_fail"], ok = f"{type(e).__name__}: {e}"[:200], False
    if ok:
        if out["n"] and j["want_last"]:
            out["last_hash"] = _hash(arr[j["offset"] + out["n"] - 1])
        return True
    why, skipped, sent, slog = out["skip_fail"], out.get("skipped", 0), out.get("sent"), out.get("skip_log")
    tries = out.get("skip_tries")
    for k in _SKIP_RESET:
        out[k] = fresh[k]
    out.update(path="opencv", skip_fail=why, skip_log=slog, skip_tries=tries, skipped=0, skipped_before_fail=skipped,
               sent=None)
    if written[0]:
        conn.send(("prog", seq, -1))
    written[0] = 0
    _log_take(); _log_reset()                  # the attempt's messages are not this slice's (the OpenCV pass prints its own)
    return False

def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I leaves the script's directory off sys.path
    cpus = os.environ.get("M2_SEG_WCPUS")
    if cpus:
        os.sched_setaffinity(0, {int(c) for c in cpus.split(",")})
    cv2.setNumThreads(1)                       # cvtColor: one thread per worker; decoding threads are per VideoCapture
    try:                                       # read FFmpeg's decode-error messages per job (see _log_take)
        _LOG["orig"] = os.dup(2)
        _LOG["fd"] = os.memfd_create("m2_seg_log", os.MFD_CLOEXEC)
        os.dup2(_LOG["fd"], 2)
    except OSError:
        _LOG["fd"] = None
    conn = Client("\0" + os.environ["M2_SEG_ADDR"], family="AF_UNIX", authkey=bytes.fromhex(os.environ["M2_SEG_AUTH"]))
    conn.send(("ready", os.getpid()))
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return                             # the API server closed the pool or died
        seq = msg[1]
        try:
            reply = ("done", seq, decode(conn, seq, msg[2]))
        except (BrokenPipeError, ConnectionError):
            return
        except Exception as e:
            reply = ("err", seq, repr(e)[:300])
        try:
            conn.send(reply)
        except (BrokenPipeError, ConnectionError, OSError):
            return

if __name__ == "__main__":
    sys.exit(main())
