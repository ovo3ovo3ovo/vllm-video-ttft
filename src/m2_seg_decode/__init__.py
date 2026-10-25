"""vLLM general plugin (mission_2 unit 5, 10-24): "切段并行解码" for the OpenCV video path (vLLM 0.31.0 only).

vLLM's decoder (vllm/multimodal/video_decoders/opencv.py:28 decode_opencv -> read_frames, :254-300) grabs every frame from
frame 0 up to the last sampled one in a single cv2.VideoCapture, so a 35-minute video decodes ~50k frames serially.  This
plugin keeps vLLM's own metadata and frame-index code, splits the sampled indices into M2_SEG_N contiguous slices and lets
a resident pool of worker processes decode them in parallel (plugins/m2_seg_decode/_worker.py), writing straight into one
shared frame buffer.  runs/u5plan/plugin_segdecode.txt has the checks, timings and the reasoning behind each step.

How the result is made exactly vLLM's (no input class is refused; review S1 and the user's 16:04 ask):
  - slice 0 decodes from frame 0 like vLLM.  Slice k > 0 starts at an anchor: the frame vLLM counts as slice k-1's last
    sampled index.  The worker finds it by timestamp (CAP_PROP_POS_MSEC is the frame's own timestamp), first at a guess
    (index / fps), and counts frames sequentially from there.
  - chain check: slice k is accepted only if its anchor frame has the same timestamp and the same pixels (hash) as slice
    k-1's last frame, and slice k-1 is accepted.  So by induction every accepted slice is aligned with vLLM's count.
  - a slice whose anchor does not chain (variable frame rate, edit list, ...) is decoded again from the exact anchor,
    whose timestamp is now known from slice k-1.  Slices after it are checked against the new result.
  - if a slice hits a grab or retrieve failure (truncated or damaged file), everything from that slice on is decoded by
    one tail job that starts at that slice's anchor and follows vLLM's rules for failures (a failed grab consumes the
    index, a failed retrieve skips the frame), so the frame count and valid indices come out as vLLM's.  Which frames
    are missing is read from grab()/retrieve() return values exactly as vLLM does (review N4), never from logs.
  - damaged streams (PLAN.txt 17:46): a slice in which FFmpeg logged a decode error is still split; its chain check
    compares timestamps only (error concealment fills damaged macroblocks from leftover decoder memory, which differs
    between workers and from vLLM; runs/u5plan/FFMPEG_EC.md), and the request status says corrupt=True and which slices.
    The 15:50-17:03 rule "decode error -> redo from frame 0 with vLLM's thread count" is withdrawn (user 17:46).
  - duplicate sampled indices collapse exactly as in vLLM (it reads into len(set(frame_idx)) frames).
  - bytes / base64 / http input has no file path: the bytes go into a memfd the workers open as /proc/<pid>/fd/<n>.
Replaced: vllm.multimodal.video_decoders.opencv.decode_opencv (decode_video looks it up by name on every call,
video_decoders/__init__.py:99-100) and VideoMediaIO.load_file (file path for the workers, and this request's status on the
returned media; vllm/multimodal/media/video.py:195).  Off by default.  Environment, read per call:
  M2_SEG_DECODE=1              turn on
  M2_SEG_N=16 / M2_SEG_THREADS=4   worker processes per pool (= most slices) / FFmpeg threads per slice
  M2_SEG_MIN_SRC_FRAMES=2642   performance gate: split only when frames to grab (last sampled index + 1) is at least this
                               (short videos are slower split; 2642 from the offline part-1 data, user 10-25 01:05;
                               correctness tests run with 0).  M2_SEG_MIN_FRAME_PIXELS=0: same for width * height
  M2_SEG_SKIP_NONREF=1         b5: H.264 pictures nobody references and nobody sampled are not decoded (_skipdec.py).
                               Per slice: the parent's frame table (_index.frame_table_ex, sent to the workers in a memfd)
                               says whether its range has such pictures; if yes the worker decodes with the skip loop
                               (status path "skip"), if not with OpenCV as before ("opencv-none": the right path for that
                               input, not a fallback).  If the skip loop cannot finish a slice the worker decodes it with
                               OpenCV ("skip_fail"); with M2_STRICT=1 that is an error unless the slice is damaged or
                               truncated (FFmpeg logged errors / a grab failed), since on a clean stream it is a bug.
  IDR start (b5): an anchor whose GOP keyframe is an IDR picture is decoded from that keyframe (nothing before an IDR can
                               be referenced); otherwise from the keyframe before it, as before.
  M2_SEG_COPY=zerocopy         copy | reuse | zerocopy (see plugin_segdecode.txt); all three feed m2_pipe_resize
  M2_SEG_WNODES=0,1 / M2_SEG_WCPUS   worker CPUs; memory follows the server's --membind
  Concurrency (main 18:46): one resident pool of M2_SEG_N workers per process, shared by every request.  Each request
                               splits as usual; a dispatcher thread hands free workers the next job of the next request
                               in rotation, so every concurrent request is split, none waits for another to finish, and
                               the FFmpeg thread total stays M2_SEG_N x M2_SEG_THREADS.  No request is sent to vLLM's
                               decoder because the pool is busy.
  M2_SEG_STALL_S=30            a worker silent this long (it reports progress every second) = stuck: the pool is killed
                               and rebuilt.  Hard cap per request max(300 s, 20x the serial estimate) x requests sharing
                               the pool (M2_SEG_TIMEOUT_S overrides the first factor)
M2_STRICT=1: every fallback below raises M2StrictError instead (the performance gate and errors vLLM raises too do
not).  Safety net only (should never trigger on normal input): stall / timeout / dead worker / stale reply (pool discarded, this
request uses vLLM's decoder; M2_STRICT=1 raises); an error vLLM would raise too (frame size differs from the header: vLLM's code runs and
raises it); frame_recovery=True (vLLM's recovery scan is not reimplemented; vLLM's code runs).
Status per request: _tls.status (decode thread) and media.m2_seg_status on the MediaWithBytes from load_file;
served=True / False, why, gated=True for the performance gate.  `last` = latest status, for offline scripts only."""
import atexit, collections, logging, mmap, os, queue, secrets, subprocess, sys, threading, time, weakref
_MARK = "_m2_segdecode"
_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_worker.py")
_log = logging.getLogger("m2_seg_decode")
_vlog = logging.getLogger("vllm.multimodal.video_decoders.opencv")
_said = set()
_tls = threading.local()
last = {}

def _on():
    return os.environ.get("M2_SEG_DECODE") == "1"

def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default

def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default

def _note(reason):
    if reason not in _said:
        _said.add(reason); _log.warning("m2_seg_decode: using vLLM's decoder (%s)", reason)

def _cpulist(s):
    out = set()
    for part in s.strip().split(","):
        if part:
            lo, _, hi = part.partition("-"); out.update(range(int(lo), int(hi or lo) + 1))
    return out

def _worker_cpus():
    if os.environ.get("M2_SEG_WCPUS"):
        return _cpulist(os.environ["M2_SEG_WCPUS"])
    cpus = set()
    for n in os.environ.get("M2_SEG_WNODES", "0,1").split(","):
        try:
            cpus |= _cpulist(open(f"/sys/devices/system/node/node{int(n)}/cpulist").read())
        except (OSError, ValueError):
            pass
    return cpus

def _clean_stale_devshm():
    """Earlier versions of this plugin left /dev/shm/m2_seg_<pid>_* behind when the server was SIGKILLed (review S5)."""
    try:
        for name in os.listdir("/dev/shm"):
            if name.startswith("m2_seg_"):
                pid = name.split("_")[2]
                if pid.isdigit() and not os.path.exists(f"/proc/{pid}"):
                    os.unlink(os.path.join("/dev/shm", name))
    except OSError:
        pass

class M2StrictError(RuntimeError):
    """M2_STRICT=1: the plugin was on but did not serve this request (no silent fallback to vLLM's decoder)."""

def _strict():
    return os.environ.get("M2_STRICT") == "1"

class _Broken(Exception):
    """The pool can no longer be trusted (stall, timeout, dead worker, stale reply): discard it."""

class _Fail(Exception):
    """The split decode cannot produce this request's frames.  vllm_fails=True: an error vLLM's own decoder raises too
    (cannot open, frame size differs from the header), so vLLM's code runs and raises it."""
    def __init__(self, msg, vllm_fails=False):
        super().__init__(msg); self.vllm_fails = vllm_fails

class _Pool:
    def __init__(self, n, cpus):
        from multiprocessing.connection import Listener
        t0 = time.perf_counter()
        addr = f"\0m2_seg_{os.getpid()}_{secrets.token_hex(6)}"     # abstract socket: no file to clean up
        auth = secrets.token_bytes(16)
        self.n, self.procs, self.conns = n, [], []
        self.buf_fd, self.buf_mm, self.buf_size = None, None, 0
        listener = Listener(addr, family="AF_UNIX", authkey=auth)
        try:
            listener._listener._socket.settimeout(30)
            env = dict(os.environ, M2_SEG_ADDR=addr[1:], M2_SEG_AUTH=auth.hex(),   # env vars cannot hold the leading NUL
                       M2_SEG_WCPUS=",".join(map(str, sorted(cpus))))
            self.procs = [subprocess.Popen([sys.executable, "-I", _WORKER], env=env, stdin=subprocess.DEVNULL,
                                           close_fds=True) for _ in range(n)]
            for _ in range(n):
                c = listener.accept()
                if not c.poll(30) or c.recv()[0] != "ready":
                    raise RuntimeError("worker did not report ready")
                self.conns.append(c)
        except Exception:
            self.close(); raise
        finally:
            listener.close()
        self.startup_s = time.perf_counter() - t0

    def reuse_buffer(self, size):
        """One memfd per pool, grown when needed, pre-faulted once (MAP_POPULATE) and kept mapped by every worker."""
        if size > self.buf_size:
            fd = os.memfd_create("m2_seg_reuse", os.MFD_CLOEXEC)
            os.ftruncate(fd, size)
            mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED | mmap.MAP_POPULATE)
            if self.buf_mm is not None:
                self.buf_mm.close(); os.close(self.buf_fd)
            self.buf_fd, self.buf_mm, self.buf_size = fd, mm, size
        return f"/proc/{os.getpid()}/fd/{self.buf_fd}", self.buf_mm

    def close(self, kill=False):
        for c in self.conns:
            try: c.close()
            except Exception: pass
        self.conns = []
        for p in self.procs:
            try:
                if kill:
                    p.kill()
                p.wait(timeout=5)
            except Exception:
                try: p.kill(); p.wait(timeout=5)
                except Exception: pass
        if self.buf_mm is not None:
            try: self.buf_mm.close(); os.close(self.buf_fd)
            except Exception: pass
            self.buf_mm = None

class _Req:
    """One request's share of the pool: its jobs waiting for a worker, and the replies routed back to it."""
    def __init__(self):
        self.inbox, self.pending, self.t0, self.first = queue.Queue(), collections.deque(), time.monotonic(), None
        self.inflight = 0

class _Sched:
    """One resident pool of M2_SEG_N workers shared by every request of the process (main 18:46).  Each request puts
    its slice jobs in its own queue; whenever a worker is free, the dispatcher thread gives it the next job of the next
    request in rotation, so C requests at once get about W/C workers each, all of them split, and none waits for
    another to finish.  The dispatcher is the only thread that talks to the workers.  Every job carries a sequence
    number the worker echoes; a reply goes back to the request that owns that job, and a reply that does not match the
    job its worker is running means the pool is out of sync and is discarded (review S3).  Discarding (stall, dead
    worker, stale reply, a request over its hard cap) kills the pool, tells every active request, and rebuilds it."""

    def __init__(self):
        self.lock = threading.Condition()
        self.pool, self.state, self.err, self.retry_at, self.gen = None, "none", None, 0.0, 0
        self.free, self.busy, self.active, self.rr, self.seq, self.cleaned = [], {}, [], 0, 0, False
        self.arriving = 0                          # requests between "plugin on" and enter(): share reserved for them
        self.rfd, self.wfd = os.pipe()
        os.set_blocking(self.rfd, False); os.set_blocking(self.wfd, False)
        self.reuse_lock, self.reuse = threading.Lock(), [None, None, 0]      # memfd, mmap, size

    def _wake(self):
        try: os.write(self.wfd, b"x")
        except BlockingIOError: pass

    def arrive(self, delta):
        """+1 when a request starts probing its video, -1 when it has entered or left without splitting."""
        with self.lock:
            self.arriving = max(0, self.arriving + delta)
        self._wake()

    def enter(self, n):
        """Register a request.  Waits while the pool is being (re)built (~0.3 s; at most n x 30 s if workers never
        connect).  Returns (req, None), or (None, reason) if the pool cannot be started."""
        with self.lock:
            if not self.cleaned:
                _clean_stale_devshm(); self.cleaned = True
            while True:
                if self.state == "ready" and self.pool.n != n and not self.active:
                    self.gen += 1; pool, self.pool, self.state = self.pool, None, "none"
                    pool.close()
                if self.state == "ready":
                    req = _Req(); req.n, req.pool_start_s = self.pool.n, self.pool.startup_s
                    self.active.append(req); self.arriving = max(0, self.arriving - 1)
                    return req, None
                if self.state == "failed" and time.monotonic() < self.retry_at:     # review S9: retry after a minute
                    self.arriving = max(0, self.arriving - 1)
                    return None, f"pool start failed: {self.err}"
                if self.state in ("none", "failed"):
                    self.state = "building"
                    threading.Thread(target=self._build, args=(n, None), name="m2_seg_build", daemon=True).start()
                self.lock.wait(1.0)

    def leave(self, req):
        with self.lock:
            if req in self.active:
                self.active.remove(req)
            req.pending.clear()
        self._wake()                               # its share goes to the others

    def submit(self, req, kind, k, job):
        with self.lock:
            if req not in self.active:
                raise _Broken("pool was discarded")
            req.pending.append((kind, k, job))
        self._wake()

    def cancel(self, req, pred):
        """Drop this request's not-yet-dispatched jobs for which pred((kind, k)) holds; returns their (kind, k)."""
        with self.lock:
            gone = [x[:2] for x in req.pending if pred(x[:2])]
            req.pending = collections.deque(x for x in req.pending if not pred(x[:2]))
            return gone

    def fail_pool(self, why):
        with self.lock:
            if self.state == "ready":
                self._broken(why)

    def reuse_buffer(self, size):
        """M2_SEG_COPY=reuse: one pre-faulted memfd (MAP_POPULATE) kept mapped by the workers, used by one request at
        a time; a concurrent request gets None and uses a buffer of its own."""
        if not self.reuse_lock.acquire(blocking=False):
            return None
        if size > self.reuse[2]:
            fd = os.memfd_create("m2_seg_reuse", os.MFD_CLOEXEC)
            os.ftruncate(fd, size)
            mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED | mmap.MAP_POPULATE)
            if self.reuse[1] is not None:
                _close_map(self.reuse[1]); os.close(self.reuse[0])
            self.reuse = [fd, mm, size]
        return f"/proc/{os.getpid()}/fd/{self.reuse[0]}", self.reuse[1]

    def reuse_release(self):
        self.reuse_lock.release()

    def close(self):
        with self.lock:
            self.gen += 1
            pool, self.pool, self.state, self.free, self.busy = self.pool, None, "none", [], {}
        if pool is not None:
            pool.close(kill=True)

    # ---- dispatcher side (the lock is held where noted)
    def _build(self, n, old):
        if old is not None:
            old.close(kill=True)
        try:
            pool, err = _Pool(n, _worker_cpus()), None
            _log.info("m2_seg_decode: %d workers up in %.2f s", n, pool.startup_s)
        except Exception as e:
            pool, err = None, repr(e)[:200]
            _note(f"pool start failed: {e!r}")
        with self.lock:
            if pool is not None:
                self.gen += 1
                self.pool, self.state, self.free, self.busy = pool, "ready", list(pool.conns), {}
                threading.Thread(target=self._loop, args=(self.gen,), name="m2_seg_dispatch", daemon=True).start()
            else:
                self.state, self.err, self.retry_at = "failed", err, time.monotonic() + 60
            self.lock.notify_all()

    def _broken(self, why):
        """Lock held.  Discard the pool, tell every active request, rebuild in the background."""
        _note(f"pool discarded: {why}")
        self.gen += 1
        pool, self.pool = self.pool, None
        for r in self.active:
            r.pending.clear(); r.inbox.put(("broken", None, None, why))
        self.active, self.free, self.busy, self.state = [], [], {}, "building"
        threading.Thread(target=self._build, args=(pool.n if pool else _env_int("M2_SEG_N", 16), pool),
                         name="m2_seg_rebuild", daemon=True).start()
        self.lock.notify_all()

    def _assign(self):
        """Lock held.  Free workers take the next job of the next request in rotation.  Fair share: a request has at
        most ceil(W / (active + arriving)) jobs running, so requests that come in together split the workers evenly
        from the start instead of the first one taking all of them.  Above its share a request only gets a free worker
        when nobody else has a job waiting and nobody is still arriving (work-conserving otherwise)."""
        while self.free and self.active:
            n, pick = len(self.active), None
            share = -(-self.pool.n // max(1, n + self.arriving))
            for over in (False, True):
                if over and self.arriving:
                    break
                for i in range(n):
                    r = self.active[(self.rr + i) % n]
                    if r.pending and (over or r.inflight < share):
                        pick, self.rr = r, (self.rr + i + 1) % n
                        break
                if pick is not None:
                    break
            if pick is None:
                return
            kind, k, job = pick.pending.popleft()
            c = self.free.pop()
            self.seq += 1
            c.send(("job", self.seq, job))
            self.busy[c] = [self.seq, pick, kind, k, time.monotonic()]
            pick.inflight += 1
            if pick.first is None:
                pick.first = time.monotonic() - pick.t0

    def _loop(self, gen):
        try:
            self._loop_inner(gen)
        except Exception as e:                     # never leave requests waiting on a dead dispatcher
            with self.lock:
                if gen == self.gen:
                    self._broken(f"dispatcher error: {e!r}"[:200])

    def _loop_inner(self, gen):
        from multiprocessing.connection import wait
        stall_s = _env_float("M2_SEG_STALL_S", 30)
        while True:
            with self.lock:
                if gen != self.gen:
                    return
                try:
                    self._assign()
                except (OSError, EOFError) as e:
                    self._broken(f"send to a worker failed: {e!r}"); return
                now = time.monotonic()
                silent = [b[3] for b in self.busy.values() if now - b[4] > stall_s]
                if silent:
                    self._broken(f"worker silent for {stall_s:.0f} s on slice {silent}"); return
                conns = list(self.busy)
            ready = wait(conns + [self.rfd], timeout=1.0)
            with self.lock:
                if gen != self.gen:
                    return
                for c in ready:
                    if c is self.rfd or c == self.rfd:
                        try: os.read(self.rfd, 4096)
                        except BlockingIOError: pass
                        continue
                    try:
                        msg = c.recv()
                    except (EOFError, OSError) as e:
                        self._broken(f"{type(e).__name__}: {e}"); return
                    b = self.busy.get(c)
                    if b is None or len(msg) < 3 or msg[1] != b[0]:
                        self._broken(f"stale reply {msg[:2]!r}"); return
                    b[4] = time.monotonic()
                    if msg[0] == "prog":
                        b[1].inbox.put(("prog", b[2], b[3], msg[2]))
                        continue
                    del self.busy[c]; self.free.append(c); b[1].inflight -= 1
                    b[1].inbox.put((msg[0], b[2], b[3], msg[2]))

_SCHED = _Sched()

_tab_lock, _tab_cache = threading.Lock(), {}

def _frame_table(path, nal="key"):
    """_index.frame_table_ex for `path` (demux only, discarded packets dropped, OpenCV's own timestamp rules; times,
    keyframes, which keyframes are IDR, and with nal="all" which H.264 pictures nobody references), cached per file;
    None if unavailable.  "token" names the file's content for the workers' own cache."""
    try:
        st = os.stat(path)
        key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, nal)
    except OSError:
        return None
    with _tab_lock:
        if key in _tab_cache:
            return _tab_cache[key]
    from . import _index
    try:
        tab = _index.frame_table_ex(path, nal=nal)
        if tab is not None:
            tab["token"] = "_".join(map(str, key))
    except Exception:
        tab = None
    with _tab_lock:
        if len(_tab_cache) > 64:
            _tab_cache.clear()
        _tab_cache[key] = tab
    return tab

def _table_memfd(tab):
    """The frame table as six .npy arrays in a memfd the workers read once per video (_skipdec.table)."""
    import numpy as np
    fd = os.memfd_create("m2_seg_tab", os.MFD_CLOEXEC)
    with os.fdopen(os.dup(fd), "wb") as fh:
        for arr in (np.asarray(tab["pts"], dtype=np.int64), np.asarray(tab["dts"], dtype=np.int64),
                    np.asarray(tab["times"], dtype=np.float64), np.frombuffer(tab["drop"], dtype=np.uint8),
                    np.asarray(tab["key_idx"], dtype=np.int64), np.asarray(tab["key_idr"], dtype=bool)):
            np.save(fh, arr, allow_pickle=False)
    return fd, f"/proc/{os.getpid()}/fd/{fd}"

def _serial_estimate(px):
    """Seconds vLLM's own decoder needs for `px` source pixels (n_src * w * h), only used to size the hard cap.  Speed
    6.4e9 px/s: R5 (35 min, 1280x720, 49,714 frames to grab) took 7.1 s with the API server pinned to node 0
    (runs/cpuprep/B_numa_pin_summary.txt)."""
    return px / _env_float("M2_SEG_SERIAL_PXPS", 6.4e9)

def _close_all():
    _SCHED.close()

atexit.register(_close_all)

def _set_status(**kw):
    st = {k: v for k, v in kw.items() if isinstance(v, (int, float, str, bool, type(None)))}
    st["path"] = "seg" if st.get("served") else ("gated" if st.get("gated") else "vllm:" + str(st.get("why")))
    _tls.status = st
    last.clear(); last.update(st)
    return st

def _close_map(mm):
    """mmap.close() raises BufferError while a view is still exported (a pipeline thread that has not let go yet); the
    mapping is then released by the garbage collector with the last view instead."""
    try:
        mm.close()
    except BufferError:
        pass

class _Buf:
    """Owner of a zerocopy frame buffer (b3 DE fix, 10-25).  The frames handed to vLLM are numpy views whose base is this
    object, not the mmap, so the thread that drops the last view (the API server's, between the processor and
    add_request: RECORD1 part 3, 7.0 ms per 100 MB) only runs the weakref.finalize callback; munmap and the memfd's
    shmem freeing then run on the _freer thread.  mmap.close() releases the GIL around munmap (CPython 3.12
    Modules/mmapmodule.c:206-212), so the freer does not stall Python threads either.  numpy takes the pointer through
    the buffer protocol (__buffer__, Python 3.12+) and keeps only a reference to this object."""
    __slots__ = ("mm", "__weakref__")
    def __init__(self, mm):
        self.mm = mm
    def __buffer__(self, flags):
        return memoryview(self.mm)
    def __release_buffer__(self, view):
        view.release()

_free_q, _free_lock, _free_pid = queue.SimpleQueue(), threading.Lock(), [None]
freed = [0, 0.0]          # buffers the freer closed, seconds it spent (offline tests read this)

def _freer():
    while True:
        mm = _free_q.get()
        t = time.perf_counter()
        _close_map(mm)
        freed[0] += 1; freed[1] += time.perf_counter() - t

def _own(mm):
    """Wrap a fresh frame mapping so its last view frees it off the dropping thread.  M2_SEG_FREE=sync keeps the old
    behaviour (the mapping dies with the last view, on that thread) for A/B timing."""
    if os.environ.get("M2_SEG_FREE", "bg") == "sync" or sys.version_info < (3, 12):
        return mm
    with _free_lock:
        if _free_pid[0] != os.getpid():                  # first buffer of this process (or of a forked child)
            threading.Thread(target=_freer, name="m2_seg_freer", daemon=True).start()
            _free_pid[0] = os.getpid()
    b = _Buf(mm)
    weakref.finalize(b, _free_q.put, mm).atexit = False   # the process exit frees the memory anyway
    return b

def _faults():
    """Test hook: M2_SEG_TEST_FAULT=hang:3,die:1,exc:0 makes those slices hang / kill their worker / raise (first
    attempt only)."""
    out = {}
    for item in os.environ.get("M2_SEG_TEST_FAULT", "").split(","):
        kind, _, k = item.partition(":")
        if kind in ("hang", "die", "exc") and k.isdigit():
            out[int(k)] = kind
    return out

def _schedule(req, base, parts, offs, uniq, fps, sink, faults, px, tab=None):
    """Submit this request's slice jobs to the shared pool, accept them by the chain check, re-run misaligned slices from
    their exact anchor, and switch to one vLLM-rule tail job (from the failing slice's anchor) at the first slice that hit
    a grab / retrieve failure.  A slice in which FFmpeg logged a decode error (damaged stream, PLAN.txt 17:46) is chained
    by timestamp only: error concealment fills damaged macroblocks from leftover decoder memory, so the same frame
    differs between workers.  Returns (n_valid, valid_indices, stats).  Raises _Broken (the pool was discarded) or _Fail;
    before _Fail leaves, the request's queued jobs are withdrawn and its running ones waited for (review N2), so no
    worker still writes into this request's buffer and no reply is left over."""
    import numpy as np
    K, T = len(parts), len(uniq)
    # Hard cap = backstop only (stuck workers are caught by the dispatcher's 30 s stall check): 20x the serial-decode
    # estimate, at least 300 s, times the number of requests sharing the pool (each gets about W/C workers).  The 16:05
    # fixed 30 s cap was hit by a normal 53-minute video.
    base_cap = _env_float("M2_SEG_TIMEOUT_S", 0) or max(300.0, 20.0 * _serial_estimate(px))
    t_start = time.monotonic()
    res, accepted = [None] * K, [False] * K
    out = {}                                       # (kind, k) submitted and not finished -> frames reported so far
    held = []                                      # the tail job, held until this request's other jobs are done
    exact_tried, restarted = set(), set()
    st = {"reruns": 0, "restarts": 0, "tail_from": None, "anchor_how": {}, "invalidated": 0, "decode_log": None,
          "corrupt_slices": [], "chain_ts_only": 0, "lead_frames": 0, "start_rejected": 0, "probe_log": None,
          "start_chosen": 0, "start_fallback": 0, "seek_overshoots": 0, "anchor_from_table": 0}
    times, keys = (tab["times"], tab["keys"]) if tab else (None, None)
    idr_at = dict(zip(keys, tab["key_idr"])) if tab else {}
    drop = np.frombuffer(tab["drop"], dtype=np.uint8) if tab and tab.get("drop") is not None else None
    kidx = np.asarray(tab["key_idx"]) if tab else None
    st.update(slices_skip=0, slices_opencv_none=0, skip_fail=0, skip_fail_why=None, pkts_skipped=0, pkts_sent=0,
              skip_withdrawn=0, idr_starts=0, skip_table_s=0.0, skip_open_s=0.0, skip_loop_s=0.0, skip_fail_damaged=0,
              skip_tries=None, skip_start_zero=0, slices_shifted=0)

    def keys_before(ms):
        """Keyframe times at or before ms, latest first (None without a table: the worker scans packets itself)."""
        if keys is None:
            return None
        return sorted((t for t in keys if t <= ms + 1e-3), reverse=True)
    tail = {"k": None, "res": None}

    def kf0_idr(kf):
        return bool(kf) and bool(idr_at.get(kf[0]))

    def exact_anchor(k, ts_only=False):
        p = res[k - 1]
        kf = keys_before(p["last_ms"])
        return {"idx": uniq[offs[k] - 1], "ms": p["last_ms"], "tol": 1e-3, "kf": kf, "kf0_idr": kf0_idr(kf),
                "hash": None if (ts_only or p["decode_log"]) else p["last_hash"]}

    def skip_flags(k, anchor):
        """b5: decode slice k with the skip loop iff its range (from the keyframe before its anchor to its last frame)
        holds pictures nobody references and nobody sampled."""
        if drop is None:
            return {}
        if (anchor is not None and not anchor.get("from_start") and anchor.get("ms") is not None
                and anchor["idx"] < len(times) and abs(times[anchor["idx"]] - anchor["ms"]) > 1e-3):
            # b5 04:55: the anchor vLLM counts as idx is not the table's frame idx: an earlier damaged GOP made the
            # decoder drop pictures, so from here on vLLM's count follows the decoder, not the packets.  The skip loop
            # numbers frames by the table, so this slice is the OpenCV path's (input-based choice, like opencv-none)
            st["slices_shifted"] += 1
            return {"skip": False}
        lo = 0
        if anchor is not None and not anchor.get("from_start"):
            kb = kidx[kidx <= anchor["idx"]]
            lo = int(kb[-2]) if len(kb) > 1 else 0
        hi = parts[k][-1]
        if "tab" not in base or not drop[lo:hi + 1].any():
            st["slices_opencv_none"] += 1
            return {"skip": False}
        st["slices_skip"] += 1
        return {"skip": True}

    def submit(kind, k, anchor):
        if kind == "tail" and out:
            held.append((kind, k, anchor)); return  # the tail rewrites the buffer from offs[k]: wait for the rest
        if kind == "slice":
            j = dict(base, want=parts[k], last=parts[k][-1], offset=offs[k], anchor=anchor, emulate=False,
                     want_last=k < K - 1, fault=faults.pop(k, None), **skip_flags(k, anchor))
        else:
            j = dict(base, want=uniq[offs[k]:], last=uniq[-1], offset=offs[k], anchor=anchor, emulate=True,
                     want_last=False, fault=None)
        out[(kind, k)] = 0
        _SCHED.submit(req, kind, k, j)

    def flush_held():
        if held and not out:
            submit(*held.pop())

    def invalidate(lo, hi):
        st["invalidated"] += 1
        if sink is not None:
            sink.invalidate_range(lo, hi)

    def advance():
        for k in range(K):
            if accepted[k]:
                continue
            if tail["k"] is not None or res[k] is None or (k > 0 and not accepted[k - 1]):
                return
            r = res[k]
            if k > 0 and r["decode_log"] and k not in restarted:
                # review N1: errors after a seek can come from the seek itself (a non-IDR start whose frames reference
                # pictures before it).  Decode the slice once more from frame 0, as vLLM does; only if FFmpeg still
                # reports errors is the stream itself damaged.
                restarted.add(k); st["restarts"] += 1
                res[k] = None; invalidate(offs[k], offs[k] + len(parts[k]))
                submit("slice", k, dict(exact_anchor(k), from_start=True))
                return
            if k > 0:
                p = res[k - 1]
                ts_only = bool(r["decode_log"] or p["decode_log"])
                same_ts = (r["anchor_found"] and r["anchor_ms"] is not None and p["last_ms"] is not None
                           and abs(r["anchor_ms"] - p["last_ms"]) <= 1e-3)
                if not (same_ts and (ts_only or r["anchor_hash"] == p["last_hash"])):
                    if k in exact_tried:
                        def brief(x):
                            return {q: x.get(q) for q in ("path", "how", "anchor_found", "anchor_ms", "last_ms", "lead_frames",
                                                          "skip_fail", "skip_tries")} | {"log": bool(x.get("decode_log"))}
                        raise _Fail(f"slice {k}: exact anchor {uniq[offs[k] - 1]} at {p['last_ms']} ms not found; "
                                    f"prev {brief(p)}; this {brief(r)}"[:900])
                    exact_tried.add(k); st["reruns"] += 1
                    res[k] = None; invalidate(offs[k], offs[k] + len(parts[k]))
                    submit("slice", k, exact_anchor(k))
                    return
                if ts_only:
                    st["chain_ts_only"] += 1
            if r["decode_log"]:
                st["corrupt_slices"].append(k)
                st["decode_log"] = st["decode_log"] or r["decode_log"][:120]
            if r["irregular"] or r["n"] != len(parts[k]):
                tail["k"] = k; st["tail_from"] = k
                for key in _SCHED.cancel(req, lambda x: x[0] == "slice" and x[1] >= k):   # superseded by the tail
                    out.pop(key, None)
                invalidate(offs[k], T)
                submit("tail", k, None if k == 0 else exact_anchor(k, ts_only=True))
                return
            accepted[k] = True

    def get():
        try:
            m = req.inbox.get(timeout=1.0)
        except queue.Empty:
            return None
        if m[0] == "broken":
            raise _Broken(m[3])
        return m

    def drain():
        for key in _SCHED.cancel(req, lambda x: True):
            out.pop(key, None)
        held.clear()
        while out:
            m = get()
            if m is not None and m[0] in ("done", "err"):
                out.pop((m[1], m[2]), None)

    for k in range(K):
        if k == 0:
            submit("slice", 0, None)
        else:
            idx = uniq[offs[k] - 1]
            if times is not None and idx < len(times):
                # main 21:03: the anchor's time comes from the frame table, not from index / average fps (on a
                # variable-frame-rate stream that estimate is off by more than a second)
                st["anchor_from_table"] += 1
                kf = keys_before(times[idx])
                submit("slice", k, {"idx": idx, "ms": times[idx], "tol": 1e-3, "kf": kf, "kf0_idr": kf0_idr(kf)})
            else:
                submit("slice", k, {"idx": idx, "ms": idx * 1000.0 / fps, "tol": 250.0 / fps})
    try:
        while out or held:
            cap = base_cap * max(1, len(_SCHED.active))
            if time.monotonic() - t_start > cap:
                raise _Fail(f"request over {cap:.0f} s")
            m = get()
            if m is None:
                continue
            typ, kind, k, payload = m
            key = (kind, k)
            superseded = kind == "slice" and tail["k"] is not None and k >= tail["k"]
            if typ == "prog" and payload == -1:
                # b5: the skip loop gave this job up; the frames it reported will be written again by OpenCV
                if key in out and out[key] > 0:
                    st["skip_withdrawn"] += 1
                    if sink is not None and not superseded:
                        sink.invalidate_range(offs[k], offs[k] + out[key])
                if key in out:
                    out[key] = 0
                continue
            if typ == "prog":
                if key in out:
                    if sink is not None and not superseded and payload > out[key]:
                        sink.ready_range(offs[k] + out[key], offs[k] + payload)
                    out[key] = max(out[key], payload)
                continue
            reported = out.pop(key, 0)
            if superseded:                             # review N6: its outcome no longer matters
                flush_held(); continue
            if typ == "err":
                raise _Fail(f"slice {k}: worker error {payload}")
            r = payload
            if r["error"]:
                raise _Fail(f"slice {k}: {r['error']}", vllm_fails=True)
            if sink is not None and r["n"] > reported:
                sink.ready_range(offs[k] + reported, offs[k] + r["n"])
            if r["how"]:
                st["anchor_how"][r["how"]] = st["anchor_how"].get(r["how"], 0) + 1
            st["lead_frames"] += r.get("lead_frames") or 0
            st["start_rejected"] += r.get("start_rejected") or 0
            st["start_chosen"] += bool(r.get("start_chosen"))
            st["start_fallback"] += bool(r.get("start_fallback"))
            st["seek_overshoots"] += r.get("seek_overshoots") or 0
            st["probe_log"] = st["probe_log"] or (r.get("probe_log") or "")[:120] or None
            st["pkts_skipped"] += r.get("skipped") or 0
            st["pkts_sent"] += r.get("sent") or 0
            st["idr_starts"] += r.get("how") == "idr"
            if r.get("path") == "skip" and r.get("skip_tries"):
                # a skip slice served after one or more starts were turned down: which and why (first one kept)
                st["skip_tries"] = st["skip_tries"] or f"slice {k}: {r['skip_tries']}"[:200]
                st["skip_start_zero"] += bool(r.get("start_fallback"))
            for kk in ("table_s", "open_s", "loop_s"):                    # summed over slices (worker CPU time scale)
                st["skip_" + kk] += (r.get("skip_t") or {}).get(kk, 0.0)
            if r.get("skip_fail"):
                st["skip_fail"] += 1
                st["skip_fail_why"] = st["skip_fail_why"] or f"slice {k}: {r['skip_fail']}"[:160]
                if r.get("skip_log") or r["decode_log"] or r["irregular"]:
                    st["skip_fail_damaged"] += 1         # the stream's damage (FFmpeg said so), not the plugin
                elif _strict():
                    # main 01:4x: switch on, a clean slice that should have skipped did not -> bug.  "Clean" = neither
                    # the skip attempt nor the OpenCV decode of the slice made FFmpeg report anything (b5 04:05: the
                    # first version looked at the OpenCV decode only; on corrupt_mid the skip lead-in crosses the
                    # damaged GOP while the OpenCV one, started elsewhere, does not)
                    raise _Fail(f"slice {k}: skip-nonref path failed on a clean slice: {r['skip_fail']}")
            if kind == "tail":
                tail["res"] = r
                if r["decode_log"]:
                    st["decode_log"] = st["decode_log"] or r["decode_log"][:120]
            else:
                res[k] = r
            advance()
            flush_held()
    except _Fail:
        drain()
        raise
    if tail["k"] is not None:
        k, r = tail["k"], tail["res"]
        if k > 0 and not r["anchor_found"]:
            raise _Fail(f"tail from slice {k}: anchor {uniq[offs[k] - 1]} not found")
        return offs[k] + r["n"], list(uniq[:offs[k]]) + list(r["valid"]), st
    if not all(accepted):
        raise _Fail(f"slices not accepted: {[k for k in range(K) if not accepted[k]]}")
    return T, list(uniq), st

def register():
    try:
        import numpy as np, cv2, vllm
        from vllm.multimodal.video_decoders import opencv as ocv
        from vllm.multimodal.media.video import VideoMediaIO
        from vllm.multimodal.video import VideoLoader
        if not vllm.__version__.startswith("0.31.0"):                     # review S7
            raise RuntimeError(f"written for vLLM 0.31.0, found {vllm.__version__}")
    except Exception as e:  # layout changed: leave everything untouched
        _log.warning("m2_seg_decode: not registered (%r)", e); return
    if getattr(ocv.decode_opencv, _MARK, False):
        return
    orig_decode = ocv.decode_opencv
    orig_load_file = VideoMediaIO.load_file
    base_read = VideoLoader.read_frames.__func__

    def load_file(self, filepath):
        _tls.path, _tls.status = str(filepath), None
        try:
            media = orig_load_file(self, filepath)
        finally:
            _tls.path = None
        st = getattr(_tls, "status", None)
        if st is not None:
            try: media.m2_seg_status = dict(st)
            except Exception: pass
        return media

    def fallback(why, loader_cls, data, target, sampling_kwargs, frame_recovery, t0, vllm_fails=False, **kw):
        """vLLM's own decoder for this request.  vllm_fails=True: vLLM raises here too, so this is not a fallback (its
        error is what the request gets with or without the plugin).  Otherwise M2_STRICT=1 raises instead."""
        if why != "off":
            _note(why)
            _set_status(served=False, why=why, **kw)
            if _strict() and not vllm_fails and not kw.get("gated"):
                raise M2StrictError(f"m2_seg_decode: {why}")
        out = orig_decode(loader_cls, data, target, sampling_kwargs, frame_recovery=frame_recovery)
        if why != "off":
            _set_status(served=False, why=why, total_s=round(time.perf_counter() - t0, 4), **kw)
        return out

    def decode_opencv(loader_cls, data, target, sampling_kwargs, *, frame_recovery=False):
        t0 = time.perf_counter()
        if not _on():
            _set_status(served=False, why="off")
            return orig_decode(loader_cls, data, target, sampling_kwargs, frame_recovery=frame_recovery)
        _SCHED.arrive(+1)
        entered = [False]
        try:
            return _decode_on(loader_cls, data, target, sampling_kwargs, frame_recovery, t0, entered)
        finally:
            if not entered[0]:
                _SCHED.arrive(-1)

    def _decode_on(loader_cls, data, target, sampling_kwargs, frame_recovery, t0, entered):
        args = (loader_cls, data, target, sampling_kwargs, frame_recovery, t0)
        if frame_recovery:
            return fallback("frame_recovery=True (vLLM's recovery scan is not reimplemented)", *args)
        if getattr(loader_cls.read_frames, "__func__", None) is not base_read:
            return fallback("loader overrides read_frames", *args)
        # same steps as decode_opencv (opencv.py:35-45), so source / frame_idx are exactly vLLM's
        cap = ocv.OpenCVVideoBackendMixin.open_video_capture(data)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ocv.check_frame_pixel_limit(w, h)
        fps = cap.get(cv2.CAP_PROP_FPS)
        vthreads = int(cap.get(cv2.CAP_PROP_N_THREADS)) or 0   # FFmpeg threads vLLM's own capture decodes with
        # b5: the rotation OpenCV applies to every retrieved frame (CAP_PROP_ORIENTATION_AUTO is on by default); the
        # skip loop converts frames itself, so it applies the same one
        rot = int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) if cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) else 0
        source = loader_cls._prepare_source(ocv.OpenCVVideoBackendMixin.get_video_metadata(cap))
        frame_idx = loader_cls.compute_frames_index_to_sample(source=source, target=target, **sampling_kwargs)
        uniq = sorted(set(frame_idx))                 # vLLM reads len(set(frame_idx)) frames in index order
        n_src = (uniq[-1] + 1) if uniq else 0
        gate = (f"below threshold ({n_src} frames to grab)" if n_src < _env_int("M2_SEG_MIN_SRC_FRAMES", 2642) else
                f"below threshold ({w}x{h})" if w * h < _env_int("M2_SEG_MIN_FRAME_PIXELS", 0) else None)
        why = ("no sampled frames" if not uniq else "no frame rate in the header" if not fps or fps <= 0 else None)
        if gate or why:
            # vLLM's own read on the same capture (what orig_decode would do next); no cap.release(): on a BytesIO
            # capture it segfaults (scripts/u5plan/repro_cap_release.py), vLLM never calls it either
            frames, valid = loader_cls.read_frames(cap, frame_idx, total_frames_num=source.total_frames_num,
                                                   frame_recovery=False)
            del cap
            _set_status(served=False, gated=gate is not None, why=gate or why, n_src=n_src,
                        total_s=round(time.perf_counter() - t0, 4))
            return frames, source, frame_idx, valid
        del cap
        px = float(n_src) * w * h
        req, why = _SCHED.enter(max(1, _env_int("M2_SEG_N", 16)))
        entered[0] = True                          # enter() took this request off the arriving count either way
        if req is None:
            return fallback(why, *args, n_src=n_src)
        try:
            return _split_decode(req, min(req.n, len(uniq)), data, uniq, frame_idx, source, w, h, fps, n_src, args,
                                 vthreads, px, rot)
        finally:
            _SCHED.leave(req)

    def _split_decode(req, n, data, uniq, frame_idx, source, w, h, fps, n_src, args, vthreads, px, rot=0):
        t0 = args[-1]
        mode = os.environ.get("M2_SEG_COPY", "zerocopy")
        threads = max(1, _env_int("M2_SEG_THREADS", 4))
        T = len(uniq)
        shape = (T, h, w, 3)
        size = T * h * w * 3
        parts = [p.tolist() for p in np.array_split(np.asarray(uniq), n) if len(p)]
        offs = np.cumsum([0] + [len(p) for p in parts[:-1]]).tolist()
        path, src_fd = getattr(_tls, "path", None), None
        if path is None or os.path.getsize(path) != len(data):
            src_fd = os.memfd_create("m2_seg_src", os.MFD_CLOEXEC)    # bytes input: give the workers a file
            os.write(src_fd, data)
            path = f"/proc/{os.getpid()}/fd/{src_fd}"
        pipe = sys.modules.get("m2_pipe_resize")
        sink = pipe.current_sink() if pipe is not None else None
        t1 = time.perf_counter()
        fd, reuse = None, None
        if mode == "reuse":
            reuse = _SCHED.reuse_buffer(size)        # None: another request holds it, use a buffer of our own
        if reuse is not None:
            buf_path, mm = reuse
        else:
            fd = os.memfd_create("m2_seg_frames", os.MFD_CLOEXEC)
            os.ftruncate(fd, size)
            mm = mmap.mmap(fd, size)
            buf_path = f"/proc/{os.getpid()}/fd/{fd}"
        view = np.ndarray(shape, dtype=np.uint8, buffer=mm if reuse is not None else _own(mm))
        if sink is not None:
            sink.begin(view)                     # m2_pipe_resize resizes each finished block while the rest decode
        base = {"path": path, "buf": buf_path, "shape": list(shape), "threads": threads, "vthreads": vthreads or threads,
                "keep": reuse is not None,
                "prog_every": sink.chunk if sink is not None else 0}
        skip_on = os.environ.get("M2_SEG_SKIP_NONREF") == "1"
        t_tab = time.perf_counter()
        tab = _frame_table(path, nal="all" if skip_on else "key")
        t_tab = time.perf_counter() - t_tab
        tab_fd = None
        if skip_on and tab is not None and tab.get("drop") is not None and any(tab["drop"]):
            tab_fd, base["tab"] = _table_memfd(tab)
            base.update(tab_token=tab["token"], rot=rot)
        failure, broken, vllm_fails = None, None, False
        try:
            n_valid, valid, sst = _schedule(req, base, parts, offs, uniq, fps, sink, _faults(), px, tab)
        except _Fail as e:
            failure, vllm_fails = str(e)[:200], e.vllm_fails
        except _Broken as e:                     # the dispatcher discarded the pool (stall, dead worker, stale reply)
            broken = str(e)[:200]
        finally:
            for x in (fd, src_fd, tab_fd):
                if x is not None:
                    os.close(x)                  # the mapping keeps the frame memory; workers have closed theirs
        t2 = time.perf_counter()
        if failure is not None or broken is not None:
            del view
            if sink is not None:
                sink.abort()                     # the fallback decode below begins a fresh pipeline on vLLM's frames
            if reuse is not None:
                _SCHED.reuse_release()
            else:
                _close_map(mm)
            why = ("pool broken: " + broken if broken else
                   ("vLLM's decoder fails here too: " if vllm_fails else "split decode failed: ") + failure)
            return fallback(why, *args[:-1], t0, vllm_fails=vllm_fails, n_src=n_src)
        if n_valid < T:
            _vlog.warning("Video loading completed with %d broken/unreadable frames. "
                          "Expected %d frames but only loaded %d frames.", T - n_valid, T, n_valid)
        frames = view[:n_valid] if n_valid < T else view
        if sink is not None and n_valid < T:
            sink.begin(frames); sink.ready(n_valid)          # fewer frames: the target size changes, start over
        if mode != "zerocopy":
            if sink is not None:
                sink.end(); sink.wait()          # the pipeline must be done with the buffer before it is reused / closed
            frames = frames.copy()
            if sink is not None:
                sink.rekey(frames)               # same bytes, new array: the precomputed result still applies
            del view
            if reuse is not None:
                _SCHED.reuse_release()
            else:
                _close_map(mm)
        t3 = time.perf_counter()
        _set_status(served=True, queue_s=round(req.first or 0.0, 4), why=None, mode=mode, n_src=n_src, n_frames=T, n_valid=n_valid, slices=len(parts),
                    threads=threads, mb=round(size / 2**20, 1), prep_s=round(t1 - t0, 4), seg_s=round(t2 - t1, 4),
                    copy_s=round(t3 - t2, 4), total_s=round(t3 - t0, 4), reruns=sst["reruns"],
                    tail_from=sst["tail_from"], anchor_how=repr(sst["anchor_how"]), invalidated=sst["invalidated"],
                    restarts=sst["restarts"], lead_frames=sst["lead_frames"], start_rejected=sst["start_rejected"],
                    start_chosen=sst["start_chosen"], start_fallback=sst["start_fallback"],
                    seek_overshoots=sst["seek_overshoots"], anchor_from_table=sst["anchor_from_table"],
                    probe_log=sst["probe_log"],
                    decode_log=sst["decode_log"], vthreads=vthreads, corrupt=bool(sst["corrupt_slices"]),
                    corrupt_slices=",".join(map(str, sst["corrupt_slices"])) or None, chain_ts_only=sst["chain_ts_only"],
                    bytes_input=src_fd is not None,
                    pool_start_s=round(req.pool_start_s, 4), sharing=len(_SCHED.active),
                    skip_nonref=skip_on, table_s=round(t_tab, 4), slices_skip=sst["slices_skip"],
                    slices_opencv_none=sst["slices_opencv_none"], skip_fail=sst["skip_fail"],
                    skip_fail_why=sst["skip_fail_why"], skip_fail_damaged=sst["skip_fail_damaged"],
                    skip_tries=sst["skip_tries"], skip_start_zero=sst["skip_start_zero"],
                    slices_shifted=sst["slices_shifted"],
                    pkts_skipped=sst["pkts_skipped"], pkts_sent=sst["pkts_sent"],
                    skip_withdrawn=sst["skip_withdrawn"], idr_starts=sst["idr_starts"],
                    decode_path=("off" if not skip_on else "skip" if not sst["slices_opencv_none"] and sst["slices_skip"]
                                 else "opencv-none" if not sst["slices_skip"] else "mixed"),
                    skip_table_s=round(sst["skip_table_s"], 4), skip_open_s=round(sst["skip_open_s"], 4),
                    skip_loop_s=round(sst["skip_loop_s"], 4))
        return frames, source, frame_idx, valid

    for f in (load_file, decode_opencv):
        setattr(f, _MARK, True)
    VideoMediaIO.load_file = load_file
    ocv.decode_opencv = decode_opencv
