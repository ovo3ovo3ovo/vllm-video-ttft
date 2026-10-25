"""Frame-time table for m2_seg_decode, read with the libavformat that OpenCV itself ships and uses (main 21:12).

vLLM numbers frames by decoding from frame 0, so frame i is the i-th picture the decoder returns, and the worker finds a
frame by its CAP_PROP_POS_MSEC.  This module reproduces those numbers and times without decoding: it demuxes the file
(av_read_frame), drops packets the demuxer marks AV_PKT_FLAG_DISCARD (mov edit lists hide leading samples this way,
libavformat/mov.c:11313; the decoder then never outputs them, libavcodec/decode.c:1559), sorts the rest by pts and
converts each pts to milliseconds exactly as OpenCV 5.0.0 does for decoded frames (modules/videoio/src/
cap_ffmpeg_impl.hpp:1745-1803 timestamp normalization, :2260-2268 dts_to_sec).  OpenCV's own raw-packet mode cannot be
used for this: it exposes no discard flag, and it normalizes against the first packet instead of the first decoded
picture, so on an edit-listed file its clock is shifted (editlist_cut: +208.5 ms, 5 hidden packets).
The structs below mirror FFmpeg 8.1.1 headers (libavformat/avformat.h, libavcodec/packet.h, libavcodec/codec_par.h); the
layout is only trusted for the exact library build it was checked against (avformat_version() == _CHECKED_VERSION),
anything else -> None.
b5 (10-25): the other streams are set to AVDISCARD_ALL after avformat_find_stream_info, so the scan reads video packets
only (audio was ~60% of the packets on the 53-minute videos; checked: the video packets, their pts / dts / flags and
order are unchanged, scripts/u5plan/b5/probe_tabcost.py).  frame_table_ex also reads the NAL unit headers of H.264 /
HEVC packets (length-prefixed avcC / hvcC only): which keyframes are IDR pictures (the IDR start, see _worker._locate)
and, with nal="all", which H.264 packets nothing references (the skip-nonref path, M2_SEG_SKIP_NONREF)."""
import ctypes, glob, os
_CHECKED_VERSION = (62 << 16) | (12 << 8) | 101       # opencv-python-headless 5.0.0.93: libavformat 62.12.101
_NOPTS = -0x8000000000000000
_KEY, _DISCARD = 0x0001, 0x0004
_AVTB = 1000000
_AVDISCARD_ALL = 48                                   # libavcodec/defs.h
_H264, _HEVC = 27, 173                                # libavcodec/codec_id.h
# H.264 NAL types a droppable packet may hold: non-IDR slice (1), access unit delimiter (9), filler (12).  Anything else
# (IDR 5, SEI 6, SPS / PPS 7 / 8, end of sequence 10, ...) changes decoder state or output timing, so such a packet is
# always sent (b5 census over 117 files: 668 of 374,535 non-reference packets carry an end-of-sequence NAL, no SEI)
_H264_DROP_OK = frozenset((1, 9, 12))
_u8 = ctypes.c_uint8.from_address

class _Rat(ctypes.Structure):
    _fields_ = [("num", ctypes.c_int), ("den", ctypes.c_int)]

class _Stream(ctypes.Structure):          # AVStream, leading fields
    _fields_ = [("av_class", ctypes.c_void_p), ("index", ctypes.c_int), ("id", ctypes.c_int),
                ("codecpar", ctypes.c_void_p), ("priv_data", ctypes.c_void_p), ("time_base", _Rat),
                ("start_time", ctypes.c_int64), ("duration", ctypes.c_int64), ("nb_frames", ctypes.c_int64),
                ("disposition", ctypes.c_int), ("discard", ctypes.c_int)]

class _Par(ctypes.Structure):             # AVCodecParameters, leading fields
    _fields_ = [("codec_type", ctypes.c_int), ("codec_id", ctypes.c_int), ("codec_tag", ctypes.c_uint32),
                ("extradata", ctypes.POINTER(ctypes.c_uint8)), ("extradata_size", ctypes.c_int),
                ("coded_side_data", ctypes.c_void_p), ("nb_coded_side_data", ctypes.c_int), ("format", ctypes.c_int),
                ("bit_rate", ctypes.c_int64), ("bits_per_coded_sample", ctypes.c_int),
                ("bits_per_raw_sample", ctypes.c_int), ("profile", ctypes.c_int), ("level", ctypes.c_int),
                ("width", ctypes.c_int), ("height", ctypes.c_int)]

class _Fmt(ctypes.Structure):             # AVFormatContext, leading fields
    _fields_ = [("av_class", ctypes.c_void_p), ("iformat", ctypes.c_void_p), ("oformat", ctypes.c_void_p),
                ("priv_data", ctypes.c_void_p), ("pb", ctypes.c_void_p), ("ctx_flags", ctypes.c_int),
                ("nb_streams", ctypes.c_uint), ("streams", ctypes.POINTER(ctypes.POINTER(_Stream))),
                ("nb_stream_groups", ctypes.c_uint), ("stream_groups", ctypes.c_void_p),
                ("nb_chapters", ctypes.c_uint), ("chapters", ctypes.c_void_p), ("url", ctypes.c_char_p),
                ("start_time", ctypes.c_int64)]

class _Pkt(ctypes.Structure):             # AVPacket, leading fields
    _fields_ = [("buf", ctypes.c_void_p), ("pts", ctypes.c_int64), ("dts", ctypes.c_int64), ("data", ctypes.c_void_p),
                ("size", ctypes.c_int), ("stream_index", ctypes.c_int), ("flags", ctypes.c_int)]

_libs = None

def _load():
    global _libs
    if _libs is None:
        import cv2
        d = os.path.join(os.path.dirname(cv2.__file__), "..", "opencv_python_headless.libs")
        def one(stem):
            hits = sorted(glob.glob(os.path.join(d, f"lib{stem}-*.so*")))
            return ctypes.CDLL(hits[0]) if hits else None
        util, fmt = one("avutil"), one("avformat")
        if util is None or fmt is None or fmt.avformat_version() != _CHECKED_VERSION:
            _libs = False
        else:
            fmt.avformat_open_input.argtypes = [ctypes.POINTER(ctypes.POINTER(_Fmt)), ctypes.c_char_p, ctypes.c_void_p,
                                                ctypes.c_void_p]
            fmt.avformat_find_stream_info.argtypes = [ctypes.POINTER(_Fmt), ctypes.c_void_p]
            fmt.av_find_best_stream.argtypes = [ctypes.POINTER(_Fmt), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                ctypes.c_void_p, ctypes.c_int]
            fmt.av_read_frame.argtypes = [ctypes.POINTER(_Fmt), ctypes.POINTER(_Pkt)]
            fmt.avformat_close_input.argtypes = [ctypes.POINTER(ctypes.POINTER(_Fmt))]
            fmt.avformat_close_input.restype = None
            codec = one("avcodec")
            codec.av_packet_alloc.restype = ctypes.POINTER(_Pkt)
            codec.av_packet_unref.argtypes = [ctypes.POINTER(_Pkt)]
            codec.av_packet_unref.restype = None
            codec.av_packet_free.argtypes = [ctypes.POINTER(ctypes.POINTER(_Pkt))]
            codec.av_packet_free.restype = None
            util.av_rescale_q.argtypes = [ctypes.c_int64, _Rat, _Rat]
            util.av_rescale_q.restype = ctypes.c_int64
            _libs = (util, codec, fmt)
    return _libs or None

def discard_other_streams(ic, vs):
    """AVDISCARD_ALL on every stream but `vs` (call after avformat_find_stream_info): av_read_frame skips them."""
    for i in range(ic.contents.nb_streams):
        if i != vs:
            ic.contents.streams[i].contents.discard = _AVDISCARD_ALL

def nal_format(par):
    """("h264" | "hevc", NAL length size) for length-prefixed H.264 / HEVC, else (None, 0)."""
    n = par.extradata_size
    if par.codec_id == _H264 and n > 4 and par.extradata[0] == 1:
        return "h264", (par.extradata[4] & 3) + 1
    if par.codec_id == _HEVC and n > 22 and par.extradata[0] == 1:
        return "hevc", (par.extradata[21] & 3) + 1
    return None, 0

def nal_types(ptr, size, nl):
    """First header byte of every NAL unit in a length-prefixed packet."""
    out, i, sa = [], 0, ctypes.string_at
    while i + nl < size:
        h = sa(ptr + i, nl + 1)
        out.append(h[nl])
        i += nl + max(int.from_bytes(h[:nl], "big"), 1)
    return out

def frame_table(path):
    """(times_ms, key_times_ms) in vLLM's frame numbering: times_ms[i] is the CAP_PROP_POS_MSEC OpenCV reports for the
    i-th decoded frame, key_times_ms the times of the keyframes among them, both ascending.  None if the library build
    is not the checked one or the file cannot be demuxed."""
    t = frame_table_ex(path, nal="none")
    return None if t is None else (t["times"], t["keys"])

def frame_table_ex(path, nal="key"):
    """frame_table plus, per frame i (vLLM's numbering): pts[i] and dts[i] (stream time base; pts is what the decoder
    returns as AVFrame.pts, dts is what av_seek_frame looks up), and per keyframe: key_idx (its frame index) and key_idr (an IDR picture: H.264 NAL type 5, HEVC 19 /
    20).  nal="all" also fills drop[i] = 1 for H.264 packets nothing references (every slice NAL has nal_ref_idc 0, no
    NAL other than _H264_DROP_OK, not AV_PKT_FLAG_DISCARD).  nal="key" parses keyframe packets only, nal="none" none.
    codec: "h264" / "hevc" / None (no NAL parsing).  None as frame_table."""
    libs = _load()
    if libs is None:
        return None
    util, codec, fmt = libs
    ic = ctypes.POINTER(_Fmt)()
    if fmt.avformat_open_input(ctypes.byref(ic), os.fsencode(path), None, None) < 0:
        return None
    pkt = codec.av_packet_alloc()
    try:
        if fmt.avformat_find_stream_info(ic, None) < 0:
            return None
        vs = fmt.av_find_best_stream(ic, 0, -1, -1, None, 0)            # AVMEDIA_TYPE_VIDEO
        if vs < 0 or vs >= ic.contents.nb_streams:
            return None
        st = ic.contents.streams[vs].contents
        if st.index != vs or st.time_base.den <= 0 or ctypes.c_int.from_address(st.codecpar).value != 0:
            return None                                                   # layout check: index and codec_type
        discard_other_streams(ic, vs)
        kind, nl = nal_format(ctypes.cast(st.codecpar, ctypes.POINTER(_Par)).contents) if nal != "none" else (None, 0)
        every = nal == "all" and kind == "h264"
        tb, avtb = _Rat(st.time_base.num, st.time_base.den), _Rat(1, _AVTB)
        pts = []                                  # (pts, key, idr, drop, dts)
        while fmt.av_read_frame(ic, pkt) >= 0:
            p = pkt.contents
            if p.stream_index == vs and not (p.flags & _DISCARD):
                t = p.pts if p.pts != _NOPTS else p.dts
                if t != _NOPTS:
                    key, idr, drop = bool(p.flags & _KEY), False, False
                    if every and not key and p.size > nl:
                        # one byte first: a reference slice as the first NAL (every packet of an I/P-only stream)
                        # decides it without walking the packet (b5 parity: 60k packets, 0.14 s -> see plugin_segdecode)
                        b0 = _u8(p.data + nl).value
                        if (b0 & 31) in (1, 5) and b0 & 0x60:
                            pts.append((t, key, False, False, p.dts if p.dts != _NOPTS else t))
                            codec.av_packet_unref(pkt)
                            continue
                    if kind is not None and (key or every):
                        hd = nal_types(p.data, p.size, nl)
                        if kind == "h264":
                            idr = any(b & 31 == 5 for b in hd)
                            if every and not key:
                                ty = {b & 31 for b in hd}
                                drop = (1 in ty and ty <= _H264_DROP_OK
                                        and all((b >> 5) & 3 == 0 for b in hd if b & 31 == 1))
                        else:
                            idr = any((b >> 1) & 63 in (19, 20) for b in hd)
                    pts.append((t, key, idr, drop, p.dts if p.dts != _NOPTS else t))
            codec.av_packet_unref(pkt)
        if not pts:
            return {"times": [], "keys": [], "pts": [], "dts": [], "key_idx": [], "key_idr": [], "drop": None, "codec": kind}
        # cap_ffmpeg_impl.hpp:1760-1803: offset decided on the first decoded picture (= smallest pts)
        cands = []
        if ic.contents.start_time != _NOPTS:
            cands.append(ic.contents.start_time)
        if st.start_time != _NOPTS:
            cands.append(util.av_rescale_q(st.start_time, tb, avtb))
        cands.append(util.av_rescale_q(min(x[0] for x in pts), tb, avtb))
        m = min(cands)
        off = -m if m < 0 else 0
        r2d = st.time_base.num / float(st.time_base.den)
        def ms(t):
            if off:
                t = util.av_rescale_q(util.av_rescale_q(t, tb, avtb) + off, avtb, tb)
            elif st.start_time != _NOPTS:
                t -= st.start_time                                        # dts_to_sec, :2265-2266
            return t * r2d * 1000
        pts.sort(key=lambda x: x[0])
        times = [ms(x[0]) for x in pts]
        ki = [i for i, x in enumerate(pts) if x[1]]
        return {"times": times, "keys": [times[i] for i in ki], "pts": [x[0] for x in pts], "dts": [x[4] for x in pts],
                "key_idx": ki,
                "key_idr": [pts[i][2] for i in ki], "codec": kind,
                "drop": bytes(x[3] for x in pts) if every else None}
    finally:
        codec.av_packet_free(ctypes.byref(pkt))
        fmt.avformat_close_input(ctypes.byref(ic))
