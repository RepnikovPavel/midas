"""Incremental extraction of known zip members from an in-order byte stream.

The archive is processed as a sequence of *spans* (one per dataset sequence).
Chunks of a span arrive in order; the extractor parses local file headers and
inflates members as soon as their compressed bytes are complete — so unpacking
runs simultaneously with downloading.

Authoritative member sizes/method/crc come from the central directory, so local
headers with data descriptors (zero sizes) are handled transparently.
"""

import os
import struct
import zlib

LFH_SIG = b"PK\x03\x04"


class MemberWriteError(Exception):
    pass


class SpanExtractor:
    """Extract one contiguous span of members written to `out_root`.

    members: central-directory entries sorted by local header offset (absolute).
    span_start: absolute offset of the first byte fed to `feed`.
    strip_prefix: path prefix removed from member names before writing.
    """

    def __init__(self, members, span_start, out_root, strip_prefix="pandaset/"):
        self.members = members
        self.pos = span_start          # absolute stream position of buffer[0]
        self.buf = bytearray()
        self.out_root = out_root
        self.strip_prefix = strip_prefix
        self.idx = 0                   # next member to extract
        self.written = 0
        self.skipped = 0
        self.bytes_written = 0

    # -- internal ---------------------------------------------------------
    def _member_paths(self, m):
        name = m["name"]
        if self.strip_prefix and name.startswith(self.strip_prefix):
            name = name[len(self.strip_prefix):]
        # sanitize: no absolute paths / traversal
        name = name.lstrip("/")
        parts = [p for p in name.split("/") if p not in ("", ".", "..")]
        return os.path.join(self.out_root, *parts) if parts else None

    def _try_extract_one(self):
        if self.idx >= len(self.members):
            return False
        m = self.members[self.idx]
        rel = m["lho"] - self.pos
        if rel < 0:
            raise MemberWriteError(f"stream passed member {m['name']} before extraction")
        if len(self.buf) < rel + 30:
            return False
        hdr = bytes(self.buf[rel:rel + 30])
        if hdr[:4] != LFH_SIG:
            raise MemberWriteError(f"bad local header for {m['name']} at {m['lho']}")
        nlen, elen = struct.unpack_from("<HH", hdr, 26)
        data_start = m["lho"] + 30 + nlen + elen
        data_rel = data_start - self.pos
        csize = m["csize"]
        if len(self.buf) < data_rel + csize:
            return False
        payload = bytes(self.buf[data_rel:data_rel + csize])
        consume = data_rel + csize

        out_path = self._member_paths(m)
        is_dir = m["name"].endswith("/") or (csize == 0 and m["usize"] == 0)
        if out_path is not None and not is_dir:
            self._write_member(m, payload, out_path)
        elif out_path is not None and is_dir:
            os.makedirs(out_path, exist_ok=True)

        del self.buf[:consume]
        self.pos += consume
        self.idx += 1
        return True

    def _write_member(self, m, payload, out_path):
        if m["method"] == 0:
            raw = payload
        elif m["method"] == 8:
            raw = zlib.decompress(payload, -15)
        else:
            raise MemberWriteError(f"unsupported method {m['method']} for {m['name']}")
        if len(raw) != m["usize"]:
            raise MemberWriteError(f"size mismatch for {m['name']}: {len(raw)} != {m['usize']}")
        if (zlib.crc32(raw) & 0xFFFFFFFF) != m["crc"]:
            raise MemberWriteError(f"crc mismatch for {m['name']}")

        if os.path.exists(out_path) and os.path.getsize(out_path) == m["usize"]:
            # resume: verify cheap crc again only if asked; size match is enough here
            self.skipped += 1
            return
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".part"
        with open(tmp, "wb") as f:
            f.write(raw)
        os.replace(tmp, out_path)
        self.written += 1
        self.bytes_written += len(raw)

    # -- public ------------------------------------------------------------
    def feed(self, data):
        """Feed the next contiguous chunk of the span (in order)."""
        self.buf += data
        n = 0
        while self._try_extract_one():
            n += 1
        return n

    def finish(self):
        """All bytes fed; drain and verify completion."""
        while self._try_extract_one():
            pass
        if self.idx != len(self.members):
            m = self.members[self.idx]
            raise MemberWriteError(f"span incomplete: stuck at member {m['name']} ({self.idx}/{len(self.members)})")
        return self.written, self.skipped, self.bytes_written
