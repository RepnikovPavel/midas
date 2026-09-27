"""CDN IP pool with liveness probing + TLS-to-IP fetching with correct SNI.

The route to HF's Xet CDN is interfered with at IP level: some A records are
poisoned/throttled, and the set changes with time. This module keeps a pool of
currently-good IPs (probed every `refresh_s` seconds) and fetches byte ranges
by opening TLS directly to a chosen IP while presenting the real hostname as
TLS SNI and HTTP Host (so certificates and signed URLs stay valid).
"""

import http.client
import random
import socket
import ssl
import threading
import time


class HttpStatusError(IOError):
    def __init__(self, status, ip=None):
        super().__init__(f"status {status}" + (f" from ip {ip}" if ip else ""))
        self.status = status
        self.ip = ip


class _IPSNIConnection(http.client.HTTPSConnection):
    """HTTPS connection to a literal IP, but TLS SNI/verification for sni_host."""

    def __init__(self, ip, sni_host, timeout, context):
        super().__init__(ip, 443, timeout=timeout, context=context)
        self._sni_host = sni_host

    def connect(self):
        sock = socket.create_connection((self.host, self.port), self.timeout,
                                        self.source_address)
        try:
            self.sock = self._context.wrap_socket(sock, server_hostname=self._sni_host)
        except Exception:
            sock.close()
            raise


class CDNIPPool:
    def __init__(self, host, refresh_s=45, probe_timeout=7, extra_headers=None):
        self.host = host
        self.refresh_s = refresh_s
        self.probe_timeout = probe_timeout
        self.extra_headers = extra_headers or {}
        self._ips = []          # current good IPs, best first
        self._lock = threading.Lock()
        self._refresh_event = threading.Event()
        self._stop = False
        self._ssl = ssl.create_default_context()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._path_query_for_probe = None

    # -- probing -----------------------------------------------------------
    def _resolve_ips(self):
        try:
            infos = socket.getaddrinfo(self.host, 443, socket.AF_INET, socket.SOCK_STREAM)
            return sorted({i[4][0] for i in infos})
        except OSError:
            return []

    def _probe(self, ip, path_query):
        n = 512 * 1024
        try:
            t0 = time.time()
            data = self.fetch_range(ip, path_query, 0, n - 1, timeout=self.probe_timeout)
            dt = time.time() - t0
            if len(data) != n:
                return None
            return n / max(dt, 1e-6)
        except Exception:  # noqa: BLE001
            return None

    def refresh(self, path_query):
        ips = self._resolve_ips()
        if not ips:
            return list(self.ips)
        results = []
        lock = threading.Lock()

        def work(ip):
            s = self._probe(ip, path_query)
            with lock:
                results.append((s or 0.0, ip))

        ths = [threading.Thread(target=work, args=(ip,)) for ip in ips]
        for t in ths:
            t.start()
        for t in ths:
            t.join(timeout=self.probe_timeout + 3)
        good = sorted((r for r in results if r[0] > 0), reverse=True)
        with self._lock:
            self._ips = [ip for _, ip in good]
        return list(self._ips)

    def _loop(self):
        while not self._stop:
            self._refresh_event.wait(self.refresh_s)
            self._refresh_event.clear()
            if self._stop:
                break
            if self._path_query_for_probe:
                try:
                    self.refresh(self._path_query_for_probe)
                except Exception:  # noqa: BLE001
                    pass

    def start(self, path_query_for_probe):
        self._path_query_for_probe = path_query_for_probe
        try:
            self.refresh(path_query_for_probe)
        except Exception:  # noqa: BLE001
            pass
        self._thread.start()
        return self

    def poke(self):
        self._refresh_event.set()

    def stop(self):
        self._stop = True
        self._refresh_event.set()

    @property
    def ips(self):
        with self._lock:
            return list(self._ips)

    def pick(self):
        ips = self.ips
        if not ips:
            return None
        n = min(len(ips), 4)
        return ips[random.randrange(n)]

    def report_bad(self, ip):
        with self._lock:
            if ip in self._ips:
                self._ips.remove(ip)
        self.poke()

    def probe_size(self, path_query, timeout=20):
        """Total content size via a 1-byte range through the pool (Content-Range)."""
        last = None
        for _ in range(max(2, len(self.ips))):
            ip = self.pick()
            if ip is None:
                break
            conn = None
            try:
                conn = _IPSNIConnection(ip, self.host, timeout, self._ssl)
                conn.putrequest("GET", path_query, skip_host=True)
                conn.putheader("Host", self.host)
                conn.putheader("Range", "bytes=0-0")
                conn.putheader("Connection", "close")
                for k, v in self.extra_headers.items():
                    conn.putheader(k, v)
                conn.endheaders()
                r = conn.getresponse()
                cr = r.headers.get("Content-Range")
                status = r.status
                conn.close()
                if status == 206 and cr:
                    return int(cr.split("/")[-1])
                last = IOError(f"ip {ip}: status {status}")
            except Exception as e:  # noqa: BLE001
                last = e
                if conn:
                    conn.close()
                self.report_bad(ip)
        raise IOError(f"probe_size failed: {last}")

    # -- low-level fetch -----------------------------------------------------
    def fetch_range(self, ip, path_query, start, end, timeout=30):
        """GET [start,end] of path_query via TLS to `ip` (SNI/Host = real host)."""
        conn = _IPSNIConnection(ip, self.host, timeout, self._ssl)
        conn.putrequest("GET", path_query, skip_host=True)
        conn.putheader("Host", self.host)
        conn.putheader("Range", f"bytes={start}-{end}")
        conn.putheader("Accept-Encoding", "identity")
        conn.putheader("Connection", "close")
        for k, v in self.extra_headers.items():
            conn.putheader(k, v)
        conn.endheaders()
        r = conn.getresponse()
        if r.status != 206:
            status = r.status
            conn.close()
            raise HttpStatusError(status, ip)
        want = end - start + 1
        chunks = []
        got = 0
        while got < want:
            c = r.read(min(1 << 20, want - got))
            if not c:
                break
            chunks.append(c)
            got += len(c)
        conn.close()
        if got != want:
            raise IOError(f"ip {ip}: short read {got}/{want}")
        return b"".join(chunks)
