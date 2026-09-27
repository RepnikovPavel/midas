"""CLI: python -m pandaset_pipe.download [options]

Runs inside the pipeline container on the server.
Defaults target the two-disk server layout.
"""

import argparse
import os

from .downloader import run

DEFAULT_URL = "https://huggingface.co/datasets/georghess/pandaset/resolve/main/pandaset.zip"


def main():
    import faulthandler
    import signal
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1)  # docker kill -s USR1 -> stack dump to logs

    p = argparse.ArgumentParser(description="PandaSet balanced streaming downloader")
    p.add_argument("--url", default=os.environ.get("PANDASET_URL", DEFAULT_URL))
    p.add_argument("--disk1-raw", default="/mnt/hdd1/datasets/pandaset_raw")
    p.add_argument("--disk1-npz", default="/mnt/hdd1/datasets/pandaset_npz")
    p.add_argument("--disk2-raw", default="/mnt/hdd2/datasets/pandaset_raw")
    p.add_argument("--disk2-npz", default="/mnt/hdd2/datasets/pandaset_npz")
    p.add_argument("--workdir", default="/work")
    p.add_argument("--streams", type=int, default=16)
    p.add_argument("--chunk-mb", type=int, default=16)
    p.add_argument("--converters", type=int, default=2)
    p.add_argument("--proxy", default=os.environ.get("PANDASET_PROXY"),
                   help="e.g. socks5h://127.0.0.1:1081 (byeDPI sidecar)")
    p.add_argument("--no-convert", action="store_true")
    p.add_argument("--only", nargs="*", default=None,
                   help="restrict to sequence ids, e.g. --only 001 002")
    args = p.parse_args()

    token = os.environ.get("PANDASET_AUTH_HEADER") or None
    if not token and os.environ.get("HF_TOKEN"):
        token = f"Bearer {os.environ['HF_TOKEN']}"

    disks = [
        {"id": "hdd1", "raw": args.disk1_raw, "npz": args.disk1_npz},
        {"id": "hdd2", "raw": args.disk2_raw, "npz": args.disk2_npz},
    ]
    for d in disks:
        os.makedirs(d["raw"], exist_ok=True)
        os.makedirs(d["npz"], exist_ok=True)

    rc = run(args.url, disks, args.workdir,
             streams=args.streams, chunk_mb=args.chunk_mb,
             auth_header=token, proxy=args.proxy,
             do_convert=not args.no_convert,
             converters=args.converters, only_seqs=args.only)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
