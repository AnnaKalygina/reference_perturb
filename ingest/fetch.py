"""Parallel, resumable downloads with receipts: several files at once, each in HTTP range parts."""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from config import CONFIG, load, path  # noqa: E402
from receipts import raw_receipt, raw_receipt_path, write_atomic  # noqa: E402


def curl(args, proxy, timeout=None):
    """Run curl with the proxy and fail loudly."""
    cmd = ["curl", "-sS", "-L", "--retry", "5", "--retry-all-errors", "--connect-timeout", "30"] + (["-x", proxy] if proxy else []) + args
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)


def remote_size(url, proxy):
    """(bytes, accepts ranges) from a HEAD request."""
    out = curl(["-I", url], proxy).stdout.lower()
    size = next((int(l.split(":")[1]) for l in out.splitlines() if l.startswith("content-length:")), None)
    return size, "accept-ranges: bytes" in out


def fetch_file(job, cfg, proxy, parts=8, min_split=64 << 20):
    """Download one file into raw/<dataset>/ unless its receipt exists; returns the receipt."""
    dataset, url = job["dataset"], job["url"]
    name = job.get("name") or url.rsplit("/", 1)[-1]
    dest = path(cfg, "raw", dataset, name)
    if raw_receipt_path(cfg, dataset, dest).exists() and dest.exists():
        return raw_receipt(cfg, dataset, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    size, ranges = remote_size(url, proxy)
    started = time.time()
    if size and ranges and size > min_split:
        bounds = [(i * size // parts, (i + 1) * size // parts - 1) for i in range(parts)]
        part_paths = [dest.with_name(f".{name}.part{i}") for i in range(parts)]
        with ThreadPoolExecutor(parts) as pool:
            list(pool.map(lambda t: fetch_range(url, proxy, *t), zip(part_paths, bounds)))
        with write_atomic(dest) as tmp, open(tmp, "wb") as out:
            for p in part_paths:
                with open(p, "rb") as fh:
                    while block := fh.read(1 << 24):
                        out.write(block)
        for p in part_paths:
            p.unlink()
    else:
        with write_atomic(dest) as tmp:
            curl(["-o", str(tmp), "-C", "-", url], proxy)
    if size and dest.stat().st_size != size:
        dest.unlink()
        raise RuntimeError(f"{dest}: {dest.stat().st_size} bytes, expected {size}")
    rec = raw_receipt(cfg, dataset, dest, url=url, upstream_md5=job.get("md5"))
    rec["seconds"] = round(time.time() - started, 1)
    return rec


def fetch_range(url, proxy, part_path, bounds, attempts=6):
    """One byte range into a part file; a complete part is kept, a short one (server error page) is retried."""
    lo, hi = bounds
    want = hi - lo + 1
    for attempt in range(attempts):
        if part_path.exists() and part_path.stat().st_size == want:
            return
        try:
            curl(["-o", str(part_path), "-r", f"{lo}-{hi}", url], proxy)
        except subprocess.CalledProcessError:
            pass
        if part_path.exists() and part_path.stat().st_size != want:
            part_path.unlink()
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{part_path}: no complete part after {attempts} attempts")


def jobs(selection):
    """Download jobs from config/downloads.yaml, filtered by dataset names (empty = all)."""
    spec = yaml.safe_load(open(CONFIG / "downloads.yaml"))
    out = []
    for dataset, files in spec.items():
        if selection and dataset not in selection:
            continue
        for f in files:
            out.append(dict(dataset=dataset, **f))
    return out


def main():
    """Fetch the selected datasets with up to --files downloads at once."""
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--files", type=int, default=6)
    ap.add_argument("--no-proxy", action="store_true")
    a = ap.parse_args()
    cfg = load()
    proxy = None if a.no_proxy else cfg["fetch"]["proxy"]
    todo = jobs(a.datasets)
    print(f"{len(todo)} files, {a.files} at once, proxy {proxy}", flush=True)
    with ThreadPoolExecutor(a.files) as pool:
        futures = {pool.submit(fetch_file, j, cfg, proxy): j for j in todo}
        for fut in as_completed(futures):
            j = futures[fut]
            try:
                rec = fut.result()
                print(f"done {j['dataset']} {rec['path']} {rec['bytes'] / 1e9:.2f} GB {rec.get('seconds', 0)} s", flush=True)
            except Exception as e:
                print(f"FAILED {j['dataset']} {j['url']}: {e}", flush=True)


if __name__ == "__main__":
    main()
