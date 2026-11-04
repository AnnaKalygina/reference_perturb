"""Receipts for raw inputs and outputs, atomic writes, and the done() rule."""
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from config import path

VCC_PROVENANCE = Path("/cluster/work/boeva/VCC/VCC2026/data/provenance")


def sha256(p, chunk=1 << 24):
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@contextmanager
def write_atomic(target):
    """Yield a temporary path in the target's directory; rename it over the target on success."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.part")
    try:
        yield tmp
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json(target, record):
    """Atomic JSON write."""
    with write_atomic(target) as tmp:
        tmp.write_text(json.dumps(record, indent=1, sort_keys=True, default=str))


def vcc2026_receipt(raw_path):
    """VCC2026's provenance receipt for an absolute raw path, or None."""
    key = hashlib.sha256(str(Path(raw_path).resolve()).encode()).hexdigest()
    p = VCC_PROVENANCE / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def raw_receipt_path(cfg, dataset, raw_path):
    """Where the raw receipt of one file lives."""
    return path(cfg, "receipts", "raw", dataset, Path(raw_path).name + ".json")


def raw_receipt(cfg, dataset, raw_path, url=None, upstream_md5=None, derived_from=None, rehash=False):
    """Receipt for one raw file: size always checked, sha256 reused from VCC2026 when present, else computed."""
    raw_path = Path(raw_path)
    target = raw_receipt_path(cfg, dataset, raw_path)
    size = raw_path.stat().st_size
    if target.exists() and not rehash:
        rec = json.loads(target.read_text())
        if rec["bytes"] != size:
            raise SystemExit(f"{raw_path}: size {size} differs from its receipt ({rec['bytes']})")
        return rec
    rec = dict(path=str(raw_path), bytes=size, url=url, upstream_md5=upstream_md5, derived_from=derived_from, created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    vcc = None if rehash else vcc2026_receipt(raw_path)
    if vcc and vcc.get("bytes") == size and vcc.get("sha256"):
        rec.update(sha256=vcc["sha256"], hashed_by="vcc2026", upstream_md5=upstream_md5 or vcc.get("upstream_md5"), url=url or vcc.get("url"))
    else:
        rec.update(sha256=sha256(raw_path), hashed_by="ours")
    write_json(target, rec)
    return rec


def output_receipt_path(cfg, stage, unit):
    """Where the output receipt of one (stage, unit) lives."""
    return path(cfg, "receipts", "outputs", stage, unit.replace("/", "__") + ".json")


def done(cfg, stage, unit, outputs):
    """True when every output exists and the receipt carries the current config hash; raises on another hash."""
    target = output_receipt_path(cfg, stage, unit)
    if not target.exists():
        return False
    rec = json.loads(target.read_text())
    if rec["config_hash"] != cfg["hash"]:
        raise SystemExit(f"{stage} {unit}: receipt has config hash {rec['config_hash']}, current is {cfg['hash']}; use a new root version")
    return all(Path(o).exists() for o in outputs)


def finish(cfg, stage, unit, outputs, inputs, started, extra=None):
    """Write the output receipt after a stage completed."""
    rec = dict(stage=stage, unit=unit, outputs=[str(o) for o in outputs], inputs=inputs, config_hash=cfg["hash"],
               code_commit=cfg["commit"], code_dirty=cfg["dirty"], seconds=round(time.time() - started, 1),
               peak_rss_gb=round(peak_rss_gb(), 2), finished=time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {}))
    write_json(output_receipt_path(cfg, stage, unit), rec)
    return rec


def peak_rss_gb():
    """Peak resident set size of this process in GB."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
