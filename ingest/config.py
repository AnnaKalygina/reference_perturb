"""Frozen configuration: one YAML, its hash, and the code commit at run time."""
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
HASHED_TABLES = ["genes_ensembl112.tsv.gz", "cell_cycle_genes.tsv"]
UNHASHED_KEYS = ["slurm", "reader"]


def load():
    """Config dict with `root`, `hash` and `commit` filled in."""
    cfg = yaml.safe_load(open(CONFIG / "ingest.yaml"))
    cfg["hash"] = config_hash(cfg)
    cfg["root"] = str(ROOT)
    cfg["commit"], cfg["dirty"] = commit()
    return cfg


def config_hash(cfg):
    """sha256 over the config without its slurm and reader sections, and the gene tables that exist."""
    h = hashlib.sha256(json.dumps({k: v for k, v in cfg.items() if k not in UNHASHED_KEYS}, sort_keys=True).encode())
    for name in HASHED_TABLES:
        p = CONFIG / name
        if p.exists():
            h.update(name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def commit():
    """(short commit, dirty flag) of the checkout, or ('none', True) outside git."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip() != ""
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "none", True


def path(cfg, *parts):
    """Absolute path under the root."""
    return Path(cfg["root"]).joinpath(*parts)
