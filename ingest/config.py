"""Frozen configuration: one YAML, its hash, the code commit at run time, and the unhashed deposit table."""
import hashlib
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
HASHED_FILES = ["ingest.yaml", "genes_ensembl112.tsv.gz", "cell_cycle_genes.tsv"]


def load():
    """Config dict with `deposits`, `root`, `hash` and `commit` filled in."""
    cfg = yaml.safe_load(open(CONFIG / "ingest.yaml"))
    cfg["deposits"] = yaml.safe_load(open(CONFIG / "deposits.yaml"))
    cfg["root"] = str(ROOT)
    cfg["hash"] = config_hash()
    cfg["commit"], cfg["dirty"] = commit()
    return cfg


def config_hash():
    """sha256 over the config file and the gene tables that exist."""
    h = hashlib.sha256()
    for name in HASHED_FILES:
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
