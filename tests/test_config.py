"""The config hash ignores the slurm and reader sections; rehash rewrites the receipts of an old hash."""
import json
import shutil
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ingest"))
import config  # noqa: E402
from ingest.__main__ import rehash  # noqa: E402


def write_config(tmp, edit):
    """ingest.yaml in tmp: the repo's config with one edit applied."""
    cfg = yaml.safe_load(open(ROOT / "config" / "ingest.yaml"))
    edit(cfg)
    yaml.safe_dump(cfg, open(tmp / "ingest.yaml", "w"))


def test_hash_ignores_slurm_and_reader(tmp_path, monkeypatch):
    for name in config.HASHED_TABLES:
        shutil.copy(ROOT / "config" / name, tmp_path / name)
    monkeypatch.setattr(config, "CONFIG", tmp_path)
    write_config(tmp_path, lambda cfg: None)
    before = config.load()["hash"]
    assert len(before) == 16 and before == config.config_hash(yaml.safe_load(open(tmp_path / "ingest.yaml")))
    write_config(tmp_path, lambda cfg: cfg["slurm"]["memory_gb"]["shard"].update(default=cfg["slurm"]["memory_gb"]["shard"]["default"] + 8))
    assert config.load()["hash"] == before
    write_config(tmp_path, lambda cfg: cfg.update(reader=dict(anything=1)))
    assert config.load()["hash"] == before
    write_config(tmp_path, lambda cfg: cfg["qc"].update(min_genes=cfg["qc"]["min_genes"] + 1))
    assert config.load()["hash"] != before


def test_rehash(tmp_path):
    outputs = tmp_path / "receipts" / "outputs" / "moments"
    outputs.mkdir(parents=True)
    (outputs / "a.json").write_text(json.dumps(dict(unit="a", config_hash="old")))
    (outputs / "b.json").write_text(json.dumps(dict(unit="b", config_hash="other")))
    cfg = dict(root=str(tmp_path), hash="new", commit="abc1234", dirty=False)
    with pytest.raises(SystemExit):
        rehash(dict(cfg, dirty=True), "old")
    with pytest.raises(SystemExit):
        rehash(cfg, "new")
    assert rehash(cfg, "old") == 1
    a = json.loads((outputs / "a.json").read_text())
    h = a["config_hash_history"][0]
    assert a["config_hash"] == "new" and (h["from"], h["to"], h["commit"], h["date"]) == ("old", "new", "abc1234", time.strftime("%Y-%m-%d"))
    assert json.loads((outputs / "b.json").read_text())["config_hash"] == "other"
    assert rehash(cfg, "old") == 0
