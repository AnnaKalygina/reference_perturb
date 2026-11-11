"""A toy context with hand-checkable numbers, built through the real shard, moments and deltas stages into a temporary root."""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ingest"))
sys.path.insert(0, str(ROOT / "ingest" / "audit"))

L_VALUES = np.array([100, 200, 400])
CONTROLS = {"b1": 6, "b2": 4, "b3": 10}
PERTS = {"P1": [("b1", 4, "g1"), ("b2", 2, "g2")], "P2": [("b3", 6, "g1")], "P3": [("b1", 1, "g1"), ("b3", 1, "g1")], "P4": [("b2", 3, "g1")], "P5": [("b3", 2, "g1")],
         "P6": [("b3", 5, "g1"), ("b3", 5, "g2")]}
COMBO = [("b1", 5, "g1")]
GENES = 20


def axis_symbols():
    """20 native features: 18 mapped (TOYG01..TOYG18), one unmapped, one duplicated id."""
    ids = [f"ENSG0000090{i:04d}" for i in range(1, 19)] + ["ENSG00009999999", "ENSG00000900018"]
    syms = [f"TOYG{i:02d}" for i in range(1, 19)] + ["UNMAPPED", "TOYG18"]
    return np.array(ids), np.array(syms)


def build_cells(rng):
    """obs and counts: controls per batch, then perturbations; target of Pk is TOYG(k) whose counts are halved (P1) or zero (P4)."""
    rows, obs = [], []
    k = 0

    def cell(batch, kind, pert, guide, target_idx=None, factor=1.0):
        nonlocal k
        L = L_VALUES[k % 3]
        base = rng.integers(1, 6, GENES)
        counts = base.astype(float)
        if target_idx is not None:
            counts[target_idx] = np.round(base[target_idx] * factor)
        counts = counts * (L / counts.sum())
        counts = np.floor(counts).astype(int)
        counts[0] += L - counts.sum()
        rows.append(counts)
        obs.append(dict(context="toy__ctx", cell_type="toy", condition="unstimulated", timepoint="d0", donor="toy", batch=batch,
                        perturbation=pert, is_control=kind == "control", control_kind="non_targeting" if kind == "control" else None,
                        guide=guide, n_guides=1 if pert.count("+") == 0 else 2, total_counts=None, n_genes=None, pct_mito=None, target_ensembl=None))
        k += 1

    for b, n in CONTROLS.items():
        for _ in range(n):
            cell(b, "control", "", "NT")
    for p, spec in PERTS.items():
        i = int(p[1:])
        for b, n, g in spec:
            for _ in range(n):
                cell(b, "single", f"TOYG{i:02d}", g, i - 1, 0.5 if p == "P1" else (0.0 if p == "P4" else 1.0))
    for b, n, g in COMBO:
        for _ in range(n):
            cell(b, "combo", "TOYG02+TOYG01", g)
    return pd.DataFrame(obs), np.array(rows)


def make_root(tmp):
    """Copy the config and gene table into a temporary root and register the toy adapter."""
    tmp = Path(tmp)
    (tmp / "config").mkdir(parents=True)
    cfg = yaml.safe_load(open(ROOT / "config" / "ingest.yaml"))
    cfg["qc"]["g0_control_detection_fraction"] = 0.05
    cfg["cell_rules"]["designed_combinations_kept"].append("toy")
    yaml.safe_dump(cfg, open(tmp / "config" / "ingest.yaml", "w"))
    ids, syms = axis_symbols()
    pd.DataFrame(dict(ensembl_id=ids[:18], symbol=syms[:18], biotype="protein_coding", chromosome="1", start=np.arange(18) * 10000 + 1,
                      end=np.arange(18) * 10000 + 5000, strand="+", tss=np.arange(18) * 10000 + 1, version="1")).to_csv(
        tmp / "config" / "genes_ensembl112.tsv.gz", sep="\t", index=False, compression="gzip")
    shutil.copy(ROOT / "config" / "cell_cycle_genes.tsv", tmp / "config" / "cell_cycle_genes.tsv")
    shutil.copy(ROOT / "config" / "deposits.yaml", tmp / "config" / "deposits.yaml")
    return tmp


class ToyAdapter:
    """Adapter over in-memory toy data."""

    dataset = "toy"

    def __init__(self, cfg, obs, counts):
        self.cfg, self._obs, self._counts = cfg, obs, counts
        self.meta = dict(deposit="toy", modality="CRISPRi", batch_field="batch", batch_kind="technical", files=[])

    def units(self):
        return ["toy"]

    def obs(self, unit):
        o = self._obs.copy()
        o["deposit_row"] = np.arange(len(o))
        return o

    def features(self, unit):
        return axis_symbols()

    def chunks(self, unit):
        m = sp.csr_matrix(self._counts)
        for s in range(0, m.shape[0], 7):
            yield s, m[s:s + 7]


def expected_axis_map():
    """Which native features map, by the rules: 18 by id, UNMAPPED dropped, the duplicate id pair dropped."""
    return dict(mapped=[f"TOYG{i:02d}" for i in range(1, 18)], dropped=["TOYG18", "UNMAPPED"])
