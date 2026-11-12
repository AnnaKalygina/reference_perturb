"""Shifrut 2018 primary T cell deposit: MatrixMarket matrices streamed in cell chunks."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "audit"))
import sources  # noqa: E402
from adapters import ADAPTERS, CHUNK, Adapter  # noqa: E402
from config import path  # noqa: E402


def mtx_header(fh):
    """(n_features, n_cells) of a MatrixMarket stream, leaving it positioned at the first entry."""
    line = fh.readline()
    while line.startswith(b"%"):
        line = fh.readline()
    return tuple(int(v) for v in line.split()[:2])


def cells_csr(t, first, n, n_features):
    """CSR of the entries of cells first..first+n-1 (0-based) over the features; fails when the entries are not sorted by cell."""
    rows = t.cell.to_numpy() - 1 - first
    if len(rows) and ((np.diff(rows) < 0).any() or rows[0] < 0 or rows[-1] >= n):
        raise SystemExit("matrix.mtx entries are not sorted by cell")
    return sp.csr_matrix((t["count"].to_numpy(), (rows, t.feature.to_numpy() - 1)), shape=(n, n_features))


def mtx_cell_chunks(fh, n_features, n_cells):
    """Yield (first cell, CSR cells x features) over CHUNK cells from a cell-sorted MatrixMarket body of integer entries."""
    pending, first = [], 0
    for part in pd.read_csv(fh, sep=" ", header=None, names=["feature", "cell", "count"], dtype=np.int32, chunksize=20000000):
        pending.append(part)
        while int(part.cell.iloc[-1]) > first + CHUNK:
            t = pd.concat(pending, ignore_index=True)
            cut = int(np.searchsorted(t.cell.to_numpy(), first + CHUNK, side="right"))
            yield first, cells_csr(t.iloc[:cut], first, CHUNK, n_features)
            pending, first = [t.iloc[cut:]], first + CHUNK
    yield first, cells_csr(pd.concat(pending, ignore_index=True), first, n_cells - first, n_features)


class Shifrut18(Adapter):
    """GEO GSE119450_RAW.tar: four Cell Ranger matrix.mtx read from the nested archives in sample order."""

    dataset, loader = "shifrut18_t_crisprko", staticmethod(sources.shifrut18_t_crisprko)

    def tar(self):
        return path(self.cfg, "raw", "shifrut18_t_crisprko", "GSE119450_RAW.tar")

    def genes(self, sample):
        return pd.read_csv(sources.shifrut_sample(self.tar(), sample).extractfile("genes.tsv"), sep="\t", header=None, names=["id", "name"])

    def features(self, unit):
        g = self.genes(next(iter(sources.SHIFRUT_SAMPLES)))
        return g.id.to_numpy(), g.name.to_numpy()

    def chunks(self, unit):
        ref = self.features(unit)[0]
        start = 0
        for sample in sources.SHIFRUT_SAMPLES:
            if not np.array_equal(self.genes(sample).id.to_numpy(), ref):
                raise SystemExit(f"shifrut18 {sample}: gene list differs from the first sample")
            fh = sources.shifrut_sample(self.tar(), sample).extractfile("matrix.mtx")
            n_features, n_cells = mtx_header(fh)
            for lo, m in mtx_cell_chunks(fh, n_features, n_cells):
                yield start + lo, m
            start += n_cells


ADAPTERS.update({a.dataset: a for a in [Shifrut18]})
