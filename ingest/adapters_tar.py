"""Adapters for GEO tar deposits read in place: the Wessels 2023 HEK293FT arms and the Adamson 2016 pilot and epistasis lanes (matrix.mtx members)."""
import gzip
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "audit"))
import sources  # noqa: E402
from adapters import ADAPTERS, chunks_of  # noqa: E402
from adapters_more import UnitAdapter  # noqa: E402
from config import path  # noqa: E402
from sources import ADAMSON_RUNS, WESSELS_TARS, tar_table  # noqa: E402


def mtx_cells(tar, member, rows, cols):
    """CSR (cells x features) of a gzipped matrix.mtx member: the given matrix columns in order over the given feature rows."""
    with tarfile.open(tar) as t, gzip.open(t.extractfile(member), "rt") as fh:
        line = "%"
        while line.startswith("%"):
            line = fh.readline()
        n_rows, n_cols = (int(v) for v in line.split()[:2])
        row_pos, col_pos = np.full(n_rows, -1), np.full(n_cols, -1)
        row_pos[rows], col_pos[cols] = np.arange(len(rows)), np.arange(len(cols))
        parts = []
        for block in pd.read_csv(fh, sep=r"\s+", header=None, names=["row", "col", "value"], dtype={"row": np.int32, "col": np.int32, "value": np.float64}, chunksize=10_000_000):
            r, c = row_pos[block.row.to_numpy() - 1], col_pos[block.col.to_numpy() - 1]
            keep = (r >= 0) & (c >= 0)
            v = block.value.to_numpy()[keep]
            if not np.array_equal(v, np.rint(v)):
                raise SystemExit(f"{member}: non-integer values")
            parts.append((c[keep], r[keep], v.astype(np.int32)))
    c, r, v = (np.concatenate(x) for x in zip(*parts))
    return sp.csr_matrix((v, (c, r)), shape=(len(cols), len(rows)))


def columns(barcodes, wanted, what):
    """Matrix columns of the wanted barcodes, in their order; every cell must be present."""
    cols = pd.Index(barcodes).get_indexer(wanted)
    if (cols < 0).any():
        raise SystemExit(f"{what}: {int((cols < 0).sum())} cells absent from the matrix barcodes")
    return cols


class TarUnits(UnitAdapter):
    """Units whose cells are located in a matrix.mtx member by their deposit barcode."""

    def obs(self, unit):
        self._obs = super().obs(unit)
        return self._obs


class WesselsHEK293FT(TarUnits):
    """GSE213957 HEK293FT: the CaRPool-seq lane (hg38 rows of its human+mouse matrix) and the stabRNA cells of the four THP-1 lanes."""

    dataset, loader = "wessels23_hek293ft", staticmethod(sources.wessels23_hek293ft)
    LANES = {"CaRPool-seq": ("carpool", "HEK293FT_and_SpeciesMixing_CaRPool-seq.GEXGDO", [""]),
             "stabRNA": ("thp1", "THP1-CaRPool-seq_and_HEK293FTstabRNA.GEXGDO", ["1", "2", "3", "4"])}

    def units(self):
        return list(self.LANES)

    def select(self, obs, unit):
        return (obs.context == f"wessels23__HEK293FT__{unit}").to_numpy()

    def tar(self, unit):
        return path(self.cfg, "raw", "wessels23_hek293ft", WESSELS_TARS[self.LANES[unit][0]])

    def gene_rows(self, unit, lane):
        """(matrix rows, ids, symbols) of the human gene features of one lane matrix."""
        f = tar_table(self.tar(unit), f"{self.LANES[unit][1]}{lane}.features.tsv.gz", sep="\t", header=None, names=["id", "name", "type"])
        gex = ((f.type == "Gene Expression") & ~f.id.str.startswith("mm10_")).to_numpy()
        return np.flatnonzero(gex), f.id[gex].str.replace("hg38_", "", regex=False).to_numpy(), f.name[gex].str.replace("hg38_", "", regex=False).to_numpy()

    def features(self, unit):
        return self.gene_rows(unit, self.LANES[unit][2][0])[1:]

    def chunks(self, unit):
        tar, prefix, lanes = self.tar(unit), self.LANES[unit][1], self.LANES[unit][2]
        ref = self.features(unit)[0]
        start = 0
        for lane in lanes:
            rows, ids, _ = self.gene_rows(unit, lane)
            if not np.array_equal(ids, ref):
                raise SystemExit(f"{self.dataset} {unit} lane {lane}: gene set differs from the first lane")
            cells = self._obs[(self._obs.batch == f"L{lane}").to_numpy()] if lane else self._obs
            cols = columns(tar_table(tar, f"{prefix}{lane}.barcodes.tsv.gz", sep="\t", header=None)[0], cells.barcode, f"{self.dataset} {unit} lane {lane}")
            for lo, piece in chunks_of(mtx_cells(tar, f"{prefix}{lane}.matrix.mtx.gz", rows, cols)):
                yield start + lo, piece
            start += len(cells)


class WesselsPerturbSeq(TarUnits):
    """GSE213957 Perturb-seq lane: one unit per Cas9 effector line, all cells from the same matrix."""

    dataset, loader = "wessels23_perturbseq", staticmethod(sources.wessels23_perturbseq)
    MODALITY = {"Cas9-nuclease": "CRISPRko (Cas9 nuclease)", "KRAB-dCas9": "CRISPRi (KRAB-dCas9)", "KRAB-dCas9-MeCP2": "CRISPRi (KRAB-dCas9-MeCP2)"}

    def units(self):
        return list(self.MODALITY)

    def select(self, obs, unit):
        return (obs.context == f"wessels23__HEK293FT__{unit}").to_numpy()

    def obs(self, unit):
        o = super().obs(unit)
        self.meta = dict(self.meta, modality=self.MODALITY[unit])
        return o

    def tar(self):
        return path(self.cfg, "raw", "wessels23_hek293ft", WESSELS_TARS["perturbseq"])

    def gene_rows(self):
        f = tar_table(self.tar(), "Perturb-seq.GEXGDO.features.tsv.gz", sep="\t", header=None, names=["id", "name", "type"])
        gex = (f.type == "Gene Expression").to_numpy()
        return np.flatnonzero(gex), f.id[gex].to_numpy(), f.name[gex].to_numpy()

    def features(self, unit):
        return self.gene_rows()[1:]

    def chunks(self, unit):
        cols = columns(tar_table(self.tar(), "Perturb-seq.GEXGDO.barcodes.tsv.gz", sep="\t", header=None)[0], self._obs.barcode, f"{self.dataset} {unit}")
        yield from chunks_of(mtx_cells(self.tar(), "Perturb-seq.GEXGDO.matrix.mtx.gz", self.gene_rows()[0], cols))


class AdamsonOtherLanes(TarUnits):
    """GSE90546_RAW.tar: one unit per GEO sample over every barcode of its matrix.mtx member."""

    dataset, loader = "adamson16_other_lanes", staticmethod(sources.adamson16_other_lanes)

    def units(self):
        return list(ADAMSON_RUNS)

    def select(self, obs, unit):
        return (obs.gsm == unit).to_numpy()

    def tar(self):
        return path(self.cfg, "raw", "adamson16_other_lanes", "GSE90546_RAW.tar")

    def features(self, unit):
        g = tar_table(self.tar(), f"{unit}_genes.tsv.gz", sep="\t", header=None, names=["id", "name"])
        return g.id.to_numpy(), g.name.to_numpy()

    def chunks(self, unit):
        cols = columns(tar_table(self.tar(), f"{unit}_barcodes.tsv.gz", sep="\t", header=None)[0], self._obs.barcode, f"{self.dataset} {unit}")
        yield from chunks_of(mtx_cells(self.tar(), f"{unit}_matrix.mtx.txt.gz", np.arange(len(self.features(unit)[0])), cols))


ADAPTERS.update({a.dataset: a for a in [WesselsHEK293FT, WesselsPerturbSeq, AdamsonOtherLanes]})
