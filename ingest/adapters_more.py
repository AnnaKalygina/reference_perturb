"""Adapters for the multi-file and gene-major deposits: H1, Orion, DLD-1, hESC, Jiang, McFaline, VIPerturb, KOLF, Feng."""
import glob
import gzip
import re
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "audit"))
import sources  # noqa: E402
from adapters import ADAPTERS, CHUNK, Adapter, chunks_of  # noqa: E402
from config import path  # noqa: E402
from h5 import col, dec, index  # noqa: E402


class UnitAdapter(Adapter):
    """Several raw units per dataset; obs rows of a unit are renumbered 0..n-1 in raw order."""

    def select(self, obs, unit):
        raise NotImplementedError

    def obs(self, unit):
        src = self.loader(self.acfg)
        self.meta = src.meta
        o = src.obs[self.select(src.obs, unit)].copy().reset_index(drop=True)
        o["deposit_row"] = np.arange(len(o))
        return o


class H1(Adapter):
    """VCC 2025 H1: three CSR h5ad files, non-training controls dropped as in the loader."""

    dataset, loader = "vcc2025_h1", staticmethod(sources.vcc2025_h1)
    SPLITS = ("Training", "Validation", "Test")

    def dir(self):
        return Path(self.cfg["paths"]["vcc_data"]) / "vcc2025"

    def features(self, unit):
        with h5py.File(self.dir() / "adata_Training.h5ad", "r") as h:
            return col(h, "var/gene_id"), index(h, "var")

    def chunks(self, unit):
        start = 0
        with h5py.File(self.dir() / "adata_Training.h5ad", "r") as h:
            ref = index(h, "var")
        for s in self.SPLITS:
            with h5py.File(self.dir() / f"adata_{s}.h5ad", "r") as h:
                if not np.array_equal(index(h, "var"), ref):
                    raise SystemExit(f"{s}: gene order differs from Training")
                keep = ~((col(h, "obs/target_gene").astype(str) == "non-targeting") & (s != "Training"))
                x = h["X"]
                for lo in range(0, int(x.attrs["shape"][0]), CHUNK):
                    hi = min(lo + CHUNK, int(x.attrs["shape"][0]))
                    ip = x["indptr"][lo:hi + 1]
                    m = sp.csr_matrix((x["data"][ip[0]:ip[-1]], x["indices"][ip[0]:ip[-1]], ip - ip[0]), shape=(hi - lo, int(x.attrs["shape"][1])))
                    m = m[keep[lo:hi]]
                    if not np.array_equal(m.data, np.rint(m.data)):
                        raise SystemExit("H1: non-integer values")
                    m.data = m.data.astype(np.int32)
                    if m.shape[0]:
                        yield start, m
                    start += m.shape[0]


class Orion(Adapter):
    """X-Atlas/Orion parquet shards: per-cell token and count lists; tokens index gene_metadata.parquet."""

    line = None

    def files(self):
        fs = glob.glob(str(Path(self.cfg["paths"]["vcc_data"]) / f"raw/orion/{self.line}_Batch*.parquet"))
        return sorted(fs, key=lambda s: int(re.search(r"Batch(\d+)", s).group(1)))

    def features(self, unit):
        gm = pd.read_parquet(Path(self.cfg["paths"]["vcc_data"]) / "orion_probe/gene_metadata.parquet").sort_values("gene_token_id")
        if not np.array_equal(gm.gene_token_id.to_numpy(), np.arange(len(gm))):
            raise SystemExit("orion: gene tokens are not 0..n-1")
        return gm.ensembl_id.astype(str).to_numpy(), gm.gene_name.astype(str).to_numpy()

    def chunks(self, unit):
        n_genes = len(self.features(unit)[0])
        start = 0
        for fp in self.files():
            t = pq.read_table(fp, columns=["gene_token_id", "gene_expression"])
            tok, val = t.column(0).combine_chunks(), t.column(1).combine_chunks()
            offsets = tok.offsets.to_numpy().astype(np.int64)
            idx = tok.values.to_numpy().astype(np.int32)
            dat = val.values.to_numpy()
            if not np.array_equal(dat, np.rint(dat)):
                raise SystemExit(f"{fp}: non-integer values")
            m = sp.csr_matrix((dat.astype(np.int32), idx, offsets), shape=(len(offsets) - 1, n_genes))
            for lo, piece in chunks_of(m):
                yield start + lo, piece
            start += m.shape[0]


class OrionHCT116(Orion):
    dataset, loader, line = "orion_hct116", staticmethod(sources.orion_hct116), "HCT116"


class OrionHEK293T(Orion):
    dataset, loader, line = "orion_hek293t", staticmethod(sources.orion_hek293t), "HEK293T"


class DLD1(Adapter):
    """GEO GSE337988 low-MOI object: dense int32 cells x genes in 1000-row chunks."""

    dataset, loader = "dld1", staticmethod(sources.dld1)

    def dir(self):
        return Path(self.cfg["paths"]["vcc_data"]) / "raw/gse337988"

    def features(self, unit):
        rd = pd.read_csv(self.dir() / "low_rowdata.csv.gz", dtype=str, keep_default_na=False)
        return rd.feature.to_numpy(), rd.Symbol.to_numpy()

    def chunks(self, unit):
        with h5py.File(self.dir() / "GSE337988_sublib2_processed_objects_Low_assays.h5", "r") as h:
            d = h["assay001"]
            for lo in range(0, d.shape[0], 10000):
                yield lo, sp.csr_matrix(np.asarray(d[lo:lo + 10000]).astype(np.int32))


class TenxLanes(Adapter):
    """Several Cell Ranger h5 files in sorted order, each per-cell contiguous (CSC over features x barcodes)."""

    def files(self):
        raise NotImplementedError

    def lane_matrix(self, fp, feature_pos=None):
        with h5py.File(fp, "r") as h:
            m = h["matrix"]
            ft = dec(m["features/feature_type"][:])
            ids = dec(m["features/id"][:])
            nf, nc = (int(v) for v in m["shape"][:])
            x = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=(nf, nc)).T.tocsr()
        gex = np.flatnonzero(ft == "Gene Expression")
        x = x[:, gex]
        if feature_pos is not None:
            order = np.array([feature_pos[g] for g in ids[gex]])
            reorder = np.argsort(order)
            if not np.array_equal(order[reorder], np.arange(len(feature_pos))):
                raise SystemExit(f"{fp}: gene set differs from the reference lane")
            x = x[:, reorder]
        x.data = x.data.astype(np.int32)
        return x


class HESC(TenxLanes):
    dataset, loader = "hesc", staticmethod(sources.hesc)

    def files(self):
        return sorted(glob.glob(str(Path(self.cfg["paths"]["vcc_data"]) / "raw/gse295214/GSM*_filtered_feature_bc_matrix.h5")))

    def features(self, unit):
        with h5py.File(self.files()[0], "r") as h:
            m = h["matrix"]
            gex = dec(m["features/feature_type"][:]) == "Gene Expression"
            return dec(m["features/id"][:])[gex], dec(m["features/name"][:])[gex]

    def chunks(self, unit):
        ref = {g: i for i, g in enumerate(self.features(unit)[0])}
        start = 0
        for fp in self.files():
            x = self.lane_matrix(fp, ref)
            for lo, piece in chunks_of(x):
                yield start + lo, piece
            start += x.shape[0]


class Extract(UnitAdapter):
    """VCC2026-style flat extracts: p/i/x int32 (CSR over cells), genes.txt, meta.csv; one unit per extract directory."""

    def extract_dir(self, unit):
        raise NotImplementedError

    def features(self, unit):
        d = self.extract_dir(unit)
        genes = np.array(open(d / "genes.txt").read().splitlines(), dtype=object)
        ens = np.array(open(d / "ensembl.txt").read().splitlines(), dtype=object) if (d / "ensembl.txt").exists() else None
        return ens, genes

    def chunks(self, unit):
        d = self.extract_dir(unit)
        info = dict(line.split(" ", 1) for line in open(d / "info.txt").read().splitlines())
        G, N = int(info["n_genes"]), int(info["n_cells"])
        p = np.fromfile(d / "p.bin", np.int32).astype(np.int64)
        i = np.memmap(d / "i.bin", np.int32, "r")
        x = np.memmap(d / "x.bin", np.int32, "r")
        for lo in range(0, N, CHUNK):
            hi = min(lo + CHUNK, N)
            a, b = p[lo], p[hi]
            yield lo, sp.csr_matrix((np.array(x[a:b]), np.array(i[a:b]), p[lo:hi + 1] - a), shape=(hi - lo, G))


class Jiang(Extract):
    dataset, loader = "jiang", staticmethod(sources.jiang)

    def units(self):
        return list(sources.JIANG_PATHWAYS)

    def select(self, obs, unit):
        return (obs.condition == sources.JIANG_PATHWAYS[unit]).to_numpy()

    def extract_dir(self, unit):
        return Path(self.cfg["paths"]["vcc_data"]) / "raw/jiang2025/extract" / unit


class McFaline(Extract):
    dataset, loader = "mcfaline", staticmethod(sources.mcfaline)

    def units(self):
        return ["A172", "T98G", "U87MG"]

    def select(self, obs, unit):
        return (obs.donor == unit).to_numpy()

    def extract_dir(self, unit):
        return Path(self.cfg["paths"]["vcc_data"]) / "raw/mcfaline2023/extract" / unit


class VIPerturb(Adapter):
    """Bradu 2026 bins A, B, C from our own Seurat extracts; non-targeting cells of B and C dropped as in the loader."""

    dataset, loader = "viperturb", staticmethod(sources.viperturb)

    def __init__(self, cfg):
        super().__init__(cfg)
        self.acfg["out_dir"] = str(path(cfg, "raw", "viperturb_extract"))

    def bins(self):
        return [path(self.cfg, "raw", "viperturb_extract", f"bin{b}") for b in "ABC"]

    def features(self, unit):
        genes = np.array(open(self.bins()[0] / "genes.txt").read().splitlines(), dtype=object)
        return None, genes

    def chunks(self, unit):
        ref = open(self.bins()[0] / "genes.txt").read().splitlines()
        start = 0
        for b, d in zip("ABC", self.bins()):
            if open(d / "genes.txt").read().splitlines() != ref:
                raise SystemExit(f"viperturb bin {b}: gene list differs from bin A")
            meta = pd.read_csv(d / f"bin{b}.meta.csv", usecols=["gene"], low_memory=False)
            keep = np.ones(len(meta), bool) if b == "A" else (meta.gene != "NO-TARGET").to_numpy()
            info = dict(line.split(" ", 1) for line in open(d / "info.txt").read().splitlines())
            G, N = int(info["n_genes"]), int(info["n_cells"])
            p = np.fromfile(d / "p.bin", np.int32).astype(np.int64)
            i = np.memmap(d / "i.bin", np.int32, "r")
            x = np.memmap(d / "x.bin", np.int32, "r")
            for lo in range(0, N, CHUNK):
                hi = min(lo + CHUNK, N)
                a, c = p[lo], p[hi]
                m = sp.csr_matrix((np.array(x[a:c]), np.array(i[a:c]), p[lo:hi + 1] - a), shape=(hi - lo, G))[keep[lo:hi]]
                if m.shape[0]:
                    yield start, m
                start += m.shape[0]


class GeneMajor(Adapter):
    """Gene-major sources: gene blocks are scattered into cell-range buckets on disk, then yielded as cell chunks."""

    def gene_blocks(self):
        """Yield (cell indices, gene indices, counts) of one gene block as int32/int32/int32 arrays."""
        raise NotImplementedError

    def n_cells(self):
        raise NotImplementedError

    def chunks(self, unit):
        n = self.n_cells()
        tmp = path(self.cfg, "tmp", self.dataset, "transpose")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        bucket = self.cfg["shard"]["bucket_rows"]
        handles = {}
        for cells, genes, counts in self.gene_blocks():
            b = cells // bucket
            order = np.argsort(b, kind="stable")
            cells, genes, counts, b = cells[order], genes[order], counts[order], b[order]
            edges = np.flatnonzero(np.r_[True, b[1:] != b[:-1], True])
            for s, e in zip(edges[:-1], edges[1:]):
                k = int(b[s])
                if k not in handles:
                    handles[k] = open(tmp / f"{k}.bin", "ab")
                handles[k].write(np.stack([(cells[s:e] % bucket).astype(np.int32), genes[s:e].astype(np.int32), counts[s:e].astype(np.int32)], 1).tobytes())
        for f in handles.values():
            f.close()
        n_genes = len(self.features(unit)[1])
        for k in range((n + bucket - 1) // bucket):
            lo, hi = k * bucket, min((k + 1) * bucket, n)
            f = tmp / f"{k}.bin"
            if f.exists():
                t = np.fromfile(f, np.int32).reshape(-1, 3)
                m = sp.csr_matrix((t[:, 2], (t[:, 0], t[:, 1])), shape=(hi - lo, n_genes))
                m.sum_duplicates()
            else:
                m = sp.csr_matrix((hi - lo, n_genes), dtype=np.int32)
            for s, piece in chunks_of(m):
                yield lo + s, piece
        shutil.rmtree(tmp)


class Kolf(GeneMajor):
    dataset, loader = "kolf", staticmethod(sources.kolf)

    def file(self):
        return Path(self.cfg["paths"]["vcc_data"]) / "raw/kolf/KOLF_Pan_Genome_QC_Filtered.h5ad"

    def features(self, unit):
        with h5py.File(self.file(), "r") as h:
            return col(h, "var/gene_ids"), index(h, "var")

    def n_cells(self):
        with h5py.File(self.file(), "r") as h:
            return int(h["layers/counts"].attrs["shape"][0])

    def gene_blocks(self):
        limit = self.cfg["shard"]["gene_block_nnz"]
        with h5py.File(self.file(), "r") as h:
            x = h["layers/counts"]
            ip = x["indptr"][:]
            g0 = 0
            while g0 < len(ip) - 1:
                g1 = g0 + 1
                while g1 < len(ip) - 1 and ip[g1 + 1] - ip[g0] <= limit:
                    g1 += 1
                a, b = int(ip[g0]), int(ip[g1])
                dat = x["data"][a:b]
                if not np.array_equal(dat, np.rint(dat)):
                    raise SystemExit("kolf: non-integer counts")
                genes = np.repeat(np.arange(g0, g1, dtype=np.int32), np.diff(ip[g0:g1 + 1]).astype(np.int64))
                yield x["indices"][a:b].astype(np.int32), genes, dat.astype(np.int32)
                g0 = g1


class Feng(GeneMajor):
    """Feng 2026: two dense csv.gz (rows = features, columns = cells); one unit per screen."""

    dataset, loader = "feng", staticmethod(sources.feng)
    FILES = {"fitness": "GenomeWideScreen_FitnessGenes_RNA-UMI-Counts.csv.gz", "nonfitness": "GenomeWideScreen_NonFitnessGenes_RNA-UMI-Counts.csv.gz"}

    def units(self):
        return list(self.FILES)

    def dir(self):
        return Path(self.cfg["paths"]["vcc_data"]) / "raw/feng"

    def header(self, unit):
        with gzip.open(self.dir() / self.FILES[unit], "rt") as fh:
            return fh.readline().rstrip("\n").split(",")[1:]

    def obs(self, unit):
        src = self.loader(self.acfg)
        self.meta = src.meta
        o = src.obs[src.obs.context.str.startswith(f"feng__{unit}__")].copy()
        header = self.header(unit)
        meta = pd.read_csv(self.dir() / ("GW_Fitness_meta.tsv.gz" if unit == "fitness" else "GW_NonFitness_meta.tsv.gz"), sep="\t", usecols=["Cell_ID"])
        ids = meta.Cell_ID.str.replace("MP-", "", n=1, regex=False)
        pos = pd.Index(header).get_indexer(ids)
        if (pos < 0).any():
            raise SystemExit(f"feng {unit}: {(pos < 0).sum()} meta cells absent from the matrix header")
        o = o.iloc[np.argsort(pos)].reset_index(drop=True)
        o["deposit_row"] = np.arange(len(o))
        self._n = len(header)
        self._unit = unit
        if len(o) != len(header):
            raise SystemExit(f"feng {unit}: {len(o)} meta cells for {len(header)} matrix columns")
        return o

    def features(self, unit):
        with gzip.open(self.dir() / self.FILES[unit], "rt") as fh:
            fh.readline()
            feats = [line.split(",", 1)[0] for line in fh]
        parts = [f.split(":") for f in feats]
        return np.array([p[0] for p in parts], dtype=object), np.array([p[1] if len(p) > 1 else p[0] for p in parts], dtype=object)

    def n_cells(self):
        return self._n

    def gene_blocks(self):
        rows_per_block = 256
        with gzip.open(self.dir() / self.FILES[self._unit], "rt") as fh:
            fh.readline()
            g = 0
            for chunk in pd.read_csv(fh, header=None, chunksize=rows_per_block, engine="c", dtype={0: str}):
                block = chunk.iloc[:, 1:].to_numpy(np.int32)
                r, c = np.nonzero(block)
                yield c.astype(np.int32), (r + g).astype(np.int32), block[r, c].astype(np.int32)
                g += block.shape[0]


ADAPTERS.update({a.dataset: a for a in [H1, OrionHCT116, OrionHEK293T, DLD1, HESC, Jiang, McFaline, VIPerturb, Kolf, Feng]})
