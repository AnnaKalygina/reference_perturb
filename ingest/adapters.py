"""One adapter per deposit: obs through the audit loaders, native features, and the expression matrix streamed in cell chunks."""
import gzip
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent / "audit"))
import sources  # noqa: E402
from h5 import col, dec, index  # noqa: E402

CHUNK = 20000


def audit_cfg(cfg):
    """Config dict in the shape the audit loaders expect."""
    return dict(root=cfg["root"], vcc_data=cfg["paths"]["vcc_data"], repo_data=cfg["paths"]["repo_data"], hgnc=cfg["paths"]["hgnc"],
                vcc_rscript=cfg["paths"]["vcc_rscript"], integer_sample_values=1000, guide_call_min_umi=cfg["guide_call_min_umi"])


def read_sparse(h, key):
    """An AnnData matrix as CSR (dense, csr or csc layouts)."""
    x = h[key]
    if isinstance(x, h5py.Dataset):
        return sp.csr_matrix(x[:])
    kind = x.attrs.get("encoding-type", "csr_matrix")
    shape = tuple(int(v) for v in x.attrs["shape"])
    cls = sp.csr_matrix if "csr" in kind else sp.csc_matrix
    return cls((x["data"][:], x["indices"][:], x["indptr"][:]), shape=shape).tocsr()


def chunks_of(m):
    """Yield (start, csr chunk) over the rows of a CSR matrix."""
    for s in range(0, m.shape[0], CHUNK):
        yield s, m[s:s + CHUNK]


def dense_rows(path, rows=CHUNK):
    """Yield (start, csr chunk) over the rows of a dense integer-valued float h5ad X, `rows` rows at a time."""
    with h5py.File(path, "r") as h:
        x = h["X"]
        for s in range(0, x.shape[0], rows):
            block = np.asarray(x[s:s + rows])
            if not np.array_equal(block, np.rint(block)):
                raise SystemExit(f"{path}: non-integer values in rows {s}:{s + rows}")
            yield s, sp.csr_matrix(block.astype(np.int32))


class Adapter:
    """Base: one dataset, units that each feed one or more contexts."""

    dataset = None
    loader = None

    def __init__(self, cfg):
        self.cfg = cfg
        self.acfg = audit_cfg(cfg)

    def units(self):
        """Unit ids (default: one unit holding every context of the dataset)."""
        return [self.dataset]

    def obs(self, unit):
        """Standardized obs for the unit in deposit order, with deposit_row."""
        src = self.loader(self.acfg)
        o = src.obs.copy()
        o["deposit_row"] = np.arange(len(o))
        self.meta = src.meta
        return o

    def features(self, unit):
        """(native ids, native symbols) of the matrix columns."""
        raise NotImplementedError

    def chunks(self, unit):
        """Yield (first deposit row, CSR chunk over the native features) in deposit order."""
        raise NotImplementedError


class ScPerturb(Adapter):
    """scPerturb-style h5ad kept whole in memory."""

    path = None
    id_key = "var/ensembl_id"

    def features(self, unit):
        with h5py.File(self.path(self.cfg), "r") as h:
            ids = col(h, self.id_key) if self.id_key and self.id_key in h else None
            return ids, col(h, "var/gene_symbol") if "var/gene_symbol" in h else index(h, "var")

    def chunks(self, unit):
        with h5py.File(self.path(self.cfg), "r") as h:
            m = read_sparse(h, "X")
        yield from chunks_of(m)


class Adamson(ScPerturb):
    dataset, loader = "scperturb_adamson", staticmethod(sources.scperturb_adamson)
    path = staticmethod(lambda cfg: Path(cfg["paths"]["vcc_data"]) / "raw/scperturb/AdamsonWeissman2016_GSM2406681_10X010.h5ad")


class Frangieh(ScPerturb):
    dataset, loader = "scperturb_frangieh", staticmethod(sources.scperturb_frangieh)
    path = staticmethod(lambda cfg: Path(cfg["paths"]["vcc_data"]) / "raw/scperturb/FrangiehIzar2021_RNA.h5ad")


class Sunshine(ScPerturb):
    dataset, loader = "scperturb_sunshine", staticmethod(sources.scperturb_sunshine)
    path = staticmethod(lambda cfg: Path(cfg["paths"]["vcc_data"]) / "raw/scperturb/SunshineHein2023.h5ad")


class Norman(ScPerturb):
    dataset, loader = "scperturb_norman", staticmethod(sources.scperturb_norman)
    path = staticmethod(lambda cfg: Path(cfg["paths"]["repo_data"]) / "norman19/norman19_downloaded.h5ad")
    id_key = "var/ensemble_id"


class Wessels(ScPerturb):
    dataset, loader = "scperturb_wessels", staticmethod(sources.scperturb_wessels)
    path = staticmethod(lambda cfg: Path(cfg["paths"]["repo_data"]) / "wessels23/wessels23_downloaded.h5ad")
    id_key = None


class ReplogleRows(Adapter):
    """Replogle 2022 / Nadig 2025 raw_singlecell_01.h5ad: dense float32 rows streamed in chunks."""

    file = None

    def path(self):
        return Path(self.cfg["paths"]["vcc_data"]) / self.file

    def features(self, unit):
        with h5py.File(self.path(), "r") as h:
            ids = index(h, "var")
            names = col(h, "var/gene_name") if "var/gene_name" in h else ids
            return ids, names

    def chunks(self, unit):
        yield from dense_rows(self.path())


class ReplogleK562(ReplogleRows):
    dataset, loader, file = "replogle22_k562", staticmethod(sources.replogle22_k562), "raw/replogle2022/K562_essential_raw_singlecell_01.h5ad"


class ReplogleRPE1(ReplogleRows):
    dataset, loader, file = "replogle22_rpe1", staticmethod(sources.replogle22_rpe1), "raw/replogle2022/rpe1_raw_singlecell_01.h5ad"


class ReplogleGWPS(ReplogleRows):
    dataset, loader, file = "replogle22_gwps", staticmethod(sources.replogle22_gwps), "raw/replogle2022/K562_gwps_raw_singlecell_01.h5ad"


class NadigHepG2(ReplogleRows):
    dataset, loader, file = "nadig25_hepg2", staticmethod(sources.nadig25_hepg2), "raw/nadig2025/GSE264667_hepg2_raw_singlecell_01.h5ad"


class NadigJurkat(ReplogleRows):
    dataset, loader, file = "nadig25_jurkat", staticmethod(sources.nadig25_jurkat), "raw/nadig2025/GSE264667_jurkat_raw_singlecell_01.h5ad"


class Replogle20(Adapter):
    """GEO GSM4367984 exp6 matrix.mtx.gz, gene-expression features only."""

    dataset, loader = "replogle20", staticmethod(sources.replogle20)

    def dir(self):
        return Path(self.cfg["paths"]["vcc_data"]) / "raw/replogle2020"

    def feature_table(self):
        return pd.read_csv(self.dir() / "GSM4367984_exp6.features.tsv.gz", sep="\t", header=None, names=["id", "name", "type"])

    def features(self, unit):
        f = self.feature_table()
        gex = f[f.type == "Gene Expression"]
        return gex.id.to_numpy(), gex.name.to_numpy()

    def chunks(self, unit):
        f = self.feature_table()
        gex = (f.type == "Gene Expression").to_numpy()
        with gzip.open(self.dir() / "GSM4367984_exp6.matrix.mtx.gz", "rt") as fh:
            header = None
            while header is None:
                line = fh.readline()
                if not line.startswith("%"):
                    header = [int(v) for v in line.split()]
            t = pd.read_csv(fh, sep=r"\s+", header=None, names=["gene", "cell", "value"], dtype={"gene": np.int32, "cell": np.int32, "value": np.float64})
        if not np.array_equal(t.value, np.rint(t.value)):
            raise SystemExit("replogle20 matrix has non-integer values")
        m = sp.csr_matrix((t.value.to_numpy(np.int32), (t.cell.to_numpy() - 1, t.gene.to_numpy() - 1)), shape=(header[1], header[0]))
        m = m[:, gex]
        yield from chunks_of(m)


ADAPTERS = {a.dataset: a for a in [Adamson, Frangieh, Sunshine, Norman, Wessels, ReplogleK562, ReplogleRPE1, ReplogleGWPS, NadigHepG2, NadigJurkat, Replogle20]}
