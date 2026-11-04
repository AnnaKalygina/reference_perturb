"""The one consumer entry point: moments, deltas (δ^A, δ^C stored; Δ^B, δ^B derived), cells, masks and scores of a context."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

SUMS = ["sum_c", "sum_c2", "sum_xB", "sum_xB2", "sum_xC", "sum_xC2"]


class Reference:
    """Read access to one reference root (default: the directory holding this package)."""

    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parent.parent)
        self.cfg = yaml.safe_load(open(self.root / "config" / "ingest.yaml"))
        self.kappa = self.cfg["kappa"]
        self.genes = pd.read_csv(self.root / "config" / "genes_ensembl112.tsv.gz", sep="\t", dtype={"chromosome": str, "version": str})

    def contexts(self):
        """Contexts with a shard, as (dataset, context) pairs."""
        return sorted((p.parent.name, p.stem) for p in (self.root / "shards").glob("*/*.h5ad"))

    def dataset(self, context):
        hits = list((self.root / "shards").glob(f"*/{context}.h5ad"))
        if len(hits) != 1:
            raise KeyError(f"{context}: {len(hits)} shards")
        return hits[0].parent.name

    def shard(self, context):
        return self.root / "shards" / self.dataset(context) / f"{context}.h5ad"

    def obs(self, context):
        return pd.read_parquet(self.shard(context).with_name(f"{context}.obs.parquet"))

    def blocks(self, context):
        return pd.read_parquet(self.shard(context).with_name(f"{context}.blocks.parquet"))

    def cells(self, context, perturbation=None, batch=None, controls=False, measured_only=True):
        """(obs rows, CSR counts) of a perturbation's cells or of the used controls, read as contiguous blocks."""
        b = self.blocks(context)
        sel = (b.kind == "control") & b.control_kind.isin(self.cfg["controls"]["is_control_kinds"]) if controls else (b.perturbation == perturbation)
        if batch is not None:
            sel &= b.batch == batch
        runs = b[sel]
        obs = self.obs(context)
        rows = np.concatenate([np.arange(s, e) for s, e in zip(runs.start, runs.end)]) if len(runs) else np.empty(0, int)
        with h5py.File(self.shard(context), "r") as h:
            x = h["X"]
            pieces = []
            for s, e in zip(runs.start, runs.end):
                ip = x["indptr"][s:e + 1]
                lo, hi = int(ip[0]), int(ip[-1])
                pieces.append(sp.csr_matrix((x["data"][lo:hi], x["indices"][lo:hi], ip - lo), shape=(e - s, int(x.attrs["shape"][1]))))
            m = sp.vstack(pieces).tocsr() if pieces else sp.csr_matrix((0, int(x.attrs["shape"][1])))
            measured = h["var/measured"][:]
        o = obs.iloc[rows]
        if controls:
            o = o[o.control_used.to_numpy(bool)]
            m = m[o.index.get_indexer(o.index) if False else np.flatnonzero(obs.iloc[rows].control_used.to_numpy(bool))]
        return o, (m[:, np.flatnonzero(measured)] if measured_only else m)

    def suffstats_dir(self, context):
        return self.root / "suffstats" / self.dataset(context) / context

    def index(self, context):
        return pd.read_parquet(self.suffstats_dir(context) / "index.parquet")

    def cell_counts(self, context):
        return pd.read_parquet(self.suffstats_dir(context) / "cell_counts.parquet")

    def measured_genes(self, context):
        return pd.read_parquet(self.suffstats_dir(context) / "genes.parquet")

    def moments(self, context, profile="authors", half="all", table="pert"):
        """(index rows, dict of sum name -> array rows x measured genes) for one table, profile and half."""
        ix = self.index(context)
        ix = ix[(ix.table == table) & (ix.profile == profile) & (ix.half == half)].sort_values("row")
        name = "pert_moments.h5" if table == "pert" else "ctrl_batch_moments.h5"
        with h5py.File(self.suffstats_dir(context) / name, "r") as h:
            sums = {s: h[f"{profile}/{half}/{s}"][:] for s in SUMS}
        return ix.reset_index(drop=True), sums

    def means(self, context, profile="authors", half="all", table="pert"):
        """Pseudobulks in the three spaces: theta_A, m_B, m_C (rows x measured genes) and per-gene variances of x^B and x^C."""
        ix, s = self.moments(context, profile, half, table)
        n = ix.n.to_numpy()[:, None]
        out = dict(theta_A=s["sum_c"] / ix.sum_L.to_numpy()[:, None], m_B=s["sum_xB"] / n, m_C=s["sum_xC"] / n)
        d = np.maximum(n - 1, 1)
        out["var_B"] = (s["sum_xB2"] - s["sum_xB"] ** 2 / n) / d
        out["var_C"] = (s["sum_xC2"] - s["sum_xC"] ** 2 / n) / d
        return ix, out

    def control_reference(self, context, profile="authors", half="all"):
        """Batch-matched control references per perturbation: (perturbation index, ref_C, ref_B_linear, ref_A_theta)."""
        cix, cm = self.means(context, profile, half, "ctrl")
        pix, _ = self.moments(context, profile, half, "pert")
        cnt = self.cell_counts(context)
        cnt = cnt[(cnt.profile == profile) & (cnt.half == half)]
        wn = self._weights(cnt, pix.perturbation, cix.batch, "n_pb")
        wl = self._weights(cnt, pix.perturbation, cix.batch, "L_pb")
        return pix, np.asarray(wn @ cm["m_C"]), np.asarray(wn @ cm["m_B"]), np.asarray(wl @ cm["theta_A"])

    @staticmethod
    def _weights(cnt, perts, batches, column):
        pi, bi = pd.Index(perts).get_indexer(cnt.perturbation), pd.Index(batches).get_indexer(cnt.batch)
        ok = (pi >= 0) & (bi >= 0) & (cnt[column].to_numpy() > 0)
        w = sp.csr_matrix((cnt[column].to_numpy(np.float64)[ok], (pi[ok], bi[ok])), shape=(len(perts), len(batches)))
        tot = np.asarray(w.sum(1)).ravel()
        return sp.diags(np.where(tot > 0, 1 / np.where(tot > 0, tot, 1), 0)) @ w

    def deltas(self, context, profile="authors", half="all", spaces=("A", "B", "C")):
        """Batch-matched deltas per perturbation: dA and dC from disk, DB (linear) and dB (log) from the moments."""
        d = self.root / "deltas" / self.dataset(context) / context
        pix, ref_c, ref_b, ref_a = self.control_reference(context, profile, half)
        out = {}
        with h5py.File(d / "deltas.h5", "r") as h:
            if "A" in spaces:
                out["dA"] = h[f"{profile}/{half}/dA"][:]
            if "C" in spaces:
                out["dC"] = h[f"{profile}/{half}/dC"][:]
        if "B" in spaces:
            _, m = self.means(context, profile, half, "pert")
            out["DB"] = m["m_B"] - ref_b
            out["dB"] = np.log1p(m["m_B"]) - np.log1p(ref_b)
        return pix, out

    def pert_qc(self, context):
        return pd.read_parquet(self.root / "deltas" / self.dataset(context) / context / "pert_qc.parquet")

    def masks(self, context, profile="authors"):
        """G0 (detected in >= 5 % of controls), per-target masks and 5 kb TSS neighbor masks over the measured genes."""
        g = self.measured_genes(context)
        info = self.genes.set_index("ensembl_id").reindex(g.ensembl_id)
        window = self.cfg["genes"]["neighbor_window_bp"]
        ids = g.ensembl_id.to_numpy()

        def neighbors(target):
            t = info.loc[target] if target in info.index else None
            if t is None or pd.isna(t.tss):
                return np.zeros(len(ids), bool)
            return ((info.chromosome.to_numpy() == t.chromosome) & (np.abs(info.tss.to_numpy() - t.tss) <= window) & (ids != target))

        return dict(g0=g[f"g0_{profile}"].to_numpy(), target=lambda t: ids == t, neighbors=neighbors, ensembl_ids=ids)

    def scores(self, context):
        p = self.root / "scores" / f"{context}.parquet"
        return pd.read_parquet(p) if p.exists() else None

    def delta_from_cells(self, context, perturbation, profile="authors"):
        """δ^C of one perturbation straight from the shard (for populations below the moment threshold)."""
        o, m = self.cells(context, perturbation)
        if profile == "strict":
            keep = o.qc_strict.to_numpy(bool)
            o, m = o[keep], m[keep]
        L = o.lib_size.to_numpy(np.float64)
        xc = m.multiply(self.kappa / L[:, None]).tocsr()
        xc.data = np.log1p(xc.data)
        cix, cm = self.means(context, profile, "all", "ctrl")
        w = pd.Series(o.batch.astype(str).to_numpy()).value_counts(normalize=True)
        ref = sum(w[b] * cm["m_C"][list(cix.batch).index(b)] for b in w.index if b in list(cix.batch))
        return np.asarray(xc.mean(0)).ravel() - ref
