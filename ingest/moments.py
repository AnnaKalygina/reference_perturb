"""Moments stage: float64 sums per perturbation and per control batch, in spaces A, B, C, for all/A/B halves and both profiles."""
import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))
from config import load, path  # noqa: E402
from receipts import done, finish, write_atomic  # noqa: E402

SUMS = ["n", "sum_c", "sum_c2", "sum_xB", "sum_xB2", "sum_xC", "sum_xC2"]
SCALARS = ["n", "sum_L", "sum_L2", "sum_invL"]
HALVES = ["all", "A", "B"]
PROFILES = ["authors", "strict"]


def read_rows(x, start, end):
    """CSR of shard rows [start, end) from an open h5 X group."""
    ip = x["indptr"][start:end + 1]
    lo, hi = int(ip[0]), int(ip[-1])
    return sp.csr_matrix((x["data"][lo:hi], x["indices"][lo:hi], ip - lo), shape=(end - start, int(x.attrs["shape"][1])))


def spaces(m, L, kappa, measured_col):
    """(counts, xB, xC) CSR matrices restricted to measured genes."""
    c = m[:, measured_col].astype(np.float64)
    xb = c.multiply(kappa / L[:, None]).tocsr()
    xc = xb.copy()
    xc.data = np.log1p(xc.data)
    return c, xb, xc


def group_sums(c, xb, xc, L, groups, n_groups):
    """Per-group float64 sums of the six per-gene series and the four scalars; groups < 0 are ignored."""
    ok = groups >= 0
    ind = sp.csr_matrix((np.ones(ok.sum()), (groups[ok], np.flatnonzero(ok))), shape=(n_groups, len(groups)))
    per_gene = [np.asarray((ind @ x).todense()) for x in (c, c.multiply(c), xb, xb.multiply(xb), xc, xc.multiply(xc))]
    n = np.bincount(groups[ok], minlength=n_groups).astype(np.float64)
    scal = [n, np.bincount(groups[ok], weights=L[ok], minlength=n_groups), np.bincount(groups[ok], weights=L[ok] ** 2, minlength=n_groups),
            np.bincount(groups[ok], weights=1.0 / L[ok], minlength=n_groups)]
    return per_gene, scal


class Table:
    """Rows of sums appended per population, written as HDF5 datasets per profile, half and sum."""

    def __init__(self, n_genes, cfg):
        self.rows, self.index, self.g, self.cfg = {}, [], n_genes, cfg

    def add(self, key, profile, half, per_gene, scal):
        self.index.append(dict(key, profile=profile, half=half, n=scal[0], sum_L=scal[1], sum_L2=scal[2], sum_invL=scal[3], row=len(self.rows.get((profile, half), []))))
        self.rows.setdefault((profile, half), []).append(np.stack([np.asarray(v).reshape(-1) for v in per_gene]))

    def write(self, target):
        opts = dict(compression="gzip", compression_opts=4, shuffle=True)
        with write_atomic(target) as tmp, h5py.File(tmp, "w") as h:
            h.attrs["config_hash"], h.attrs["code_commit"] = self.cfg["hash"], self.cfg["commit"]
            for (profile, half), rows in self.rows.items():
                block = np.stack(rows)
                for i, name in enumerate(SUMS[1:]):
                    h.create_dataset(f"{profile}/{half}/{name}", data=block[:, i, :], chunks=(min(32, len(rows)), min(2048, self.g)), **opts)
        return pd.DataFrame(self.index)


def run(context, cfg):
    """Accumulate the moments of one context from its sorted shard."""
    started = time.time()
    dataset = context_dataset(context, cfg)
    shard = path(cfg, "shards", dataset, f"{context}.h5ad")
    out = path(cfg, "suffstats", dataset, context)
    outputs = [out / "pert_moments.h5", out / "ctrl_batch_moments.h5", out / "cell_counts.parquet", out / "index.parquet", out / "genes.parquet"]
    if done(cfg, "moments", context, outputs):
        print(f"{context}: done", flush=True)
        return
    obs = pd.read_parquet(shard.with_name(f"{context}.obs.parquet"))
    blocks = pd.read_parquet(shard.with_name(f"{context}.blocks.parquet"))
    kappa, min_cells = cfg["kappa"], cfg["populations"]["min_cells_for_moments"]
    with h5py.File(shard, "r") as h:
        measured = h["var/measured"][:]
        gene_ids = h["var/ensembl_id"][:].astype(str)
        measured_col = np.flatnonzero(measured)
        x = h["X"]
        strict = obs.qc_strict.to_numpy(bool)
        half = obs.half.astype(str).to_numpy()
        L = obs.lib_size.to_numpy(np.float64)
        batch = obs.batch.astype(str).to_numpy()
        ctrl_rows = blocks[(blocks.kind == "control")]
        used = obs.control_used.to_numpy(bool)
        ctrl = Table(len(measured_col), cfg)
        ctrl_mean = {p: {} for p in PROFILES}
        ctrl_n_half = {}
        detect = {p: np.zeros(len(measured_col)) for p in PROFILES}
        n_ctrl = {p: 0 for p in PROFILES}
        target_cols = {}
        ctrl_xb_target = {}
        targets = sorted({t for ts in obs.target_ensembl.dropna().astype(str) for t in ts.split("+")} & set(gene_ids[measured]))
        gene_pos = {g: i for i, g in enumerate(gene_ids[measured_col])}
        tcols = np.array([gene_pos[t] for t in targets], int)
        for b, grp in ctrl_rows.groupby("batch", sort=False):
            rows = np.concatenate([np.arange(s, e) for s, e in zip(grp.start, grp.end)])
            rows = rows[used[rows]]
            if len(rows) == 0:
                continue
            m = read_rows(x, int(rows.min()), int(rows.max()) + 1)[rows - rows.min()]
            c, xb, xc = spaces(m, L[rows], kappa, measured_col)
            for p in PROFILES:
                mask = np.ones(len(rows), bool) if p == "authors" else strict[rows]
                g = np.where(mask, np.select([half[rows] == "A", half[rows] == "B"], [1, 2], 0), -1)
                per_gene, scal = group_sums(c, xb, xc, L[rows], np.where(g >= 0, 0, -1), 1)
                ctrl_n_half[(p, b, "all")] = int(scal[0][0])
                if scal[0][0] > 0:
                    ctrl.add(dict(batch=b), p, "all", per_gene, [s[0] for s in scal])
                    ctrl_mean[p][(b, "all")] = (per_gene[0][0] / scal[1][0], per_gene[2][0] / scal[0][0], per_gene[4][0] / scal[0][0])
                    detect[p] += np.asarray((c[mask] > 0).sum(0)).ravel()
                    n_ctrl[p] += int(scal[0][0])
                for hname, hid in (("A", 1), ("B", 2)):
                    sel = np.where(g == hid, 0, -1)
                    pg, sc = group_sums(c, xb, xc, L[rows], sel, 1)
                    ctrl_n_half[(p, b, hname)] = int(sc[0][0])
                    if sc[0][0] > 0:
                        ctrl.add(dict(batch=b), p, hname, pg, [s[0] for s in sc])
                        ctrl_mean[p][(b, hname)] = (pg[0][0] / sc[1][0], pg[2][0] / sc[0][0], pg[4][0] / sc[0][0])
            ctrl_xb_target[b] = (xb[:, tcols].toarray() if len(tcols) else np.zeros((len(rows), 0)), strict[rows])
        g0 = {p: detect[p] >= cfg["qc"]["g0_control_detection_fraction"] * max(n_ctrl[p], 1) for p in PROFILES}
        pert = Table(len(measured_col), cfg)
        counts, qc_rows = [], []
        perts = blocks[blocks.kind != "control"]
        for (kind, label), grp in perts.groupby(["kind", "perturbation"], sort=False):
            start, end = int(grp.start.min()), int(grp.end.max())
            rows = np.arange(start, end)
            m = read_rows(x, start, end)
            c, xb, xc = spaces(m, L[rows], kappa, measured_col)
            bt = batch[rows]
            guides = obs.guide.astype(str).to_numpy()[rows]
            for p in PROFILES:
                pmask = np.ones(len(rows), bool) if p == "authors" else strict[rows]
                for hname in HALVES:
                    hmask = pmask & ((half[rows] == hname) if hname != "all" else True)
                    matched = np.array([ctrl_n_half.get((p, b, hname), 0) > 0 for b in bt])
                    inc = hmask & matched
                    excluded = int((hmask & ~matched).sum())
                    per_gene, scal = group_sums(c, xb, xc, L[rows], np.where(inc, 0, -1), 1)
                    n = int(scal[0][0])
                    for b in np.unique(bt[hmask]):
                        sel = hmask & (bt == b)
                        counts.append(dict(perturbation=label, kind=kind, batch=b, profile=p, half=hname, n_pb=int((sel & matched).sum()), L_pb=float(L[rows][sel & matched].sum()),
                                           n_excluded=int((sel & ~matched).sum())))
                    key = dict(perturbation=label, kind=kind, n_excluded=excluded, n_batches=int(len(np.unique(bt[inc]))))
                    if n >= min_cells:
                        pert.add(key, p, hname, per_gene, [s[0] for s in scal])
                    if hname == "all":
                        qc_rows.append(dict(key, profile=p, n=n, **guide_agreement(c, xc, L[rows], bt, guides, inc, ctrl_mean[p], g0[p], kappa, min_cells),
                                            **knockdown(label, xb, bt, inc, targets, tcols, ctrl_xb_target, p, cfg, context)))
        out.mkdir(parents=True, exist_ok=True)
        pert_index = pert.write(out / "pert_moments.h5")
        ctrl_index = ctrl.write(out / "ctrl_batch_moments.h5")
        pert_index["table"], ctrl_index["table"] = "pert", "ctrl"
        pd.concat([pert_index, ctrl_index], ignore_index=True).to_parquet(out / "index.parquet")
        pd.DataFrame(counts).to_parquet(out / "cell_counts.parquet")
        pd.DataFrame(qc_rows).to_parquet(out / "guide_knockdown.parquet")
        pd.DataFrame(dict(ensembl_id=gene_ids[measured_col], axis_index=measured_col, g0_authors=g0["authors"], g0_strict=g0["strict"],
                          detected_controls_authors=detect["authors"], detected_controls_strict=detect["strict"])).to_parquet(out / "genes.parquet")
    finish(cfg, "moments", context, outputs, [str(shard)], started, dict(n_pert_rows=int(len(pert_index)), n_ctrl_rows=int(len(ctrl_index))))
    print(f"{context}: {len(pert_index)} perturbation rows, {len(ctrl_index)} control rows, {round(time.time() - started)} s", flush=True)


CONTROL_PART = re.compile(r"negctrl|non.?target|safe.?target|^nt$|^ctrl", re.I)


def guide_key(guide):
    """Targeting components of a construct id (parts split on | or ;), so a non-targeting partner does not create a new guide."""
    parts = [p for p in re.split(r"[|;]", guide) if not CONTROL_PART.search(p.strip())]
    return "|".join(parts) if parts else guide


def guide_agreement(c, xc, L, bt, guides, inc, ctrl_mean, g0, kappa, min_cells, max_guides=6):
    """Pearson r over G0 between the C-space deltas of the perturbation's guides (min and mean over the largest guides); NaN below two guides."""
    keys = np.array([guide_key(g) for g in guides], object)
    sizes = pd.Series(keys[inc]).value_counts()
    sizes = sizes[~sizes.index.isin(["", "nan", "None"])]
    used = sizes[sizes >= min_cells].head(max_guides)
    if len(used) < 2:
        return dict(n_guides=int(len(sizes)), n_guides_used=int(len(used)), guide_r_min=np.nan, guide_r_mean=np.nan)
    deltas = []
    for g in used.index:
        sel = inc & (keys == g)
        mean = np.asarray(xc[sel].mean(0)).ravel()
        w = pd.Series(bt[sel]).value_counts(normalize=True)
        ref = sum(w[b] * ctrl_mean[(b, "all")][2] for b in w.index if (b, "all") in ctrl_mean)
        deltas.append((mean - ref)[g0])
    r = [np.corrcoef(deltas[i], deltas[j])[0, 1] for i in range(len(deltas)) for j in range(i + 1, len(deltas))]
    return dict(n_guides=int(len(sizes)), n_guides_used=int(len(used)), guide_r_min=float(np.nanmin(r)), guide_r_mean=float(np.nanmean(r)))


def knockdown(label, xb, bt, inc, targets, tcols, ctrl_xb_target, profile, cfg, context):
    """Knockdown ratio and directional Mann-Whitney p per target of the perturbation; NaN where the target is not measured."""
    out = {}
    direction = "up" if context_dataset(context, cfg) in cfg["perturbation_direction"]["up"] else cfg["perturbation_direction"]["default"]
    pos = {t: i for i, t in enumerate(targets)}
    for j, t in enumerate(label_targets(label, cfg, context)):
        suffix = "" if j == 0 else f"_{j + 1}"
        if t not in pos or inc.sum() == 0:
            out.update({f"target{suffix}": t, f"knockdown_ratio{suffix}": np.nan, f"knockdown_p{suffix}": np.nan})
            continue
        col = pos[t]
        vals = xb[inc][:, tcols[col]].toarray().ravel()
        w = pd.Series(bt[inc]).value_counts(normalize=True)
        sample, ref = [], 0.0
        rng = np.random.default_rng(int.from_bytes(hashlib.blake2b(f"{cfg['seed']}|{context}|{label}|mwu".encode(), digest_size=8).digest(), "little"))
        total = cfg["populations"]["mwu_control_sample"]
        for b in w.index:
            cx, cs = ctrl_xb_target[b]
            cx = cx[:, col] if profile == "authors" else cx[cs, col]
            if len(cx) == 0:
                continue
            ref += w[b] * cx.mean()
            m = min(len(cx), int(round(total * w[b])))
            sample.append(rng.choice(cx, m, replace=False) if m < len(cx) else cx)
        sample = np.concatenate(sample) if sample else np.empty(0)
        p = mannwhitneyu(vals, sample, alternative="less" if direction == "down" else "greater", method="asymptotic", use_continuity=True).pvalue if len(sample) and len(vals) else np.nan
        out.update({f"target{suffix}": t, f"knockdown_ratio{suffix}": float(vals.mean() / ref) if ref > 0 else np.nan, f"knockdown_p{suffix}": float(p)})
    return out


def label_targets(label, cfg, context):
    """Ensembl ids of a perturbation label, from the shard obs mapping cached per context."""
    return LABEL_TARGETS.get(context, {}).get(label, [])


LABEL_TARGETS = {}


def context_dataset(context, cfg):
    """Dataset name of a context from the shard directory layout."""
    hits = list(path(cfg, "shards").glob(f"*/{context}.h5ad"))
    if len(hits) != 1:
        raise SystemExit(f"{context}: {len(hits)} shards found")
    return hits[0].parent.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("context")
    a = ap.parse_args()
    cfg = load()
    obs = pd.read_parquet(path(cfg, "shards", context_dataset(a.context, cfg), f"{a.context}.obs.parquet"), columns=["perturbation", "target_ensembl"])
    LABEL_TARGETS[a.context] = {k: str(v).split("+") for k, v in obs.dropna().drop_duplicates("perturbation").set_index("perturbation").target_ensembl.items()}
    run(a.context, cfg)


if __name__ == "__main__":
    main()
