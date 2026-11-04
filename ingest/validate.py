"""Validation stage: independent recomputation of deltas from shard blocks, pseudo-perturbation nulls, and the VCC2026 comparison."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from config import load, path  # noqa: E402
from moments import context_dataset  # noqa: E402
from receipts import done, finish, write_json  # noqa: E402

VCC = Path("/cluster/work/boeva/VCC/VCC2026/data")
VCC_CONTEXTS = {"replogle22_essential__K562": "replogle_nadig__k562", "replogle22_essential__RPE1": "replogle_nadig__rpe1", "nadig25__HepG2": "replogle_nadig__hepg2"}
POOL = 20000
TALK_CONTEXT = "replogle22_essential__K562"


def rng_for(cfg, context, purpose):
    return np.random.default_rng(int.from_bytes(hashlib.blake2b(f"{cfg['seed']}|{context}|{purpose}".encode(), digest_size=8).digest(), "little"))


def block(x, start, end, measured_col):
    """CSR of contiguous shard rows restricted to the measured genes."""
    ip = x["indptr"][start:end + 1]
    lo, hi = int(ip[0]), int(ip[-1])
    return sp.csr_matrix((x["data"][lo:hi], x["indices"][lo:hi], ip - lo), shape=(end - start, int(x.attrs["shape"][1])))[:, measured_col]


def xc(m, L, kappa):
    """log1p(kappa c / L) per cell, dense float64."""
    return np.log1p(kappa * m.toarray() / L[:, None])


class Controls:
    """Per-batch control cells of a context: full per-batch means, and a seeded dense pool for the nulls."""

    def __init__(self, x, obs, blocks, measured_col, kappa, rng):
        self.rows, self.mean, self.var, self.n = {}, {}, {}, {}
        L = obs.lib_size.to_numpy(np.float64)
        used = obs.control_used.to_numpy(bool)
        pool_rows, pool_xc = [], []
        for b, g in blocks[blocks.kind == "control"].groupby("batch", sort=False):
            parts = []
            for s, e in zip(g.start, g.end):
                rows = np.arange(s, e)
                rows = rows[used[rows]]
                if len(rows) == 0:
                    continue
                m = block(x, int(rows[0]), int(rows[-1]) + 1, measured_col)[rows - rows[0]]
                parts.append((rows, xc(m, L[rows], kappa)))
            if not parts:
                continue
            rows = np.concatenate([r for r, _ in parts])
            v = np.concatenate([d for _, d in parts])
            self.rows[str(b)], self.n[str(b)] = rows, len(rows)
            self.mean[str(b)], self.var[str(b)] = v.mean(0), v.var(0, ddof=1) if len(rows) > 1 else np.zeros(v.shape[1])
            pick = rng.choice(len(rows), min(len(rows), max(1, int(POOL * len(rows) / used.sum()))), replace=False)
            pool_rows.append(rows[pick])
            pool_xc.append(v[pick].astype(np.float32))
        self.pool_rows = np.concatenate(pool_rows)
        self.pool_xc = np.concatenate(pool_xc)
        self.pool_batch = obs.batch.astype(str).to_numpy()[self.pool_rows]


def first_form_delta(pert_xc, batches, ctrl_means):
    """δ^C = Σ_b w_b [mean(p, b) − mean(ctrl, b)] over the batches that have a control mean."""
    delta, n = 0.0, 0
    for b in np.unique(batches):
        if b not in ctrl_means:
            continue
        sel = batches == b
        delta = delta + sel.sum() * (pert_xc[sel].mean(0) - ctrl_means[b])
        n += int(sel.sum())
    return delta / max(n, 1)


def run(context, cfg):
    """Checks (a) and (b) for every context, (d) where a VCC2026 reference exists; writes validation/<context>.json."""
    started = time.time()
    dataset = context_dataset(context, cfg)
    out = path(cfg, "validation", f"{context}.json")
    if done(cfg, "validate", context, [out]):
        print(f"{context}: done", flush=True)
        return
    kappa = cfg["kappa"]
    shard = path(cfg, "shards", dataset, f"{context}.h5ad")
    obs = pd.read_parquet(shard.with_name(f"{context}.obs.parquet"))
    blocks = pd.read_parquet(shard.with_name(f"{context}.blocks.parquet"))
    L = obs.lib_size.to_numpy(np.float64)
    batch = obs.batch.astype(str).to_numpy()
    ddir = path(cfg, "deltas", dataset, context)
    qc = pd.read_parquet(ddir / "pert_qc.parquet")
    result = dict(context=context, dataset=dataset, config_hash=cfg["hash"], code_commit=cfg["commit"])
    with h5py.File(shard, "r") as h, h5py.File(ddir / "deltas.h5", "r") as hd:
        x = h["X"]
        measured_col = np.flatnonzero(h["var/measured"][:])
        ctrl = Controls(x, obs, blocks, measured_col, kappa, rng_for(cfg, context, "pool"))
        rows_by_pert = {p: (int(g.start.min()), int(g.end.max())) for p, g in blocks[blocks.kind != "control"].groupby("perturbation")}
        stored = qc[(qc.profile == "authors") & (qc.half == "all") & (qc.delta_row >= 0)]
        pick = rng_for(cfg, context, "validate").choice(stored.perturbation.to_numpy(), min(20, len(stored)), replace=False)
        dC = hd["authors/all/dC"]
        excess = []
        for p in pick:
            s, e = rows_by_pert[p]
            v = xc(block(x, s, e, measured_col), L[s:e], kappa)
            direct = first_form_delta(v, batch[s:e], ctrl.mean)
            sto = dC[int(stored[stored.perturbation == p].delta_row.iloc[0])].astype(np.float64)
            excess.append(float(np.max(np.abs(direct - sto) - 1e-6 * np.abs(sto))))
        result["a_recompute"] = dict(n=len(pick), max_excess_over_tolerance=max(excess) if excess else None, passed=bool(excess and max(excess) <= 1e-6))
        n_med = int(np.median(stored.n)) if len(stored) else 20
        result["b_null"] = {}
        for family in ("control_mix", "perturbation_mix"):
            rng = rng_for(cfg, context, f"null_{family}")
            zs, sds = [], []
            for _ in range(50):
                if family == "control_mix":
                    cells = rng.choice(len(ctrl.pool_rows), min(n_med, len(ctrl.pool_rows)), replace=False)
                else:
                    p = rng.choice(stored.perturbation.to_numpy())
                    s, e = rows_by_pert[p]
                    mix = pd.Series(batch[s:e]).value_counts(normalize=True)
                    cells = np.concatenate([rng.choice(np.flatnonzero(ctrl.pool_batch == b), min(max(1, int(round(n_med * f))), int((ctrl.pool_batch == b).sum())), replace=False)
                                            for b, f in mix.items() if (ctrl.pool_batch == b).any()])
                rest = np.setdiff1d(np.arange(len(ctrl.pool_rows)), cells)
                means, counts = {}, {}
                for b in np.unique(ctrl.pool_batch[rest]):
                    sel = rest[ctrl.pool_batch[rest] == b]
                    means[b], counts[b] = ctrl.pool_xc[sel].astype(np.float64).mean(0), len(sel)
                sample = ctrl.pool_xc[cells].astype(np.float64)
                delta = first_form_delta(sample, ctrl.pool_batch[cells], means)
                w = pd.Series(ctrl.pool_batch[cells]).value_counts(normalize=True)
                within = sum((w[b] ** 2) * (sample[ctrl.pool_batch[cells] == b].var(0, ddof=1) if (ctrl.pool_batch[cells] == b).sum() > 1 else ctrl.var[b]) / (ctrl.pool_batch[cells] == b).sum() for b in w.index)
                noise_g = within + sum((w[b] ** 2) * ctrl.var[b] / counts[b] for b in w.index if b in counts)
                noise = float(noise_g.sum())
                zs.append((float((delta ** 2).sum()) - noise) / noise)
                sds.append(float(np.sqrt(2 * (noise_g ** 2).sum())) / noise)
            z, sd_null = np.array(zs), float(np.median(sds))
            se = float(z.std(ddof=1) / np.sqrt(len(z)))
            result["b_null"][family] = dict(mean_z=float(z.mean()), median_z=float(np.median(z)), sd_z=float(z.std(ddof=1)), sd_null_gaussian=sd_null,
                                             n_cells=n_med, n_draws=len(z), passed=bool(abs(z.mean()) <= max(3 * se, 0.02)))
    if context in VCC_CONTEXTS:
        result["d_vcc2026"] = compare_vcc(context, VCC_CONTEXTS[context], cfg)
    if context == TALK_CONTEXT:
        result["c_talk"] = compare_talk(context, cfg)
    write_json(out, result)
    finish(cfg, "validate", context, [out], [str(shard)], started)
    print(f"{context}: a {result['a_recompute']['passed']}, b {[v['passed'] for v in result['b_null'].values()]}, d {result.get('d_vcc2026', {}).get('passed')}, {round(time.time() - started)} s", flush=True)


def compare_talk(context, cfg):
    """K562 essential against the talk of 2026-10-20: perturbations with n >= 20, median M, median rho and the share rho >= 0.5."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from reference_perturb import Reference

    R = Reference(cfg["root"])
    talk = pd.read_csv(Path(cfg["root"]) / "config" / "validation_k562_essential_talk.csv")
    qc = R.pert_qc(context)
    singles = qc[(qc.profile == "authors") & (qc.half == "all") & (qc.kind == "single")]
    n20 = singles[singles.n >= 20]
    pix, d = R.deltas(context, spaces=("C",))
    ix, mom = R.means(context)
    cix, cm = R.means(context, table="ctrl")
    cnt = R.cell_counts(context)
    cnt = cnt[(cnt.profile == "authors") & (cnt.half == "all")]
    g0 = R.masks(context)["g0"]
    ids = R.measured_genes(context).ensembl_id.to_numpy()
    pa, da = R.deltas(context, half="A", spaces=("C",))
    pb, db = R.deltas(context, half="B", spaces=("C",))
    rowA, rowB = {p: i for i, p in enumerate(pa.perturbation)}, {p: i for i, p in enumerate(pb.perturbation)}
    cn = dict(zip(cix.batch, cix.n))
    rows = []
    for i, p in enumerate(pix.perturbation):
        if p not in set(n20.perturbation):
            continue
        t = n20[n20.perturbation == p].iloc[0]
        target = str(t.target) if isinstance(t.target, str) else None
        mask = g0 & (ids != target)
        w = cnt[cnt.perturbation == p]
        wb = dict(zip(w.batch, w.n_pb / w.n_pb.sum()))
        ctrl_var = sum((wb[b] ** 2) * cm["var_C"][list(cix.batch).index(b)] / cn[b] for b in wb if b in cn)
        noise = mom["var_C"][i] / ix.n.iloc[i] + ctrl_var
        M = float(((d["dC"][i].astype(np.float64) ** 2) - noise)[mask].sum())
        r = np.corrcoef(da["dC"][rowA[p]][mask], db["dC"][rowB[p]][mask])[0, 1] if p in rowA and p in rowB else np.nan
        rows.append(dict(perturbation=p, n=int(t.n), M=M, rho=2 * r / (1 + r) if np.isfinite(r) else np.nan))
    ours = pd.DataFrame(rows)
    shared = ours.merge(talk[["perturbation", "M", "rho"]], on="perturbation", suffixes=("", "_talk"))
    out = dict(n_perturbations=int(len(n20)), n_talk=int(len(talk)), n_shared=int(len(shared)), median_M=float(ours.M.median()), median_M_talk=float(talk.M.median()),
               median_rho=float(ours.rho.median()), median_rho_talk=float(talk.rho.median()), share_rho_05=float((ours.rho >= 0.5).mean()), share_rho_05_talk=float((talk.rho >= 0.5).mean()),
               spearman_M=float(shared[["M", "M_talk"]].corr("spearman").iloc[0, 1]), spearman_rho=float(shared[["rho", "rho_talk"]].corr("spearman").iloc[0, 1]))
    gate = json.loads((Path(cfg["root"]) / "manifest" / f"{context}.json").read_text())["gate"]
    out["gate_excluded_single_labels"] = [g["label"] for g in gate if g["n_targets"] == 1]
    out["passed"] = bool(0 <= out["n_talk"] - out["n_perturbations"] <= len(out["gate_excluded_single_labels"]) and abs(out["median_M"] / out["median_M_talk"] - 1) <= 0.05
                         and abs(out["median_rho"] - out["median_rho_talk"]) <= 0.03 and abs(out["share_rho_05"] - out["share_rho_05_talk"]) <= 0.03)
    return out


def compare_vcc(context, vcc_context, cfg):
    """Spearman per shared target between our log ratio of batch-matched B means and VCC2026's lfc over its measured genes with control CPM >= 1."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from reference_perturb import Reference

    R = Reference(cfg["root"])
    pix, ref_c, ref_b, ref_a = R.control_reference(context)
    _, m = R.means(context)
    ours = np.log(m["m_B"] / ref_b)
    genes = R.measured_genes(context).ensembl_id.to_numpy()
    sym = R.genes.set_index("ensembl_id").symbol.reindex(genes).to_numpy()
    with h5py.File(VCC / f"{vcc_context}.h5", "r") as h:
        vg = h["genes"][:].astype(str)
        keep_v = h["measured"][:] & (h["control/mean_cpm"][:] >= 1)
        targets = h["pert/targets"][:].astype(str)
        lfc = h["pert/lfc"][:]
    vpos = {g: i for i, g in enumerate(vg)}
    shared = [(i, vpos[s]) for i, s in enumerate(sym) if isinstance(s, str) and s in vpos and keep_v[vpos[s]]]
    oi, vi = np.array([a for a, _ in shared]), np.array([b for _, b in shared])
    tpos = {t: i for i, t in enumerate(targets)}
    rhos = []
    for i, p in enumerate(pix.perturbation):
        if p in tpos:
            a, b = ours[i, oi], lfc[tpos[p], vi]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 100:
                rhos.append(spearmanr(a[ok], b[ok]).correlation)
    rhos = np.array(rhos)
    return dict(n_targets=int(len(rhos)), n_genes=int(len(oi)), median_spearman=float(np.median(rhos)) if len(rhos) else None,
                p5_spearman=float(np.percentile(rhos, 5)) if len(rhos) else None, passed=bool(len(rhos) and np.median(rhos) >= 0.95))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("context")
    a = ap.parse_args()
    run(a.context, load())
