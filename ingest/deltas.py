"""Deltas stage: batch-matched deltas in spaces A and C from the stored moments, n_eff, flags, knockdown and guide agreement in pert_qc."""
import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from config import load, path  # noqa: E402
from moments import HALVES, PROFILES, context_dataset  # noqa: E402
from receipts import done, finish, write_atomic  # noqa: E402

BLOCK = 1024


def weights(counts, perts, batches, column):
    """Row-normalized sparse (perts x batches) weight matrix from a cell_counts column."""
    pi = pd.Index(perts).get_indexer(counts.perturbation)
    bi = pd.Index(batches).get_indexer(counts.batch)
    ok = (pi >= 0) & (bi >= 0) & (counts[column].to_numpy() > 0)
    w = sp.csr_matrix((counts[column].to_numpy(np.float64)[ok], (pi[ok], bi[ok])), shape=(len(perts), len(batches)))
    tot = np.asarray(w.sum(1)).ravel()
    return sp.diags(np.where(tot > 0, 1 / np.where(tot > 0, tot, 1), 0)) @ w


def run(context, cfg):
    """Compute and store the deltas and pert_qc of one context."""
    started = time.time()
    dataset = context_dataset(context, cfg)
    src = path(cfg, "suffstats", dataset, context)
    out = path(cfg, "deltas", dataset, context)
    outputs = [out / "deltas.h5", out / "pert_qc.parquet"]
    if done(cfg, "deltas", context, outputs):
        print(f"{context}: done", flush=True)
        return
    kappa, pop = cfg["kappa"], cfg["populations"]
    index = pd.read_parquet(src / "index.parquet")
    counts = pd.read_parquet(src / "cell_counts.parquet")
    gk = pd.read_parquet(src / "guide_knockdown.parquet")
    genes = pd.read_parquet(src / "genes.parquet")
    qc_rows = []
    out.mkdir(parents=True, exist_ok=True)
    opts = dict(compression="gzip", compression_opts=4, shuffle=True)
    with h5py.File(src / "pert_moments.h5", "r") as hp, h5py.File(src / "ctrl_batch_moments.h5", "r") as hc, write_atomic(out / "deltas.h5") as tmp, h5py.File(tmp, "w") as ho:
        ho.attrs["config_hash"], ho.attrs["code_commit"] = cfg["hash"], cfg["commit"]
        for profile in PROFILES:
            for half in HALVES:
                pi = index[(index.table == "pert") & (index.profile == profile) & (index.half == half)].sort_values("row")
                ci = index[(index.table == "ctrl") & (index.profile == profile) & (index.half == half)].sort_values("row")
                cnt = counts[(counts.profile == profile) & (counts.half == half)]
                ctrl_n = dict(zip(ci.batch, ci.n))
                if len(pi) == 0:
                    continue
                batches = ci.batch.tolist()
                c_sum_c, c_sum_xc = hc[f"{profile}/{half}/sum_c"][:], hc[f"{profile}/{half}/sum_xC"][:]
                theta_c = c_sum_c / ci.sum_L.to_numpy()[:, None]
                mean_c = c_sum_xc / ci.n.to_numpy()[:, None]
                perts = pi.perturbation.tolist()
                wn = weights(cnt, perts, batches, "n_pb")
                wl = weights(cnt, perts, batches, "L_pb")
                ref_c = np.asarray(wn @ mean_c)
                ref_a = np.asarray(wl @ theta_c)
                ga = ho.create_dataset(f"{profile}/{half}/dA", shape=(len(pi), theta_c.shape[1]), dtype=np.float32, chunks=(min(32, len(pi)), min(2048, theta_c.shape[1])), **opts)
                gc = ho.create_dataset(f"{profile}/{half}/dC", shape=(len(pi), theta_c.shape[1]), dtype=np.float32, chunks=(min(32, len(pi)), min(2048, theta_c.shape[1])), **opts)
                for s in range(0, len(pi), BLOCK):
                    rows = pi.iloc[s:s + BLOCK]
                    sum_c = hp[f"{profile}/{half}/sum_c"][s:s + len(rows)]
                    sum_xc = hp[f"{profile}/{half}/sum_xC"][s:s + len(rows)]
                    ga[s:s + len(rows)] = np.log1p(kappa * sum_c / rows.sum_L.to_numpy()[:, None]) - np.log1p(kappa * ref_a[s:s + len(rows)])
                    gc[s:s + len(rows)] = sum_xc / rows.n.to_numpy()[:, None] - ref_c[s:s + len(rows)]
        for profile in PROFILES:
            for half in HALVES:
                cnt = counts[(counts.profile == profile) & (counts.half == half)]
                ci = index[(index.table == "ctrl") & (index.profile == profile) & (index.half == half)]
                pi = index[(index.table == "pert") & (index.profile == profile) & (index.half == half)]
                if len(cnt) == 0 or len(ci) == 0:
                    continue
                ctrl_n = dict(zip(ci.batch, ci.n))
                pops = cnt.groupby(["perturbation", "kind"], as_index=False).agg(n=("n_pb", "sum"), n_excluded=("n_excluded", "sum"), n_batches=("batch", "nunique"))
                perts = pops.perturbation.tolist()
                wn = weights(cnt, perts, list(ci.batch), "n_pb").tocsr()
                ctrl_arr = np.array([ctrl_n[b] for b in ci.batch], float)
                n_eff = 1 / np.maximum(np.asarray(wn.multiply(wn) @ (1 / ctrl_arr)).ravel(), 1e-300)
                min_ctrl = np.array([ctrl_arr[wn.indices[wn.indptr[i]:wn.indptr[i + 1]]].min() if wn.indptr[i + 1] > wn.indptr[i] else np.inf for i in range(len(perts))])
                rows = dict(zip(pi.perturbation, pi.row))
                for i, r in enumerate(pops.itertuples()):
                    n = int(r.n)
                    qc_rows.append(dict(perturbation=r.perturbation, kind=r.kind, profile=profile, half=half, n=n, n_batches=int(r.n_batches), n_excluded=int(r.n_excluded),
                                        n_eff=float(n_eff[i]) if n else np.nan, n_eff_over_n=float(n_eff[i] / n) if n else np.nan,
                                        n_eff_below_4x_cells=bool(n and n_eff[i] < pop["n_eff_flag_ratio"] * n), batch_below_5_controls=bool(min_ctrl[i] < pop["min_controls_per_batch_floor"]),
                                        n_below_20=bool(n < pop["n_cells_flag"]), delta_row=int(rows.get(r.perturbation, -1))))
    qc = pd.DataFrame(qc_rows)
    gk_cols = [c for c in gk.columns if c.startswith(("target", "knockdown", "guide", "n_guides"))]
    qc = qc.merge(gk[["perturbation", "profile"] + gk_cols], on=["perturbation", "profile"], how="left")
    qc["target_not_measured"] = qc.get("target", pd.Series(index=qc.index, dtype=object)).isna() | qc.get("knockdown_ratio", pd.Series(index=qc.index)).isna()
    qc.to_parquet(out / "pert_qc.parquet")
    genes.to_parquet(out / "genes.parquet")
    finish(cfg, "deltas", context, outputs, [str(src / "pert_moments.h5")], started, dict(rows=int(len(qc))))
    print(f"{context}: {len(qc)} pert_qc rows, {round(time.time() - started)} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("context")
    a = ap.parse_args()
    run(a.context, load())
