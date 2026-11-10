"""Scores stage: scrublet doublet scores per batch and cell-cycle scores per cell, flag-only, in scores/<context>.parquet."""
import argparse
import multiprocessing as mp
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from config import CONFIG, load, path  # noqa: E402
from moments import context_dataset, read_rows  # noqa: E402
from receipts import done, finish, write_atomic  # noqa: E402


def cell_cycle_lists(gene_ids):
    """(S, G2M) lists of measured Ensembl ids from the Tirosh symbols through the gene table."""
    cc = pd.read_csv(CONFIG / "cell_cycle_genes.tsv", sep="\t")
    table = pd.read_csv(CONFIG / "genes_ensembl112.tsv.gz", sep="\t", usecols=["ensembl_id", "symbol"])
    sym_to_id = table.drop_duplicates("symbol").set_index("symbol").ensembl_id
    present = set(gene_ids)
    pick = lambda phase: [sym_to_id[s] for s in cc[cc.phase == phase].symbol if s in sym_to_id.index and sym_to_id[s] in present]
    return pick("S"), pick("G2M")


def batch_rows(obs, blocks, b):
    """Shard rows of one batch (controls and perturbed cells together)."""
    runs = blocks[blocks.batch.astype(str) == b]
    return np.concatenate([np.arange(s, e) for s, e in zip(runs.start, runs.end)])


def score_batch(shard, b, rows, cells, measured_col, gene_ids, s_genes, g2m_genes, seed, kappa):
    """Scrublet and cell-cycle scores of one batch, read straight from the shard, on one thread."""
    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with h5py.File(shard, "r") as h:
            m = sp.vstack([read_rows(h["X"], int(s), int(e)) for s, e in contiguous(rows)]).tocsr()[:, measured_col]
        a = ad.AnnData(X=m.astype(np.float32), obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=gene_ids))
        frame = pd.DataFrame(index=a.obs_names)
        frame["batch"] = b
        if a.n_obs >= 50:
            sc.pp.scrublet(a, batch_key=None, random_state=seed, verbose=False)
            frame["doublet_score"], frame["predicted_doublet"] = a.obs.doublet_score.to_numpy(), a.obs.predicted_doublet.astype(bool).to_numpy()
        else:
            frame["doublet_score"], frame["predicted_doublet"] = np.nan, pd.NA
        sc.pp.normalize_total(a, target_sum=kappa)
        sc.pp.log1p(a)
        if len(s_genes) >= 5 and len(g2m_genes) >= 5:
            sc.tl.score_genes_cell_cycle(a, s_genes=s_genes, g2m_genes=g2m_genes)
            frame["S_score"], frame["G2M_score"], frame["phase"] = a.obs.S_score.to_numpy(), a.obs.G2M_score.to_numpy(), a.obs.phase.astype(str).to_numpy()
        else:
            frame["S_score"], frame["G2M_score"], frame["phase"] = np.nan, np.nan, None
    return frame


def run(context, cfg):
    """Score every batch of the context, largest batches first, in parallel over the job's CPUs."""
    started = time.time()
    dataset = context_dataset(context, cfg)
    shard = path(cfg, "shards", dataset, f"{context}.h5ad")
    out = path(cfg, "scores", f"{context}.parquet")
    if done(cfg, "scores", context, [out]):
        print(f"{context}: done", flush=True)
        return
    obs = pd.read_parquet(shard.with_name(f"{context}.obs.parquet"))
    blocks = pd.read_parquet(shard.with_name(f"{context}.blocks.parquet"))
    with h5py.File(shard, "r") as h:
        measured_col = np.flatnonzero(h["var/measured"][:])
        gene_ids = h["var/ensembl_id"][:].astype(str)[measured_col]
    s_genes, g2m_genes = cell_cycle_lists(gene_ids)
    cells = obs.index.astype(str)
    rows = {b: np.sort(batch_rows(obs, blocks, b)) for b in obs.batch.astype(str).unique()}
    batches = sorted(rows, key=lambda b: -len(rows[b]))
    workers = len(os.sched_getaffinity(0))
    frames = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        results = pool.map(score_batch, repeat(shard), batches, [rows[b] for b in batches], [cells[rows[b]] for b in batches],
                           repeat(measured_col), repeat(gene_ids), repeat(s_genes), repeat(g2m_genes), repeat(cfg["seed"]), repeat(cfg["kappa"]))
        for b, frame in zip(batches, results):
            frames.append(frame)
            print(f"{context}: batch {b} {len(frame):,} cells, {round(time.time() - started)} s", flush=True)
    scores = pd.concat(frames).reindex(cells)
    scores["predicted_doublet"] = scores.predicted_doublet.astype("boolean")
    with write_atomic(out) as tmp:
        scores.to_parquet(tmp)
    finish(cfg, "scores", context, [out], [str(shard)], started, dict(n_s_genes=len(s_genes), n_g2m_genes=len(g2m_genes), workers=workers))
    print(f"{context}: {len(scores):,} cells scored by {workers} workers, {round(time.time() - started)} s", flush=True)


def contiguous(rows):
    """(start, end) runs of consecutive integers."""
    if len(rows) == 0:
        return []
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    starts = np.r_[rows[0], rows[breaks]]
    ends = np.r_[rows[breaks - 1] + 1, rows[-1] + 1]
    return list(zip(starts, ends))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("context")
    a = ap.parse_args()
    run(a.context, load())
