"""Shard stage: one sorted raw-count CSR h5ad per context on the global gene axis, with obs, blocks and manifest."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "audit"))
import cells  # noqa: E402
import gene_axis  # noqa: E402
from adapters import ADAPTERS  # noqa: E402
from config import load, path  # noqa: E402
from genes import HGNC  # noqa: E402
from receipts import done, finish, raw_receipt, write_atomic, write_json  # noqa: E402

OBS_COLUMNS = ["dataset", "context", "cell_type", "condition", "timepoint", "modality", "donor", "batch", "batch_kind", "kind", "perturbation",
               "target_ensembl", "n_targets", "guide", "n_guides", "is_control", "control_kind", "control_used", "deposit_row", "lib_size",
               "total_counts_stored", "total_counts_author", "n_genes", "pct_mito", "pct_mito_author", "pct_mito_lower_bound", "guide_umi",
               "assigned_by", "qc_genes_500", "qc_mito_20", "qc_guide_20", "qc_umi_1000", "qc_lane_flag", "qc_strict", "half", "high_moi_design"]


def prepare_obs(obs, dataset, adapter, cfg, fmap, axis):
    """Per-cell decisions that need no matrix: rules, label gate, halves, sort order; returns kept obs in shard order, drops, gate log."""
    obs = obs.copy()
    obs["dataset"] = dataset
    kind, reason = cells.apply_rules(obs, dataset, cfg)
    obs["kind"], obs["drop_reason"] = kind, reason
    native_ids = fmap.native_id.to_numpy() if "native_id" in fmap else None
    deposit_ids = cells.deposit_target_ids(obs, native_ids, fmap.native_symbol.to_numpy())
    sym, ens, nt, cls, table = cells.gate_labels(obs, obs.kind, deposit_ids, axis.ensembl_id.to_numpy())
    perturbed = obs.kind.isin(["single", "combo"])
    obs.loc[perturbed & sym.isna() & (obs.drop_reason == ""), "drop_reason"] = "label_gate"
    obs["perturbation"] = np.where(perturbed, sym.fillna(obs.perturbation.astype(str)), "")
    obs["target_ensembl"] = np.where(perturbed, ens, None)
    obs["n_targets"] = np.where(perturbed, nt, 0)
    obs["control_used"] = (obs.kind == "control") & obs.control_kind.isin(cfg["controls"]["is_control_kinds"] + (["unassigned"] if dataset in cfg["controls"]["including_unassigned"] else []))
    obs["high_moi_design"] = dataset in cfg["cell_rules"].get("high_moi_combinations_kept", [])
    obs["modality"] = adapter.meta.get("modality", "")
    obs["batch_kind"] = adapter.meta.get("batch_kind", "")
    if "assigned_by" not in obs:
        obs["assigned_by"] = "authors"
    obs["pct_mito_author"] = pd.to_numeric(obs.pct_mito, errors="coerce")
    drops = obs[obs.drop_reason != ""].groupby(["context", "batch", "drop_reason"], observed=True).size().rename("n")
    gate_log = pd.DataFrame([(lab, *v) for lab, v in table.items()], columns=["label", "symbol", "ensembl", "n_targets", "gate"])
    keep = obs[obs.drop_reason == ""].drop(columns="drop_reason").copy()
    halves = np.full(len(keep), "all", object)
    for c, idx in keep.groupby(keep.context.astype(str), observed=True).indices.items():
        sub = keep.iloc[idx]
        key = sub.assign(k=np.where(sub.kind == "control", "", sub.perturbation.astype(str)))
        halves[idx] = cells.half_of(key, ["k", "batch"], cfg["seed"], c)
    keep["half"] = halves
    keep = keep.iloc[cells.sort_key(keep)].reset_index(drop=True)
    return keep, drops, gate_log


def segments(indptr, rows):
    """Flat positions of the nonzeros of the given rows, in row order."""
    starts, lens = indptr[rows], indptr[rows + 1] - indptr[rows]
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, np.int64)
    offs = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
    return np.arange(total) + offs


class Buckets:
    """Append-only on-disk buckets of row segments keyed by sorted row // bucket_rows; read back in local row order."""

    def __init__(self, tmp, bucket_rows):
        self.tmp, self.b = Path(tmp), bucket_rows
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.files = {}

    def handles(self, k):
        if k not in self.files:
            self.files[k] = tuple(open(self.tmp / f"{k}.{name}", "ab") for name in ("local", "nnz", "idx", "dat"))
        return self.files[k]

    def append(self, sorted_rows, m):
        """Scatter the rows of a canonical CSR chunk into their buckets."""
        buckets, local = sorted_rows // self.b, (sorted_rows % self.b).astype(np.uint16)
        nnz = np.diff(m.indptr).astype(np.uint32)
        for k in np.unique(buckets):
            rows = np.flatnonzero(buckets == k)
            seg = segments(m.indptr, rows)
            fl, fn, fi, fd = self.handles(k)
            fl.write(local[rows].tobytes()); fn.write(nnz[rows].tobytes())
            fi.write(m.indices[seg].astype(np.int32).tobytes()); fd.write(m.data[seg].astype(np.int32).tobytes())

    def close(self):
        for fs in self.files.values():
            for f in fs:
                f.close()
        self.files = {}

    def read(self, k):
        """(indptr, indices, data) of bucket k over all its local rows in order; absent rows are empty."""
        p = self.tmp / f"{k}."
        if not Path(f"{p}local").exists():
            return np.zeros(1, np.int64), np.empty(0, np.int32), np.empty(0, np.int32)
        local = np.fromfile(f"{p}local", np.uint16).astype(np.int64)
        nnz = np.fromfile(f"{p}nnz", np.uint32).astype(np.int64)
        idx, dat = np.fromfile(f"{p}idx", np.int32), np.fromfile(f"{p}dat", np.int32)
        order = np.argsort(local, kind="stable")
        indptr_src = np.concatenate([[0], np.cumsum(nnz)])
        gather = segments(indptr_src, order)
        counts = np.bincount(local, weights=nnz, minlength=int(local.max()) + 1).astype(np.int64)
        return np.concatenate([[0], np.cumsum(counts)]), idx[gather], dat[gather]

    def rows(self, start, end):
        """Yield (indptr, indices, data) pieces covering sorted rows [start, end)."""
        for k in range(start // self.b, (end - 1) // self.b + 1):
            ip, idx, dat = self.read(k)
            lo, hi = max(start - k * self.b, 0), min(end - k * self.b, self.b)
            if len(ip) - 1 < hi:
                ip = np.concatenate([ip, np.full(hi - (len(ip) - 1), ip[-1])])
            yield ip[lo:hi + 1] - ip[lo], idx[ip[lo]:ip[hi]], dat[ip[lo]:ip[hi]]


def remap(m, col_map, n_cols):
    """CSR over native features to canonical CSR over the axis, dropping unmapped features."""
    coo = m.tocoo()
    keep = col_map[coo.col] >= 0
    out = sp.csr_matrix((coo.data[keep], (coo.row[keep], col_map[coo.col[keep]])), shape=(m.shape[0], n_cols))
    out.sum_duplicates()
    out.sort_indices()
    return out


def write_h5ad(target, obs, var, uns, pieces, n_cols, cfg):
    """Write an anndata-layout h5 whose CSR X is streamed from sorted pieces."""
    comp = dict(compression=cfg["shard"]["compression"], compression_opts=cfg["shard"]["compression_level"], shuffle=True)
    chunk = (cfg["shard"]["chunk_elements"],)
    with write_atomic(target) as tmp, h5py.File(tmp, "w") as h:
        h.attrs["encoding-type"], h.attrs["encoding-version"] = "anndata", "0.1.0"
        x = h.create_group("X")
        x.attrs["encoding-type"], x.attrs["encoding-version"], x.attrs["shape"] = "csr_matrix", "0.1.0", np.array([len(obs), n_cols], np.int64)
        data = x.create_dataset("data", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=chunk, **comp)
        indices = x.create_dataset("indices", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=chunk, **comp)
        indptr = np.zeros(len(obs) + 1, np.int64)
        row, total = 0, 0
        for ip, idx, dat in pieces:
            n = len(ip) - 1
            data.resize(total + len(dat), 0)
            indices.resize(total + len(idx), 0)
            data[total:] = dat
            indices[total:] = idx
            indptr[row + 1:row + n + 1] = total + ip[1:]
            row += n
            total += len(dat)
        if row != len(obs):
            raise SystemExit(f"{target}: wrote {row} rows for {len(obs)} cells")
        x.create_dataset("indptr", data=indptr)
        write_dataframe(h.create_group("obs"), obs, "cell_id")
        write_dataframe(h.create_group("var"), var, "ensembl_id")
        u = h.create_group("uns")
        u.attrs["encoding-type"], u.attrs["encoding-version"] = "dict", "0.1.0"
        for k, v in uns.items():
            u.create_dataset(k, data=v if isinstance(v, str) else json.dumps(v, default=str))


def write_dataframe(g, df, index_name):
    """AnnData dataframe layout: strings as categoricals, numbers as arrays, booleans with NA as nullable."""
    g.attrs["encoding-type"], g.attrs["encoding-version"], g.attrs["_index"] = "dataframe", "0.2.0", index_name
    g.attrs["column-order"] = np.array(list(df.columns), dtype=object)
    g.create_dataset(index_name, data=np.asarray(df.index.astype(str), dtype=object), dtype=h5py.string_dtype())
    for c in df.columns:
        v = df[c]
        if isinstance(v.dtype, pd.BooleanDtype):
            sub = g.create_group(c)
            sub.attrs["encoding-type"], sub.attrs["encoding-version"] = "nullable-boolean", "0.1.0"
            sub.create_dataset("values", data=v.fillna(False).to_numpy(bool))
            sub.create_dataset("mask", data=v.isna().to_numpy())
        elif v.dtype == bool:
            g.create_dataset(c, data=v.to_numpy())
        elif v.dtype == object or isinstance(v.dtype, pd.CategoricalDtype):
            cat = pd.Categorical(v.astype(object).where(v.notna(), "").astype(str))
            sub = g.create_group(c)
            sub.attrs["encoding-type"], sub.attrs["encoding-version"], sub.attrs["ordered"] = "categorical", "0.2.0", False
            sub.create_dataset("codes", data=cat.codes.astype(np.int32))
            sub.create_dataset("categories", data=np.asarray(cat.categories, dtype=object), dtype=h5py.string_dtype())
        else:
            g.create_dataset(c, data=v.to_numpy(np.float64) if v.isna().any() else v.to_numpy())


def blocks(o):
    """Contiguous runs of (kind, perturbation, batch, control_kind) in a sorted obs."""
    key = (o.kind.astype(str) + "|" + o.perturbation.astype(str) + "|" + o.batch.astype(str) + "|" + o.control_kind.fillna("").astype(str)).to_numpy()
    change = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    return pd.DataFrame(dict(kind=o.kind.to_numpy()[change], perturbation=o.perturbation.to_numpy()[change], batch=o.batch.astype(str).to_numpy()[change],
                             control_kind=o.control_kind.to_numpy()[change], start=change, end=np.r_[change[1:], len(o)]))


def manifest(context, o, dataset, adapter, drops, gate_log, uns):
    """Per-context manifest; every field names its source."""
    d = drops.loc[context].reset_index().to_dict("records") if context in drops.index.get_level_values(0) else []
    return dict(context=context, dataset=dataset, deposit=adapter.meta.get("deposit"), modality=adapter.meta.get("modality"),
                batch_definition=adapter.meta.get("batch_field"), batch_kind=adapter.meta.get("batch_kind"),
                control_definition=uns["control_definition"], perturbation_direction=uns["perturbation_direction"], lib_size_rule=uns["lib_size_rule"],
                guide_assignment=adapter.meta.get("guide_assignment"), cells_kept=int(len(o)),
                cells_per_batch={str(k): int(v) for k, v in o.groupby(o.batch.astype(str)).size().items()},
                controls_per_batch={str(k): int(v) for k, v in o[o.control_used].groupby(o.batch.astype(str)).size().items()},
                dropped_cells=d, gate=gate_log[gate_log.gate != "kept"].to_dict("records"), gene_mapping=uns["gene_mapping"],
                sources=dict(modality="audit loader meta (paper)", batch_definition="audit loader meta (deposit column)", control_definition="config",
                             perturbation_direction="config", lib_size_rule="config", guide_assignment="audit loader meta", dropped_cells="cell rules",
                             gate="label gate", harvest_time=None, moi_design=None, cas_system=None, selection_protocol=None))


def run(dataset, unit, cfg):
    """Build the shards of one unit: obs decisions, one streaming pass into sorted buckets, one h5ad per context."""
    started = time.time()
    adapter = ADAPTERS[dataset](cfg)
    unit = unit or dataset
    obs = adapter.obs(unit)
    contexts = sorted(obs.context.astype(str).unique())
    outputs = [path(cfg, "shards", dataset, f"{c}.h5ad") for c in contexts]
    if done(cfg, "shard", unit, outputs):
        print(f"{unit}: done", flush=True)
        return
    axis = gene_axis.axis()
    native_ids, native_symbols = adapter.features(unit)
    fmap = gene_axis.map_features(native_ids, native_symbols, HGNC(cfg["paths"]["hgnc"]), axis.ensembl_id.to_numpy())
    keep, drops, gate_log = prepare_obs(obs, dataset, adapter, cfg, fmap, axis)
    col_map = fmap.axis_index.to_numpy(dtype=float)
    col_map = np.where(np.isnan(col_map), -1, col_map).astype(np.int64)
    mt_native = fmap.ensembl.isin(set(axis.ensembl_id[axis.chromosome == "MT"])).to_numpy()
    sort_pos = np.full(len(obs), -1, np.int64)
    sort_pos[keep.deposit_row.to_numpy()] = np.arange(len(keep))
    stored, ngenes, mito = (np.zeros(len(obs), np.int64) for _ in range(3))
    tmp = path(cfg, "tmp", dataset, unit.replace("/", "__"))
    if tmp.exists():
        shutil.rmtree(tmp)
    buckets = Buckets(tmp, cfg["shard"]["bucket_rows"])
    for start, m in adapter.chunks(unit):
        m = m.tocsr()
        if m.shape[1] != len(fmap):
            raise SystemExit(f"{unit}: chunk has {m.shape[1]} features, expected {len(fmap)}")
        r = np.arange(start, start + m.shape[0])
        stored[r], ngenes[r] = np.asarray(m.sum(1)).ravel(), np.diff(m.indptr)
        mito[r] = np.asarray(m[:, mt_native].sum(1)).ravel() if mt_native.any() else 0
        kept = np.flatnonzero(sort_pos[r] >= 0)
        if len(kept):
            buckets.append(sort_pos[r[kept]], remap(m[kept], col_map, len(axis)))
    buckets.close()
    rows = keep.deposit_row.to_numpy()
    keep["total_counts_stored"] = stored[rows]
    keep["total_counts_author"] = pd.to_numeric(keep.total_counts, errors="coerce")
    partial = dataset in cfg["lib_size"]["author_total_when_partial"]
    keep["lib_size"] = np.where(partial & keep.total_counts_author.notna().to_numpy(), keep.total_counts_author, keep.total_counts_stored)
    keep["n_genes"] = ngenes[rows]
    has_mt = bool(mt_native.any())
    keep["pct_mito"] = 100 * mito[rows] / np.maximum(stored[rows], 1) if has_mt else np.nan
    keep["pct_mito_lower_bound"] = has_mt and partial
    for c, v in cells.qc_flags(keep, cfg).items():
        keep[c] = v
    var = axis[["ensembl_id", "symbol", "biotype", "chromosome", "tss"]].copy()
    measured = np.zeros(len(axis), bool)
    measured[col_map[col_map >= 0]] = True
    var["measured"] = measured
    var = var.set_index("ensembl_id")
    uns = dict(dataset=dataset, unit=unit, deposit=adapter.meta.get("deposit"), config_hash=cfg["hash"], code_commit=cfg["commit"], seed=cfg["seed"],
               control_definition="unassigned + non-targeting" if dataset in cfg["controls"]["including_unassigned"] else "control guides only",
               lib_size_rule="total_counts_author" if partial else "total_counts_stored",
               perturbation_direction="up" if dataset in cfg["perturbation_direction"]["up"] else cfg["perturbation_direction"]["default"],
               cell_rules=cfg["cell_rules"].get(dataset, cfg["cell_rules"]["default"]), gene_mapping=gene_axis.mapping_summary(fmap))
    ctx = keep.context.astype(str).to_numpy()
    for c in contexts:
        where = np.flatnonzero(ctx == c)
        start, end = int(where[0]), int(where[-1]) + 1
        o = keep.iloc[start:end].copy()
        o.index = [f"{dataset}:{r}" for r in o.deposit_row]
        o = o[[k for k in OBS_COLUMNS if k in o.columns] + [k for k in o.columns if k not in OBS_COLUMNS and k != "total_counts"]]
        target = path(cfg, "shards", dataset, f"{c}.h5ad")
        write_h5ad(target, o, var, dict(uns, context=c), buckets.rows(start, end), len(axis), cfg)
        o.to_parquet(target.with_name(f"{c}.obs.parquet"))
        blocks(o).to_parquet(target.with_name(f"{c}.blocks.parquet"))
        write_json(path(cfg, "manifest", f"{c}.json"), manifest(c, o, dataset, adapter, drops, gate_log, uns))
    path(cfg, "genes").mkdir(exist_ok=True)
    fmap.to_parquet(path(cfg, "genes", f"{dataset}.map.parquet"))
    shutil.rmtree(tmp)
    inputs = [raw_receipt(cfg, dataset, f)["sha256"] for f in adapter.meta.get("files", []) if Path(f).is_file()]
    finish(cfg, "shard", unit, outputs, inputs, started, dict(contexts=contexts, cells=int(len(keep)), cells_in_deposit=int(len(obs))))
    print(f"{unit}: {len(contexts)} contexts, {len(keep):,} cells kept of {len(obs):,}, {round(time.time() - started)} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--unit")
    a = ap.parse_args()
    run(a.dataset, a.unit, load())
