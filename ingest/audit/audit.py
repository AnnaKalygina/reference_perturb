"""Audit one deposit: batches, controls per batch, perturbation spread, integer counts, gene ids."""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from genes import HGNC  # noqa: E402
from h5 import col  # noqa: E402
from sources import SOURCES  # noqa: E402

CFG = yaml.safe_load(open(Path(__file__).with_name("thresholds.yaml")))


def category(obs):
    """control / single / combo / unassigned per cell."""
    p = obs.perturbation
    return np.select([obs.is_control, p.str.contains(r"\+", regex=True), p != ""], ["control", "combo", "single"], "unassigned")


def batch_table(obs):
    """Cells, controls and perturbed cells per (context, batch)."""
    g = obs.groupby(["context", "batch"], observed=True)
    return pd.DataFrame(dict(n_cells=g.size(), n_controls=g.is_control.sum(), n_perturbed=g.is_perturbed.sum())).reset_index()


def pert_table(obs, bt, bins):
    """Per (context, perturbation): cells, batches, and share of cells in batches with few controls."""
    p = obs[obs.kind.isin(["single", "combo"])][["context", "batch", "perturbation"]]
    p = p.merge(bt[["context", "batch", "n_controls"]], on=["context", "batch"], how="left")
    for b in bins:
        p[f"lt_{b}"] = p.n_controls < b
    g = p.groupby(["context", "perturbation"], observed=True)
    out = pd.DataFrame(dict(n_cells=g.size(), n_batches=g.batch.nunique()))
    for b in bins:
        out[f"share_cells_ctrl_lt_{b}"] = g[f"lt_{b}"].mean()
    out["combo"] = out.index.get_level_values("perturbation").str.contains(r"\+", regex=True)
    return out.reset_index(), p


def threshold_effect(p, cands, min_cells):
    """Cells and perturbations lost for each minimum-controls-per-batch candidate."""
    rows = []
    tot = p.groupby(["context", "perturbation"], observed=True).size()
    for t in cands:
        keep = p[p.n_controls >= t].groupby(["context", "perturbation"], observed=True).size().reindex(tot.index, fill_value=0)
        rows.append(dict(min_controls=t, perturbed_cells_dropped=int(len(p) - keep.sum()),
                         share_cells_dropped=float(1 - keep.sum() / max(1, len(p))),
                         perts_lost_all=int((keep == 0).sum()), perts_below_min_cells=int((keep < min_cells).sum()),
                         n_perts=int(len(tot))))
    return rows


def nesting(obs):
    """How batch labels relate to conditions and cell types."""
    g = obs.groupby("batch", observed=True)
    ncond, nct, nctx = g.condition.nunique(), g.cell_type.nunique(), g.context.nunique()
    return dict(n_batch_labels=int(len(ncond)), labels_spanning_conditions=int((ncond > 1).sum()),
                labels_spanning_cell_types=int((nct > 1).sum()), labels_spanning_contexts=int((nctx > 1).sum()),
                verdict=("single context" if obs.context.nunique() == 1 else
                         "batches crossed with contexts" if (nctx > 1).any() else "batches nested within contexts"))


def ints(v):
    """Integer check on a sample of stored values."""
    v = np.asarray(v, float)
    nz = v[v != 0]
    return dict(n_sampled=int(len(v)), n_nonzero=int(len(nz)), non_integer=int((nz != np.round(nz)).sum()),
                negative=int((nz < 0).sum()), min_nonzero=float(nz.min()) if len(nz) else None,
                max=float(nz.max()) if len(nz) else None)


def k562_targets():
    """Target symbols of Replogle 2022 K562 essential."""
    with h5py.File(CFG["k562_essential"], "r") as h:
        return set(col(h, "obs/gene").astype(str)) - {"non-targeting"}


def q(s):
    """min / median / max of a series."""
    return dict(min=float(s.min()), median=float(s.median()), max=float(s.max())) if len(s) else dict(min=None, median=None, max=None)


def run(name):
    """Audit one source and write its tables."""
    t0 = time.time()
    out = Path(CFG["out_dir"]) / name
    out.mkdir(parents=True, exist_ok=True)
    src = SOURCES[name](CFG)
    obs = src.obs
    obs["kind"] = category(obs)
    obs["is_perturbed"] = obs.kind.isin(["single", "combo"])
    obs["batch"] = obs.batch.astype(str)
    bins, cands = CFG["control_bins"], CFG["min_control_candidates"]
    bt = batch_table(obs)
    pt, p = pert_table(obs, bt, bins)
    ess = k562_targets()
    hg = HGNC(CFG["hgnc"])
    contexts = []
    for c, o in obs.groupby("context", observed=True):
        b = bt[bt.context == c]
        pc = pt[pt.context == c]
        singles = pc[~pc.combo]
        tgt = set(singles.perturbation) | {x for s in pc[pc.combo].perturbation for x in s.split("+")}
        row = dict(context=c, cell_type=str(o.cell_type.iloc[0]), condition=str(o.condition.iloc[0]),
                   timepoint=str(o.timepoint.iloc[0]), donor=str(o.donor.iloc[0]), n_cells=len(o),
                   n_controls=int(o.is_control.sum()), n_single=int((o.kind == "single").sum()),
                   n_combo=int((o.kind == "combo").sum()), n_unassigned=int((o.kind == "unassigned").sum()),
                   n_perts=int(len(singles)), n_combo_perts=int(pc.combo.sum()), n_batches=int(len(b)),
                   control_free_batches=int((b.n_controls == 0).sum()),
                   control_free_batches_with_perturbed=int(((b.n_controls == 0) & (b.n_perturbed > 0)).sum()),
                   ctrl_per_batch=q(b.n_controls), cells_per_pert=q(singles.n_cells),
                   batches_per_pert=q(pc.n_batches),
                   k562_essential_overlap=int(len(tgt & ess)), n_targets=int(len(tgt)))
        for t in bins:
            row[f"batches_ctrl_lt_{t}"] = int((b.n_controls < t).sum())
            pp = p[p.context == c]
            row[f"share_pert_cells_in_batches_ctrl_lt_{t}"] = float((pp.n_controls < t).mean()) if len(pp) else None
        ng = pd.to_numeric(o.n_guides, errors="coerce")
        row["n_guides"] = {str(k): int(v) for k, v in ng.clip(upper=3).value_counts(dropna=False).sort_index().items()}
        row["control_kinds"] = {str(k): int(v) for k, v in o.control_kind[o.is_control].value_counts().items()}
        contexts.append(row)
    summary = dict(source=name, meta=src.meta, n_cells=len(obs), n_contexts=int(obs.context.nunique()),
                   n_controls=int(obs.is_control.sum()), n_perturbed=int(obs.kind.isin(["single", "combo"]).sum()),
                   n_unassigned=int((obs.kind == "unassigned").sum()), n_batches=int(len(bt)),
                   integer_check=ints(src.values), genes=hg.summary(src.genes, src.ensembl),
                   targets=hg.targets(sorted({x for s in pt.perturbation for x in s.split("+")})),
                   nesting=nesting(obs), thresholds=threshold_effect(p[~p.perturbation.str.contains("+", regex=False)], cands, CFG["min_cells_per_perturbation"]),
                   qc_columns={c: bool(obs[c].notna().any()) for c in ("total_counts", "n_genes", "pct_mito")},
                   contexts=contexts, seconds=round(time.time() - t0))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    bt.to_csv(out / "batches.csv", index=False)
    pt.to_csv(out / "perturbations.csv.gz", index=False)
    print(f"{name}: {len(obs):,} cells, {summary['n_contexts']} contexts, {len(bt)} batches, {summary['seconds']} s")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
