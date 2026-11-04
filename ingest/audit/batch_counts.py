"""Per (context, batch): cells of each control kind and of each single-target perturbation; per label: the deposit's target Ensembl id."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from audit import CFG  # noqa: E402
from sources import SOURCES  # noqa: E402


def target_ensembl(single, src):
    """Per label, the target's versionless Ensembl id as the deposit gives it: per-cell annotation first, else the deposit's own feature table."""
    labels = pd.Series(sorted(single.perturbation.unique()), name="perturbation")
    out = pd.DataFrame(dict(perturbation=labels, deposit_ensembl=None, deposit_ensembl_source=None))
    e = single.target_ensembl.dropna().astype(str).str.split(".").str[0]
    per = e[~e.isin(["", "nan", "None"])].groupby(single.perturbation).unique()
    one = per[per.map(len) == 1].map(lambda v: v[0])
    hit = out.perturbation.isin(one.index)
    out.loc[hit, "deposit_ensembl"] = out.perturbation[hit].map(one)
    out.loc[hit, "deposit_ensembl_source"] = "per-cell target annotation"
    out.loc[out.perturbation.isin(per.index[per.map(len) > 1]), "deposit_ensembl_source"] = "per-cell target annotation: several ids, unused"
    if src.genes is not None and src.ensembl is not None:
        f = pd.DataFrame(dict(symbol=pd.Series(src.genes, dtype=str), ens=pd.Series(src.ensembl, dtype=str).str.split(".").str[0]))
        f = f[f.ens.str.startswith("ENSG")].drop_duplicates()
        unique = f[~f.symbol.duplicated(keep=False)].set_index("symbol").ens
        todo = out.deposit_ensembl.isna() & out.deposit_ensembl_source.isna() & out.perturbation.isin(unique.index)
        out.loc[todo, "deposit_ensembl"] = out.perturbation[todo].map(unique)
        out.loc[todo, "deposit_ensembl_source"] = "deposit feature table (symbol to Ensembl)"
    return out


def run(name):
    """Write control cells per batch and kind, and single-target cells per batch and perturbation, for one source."""
    out = Path(CFG["out_dir"]) / name
    out.mkdir(parents=True, exist_ok=True)
    src = SOURCES[name](CFG)
    obs = src.obs
    obs["batch"] = obs.batch.astype(str)
    ctrl = obs[obs.control_kind.notna()].groupby(["context", "batch", "control_kind"], observed=True).size().rename("n_cells")
    single = obs[~obs.is_control & (obs.perturbation != "") & ~obs.perturbation.str.contains("+", regex=False)]
    pert = single.groupby(["context", "batch", "perturbation"], observed=True).size().rename("n_cells")
    desc = obs.groupby("context", observed=True)[["cell_type", "condition", "timepoint", "donor"]].first()
    target_ensembl(single, src).to_csv(out / "target_ensembl.csv", index=False)
    ctrl.reset_index().to_csv(out / "controls_per_batch.csv", index=False)
    pert.reset_index().to_csv(out / "single_target_cells_per_batch.csv.gz", index=False)
    tc = pd.to_numeric(single.total_counts, errors="coerce")
    lib = tc.groupby([single.context, single.perturbation], observed=True).agg(["size", "count", "mean"])
    lib.columns = ["n_cells", "n_cells_with_total_counts", "mean_total_counts"]
    lib.reset_index().to_csv(out / "single_target_library_size.csv.gz", index=False)
    desc.reset_index().to_csv(out / "context_description.csv", index=False)
    print(f"{name}: {len(ctrl)} control batches, {len(pert):,} (context, batch, perturbation) rows", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
