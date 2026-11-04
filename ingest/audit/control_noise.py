"""Per (context, perturbation): effective number of controls of the batch-matched delta and its flag."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from batch_table import CFG, OUT, STUDY, controls, perturbed  # noqa: E402


def per_perturbation(name):
    """Variance factors in units of the per-cell variance, from cell counts only."""
    ctrl, _ = controls(name)
    p = perturbed(name).merge(ctrl[["context", "batch", "n_control_cells"]], on=["context", "batch"], how="left")
    p["n_control_cells"] = p.n_control_cells.fillna(0)
    m = p[p.n_control_cells > 0].copy()
    key = ["context", "perturbation"]
    n_p = m.groupby(key).n_cells.transform("sum")
    m["w2_over_nc"] = (m.n_cells / n_p) ** 2 / m.n_control_cells
    g = m.groupby(key)
    out = pd.DataFrame(dict(n_cells_matched=g.n_cells.sum(), n_batches=g.size(), n_eff=1 / g.w2_over_nc.sum()))
    allc = p.groupby(key).n_cells.sum()
    out = out.reindex(allc.index)
    out["n_cells"] = allc
    out["n_cells_in_batches_without_controls"] = allc - out.n_cells_matched.fillna(0)
    out["n_eff_over_n_cells"] = out.n_eff / out.n_cells_matched
    out["variance_batch_matched"] = 1 / out.n_cells_matched + 1 / out.n_eff
    out["variance_pooled_controls"] = 1 / out.n_cells_matched + 1 / out.index.get_level_values("context").map(
        ctrl.groupby("context").n_control_cells.sum()).to_numpy()
    out[f"n_eff_below_{CFG['n_eff_flag_ratio']}x_cells"] = out.n_eff_over_n_cells < CFG["n_eff_flag_ratio"]
    out.insert(0, "source", name)
    return out.reset_index()


def context_summary(t):
    """Context-level columns for the batch table."""
    r = CFG["n_eff_flag_ratio"]
    g = t.groupby("context")
    return pd.DataFrame({
        "n_eff_median": g.n_eff.median(), "n_eff_min": g.n_eff.min(),
        "n_eff_over_n_cells_median": g.n_eff_over_n_cells.median(),
        "share_perturbations_n_eff_below_1x_cells": g.n_eff_over_n_cells.apply(lambda x: (x < 1).mean()),
        f"share_perturbations_n_eff_below_{r}x_cells": g[f"n_eff_below_{r}x_cells"].mean(),
        "control_share_of_delta_variance_median": (t["variance_batch_matched"] - 1 / t.n_cells_matched).div(t.variance_batch_matched).groupby(t.context).median(),
        "batch_matching_variance_cost_median": (t.variance_batch_matched / t.variance_pooled_controls).groupby(t.context).median(),
        "perturbed_cells_in_batches_without_controls": g.n_cells_in_batches_without_controls.sum(),
    }).reset_index()


def main():
    """Write the per-perturbation table and its per-context summary."""
    t = pd.concat([per_perturbation(n) for n in STUDY], ignore_index=True)
    t.to_csv(OUT / "perturbation_control_noise.csv.gz", index=False, float_format="%.5g")
    c = context_summary(t)
    c.to_csv(OUT / "context_control_noise.csv", index=False, float_format="%.4g")
    print(f"{len(t):,} perturbation rows, {len(c)} contexts")
    print(c.drop(columns="context").describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
