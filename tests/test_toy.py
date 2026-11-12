"""The toy context through shard, moments and deltas, with hand-checkable identities."""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "ingest"))
sys.path.insert(0, str(ROOT / "ingest" / "audit"))
import toy  # noqa: E402


def build(tmp, shallow=()):
    """Run the three stages on the toy context inside a temporary root."""
    import config
    import adapters
    import shard
    import moments
    import deltas

    tmp = toy.make_root(tmp)
    config.ROOT, config.CONFIG = tmp, tmp / "config"
    cfg = config.load()
    rng = np.random.default_rng(1)
    obs, counts = toy.build_cells(rng, shallow)
    adapters.ADAPTERS["toy"] = lambda cfg: toy.ToyAdapter(cfg, obs, counts)
    import gene_axis
    gene_axis.TABLE = tmp / "config" / "genes_ensembl112.tsv.gz"
    shard.run("toy", None, cfg)
    moments.LABEL_TARGETS["toy__ctx"] = {k: str(v).split("+") for k, v in pd.read_parquet(tmp / "shards/toy/toy__ctx.obs.parquet").dropna(subset=["target_ensembl"]).drop_duplicates("perturbation").set_index("perturbation").target_ensembl.items()}
    moments.run("toy__ctx", cfg)
    deltas.run("toy__ctx", cfg)
    sys.path.insert(0, str(ROOT))
    from reference_perturb import Reference
    return dict(root=tmp, cfg=cfg, obs=obs, counts=counts, R=Reference(tmp))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build(tmp_path_factory.mktemp("root"))


@pytest.fixture(scope="module")
def built_shallow(tmp_path_factory):
    return build(tmp_path_factory.mktemp("root_shallow"), shallow=("b2",))


def test_row_order_and_values(built):
    o = pd.read_parquet(built["root"] / "shards/toy/toy__ctx.obs.parquet")
    kinds = o.kind.map({"control": 0, "single": 1, "combo": 2}).to_numpy()
    assert (np.diff(kinds) >= 0).all()
    with h5py.File(built["root"] / "shards/toy/toy__ctx.h5ad") as h:
        x = h["X"]
        m = sp.csr_matrix((x["data"][:], x["indices"][:], x["indptr"][:]), shape=tuple(x.attrs["shape"]))
        measured = h["var/measured"][:]
    assert measured.sum() == 17
    native = built["counts"][o.deposit_row.to_numpy()][:, 0:17]
    assert np.array_equal(m[:, np.flatnonzero(measured)].toarray(), native)
    assert np.array_equal(o.total_counts_stored.to_numpy(), built["counts"][o.deposit_row.to_numpy()].sum(1))


def test_gate_and_rules(built):
    o = pd.read_parquet(built["root"] / "shards/toy/toy__ctx.obs.parquet")
    assert set(o[o.kind == "combo"].perturbation) == {"TOYG01+TOYG02"}
    assert o[o.kind == "single"].perturbation.isin(["TOYG01", "TOYG02", "TOYG03", "TOYG04", "TOYG05", "TOYG06"]).all()
    assert (o[o.kind == "combo"].n_targets == 2).all()


def test_halves(built):
    o = pd.read_parquet(built["root"] / "shards/toy/toy__ctx.obs.parquet")
    g = o.groupby([o.kind == "control", o.perturbation, o.batch]).half.value_counts().unstack(fill_value=0)
    assert (abs(g.get("A", 0) - g.get("B", 0)) <= 1).all()


def test_pooled_equals_weighted_per_batch(built):
    R = built["R"]
    pix, m = R.means("toy__ctx")
    cnt = R.cell_counts("toy__ctx")
    o, cells = R.cells("toy__ctx", "TOYG01")
    L = o.lib_size.to_numpy(np.float64)
    xc = np.log1p(1e4 * cells.toarray() / L[:, None])
    per_batch = sum((o.batch.astype(str) == b).sum() / len(o) * xc[(o.batch.astype(str) == b).to_numpy()].mean(0) for b in o.batch.astype(str).unique())
    i = list(pix.perturbation).index("TOYG01")
    assert np.allclose(m["m_C"][i], per_batch, atol=1e-12)
    assert np.allclose(m["m_C"][i], xc.mean(0), atol=1e-12)


def test_space_a_identity(built):
    R = built["R"]
    o, cells = R.cells("toy__ctx", "TOYG01")
    L = o.lib_size.to_numpy(np.float64)
    c = cells.toarray()
    pooled = c.sum(0) / L.sum()
    b = o.batch.astype(str).to_numpy()
    weighted = sum(L[b == k].sum() / L.sum() * c[b == k].sum(0) / L[b == k].sum() for k in np.unique(b))
    assert np.allclose(pooled, weighted, atol=1e-12)


def test_stored_delta_matches_first_form(built):
    R = built["R"]
    pix, d = R.deltas("toy__ctx")
    i = list(pix.perturbation).index("TOYG02")
    direct = R.delta_from_cells("toy__ctx", "TOYG02")
    assert np.allclose(d["dC"][i], direct, atol=1e-6)
    assert np.all(np.log1p(d["DB"][i] + R.control_reference("toy__ctx")[2][i]) >= d["dC"][i] + R.control_reference("toy__ctx")[1][i] - 1e-9)


def test_n_eff_and_flags(built):
    qc = built["R"].pert_qc("toy__ctx")
    q = qc[(qc.profile == "authors") & (qc.half == "all")].set_index("perturbation")
    assert np.isclose(q.loc["TOYG01", "n_eff"], 1 / ((4 / 6) ** 2 / 6 + (2 / 6) ** 2 / 4), atol=1e-9)
    assert np.isclose(q.loc["TOYG02", "n_eff"], 10)
    assert np.isclose(q.loc["TOYG03", "n_eff"], 1 / (0.25 / 6 + 0.25 / 10))
    assert np.isclose(q.loc["TOYG04", "n_eff"], 4)
    assert bool(q.loc["TOYG04", "batch_below_5_controls"]) and not bool(q.loc["TOYG02", "batch_below_5_controls"])
    assert bool(q.loc["TOYG03", "n_below_20"])
    assert q.loc["TOYG05", "delta_row"] == -1 and q.loc["TOYG05", "n"] == 2


def test_knockdown_direction(built):
    qc = built["R"].pert_qc("toy__ctx")
    q = qc[(qc.profile == "authors") & (qc.half == "all")].set_index("perturbation")
    assert q.loc["TOYG04", "knockdown_ratio"] == 0 and q.loc["TOYG04", "knockdown_p"] < 0.05
    assert q.loc["TOYG01", "knockdown_ratio"] < 0.8
    assert q.loc["TOYG01", "n_guides"] == 2 and q.loc["TOYG01", "n_guides_used"] == 0 and np.isnan(q.loc["TOYG01", "guide_r_min"])
    assert q.loc["TOYG06", "n_guides_used"] == 2 and np.isclose(q.loc["TOYG06", "guide_r_min"], q.loc["TOYG06", "guide_r_mean"])


def test_g0_and_masks(built):
    mk = built["R"].masks("toy__ctx")
    assert mk["g0"].sum() == 17
    assert mk["target"]("ENSG00000900002").sum() == 1
    assert mk["neighbors"]("ENSG00000900002").sum() == 0


def test_variance_matches_numpy(built):
    R = built["R"]
    pix, m = R.means("toy__ctx")
    o, cells = R.cells("toy__ctx", "TOYG01")
    L = o.lib_size.to_numpy(np.float64)
    xc = np.log1p(1e4 * cells.toarray() / L[:, None])
    i = list(pix.perturbation).index("TOYG01")
    assert np.allclose(m["var_C"][i], xc.var(0, ddof=1), atol=1e-12)


def test_scores_without_predicted_doublet(built):
    assert built["R"].scores("toy__ctx") is None


def test_batch_qc_shape(built):
    q = built["R"].batch_qc("toy__ctx")
    assert q.batch.tolist() == ["b1", "b2", "b3"]
    assert q.cells.tolist() == [16, 9, 29] and q.controls_used.tolist() == [6, 4, 10]
    assert q.columns.tolist() == ["batch", "cells", "controls_used", "lib_size_median", "n_genes_median", "pct_mito_median", "qc_strict_share", "doublet_score_median",
                                  "control_shift", "depth_low", "control_shift_high"]
    assert q.control_shift.notna().all() and not q.depth_low.any()


def test_batch_qc_flags_shallow_batch(built_shallow):
    q = built_shallow["R"].batch_qc("toy__ctx").set_index("batch")
    assert q.depth_low.tolist() == [False, True, False]
    assert q.loc["b2", "lib_size_median"] < 0.5 * q.lib_size_median.median()


def test_batch_effect_size(built):
    e = built["R"].batch_effect_size("toy__ctx")
    assert set(e) == {"batch_offset_scale", "perturbation_effect_scale", "ratio"}
    assert e["batch_offset_scale"] > 0 and e["perturbation_effect_scale"] > 0
    assert np.isclose(e["ratio"], e["batch_offset_scale"] / e["perturbation_effect_scale"])
