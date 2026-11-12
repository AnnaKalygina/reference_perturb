"""Per-cell decisions before the matrix is touched: cell rules, control kinds, QC flags, perturbation labels, halves, sort order."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "audit"))
from labels import classify  # noqa: E402

KIND_ORDER = {"control": 0, "single": 1, "combo": 2}


def kinds(obs):
    """control / single / combo / unassigned per cell, from the standardized obs."""
    p = obs.perturbation.fillna("").astype(str)
    return np.select([obs.is_control.to_numpy(bool), p.str.contains("+", regex=False), p != ""], ["control", "combo", "single"], "unassigned")


def apply_rules(obs, dataset, cfg):
    """keep / drop_reason per cell from the dataset's cell rules; drops are counted, never silent."""
    rules = cfg["cell_rules"]
    kind = kinds(obs)
    ng = pd.to_numeric(obs.n_guides, errors="coerce")
    reason = np.full(len(obs), "", object)
    reason[kind == "unassigned"] = "no_guide"
    reason[(kind == "unassigned") & (ng > 1)] = "multi_guide"
    if dataset not in rules.get("designed_combinations_kept", []):
        reason[kind == "combo"] = "multi_guide"
    if dataset == "feng":
        feng_ctrl = (kind == "unassigned") & obs.control_kind.eq("unassigned").to_numpy()
        reason[feng_ctrl] = ""
        kind = np.where(feng_ctrl, "control", kind)
    low = obs.get("low_quality")
    if dataset == "dld1" and low is not None:
        reason[(reason == "") & low.fillna(False).astype(bool).to_numpy()] = "authors_low_quality"
    return pd.Series(kind, index=obs.index), pd.Series(reason, index=obs.index)


def gate_labels(obs, kind, deposit_ids, axis_ids):
    """Per-cell perturbation label after the gate: symbols in Ensembl order, target ids, n_targets, gate class."""
    labels = obs.perturbation.fillna("").astype(str)
    uniq = labels[kind.isin(["single", "combo"])].unique()
    pos = {g: i for i, g in enumerate(axis_ids)}
    table = {}
    for lab in uniq:
        parts = lab.split("+")
        hits = [classify(t, deposit_ids.get(t)) for t in parts]
        ok = all(h[3] and h[2] in pos for h in hits)
        if ok:
            order = sorted(range(len(parts)), key=lambda i: hits[i][2])
            table[lab] = ("+".join(hits[i][1] for i in order), "+".join(hits[i][2] for i in order), len(parts), "kept")
        else:
            table[lab] = (None, None, len(parts), ";".join(f"{t}:{h[0]}" for t, h in zip(parts, hits) if not (h[3] and h[2] in pos)))
    sym = labels.map(lambda x: table.get(x, (None, None, 0, ""))[0])
    ens = labels.map(lambda x: table.get(x, (None, None, 0, ""))[1])
    nt = labels.map(lambda x: table.get(x, (None, None, 0, ""))[2]).astype(int)
    cls = labels.map(lambda x: table.get(x, (None, None, 0, ""))[3])
    return sym, ens, nt, cls, table


def deposit_target_ids(obs, native_ids, native_symbols):
    """Label -> versionless deposit Ensembl id, from per-cell target ids first, else the deposit's own feature table."""
    out = {}
    if "target_ensembl" in obs and obs.target_ensembl.notna().any():
        t = obs[["perturbation", "target_ensembl"]].dropna()
        t = t[~t.perturbation.astype(str).str.contains("+", regex=False)]
        e = t.target_ensembl.astype(str).str.split(".").str[0]
        per = e.groupby(t.perturbation.astype(str)).unique()
        out.update({k: v[0] for k, v in per.items() if len(v) == 1 and v[0].startswith("ENSG")})
    if native_ids is not None and native_symbols is not None:
        f = pd.DataFrame(dict(s=pd.Series(native_symbols, dtype="object").astype(str), e=pd.Series(native_ids, dtype="object").astype(str).str.split(".").str[0]))
        f = f[f.e.str.startswith("ENSG")].drop_duplicates()
        unique = f[~f.s.duplicated(keep=False)]
        for s, e in zip(unique.s, unique.e):
            out.setdefault(s, e)
    return out


def half_of(obs, key_cols, seed, context):
    """A/B assignment: within each (key, batch) group, cells in deposit order are permuted by a seeded rng and alternate A, B."""
    half = np.full(len(obs), "all", object)
    groups = obs.groupby(key_cols, sort=False, observed=True).indices
    for key, idx in groups.items():
        key = key if isinstance(key, tuple) else (key,)
        digest = hashlib.blake2b("|".join([str(seed), context, *map(str, key)]).encode(), digest_size=8).digest()
        order = np.random.default_rng(int.from_bytes(digest, "little")).permutation(len(idx))
        ranks = np.empty(len(idx), int)
        ranks[order] = np.arange(len(idx))
        half[idx] = np.where(ranks % 2 == 0, "A", "B")
    return half


def qc_flags(obs, cfg):
    """Flag columns (never filters); NaN inputs give NaN flags, which never pass."""
    q = cfg["qc"]
    f = pd.DataFrame(index=obs.index)
    f["qc_genes_500"] = flag(pd.to_numeric(obs.n_genes, errors="coerce") >= q["min_genes"], obs.n_genes)
    f["qc_mito_20"] = flag(pd.to_numeric(obs.pct_mito, errors="coerce") < q["max_pct_mito"], obs.pct_mito)
    gu = pd.to_numeric(obs.get("guide_umi"), errors="coerce") if "guide_umi" in obs else pd.Series(np.nan, index=obs.index)
    f["qc_guide_20"] = flag(gu >= q["min_guide_umi"], gu)
    f["qc_umi_1000"] = flag(pd.to_numeric(obs.lib_size, errors="coerce") >= q["min_umi"], obs.lib_size)
    lanes = set(q.get("lane_flags", {}).get(str(obs.context.iloc[0]), []))
    f["qc_lane_flag"] = obs.batch.astype(str).isin(lanes).to_numpy()
    strict = np.ones(len(obs), bool)
    for c in cfg["profiles"]["strict"]:
        v = f[c]
        strict &= v.fillna(False).to_numpy(bool) if c != "qc_guide_20" else (v.isna() | v.fillna(False)).to_numpy(bool)
    f["qc_strict"] = strict
    return f


def flag(cond, source):
    """Boolean flag with NaN where the source value is missing."""
    out = cond.astype("boolean")
    out[pd.to_numeric(source, errors="coerce").isna().to_numpy()] = pd.NA
    return out


def sort_key(obs):
    """Row order of the shard: context, kind, perturbation, batch, control kind, deposit row."""
    k = pd.DataFrame(dict(context=obs.context.astype(str), kind=obs.kind.map(KIND_ORDER), perturbation=obs.perturbation.fillna("").astype(str),
                          batch=obs.batch.astype(str), control_kind=obs.control_kind.fillna("").astype(str), row=obs.deposit_row))
    return np.lexsort([k.row.to_numpy(), k.control_kind.to_numpy(), k.batch.to_numpy(), k.perturbation.to_numpy(), k.kind.to_numpy(), k.context.to_numpy()])
