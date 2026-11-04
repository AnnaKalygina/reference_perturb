"""One CSV row per (context, batch): study, source, controls and single-target cells per perturbation in that batch."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from audit import CFG  # noqa: E402
from labels import gate  # noqa: E402

OUT = Path(CFG["out_dir"])
R22 = ("ReplogleWeissman2022", "https://doi.org/10.1016/j.cell.2022.05.013", "10x Genomics Chromium 3' v3 with direct guide capture")
NADIG = ("NadigOConnor2025", "https://doi.org/10.1038/s41588-025-02169-3", "10x Genomics Chromium 3' v3 with direct guide capture")
ORION = ("HuangChu2025", "https://doi.org/10.1101/2025.06.11.659105", "10x Genomics GEM-X 5' v3 with CRISPR feature barcoding, fixed and cryopreserved cells (FiCS)")
STUDY = {
    "scperturb_adamson": ("AdamsonWeissman2016", "https://doi.org/10.1016/j.cell.2016.11.048", "10x Genomics 3' (Perturb-seq, guide barcode in an expressed transcript)"),
    "scperturb_frangieh": ("FrangiehIzar2021", "https://doi.org/10.1038/s41588-021-00779-1", "10x Genomics 3' RNA + CITE-seq protein (Perturb-CITE-seq)"),
    "scperturb_sunshine": ("SunshineHein2023", "https://doi.org/10.1038/s41467-023-41788-4", "10x Genomics 3' with guide capture (kit not verified)"),
    "scperturb_norman": ("NormanWeissman2019", "https://doi.org/10.1126/science.aax4438", "10x Genomics Chromium 3' v2"),
    "scperturb_wessels": ("WesselsSatija2023", "https://doi.org/10.1038/s41592-022-01705-x", "10x Genomics 3' with crRNA capture and cell hashing (CaRPool-seq; kit not verified)"),
    "replogle22_k562": R22, "replogle22_rpe1": R22, "replogle22_gwps": R22,
    "nadig25_hepg2": NADIG, "nadig25_jurkat": NADIG,
    "replogle20": ("ReplogleAdamson2020", "https://doi.org/10.1038/s41587-020-0470-y", "10x Genomics Chromium 3' v3 with direct guide capture"),
    "orion_hct116": ORION, "orion_hek293t": ORION,
    "kolf": ("NourreddineMali2026", "https://doi.org/10.1038/s41587-026-03199-w", "10x Genomics 3' HT"),
    "vcc2025_h1": ("RoohaniBurke2025", "https://doi.org/10.1016/j.cell.2025.06.008 (challenge paper; the data have no own publication)", "10x Genomics Flex (fixed, probe-based)"),
    "feng": ("FengVelten2026", "https://doi.org/10.1016/j.xgen.2025.101076", "10x Genomics Chromium Next GEM 5' v2"),
    "dld1": ("YeungXie2026", "https://doi.org/10.64898/2026.07.10.737863", "10x Genomics Chromium 3' GEM-X v4 + sgRNA capture + MULTI-seq CMO hashing"),
    "hesc": ("PanZhuang2026", "https://doi.org/10.64898/2026.05.16.725005", "10x Genomics 5' (per GEO sample protocol)"),
    "cd4t": ("ZhuMarson2026", "https://doi.org/10.1016/j.cell.2026.08.002", "10x Genomics Flex (GEM-X Flex v1, probe-based)"),
    "jiang": ("JiangSatija2025", "https://doi.org/10.1038/s41556-025-01622-z", "Parse Biosciences Evercode WT Mega (split-pool barcoding)"),
    "mcfaline": ("McFalineFigueroaTrapnell2024", "https://doi.org/10.1016/j.xgen.2023.100487", "sci-RNA-seq3 with sci-Plex hashing (sci-Plex-GxE)"),
    "viperturb": ("BraduSatija2026", "https://doi.org/10.1101/2026.02.12.705613", "10x Genomics Flex, fixed cells (kit version v1 or v2 not verified)"),
}


def describe(r):
    """Context in words from its standardized fields."""
    parts = [f"context id {r.context}", f"cell type {r.cell_type}", f"donor or line {r.donor}", f"condition {r.condition}"]
    if isinstance(r.timepoint, str) and r.timepoint:
        parts.append(f"timepoint {r.timepoint}")
    return "; ".join(parts)


def controls(name):
    """Control cells per (context, batch) under the source's control definition, and whether unassigned cells count."""
    kinds = pd.read_csv(OUT / name / "controls_per_batch.csv", dtype={"batch": str})
    with_unassigned = name in CFG["controls_including_unassigned"]
    excluded = set(CFG["control_kinds_not_used"]) - ({"unassigned"} if with_unassigned else set())
    used = kinds[~kinds.control_kind.isin(excluded)]
    ctrl = used.groupby(["context", "batch"]).n_cells.sum().rename("n_control_cells").reset_index()
    label = kinds.control_kind + np.where(kinds.control_kind.isin(excluded), " (not used)", "") + " " + kinds.n_cells.astype(str)
    shown = label.groupby([kinds.context, kinds.batch]).agg(", ".join).rename("control_kinds_in_batch").reset_index()
    ctrl = ctrl.merge(shown, on=["context", "batch"], how="outer")
    ctrl["n_control_cells"] = ctrl.n_control_cells.fillna(0).astype(int)
    return ctrl, with_unassigned


def perturbed(name, log=False):
    """Single-target cells per (context, batch, perturbation) that pass the label gate, optionally with the gate log."""
    kept, gate_log = gate(pd.read_csv(OUT / name / "single_target_cells_per_batch.csv.gz", dtype={"batch": str}, keep_default_na=False),
                          pd.read_csv(OUT / name / "target_ensembl.csv", dtype=str, keep_default_na=False).replace("", None))
    return (kept, gate_log) if log else kept


def source_rows(name):
    """Batch rows of one source."""
    d = OUT / name
    meta = json.load(open(d / "summary.json"))["meta"]
    study, url, tech = STUDY[name]
    ctrl, with_unassigned = controls(name)
    pert = perturbed(name)
    desc = {r.context: describe(r) for r in pd.read_csv(d / "context_description.csv", dtype=str).itertuples()}
    g = pert.groupby(["context", "batch"]).n_cells
    pb = pd.DataFrame(dict(n_perturbed_cells=g.sum(), n_perturbations_present=g.size(), mean=g.mean(),
                           min=g.min(), max=g.max(), var=g.var(ddof=1))).reset_index()
    pb["se"] = np.sqrt(pb["var"] / pb.n_perturbations_present)
    b = ctrl.merge(pb, on=["context", "batch"], how="outer")
    n_perts = pert.groupby("context").perturbation.nunique()
    b["n_perturbations"] = b.context.map(n_perts).fillna(0).astype(int)
    b["n_control_cells"] = b.n_control_cells.fillna(0).astype(int)
    b["n_perturbed_cells"] = b.n_perturbed_cells.fillna(0).astype(int)
    b["n_perturbations_present"] = b.n_perturbations_present.fillna(0).astype(int)
    out = pd.DataFrame(dict(
        study=study, publication=url, data_source=meta["deposit"], perturbation_modality=meta["modality"],
        sequencing_technology=tech, context=b.context,
        context_description=b.context.map(desc),
        n_perturbations=b.n_perturbations, batch_kind=meta["batch_kind"],
        control_definition="unassigned + non-targeting cells (authors' convention)" if with_unassigned else "control guides only",
        control_batch=np.where(b.n_control_cells > 0, b.batch, ""), n_control_cells=b.n_control_cells,
        control_kinds_in_batch=b.control_kinds_in_batch.fillna(""),
        perturbed_batch=np.where(b.n_perturbed_cells > 0, b.batch, ""), n_perturbed_cells=b.n_perturbed_cells,
        n_perturbations_present=b.n_perturbations_present,
        share_of_perturbations_present=np.where(b.n_perturbations > 0, b.n_perturbations_present / b.n_perturbations.clip(lower=1), np.nan),
        cells_per_perturbation_mean=b["mean"], cells_per_perturbation_min=b["min"], cells_per_perturbation_max=b["max"],
        cells_per_perturbation_se=b["se"], cells_per_perturbation_var=b["var"]))
    return out.sort_values(["context", "control_batch", "perturbed_batch"], key=lambda s: s.astype(str))


def main():
    """Assemble the table over all sources."""
    t = pd.concat([source_rows(n) for n in STUDY], ignore_index=True)
    pd.concat([perturbed(n, log=True)[1].assign(source=n) for n in STUDY], ignore_index=True).to_csv(OUT / "perturbation_label_gate.csv", index=False)
    t = t.merge(pd.read_csv(OUT / "context_control_noise.csv"), on="context", how="left")
    t.to_csv(OUT / "context_batch_table.csv", index=False, float_format="%.4g")
    shown = t.astype(object)
    study_cols = ["study", "publication", "data_source", "perturbation_modality", "sequencing_technology"]
    context_cols = ["context", "context_description", "n_perturbations", "batch_kind", "control_definition",
                    *[c for c in t.columns if c in set(pd.read_csv(OUT / "context_control_noise.csv", nrows=0).columns) - {"context"}]]
    shown.loc[t.study.eq(t.study.shift()), study_cols] = ""
    shown.loc[t.context.eq(t.context.shift()), context_cols] = ""
    shown.to_csv(OUT / "context_batch_table_display.csv", index=False, float_format="%.4g")
    print(f"{len(t):,} batch rows, {t.context.nunique()} contexts")


if __name__ == "__main__":
    main()
