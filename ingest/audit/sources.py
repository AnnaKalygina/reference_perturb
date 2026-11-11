"""One read-only loader per deposit: standardized per-cell obs plus deposit metadata."""
import glob
import gzip
import json
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from h5 import col, dec, index, matrix_values, tenx_features, tenx_values

STD = ["context", "cell_type", "condition", "timepoint", "donor", "batch", "perturbation", "target_ensembl", "is_control",
       "control_kind", "guide", "n_guides", "total_counts", "n_genes", "pct_mito"]


def frame(n, **cols):
    """Per-cell standardized obs; scalars broadcast, missing columns are NA."""
    d = {}
    for c in STD:
        v = cols.get(c)
        if v is None:
            d[c] = pd.Series([pd.NA] * n, dtype="object") if c not in ("is_control",) else np.zeros(n, bool)
        elif np.isscalar(v):
            d[c] = np.full(n, v, dtype=object)
        else:
            d[c] = np.asarray(v)
    df = pd.DataFrame(d)
    df["perturbation"] = df["perturbation"].fillna("").astype(str)
    df.loc[df.is_control, "perturbation"] = ""
    return df


def join_targets(parts):
    """Target symbols to one label: sorted unique, joined by '+'."""
    s = sorted({p for p in parts if p})
    return "+".join(s)


def ctx(*parts):
    """Context id from its parts."""
    return "__".join(str(p) for p in parts if p not in (None, ""))


class Source:
    """A loaded deposit."""

    def __init__(self, name, obs, meta, genes, ensembl, values):
        self.name, self.obs, self.meta, self.genes, self.ensembl, self.values = name, obs, meta, genes, ensembl, values


def scperturb_adamson(cfg):
    """scPerturb AdamsonWeissman2016 GSM2406681 (10X010), identical file in repo and VCC2026."""
    p = Path(cfg["vcc_data"]) / "raw/scperturb/AdamsonWeissman2016_GSM2406681_10X010.h5ad"
    with h5py.File(p, "r") as h:
        pert = col(h, "obs/perturbation")
        n = len(pert)
        s = pd.Series(pert, dtype="object").fillna("")
        is_ctrl = s.str.contains(r"\(", regex=True).to_numpy()
        lab = s.str.split("_").str[0].to_numpy()
        obs = frame(n, context="adamson16__K562", cell_type="K562", condition="unstimulated", donor="K562",
                    batch="10X010", perturbation=lab, is_control=is_ctrl,
                    control_kind=np.where(is_ctrl, "non_targeting", None), guide=s.to_numpy(),
                    n_guides=col(h, "obs/nperts"), total_counts=col(h, "obs/ncounts"), n_genes=col(h, "obs/ngenes"),
                    pct_mito=col(h, "obs/percent_mito"))
        genes, ens = col(h, "var/gene_symbol"), col(h, "var/ensembl_id")
        vals = matrix_values(h, "X", cfg["integer_sample_values"])
    meta = dict(deposit="Zenodo 7416068/10044268 AdamsonWeissman2016_GSM2406681_10X010.h5ad (scPerturb); GEO GSM2406681",
                files=[str(p), str(Path(cfg["repo_data"]) / "adamson16/adamson16_downloaded.h5ad")],
                batch_field="none (single 10x lane GSM2406681 / 10X010)", batch_kind="none",
                modality="CRISPRi", cell_calling="authors (Cell Ranger, scPerturb copy)",
                guide_assignment="authors' perturbation label; nperts counts guide+plasmid; unassigned cells have no label",
                matrix="X csc float32 integer-valued", gene_ids="symbols + ensembl_id (Ensembl release not stated)",
                flags=["only GSM2406681 of the Adamson study; other experiments (GSM2406675/77) not in scPerturb file"])
    return Source("adamson16", obs, meta, genes, ens, vals)


FRANGIEH_CTRL = ("NO_SITE", "ONE_NON-GENE_SITE")


def scperturb_frangieh(cfg):
    """scPerturb FrangiehIzar2021_RNA, identical file in repo and VCC2026."""
    p = Path(cfg["vcc_data"]) / "raw/scperturb/FrangiehIzar2021_RNA.h5ad"
    rx = re.compile(r"^(.+)_(\d+)$")
    with h5py.File(p, "r") as h:
        gid = col(h, "obs/guide_id")
        cond = col(h, "obs/perturbation_2")
        n = len(gid)
        lab, is_ctrl, kind, ng = [], np.zeros(n, bool), np.full(n, None, object), np.zeros(n)
        for i, g in enumerate(gid):
            if g is None or str(g) in ("", "nan"):
                lab.append("")
                continue
            parts = str(g).split(";")
            ng[i] = len(parts)
            gs = [rx.match(t).group(1) if rx.match(t) else t for t in parts]
            tg = {x for x in gs if x not in FRANGIEH_CTRL}
            if not tg:
                is_ctrl[i] = True
                kind[i] = "+".join(sorted({"non_targeting" if x == "NO_SITE" else "safe_targeting" for x in gs}))
                lab.append("")
            else:
                lab.append(join_targets(tg))
        cmap = {"Control": "control", "IFNγ": "IFNG", "Co-culture": "TIL_coculture"}
        c = np.array([cmap.get(str(x), str(x)) for x in cond], object)
        obs = frame(n, context=np.array(["frangieh21__melanoma__" + x for x in c], object), cell_type="patient-derived melanoma",
                    condition=c, donor="melanoma_patient", batch="unknown", perturbation=np.array(lab, object),
                    is_control=is_ctrl, control_kind=kind, guide=gid, n_guides=ng,
                    total_counts=col(h, "obs/ncounts"), n_genes=col(h, "obs/ngenes"), pct_mito=col(h, "obs/percent_mito"))
        genes, ens = col(h, "var/gene_symbol"), col(h, "var/ensembl_id")
        vals = matrix_values(h, "X", cfg["integer_sample_values"])
    meta = dict(deposit="Zenodo 7041849 FrangiehIzar2021_RNA.h5ad (scPerturb); original Single Cell Portal SCP1064",
                files=[str(p), str(Path(cfg["repo_data"]) / "frangieh21/frangieh21_downloaded.h5ad")],
                batch_field="none in scPerturb copy (SCP1064 metadata may carry 10x lane/sample)", batch_kind="none",
                modality="CRISPRko (Cas9, Perturb-CITE-seq)", cell_calling="authors",
                guide_assignment="authors' guide_id (';'-joined for MOI>1); scPerturb labels guide-less cells 'control'",
                matrix="X csc float32 integer-valued", gene_ids="symbols + ensembl_id",
                flags=["scPerturb 'control' includes 23,028 guide-less cells; true controls are NO_SITE/ONE_NON-GENE_SITE guides",
                       "no batch field; check SCP1064 raw metadata before ingestion"])
    return Source("frangieh21", obs, meta, genes, ens, vals)


def scperturb_sunshine(cfg):
    """scPerturb SunshineHein2023, identical file in repo and VCC2026."""
    p = Path(cfg["vcc_data"]) / "raw/scperturb/SunshineHein2023.h5ad"
    with h5py.File(p, "r") as h:
        pert = pd.Series(col(h, "obs/perturbation"), dtype="object").fillna("")
        n = len(pert)
        is_ctrl = (pert == "control").to_numpy()
        lab = np.array([join_targets(x.split("_")) if x and x != "control" else "" for x in pert], object)
        obs = frame(n, context="sunshine23__Calu3", cell_type="Calu-3", condition="unstimulated", donor="Calu-3",
                    batch=col(h, "obs/gem_group").astype(str), perturbation=lab, is_control=is_ctrl,
                    control_kind=np.where(is_ctrl, "non_targeting", None), guide=col(h, "obs/guide_id"),
                    n_guides=col(h, "obs/nperts"), total_counts=col(h, "obs/ncounts"), n_genes=col(h, "obs/ngenes"),
                    pct_mito=col(h, "obs/percent_mito"))
        genes, ens = col(h, "var/gene_symbol"), col(h, "var/ensembl_id")
        vals = matrix_values(h, "X", cfg["integer_sample_values"])
    meta = dict(deposit="Zenodo 10044268 SunshineHein2023.h5ad (scPerturb)",
                files=[str(p), str(Path(cfg["repo_data"]) / "sunshine23/sunshine23_downloaded.h5ad")],
                batch_field="obs/gem_group (10x GEM group = lane)", batch_kind="technical",
                modality="CRISPRi (high MOI)", cell_calling="authors", guide_assignment="authors; high MOI, nperts up to 40+",
                matrix="X csc int64", gene_ids="symbols + ensembl_id",
                flags=["high-MOI design: most perturbed cells carry several targets"])
    return Source("sunshine23", obs, meta, genes, ens, vals)


def scperturb_norman(cfg):
    """scPerturb NormanWeissman2019_filtered (repo only)."""
    p = Path(cfg["repo_data"]) / "norman19/norman19_downloaded.h5ad"
    with h5py.File(p, "r") as h:
        pert = pd.Series(col(h, "obs/perturbation"), dtype="object").fillna("")
        n = len(pert)
        is_ctrl = (pert == "control").to_numpy()
        lab = np.array([join_targets(x.split("_")) if x and x != "control" else "" for x in pert], object)
        obs = frame(n, context="norman19__K562", cell_type="K562", condition="unstimulated", donor="K562",
                    batch=col(h, "obs/gemgroup").astype(str), perturbation=lab, is_control=is_ctrl,
                    control_kind=np.where(is_ctrl, "non_targeting", None), guide=col(h, "obs/guide_id"),
                    n_guides=col(h, "obs/nperts"), total_counts=col(h, "obs/ncounts"), n_genes=col(h, "obs/ngenes"),
                    pct_mito=col(h, "obs/percent_mito"))
        genes, ens = index(h, "var"), col(h, "var/ensemble_id")
        vals = matrix_values(h, "X", cfg["integer_sample_values"])
    meta = dict(deposit="Zenodo 7041849 NormanWeissman2019_filtered.h5ad (scPerturb); GEO GSE133344",
                files=[str(p)], batch_field="obs/gemgroup (10x GEM group)", batch_kind="technical",
                modality="CRISPRa single + dual", cell_calling="authors",
                guide_assignment="authors' guide_id; NegCtrl guides; combos 'A_B'",
                matrix="X csc float32 integer-valued", gene_ids="symbols + ensemble_id",
                flags=["Norman 2019 is CRISPRa (not CRISPRi as the repo catalog says)"])
    return Source("norman19", obs, meta, genes, ens, vals)


def scperturb_wessels(cfg):
    """scPerturb WesselsSatija2023 CaRPool-seq (Cas13) THP-1 (repo only)."""
    p = Path(cfg["repo_data"]) / "wessels23/wessels23_downloaded.h5ad"
    with h5py.File(p, "r") as h:
        pert = pd.Series(col(h, "obs/perturbation"), dtype="object").fillna("")
        cls = col(h, "obs/Guide.Class")
        n = len(pert)
        is_ctrl = (cls == "NT")
        lab = np.array([join_targets([t for t in x.split("_") if t not in ("NT", "control")]) for x in pert], object)
        obs = frame(n, context="wessels23__THP1", cell_type="THP-1", condition="unstimulated", donor="THP-1",
                    batch=col(h, "obs/10X_lane").astype(str), perturbation=lab, is_control=is_ctrl,
                    control_kind=np.where(is_ctrl, "non_targeting", None), guide=col(h, "obs/guides"),
                    n_guides=col(h, "obs/nperts"), total_counts=col(h, "obs/ncounts"), n_genes=col(h, "obs/ngenes"),
                    pct_mito=col(h, "obs/percent_mito"))
        genes = col(h, "var/gene_symbol")
        vals = matrix_values(h, "X", cfg["integer_sample_values"])
    meta = dict(deposit="Zenodo 13350497 WesselsSatija2023.h5ad (scPerturb); GEO GSE213957",
                files=[str(p)], batch_field="obs/10X_lane (4 lanes)", batch_kind="technical",
                modality="Cas13 (CaRPool-seq, RfxCas13d)", cell_calling="authors (HTO demux)",
                guide_assignment="authors' Guide.Class Single/Dual/NT", matrix="X csr int64", gene_ids="symbols only",
                flags=["only ~424 NT control cells over 4 lanes", "the repo catalog calls this CRISPRi; it is Cas13"])
    return Source("wessels23", obs, meta, genes, None, vals)


def replogle_rows(cfg, name, path, cell_type, deposit, timepoint, extra_flags=()):
    """Replogle 2022 / Nadig 2025 raw_singlecell_01.h5ad (figshare / GEO)."""
    with h5py.File(path, "r") as h:
        g = col(h, "obs/gene").astype(str)
        n = len(g)
        is_ctrl = g == "non-targeting"
        umi = col(h, "obs/UMI_count")
        mito = col(h, "obs/mitopercent") if "obs/mitopercent" in h else col(h, "obs/percent_mito")
        gem = col(h, "obs/gem_group") if "obs/gem_group" in h else col(h, "obs/batch")
        guide = col(h, "obs/sgID_AB") if "obs/sgID_AB" in h else col(h, "obs/guide_id")
        obs = frame(n, context=ctx(name, cell_type), cell_type=cell_type, condition="unstimulated", timepoint=timepoint,
                    donor=cell_type, batch=np.asarray(gem).astype(str), perturbation=np.where(is_ctrl, "", g),
                    target_ensembl=np.where(is_ctrl, None, col(h, "obs/gene_id").astype(str)),
                    is_control=is_ctrl, control_kind=np.where(is_ctrl, "non_targeting", None), guide=guide,
                    n_guides=2.0, total_counts=umi, pct_mito=mito)
        genes = col(h, "var/gene_name") if "var/gene_name" in h else index(h, "var")
        ens = index(h, "var") if str(index(h, "var")[0]).startswith("ENSG") else (col(h, "var/gene_id") if "var/gene_id" in h else None)
        vals = matrix_values(h, "X", cfg["integer_sample_values"])
        shape = h["X"].shape if isinstance(h["X"], h5py.Dataset) else tuple(h["X"].attrs["shape"])
    meta = dict(deposit=deposit, files=[str(path)], batch_field="obs/gem_group (10x GEM group)", batch_kind="technical",
                modality="CRISPRi (dual-guide, dCas9-KRAB)", cell_calling="authors (Cell Ranger + authors' filter)",
                guide_assignment="authors; only cells with one assigned dual-guide construct are deposited (n_guides=2 by design)",
                matrix=f"X dense float32 {shape}, only genes > 0.01 UMI/cell kept by the authors",
                gene_ids="Ensembl ids (var index) + gene_name", flags=["deposit keeps ~8-9k expressed genes only; "
                                                                      "UMI_count is total over all genes"] + list(extra_flags))
    return Source(name, obs, meta, genes, ens, vals)


def replogle22_k562(cfg):
    p = Path(cfg["vcc_data"]) / "raw/replogle2022/K562_essential_raw_singlecell_01.h5ad"
    return replogle_rows(cfg, "replogle22_essential", p, "K562", "figshare 20029387 / 35773219 K562_essential_raw_singlecell_01.h5ad; "
                         "repo copy Zenodo 10044268 ReplogleWeissman2022_K562_essential.h5ad", "day 6")


def replogle22_rpe1(cfg):
    p = Path(cfg["vcc_data"]) / "raw/replogle2022/rpe1_raw_singlecell_01.h5ad"
    return replogle_rows(cfg, "replogle22_essential", p, "RPE1", "figshare 35775606 rpe1_raw_singlecell_01.h5ad; "
                         "repo copy Zenodo 10044268 ReplogleWeissman2022_rpe1.h5ad", "day 7")


def replogle22_gwps(cfg):
    p = Path(cfg["vcc_data"]) / "raw/replogle2022/K562_gwps_raw_singlecell_01.h5ad"
    return replogle_rows(cfg, "replogle22_gwps", p, "K562", "figshare 35775507 K562_gwps_raw_singlecell_01.h5ad", "day 8")


def nadig25_hepg2(cfg):
    p = Path(cfg["vcc_data"]) / "raw/nadig2025/GSE264667_hepg2_raw_singlecell_01.h5ad"
    return replogle_rows(cfg, "nadig25", p, "HepG2", "GEO GSE264667 GSE264667_hepg2_raw_singlecell_01.h5ad", "day 7")


def nadig25_jurkat(cfg):
    p = Path(cfg["vcc_data"]) / "raw/nadig2025/GSE264667_jurkat_raw_singlecell_01.h5ad"
    return replogle_rows(cfg, "nadig25", p, "Jurkat", "GEO GSE264667 GSE264667_jurkat_raw_singlecell_01.h5ad", "day 7")


def replogle20(cfg):
    """GEO GSM4367984 Replogle 2020 exp6 (same files in repo and VCC2026)."""
    d = Path(cfg["vcc_data"]) / "raw/replogle2020"
    f = pd.read_csv(d / "GSM4367984_exp6.features.tsv.gz", sep="\t", header=None, names=["id", "name", "type"])
    name_of = dict(zip(f["id"], f["name"]))
    bc = pd.read_csv(d / "GSM4367984_exp6.barcodes.tsv.gz", header=None)[0].to_numpy()
    ids = pd.read_csv(d / "GSM4367984_exp6.cell_identities.csv.gz").set_index("cell_barcode").reindex(bc)
    n = len(bc)
    ctrl = (ids["control"] == 1).to_numpy()
    tA = ids["target_A"].map(lambda x: name_of.get(x, x) if isinstance(x, str) and x != "Non-Targeting" else "")
    tB = ids["target_B"].map(lambda x: name_of.get(x, x) if isinstance(x, str) and x != "Non-Targeting" else "")
    lab = np.array([join_targets([a, b]) for a, b in zip(tA, tB)], object)
    lab[~(ids[["single", "double"]].fillna(0).sum(1) > 0).to_numpy()] = ""
    obs = frame(n, context="replogle20__K562", cell_type="K562", condition="unstimulated", donor="K562",
                batch=pd.Series(bc).str.split("-").str[1].to_numpy(), perturbation=lab, is_control=ctrl,
                control_kind=np.where(ctrl, "non_targeting", None), guide=ids["guide_identity"].to_numpy() if "guide_identity" in ids else None,
                n_guides=ids["num_guides"].to_numpy())
    with gzip.open(d / "GSM4367984_exp6.matrix.mtx.gz", "rt") as fh:
        rows = []
        for line in fh:
            if line.startswith("%"):
                continue
            if not rows:
                rows.append(line)
                continue
            rows.append(float(line.split()[2]))
            if len(rows) > 200000:
                break
    gex = (f["type"] == "Gene Expression").to_numpy()
    meta = dict(deposit="GEO GSM4367984 (GSE146194) exp6 matrix.mtx + cell_identities.csv", files=[str(d), str(Path(cfg["repo_data"]) / "replogle20")],
                batch_field="barcode suffix = 10x GEM group", batch_kind="technical", modality="CRISPRi (dual-guide direct capture)",
                cell_calling="authors (all Cell Ranger barcodes deposited; cell_identities lists guide-called cells)",
                guide_assignment="authors' cell_identities: single / double / control, num_guides",
                matrix="matrix.mtx integer", gene_ids="Ensembl ids + symbols (Cell Ranger reference)",
                flags=["barcodes without a cell_identities row are unassigned"])
    return Source("replogle20", obs, meta, f["name"][gex].to_numpy(), f["id"][gex].to_numpy(), np.array(rows[1:], float))


def orion(cfg, line):
    """X-Atlas/Orion parquet shards (HF Xaira-Therapeutics/X-Atlas-Orion)."""
    files = sorted(glob.glob(str(Path(cfg["vcc_data"]) / f"raw/orion/{line}_Batch*.parquet")),
                   key=lambda s: int(re.search(r"Batch(\d+)", s).group(1)))
    parts = []
    for fp in files:
        t = pq.read_table(fp, columns=["sample", "gene_target", "guide_target", "num_features", "pass_guide_filter",
                                       "total_counts", "n_genes_by_counts", "pct_counts_mt"]).to_pandas()
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    n = len(t)
    is_ctrl = (t.gene_target == "Non-Targeting").to_numpy()
    passed = t.pass_guide_filter.astype(bool).to_numpy()
    lab = np.where(passed & ~is_ctrl, t.gene_target.astype(str), "")
    obs = frame(n, context=f"orion__{line}", cell_type=line, condition="unstimulated", timepoint="day 7", donor=line,
                batch=t["sample"].astype(str).to_numpy(), perturbation=lab, is_control=is_ctrl & passed,
                control_kind=np.where(is_ctrl & passed, "non_targeting", None), guide=t.guide_target.to_numpy(),
                n_guides=t.num_features.to_numpy(), total_counts=t.total_counts.to_numpy(), n_genes=t.n_genes_by_counts.to_numpy(),
                pct_mito=t.pct_counts_mt.to_numpy())
    gm = pd.read_parquet(Path(cfg["vcc_data"]) / "orion_probe/gene_metadata.parquet").sort_values("gene_token_id")
    v = pq.read_table(files[0], columns=["gene_expression"]).column(0).combine_chunks().values.to_numpy()[: cfg["integer_sample_values"]]
    meta = dict(deposit=f"HF Xaira-Therapeutics/X-Atlas-Orion data/{line}_Batch*.parquet ({len(files)} shards)",
                files=[str(Path(cfg["vcc_data"]) / "raw/orion"), str(Path(cfg["repo_data"]) / "xatlas_orion/xatlas_orion_processed.h5ad")],
                batch_field="parquet shard / obs sample (one 10x GEM-X sample each)", batch_kind="technical",
                modality="CRISPRi (dual-guide)", cell_calling="authors",
                guide_assignment="authors' guide_target / gene_target + pass_guide_filter (num_features guides per cell)",
                matrix="gene_token_id/gene_expression lists, integer", gene_ids="gene_metadata.parquet: ensembl_id + gene_name",
                flags=[f"{int((~passed).sum())} cells fail pass_guide_filter", "repo xatlas_orion_processed.h5ad is log-normalized, not raw"])
    return Source(f"orion_{line}", obs, meta, gm.gene_name.astype(str).to_numpy(), gm.ensembl_id.astype(str).to_numpy(), v)


def orion_hct116(cfg):
    return orion(cfg, "HCT116")


def orion_hek293t(cfg):
    return orion(cfg, "HEK293T")


def kolf(cfg):
    """KOLF2.1J hiPSC pan-genome (figshare+ 64650261)."""
    p = Path(cfg["vcc_data"]) / "raw/kolf/KOLF_Pan_Genome_QC_Filtered.h5ad"
    with h5py.File(p, "r") as h:
        pert = col(h, "obs/perturbed").astype(str)
        g = col(h, "obs/gene_target").astype(str)
        n = len(g)
        is_ctrl = pert == "False"
        obs = frame(n, context="kolf__KOLF2.1J", cell_type="KOLF2.1J hiPSC", condition="unstimulated", timepoint="day 6",
                    donor="KOLF2.1J", batch=col(h, "obs/channel").astype(str), perturbation=np.where(is_ctrl, "", g),
                    target_ensembl=np.where(is_ctrl, None, col(h, "obs/gene_target_ensembl_id").astype(str)),
                    is_control=is_ctrl, control_kind=np.where(is_ctrl, "non_targeting", None), guide=col(h, "obs/gRNA"),
                    n_guides=col(h, "obs/n_gRNA"), total_counts=col(h, "obs/total_counts"), n_genes=col(h, "obs/n_genes_by_counts"),
                    pct_mito=col(h, "obs/pct_counts_mt"))
        obs["sublibrary"] = col(h, "obs/batch").astype(str)
        genes = index(h, "var")
        ens = col(h, "var/gene_ids")
        vals = matrix_values(h, "layers/counts", cfg["integer_sample_values"])
        xvals = matrix_values(h, "X", 200000)
    meta = dict(deposit="figshare+ 64650261 KOLF_Pan_Genome_QC_Filtered.h5ad (layers/counts)", files=[str(p)],
                batch_field="obs/channel (10x chip channel, 90); obs/batch = sub-library ALPHA/BETA/GAMMA (each target in one)",
                batch_kind="technical", modality="CRISPRi (KRAB-ZIM3-dCas9)", cell_calling="authors (QC-filtered object)",
                guide_assignment="authors; n_gRNA == 1 for every cell (single-guide cells only deposited)",
                matrix="layers/counts CSC", gene_ids="var gene_ids (Ensembl) + symbols",
                flags=["deposited object is already QC-filtered by the authors",
                       f"X non-integer sample count {int((xvals != np.round(xvals)).sum())} (X normalized; counts in layers/counts)"])
    return Source("kolf", obs, meta, genes, ens, vals)


def vcc2025_h1(cfg):
    """VCC 2025 H1 hESC (Arc), three splits, shared controls."""
    d = Path(cfg["vcc_data"]) / "vcc2025"
    parts, vals = [], None
    for s in ("Training", "Validation", "Test"):
        with h5py.File(d / f"adata_{s}.h5ad", "r") as h:
            t = pd.DataFrame(dict(cell=index(h, "obs"), tg=col(h, "obs/target_gene").astype(str),
                                  batch=col(h, "obs/batch").astype(str), guide=col(h, "obs/guide_id")))
            t["split"] = s
            parts.append(t)
            if vals is None:
                vals = matrix_values(h, "X", cfg["integer_sample_values"])
                genes, ens = index(h, "var"), col(h, "var/gene_id")
    t = pd.concat(parts, ignore_index=True)
    t = t[~((t.tg == "non-targeting") & (t.split != "Training"))].reset_index(drop=True)
    is_ctrl = (t.tg == "non-targeting").to_numpy()
    obs = frame(len(t), context="vcc2025__H1", cell_type="H1 hESC", condition="unstimulated", donor="H1", batch=t.batch.to_numpy(),
                perturbation=np.where(is_ctrl, "", t.tg), is_control=is_ctrl, control_kind=np.where(is_ctrl, "non_targeting", None),
                guide=t.guide.to_numpy(), n_guides=2.0)
    meta = dict(deposit="VCC 2025 challenge data (Arc Institute) adata_{Training,Validation,Test}.h5ad", files=[str(d)],
                batch_field="obs/batch (48 10x Flex batches)", batch_kind="technical", modality="CRISPRi (dual guide)",
                cell_calling="Arc", guide_assignment="Arc target_gene/guide_id", matrix="X csr integer",
                gene_ids="symbols + gene_id (Ensembl)", flags=["controls identical across the three splits (deduplicated)",
                                                               "licence/terms of VCC 2025 challenge data not verified"])
    return Source("vcc2025_h1", obs, meta, genes, ens, vals)


def feng(cfg):
    """Feng hPSC genome-wide CRISPRi (figshare 27989294), fitness + non-fitness screens, many lines."""
    d = Path(cfg["vcc_data"]) / "raw/feng"
    parts = []
    for screen, meta_f in (("fitness", "GW_Fitness_meta.tsv.gz"), ("nonfitness", "GW_NonFitness_meta.tsv.gz")):
        m = pd.read_csv(d / meta_f, sep="\t")
        m["screen"] = screen
        parts.append(m)
    m = pd.concat(parts, ignore_index=True)
    gcall = m.Guide_Call.astype(str)
    tgt = gcall.str.rsplit("_", n=1).str[0]
    nontarget = gcall.str.startswith("NonTarget") | (tgt == "NonTarget")
    unassigned = gcall == "unassigned"
    day = m.Batch.str.extract(r"_(Day\d+)_")[0].fillna("NA")
    pool = m.Batch.str.extract(r"_(Pool\d+)_")[0].fillna("NA")
    line = m.Cell_Line.fillna("NA").astype(str)
    lab = np.where(nontarget | unassigned, "", tgt)
    obs = frame(len(m), context=("feng__" + m.screen + "__" + line + "__" + day).to_numpy(), cell_type="hPSC",
                condition="unstimulated", timepoint=day.to_numpy(), donor=line.to_numpy(), batch=m.Batch.astype(str).to_numpy(),
                perturbation=lab, is_control=nontarget.to_numpy(),
                control_kind=np.select([nontarget, unassigned], ["non_targeting", "unassigned"], None),
                guide=gcall.to_numpy(), n_guides=np.where(unassigned, 0, 1))
    obs["unassigned"] = unassigned.to_numpy()
    obs["pool"] = pool.to_numpy()
    with gzip.open(d / "GenomeWideScreen_FitnessGenes_RNA-UMI-Counts.csv.gz", "rt") as fh:
        fh.readline()
        vals = []
        for _ in range(30):
            r = fh.readline().rstrip().split(",")[1:]
            vals.extend(float(x) for x in r if x not in ("", "0"))
    meta = dict(deposit="figshare article 27989294 (files 51059483, 51059558) RNA-UMI-Counts.csv.gz + GW_*_meta.tsv.gz",
                files=[str(d)], batch_field="meta Batch = screen_pool_day_inlet (10x inlet = channel); cell line = donor genotype",
                batch_kind="technical (inlet) within pool/day", modality="CRISPRi",
                cell_calling="authors", guide_assignment="authors' Guide_Call (guide or 'unassigned'); ~30% assigned",
                matrix="dense gene x cell csv.gz, integer", gene_ids="'ENSG:SYMBOL:Gene-Expression' features",
                flags=["only ~48 NonTarget cells: authors (and VCC2026) use 'unassigned' cells as controls, which are cells "
                       "with an undetected guide, not true controls", "cells of many hPSC lines pooled (donor genotype -> context)"])
    return Source("feng", obs, meta, None, None, np.array(vals))


def dld1(cfg):
    """GEO GSE337988 DLD-1 CRISPRi (Genentech), Low-MOI object."""
    d = Path(cfg["vcc_data"]) / "raw/gse337988"
    c = pd.read_csv(d / "low_coldata.csv.gz", dtype=str, keep_default_na=False)
    rd = pd.read_csv(d / "low_rowdata.csv.gz", dtype=str, keep_default_na=False)
    n = len(c)
    ntc = (c.assignment_v2 == "NTC").to_numpy()
    spike = c.assignment_v2.str.startswith("NTC_MENG").to_numpy()
    single = (c.demux_type_v2 == "singlet").to_numpy() & (c.demux_type == "singlet").to_numpy()
    is_ctrl = ntc & single
    lab = np.where(single & ~ntc & ~spike, c.assignment_v2, "")
    obs = frame(n, context="gse337988__DLD1", cell_type="DLD-1", condition="unstimulated",
                timepoint=(c.Timepoint_hrs + "h").to_numpy(), donor="DLD-1", batch=c.Sample.to_numpy(), perturbation=lab,
                is_control=is_ctrl, control_kind=np.select([is_ctrl, spike & single], ["non_targeting", "non_targeting_spike_in"], None),
                guide=c.guide_name.to_numpy(), n_guides=np.where(c.demux_type_v2 == "singlet", 1, np.where(c.demux_type_v2 == "doublet", 2, 0)),
                total_counts=pd.to_numeric(c["total.counts"]).to_numpy(), pct_mito=100 * pd.to_numeric(c["frac.mt"]).to_numpy())
    obs["biological_replicate"] = c.Biological_replicate.to_numpy()
    with h5py.File(d / "GSE337988_sublib2_processed_objects_Low_assays.h5", "r") as h:
        vals = np.asarray(h["assay001"][:200]).ravel()
    meta = dict(deposit="GEO GSE337988 sublib2_processed_objects_Low_assays.h5 + low_coldata/rowdata.csv.gz",
                files=[str(d)], batch_field="colData Sample (10x channel); Biological_replicate also present",
                batch_kind="technical (channel) nested in biological replicate", modality="CRISPRi (Zim3-dCas9, TSS-level guides)",
                cell_calling="authors (hashtag + guide demux, QC flags is.bad.cell)", guide_assignment="authors' demux_type_v2 / assignment_v2 / promoter",
                matrix="dense h5 assay001 cells x genes", gene_ids="rowData feature (Ensembl) + Symbol",
                flags=["only the Low-MOI sub-library 2 object audited; other GSE337988 objects not on disk",
                       "targets are TSS groups (gene_promoter); several promoters per gene"])
    return Source("gse337988", obs, meta, rd["Symbol"].to_numpy(), rd["feature"].to_numpy(), vals)


CONTROL_RE = re.compile(r"^(non[-_]?targeting|ntc|negative|neg)", re.I)


def hesc(cfg):
    """GEO GSE295214 hESC, 85 Cell Ranger h5; per-sgRNA calls by the authors' rule (no author calls on disk)."""
    min_umi = cfg["guide_call_min_umi"]["hesc"]
    files = sorted(glob.glob(str(Path(cfg["vcc_data"]) / "raw/gse295214/GSM*_filtered_feature_bc_matrix.h5")))
    parts, genome = [], None
    for fp in files:
        with h5py.File(fp, "r") as h:
            m = h["matrix"]
            ft = dec(m["features/feature_type"][:])
            names = dec(m["features/name"][:])
            nf, nc = (int(x) for x in m["shape"][:])
            ip = m["indptr"][:]
            ix = m["indices"][:]
            dat = m["data"][:]
            if genome is None:
                genome = pd.Series(dec(m["features/genome"][:])).value_counts().to_dict() if "genome" in m["features"] else None
        is_g = ft == "Gene Expression"
        gfeat = np.flatnonzero(~is_g)
        tg = np.array(["non-targeting" if CONTROL_RE.match(x) else x.split("_")[0] for x in names[gfeat]], object)
        col_of = np.full(nf, -1)
        col_of[gfeat] = np.arange(len(gfeat))
        cell = np.repeat(np.arange(nc), np.diff(ip))
        sel = col_of[ix] >= 0
        tot = np.bincount(cell[~sel], weights=dat[~sel], minlength=nc)
        ngen = np.bincount(cell[~sel], minlength=nc)
        called = sel & (dat >= min_umi)
        nguide = np.bincount(cell[called], minlength=nc)
        best = np.full(nc, "", object)
        best[cell[called]] = tg[col_of[ix[called]]]
        ok = nguide == 1
        lane = Path(fp).name.split("_filtered")[0]
        is_ctrl = ok & (best == "non-targeting")
        parts.append(frame(nc, context="gse295214__hESC", cell_type="hESC", condition="unstimulated", donor="hESC", batch=lane,
                           perturbation=np.where(ok & ~is_ctrl, best, ""), is_control=is_ctrl,
                           control_kind=np.where(is_ctrl, "non_targeting", None), guide=None,
                           n_guides=nguide, total_counts=tot, n_genes=ngen))
    obs = pd.concat(parts, ignore_index=True)
    obs["run"] = obs.batch.str.extract(r"_(jan_\d+|[a-z]{3}_\d+)_")[0].fillna("NA")
    ids, names, ft, _ = tenx_features(files[0])
    gex = ft == "Gene Expression"
    meta = dict(deposit="GEO GSE295214, 85 GSM*_filtered_feature_bc_matrix.h5", files=[str(Path(files[0]).parent)],
                batch_field="GEO sample file (one 10x lane each); date prefix (e.g. jan_10) = run", batch_kind="technical",
                modality="CRISPRi", cell_calling="Cell Ranger filtered barcodes (authors)",
                guide_assignment=f"NOT deposited as calls: computed per sgRNA as in the paper's methods (exactly one sgRNA with >= {min_umi} UMI); the paper also keeps only cells with >= 500 genes (n_genes here, not applied)",
                matrix="10x CSC integer", gene_ids=f"Cell Ranger features (Ensembl); genome field {genome}",
                flags=["no author guide calls on disk: check GEO supplementary for protospacer_calls before ingestion",
                       f"{int(gex.sum())} gene features (more than one reference?)"])
    return Source("gse295214", obs, meta, names[gex], ids[gex], tenx_values(files[0], cfg["integer_sample_values"]))


def tian21_lanes(tar, arm):
    """Cell Ranger h5 members of one Tian 2021 arm in lane order."""
    return sorted((m for m in tar.getmembers() if f"_{arm}_" in m.name), key=lambda m: m.name)


def tian21_neurons(cfg, name, arm):
    """Tian 2021 CROP-seq in iPSC-derived neurons: GEO lane h5 inside GSE152988_RAW.tar + the authors' single-sgRNA maps (maps.zip)."""
    d = Path(cfg["root"]) / "raw/tian21_neurons"
    maps = zipfile.ZipFile(d / "maps.zip")
    parts, outside = [], 0
    with tarfile.open(d / "GSE152988_RAW.tar") as tar:
        lanes = tian21_lanes(tar, arm)
        for m in lanes:
            lane = m.name.rsplit("_", 1)[1].split(".")[0]
            with h5py.File(tar.extractfile(m), "r") as h:
                bc = dec(h["matrix/barcodes"][:])
            sg = pd.read_csv(maps.open(f"maps/mapped_single_sgRNA_to_cell_{'CRISPRa_' if arm == 'CRISPRa' else ''}{lane}.txt"), sep="\t", header=None, index_col=0)[1]
            outside += int((~sg.index.isin(bc)).sum())
            guide = sg.reindex(bc).to_numpy()
            mapped = pd.notna(guide)
            target = np.where(mapped, pd.Series(guide).str.split("_").str[0], "")
            is_ctrl = mapped & (target == "non-targeting")
            parts.append(frame(len(bc), context=ctx(name, "iNeuron"), cell_type="iPSC-derived glutamatergic neuron", condition="unstimulated",
                               timepoint="day 10", donor="WTC11", batch=lane, perturbation=np.where(mapped & ~is_ctrl, target, ""),
                               is_control=is_ctrl, control_kind=np.where(is_ctrl, "non_targeting", None), guide=guide,
                               n_guides=np.where(mapped, 1.0, np.nan)))
        ids, names, ft, _ = tenx_features(tar.extractfile(lanes[0]))
        vals = tenx_values(tar.extractfile(lanes[0]), cfg["integer_sample_values"])
    gex = ft == "Gene Expression"
    meta = dict(deposit=f"GEO GSE152988 {arm} lanes {', '.join(m.name.split('_')[0] for m in lanes)} (filtered_feature_bc_matrix h5 in GSE152988_RAW.tar) + "
                "the authors' mapped_single_sgRNA_to_cell maps (kampmannlab.ucsf.edu/crop-seq maps.zip)",
                files=[str(d / "GSE152988_RAW.tar"), str(d / "maps.zip")], batch_field="GEO sample = 10x lane (file suffix lib<n>; about 25,000 cells loaded per lane)",
                batch_kind="technical", modality="CRISPRi (dCas9-KRAB, CROP-seq)" if arm == "CRISPRi" else "CRISPRa (DHFR-dCas9-VPH, CROP-seq)",
                cell_calling="Cell Ranger 3.1.0 filtered barcodes (GEO); the authors instead kept raw barcodes with > 2,000 genes, > 8,000 UMI and < 15% mito",
                guide_assignment="authors' single-sgRNA maps (sgRNA UMI > mean + 4 SD over the cell's sgRNAs, exactly one such sgRNA); "
                                 "cells absent from the maps have no or several sgRNAs, not distinguishable (n_guides NA)",
                matrix="10x CSC int32", gene_ids="Cell Ranger GRCh38 features (Ensembl ids + symbols)",
                flags=[f"{outside} mapped barcodes of this arm are outside the Cell Ranger filtered set (the authors mapped raw matrices)",
                       "no target Ensembl ids in the deposit; TMEM55A, ATP5C1 and ATP5H are old symbols the authors renamed"])
    return Source(name, pd.concat(parts, ignore_index=True), meta, names[gex], ids[gex], vals)


def tian21_neurons_crispri(cfg):
    return tian21_neurons(cfg, "tian21_neurons_crispri", "CRISPRi")


def tian21_neurons_crispra(cfg):
    return tian21_neurons(cfg, "tian21_neurons_crispra", "CRISPRa")


def cd4t(cfg):
    """Marson 2025 genome-wide CRISPRi in primary CD4+ T cells (GSE314342 + authors' S3 obs)."""
    d = Path(cfg["vcc_data"]) / "raw/cd4t_marson2025"
    parts = []
    for fp in sorted((d / "obs").glob("*.obs.parquet")):
        donor, cond = fp.name.split(".")[0].split("_", 1)
        o = pd.read_parquet(fp, columns=["lane_id", "guide_group", "guide_type", "low_quality", "perturbed_gene_name", "perturbed_gene_id",
                                         "guide_id", "total_counts", "n_genes_by_counts", "pct_counts_mt"])
        single = (o.guide_group == "targeting single sgRNA").to_numpy()
        nt = (o.guide_type == "non-targeting").to_numpy()
        is_ctrl = single & nt
        ng = np.select([o.guide_group == "no sgRNA", o.guide_group == "targeting single sgRNA", o.guide_group == "multi sgRNA"], [0, 1, 2], -1)
        parts.append(frame(len(o), context=f"cd4t_marson25__{donor}__{cond}", cell_type="primary CD4+ T", condition=cond,
                           timepoint=cond, donor=donor, batch=o.lane_id.astype(str).to_numpy(),
                           perturbation=np.where(single & ~nt, o.perturbed_gene_name.astype(str), ""),
                           target_ensembl=np.where(single & ~nt, o.perturbed_gene_id.astype(str), None), is_control=is_ctrl,
                           control_kind=np.where(is_ctrl, "non_targeting", None), guide=o.guide_id.astype(str).to_numpy(),
                           n_guides=ng, total_counts=o.total_counts.to_numpy(), n_genes=o.n_genes_by_counts.to_numpy(),
                           pct_mito=o.pct_counts_mt.to_numpy()))
        parts[-1]["low_quality"] = o.low_quality.astype(bool).to_numpy()
    obs = pd.concat(parts, ignore_index=True)
    lane_files = sorted(d.glob("GSM*_filtered_feature_bc_matrix.h5"))
    ids, names, ft, _ = tenx_features(lane_files[0])
    gex = ft == "Gene Expression"
    meta = dict(deposit="GEO GSE314342 per-lane *_sample_filtered_feature_bc_matrix.h5 + s3://genome-scale-tcell-perturb-seq/marson2025_data/*.assigned_guide.h5ad obs",
                files=[str(d)], batch_field="obs lane_id (10x Flex lane; each lane multiplexes all 4 donors, one file per lane x donor)",
                batch_kind="technical (lane) within donor; donors are separate contexts", modality="CRISPRi",
                cell_calling="authors (Cell Ranger multi Flex, per-sample filtered)", guide_assignment="authors' guide_group / guide_id / low_quality",
                matrix="10x Flex h5 integer", gene_ids="Flex probe-set genes (Ensembl ids)",
                flags=["~22M cells; obs from authors' S3 h5ad, counts from GEO lane matrices",
                       "two lane matrices truncated on GEO (VCC2026 used S3 copies)", "authors flag ineffective guides (guide_kd_efficiency table)"])
    return Source("cd4t_marson25", obs, meta, names[gex], ids[gex], tenx_values(lane_files[0], cfg["integer_sample_values"]))


JIANG_PATHWAYS = {"IFNB": "IFNB", "IFNG": "IFNG", "INS": "INS", "TGFB": "TGFB1", "TNFA": "TNFA"}


def jiang(cfg):
    """Jiang 2025 (Satija) five pathway Seurat objects, VCC2026 flat extracts of the raw counts."""
    d = Path(cfg["vcc_data"]) / "raw/jiang2025/extract"
    parts = []
    for pw, label in JIANG_PATHWAYS.items():
        m = pd.read_csv(d / pw / "meta.csv", dtype=str, usecols=["cell", "sample", "cell_type", "pathway", "Batch_info", "gene",
                                                                  "guide", "nCount_RNA", "nFeature_RNA", "percent.mito"])
        s = m["sample"].str.replace("BXCP3", "BXPC3", regex=False)
        line = s.str.split("_").str[0]
        is_ctrl = (m.gene == "NT").to_numpy()
        parts.append(frame(len(m), context=("jiang25__" + line + "__" + label).to_numpy(), cell_type=line.to_numpy(),
                           condition=label, timepoint="24h", donor=line.to_numpy(), batch=m.Batch_info.astype(str).to_numpy(),
                           perturbation=np.where(is_ctrl, "", m.gene.fillna("")), is_control=is_ctrl,
                           control_kind=np.where(is_ctrl, "non_targeting", None), guide=m.guide.to_numpy(), n_guides=1.0,
                           total_counts=pd.to_numeric(m.nCount_RNA).to_numpy(), n_genes=pd.to_numeric(m.nFeature_RNA).to_numpy(),
                           pct_mito=pd.to_numeric(m["percent.mito"]).to_numpy()))
        parts[-1]["sample_raw"] = m["sample"].to_numpy()
    obs = pd.concat(parts, ignore_index=True)
    genes = np.array(open(d / "IFNB/genes.txt").read().splitlines())
    vals = np.fromfile(d / "IFNB/x.bin", np.int32, count=cfg["integer_sample_values"]).astype(float)
    meta = dict(deposit="Zenodo 14518762 Seurat_object_{IFNB,IFNG,INS,TGFB,TNFA}_Perturb_seq.rds (RNA counts)",
                files=[str(d.parent), str(Path(cfg["repo_data"]) / "jiang24/jiang24_downloaded.h5ad")],
                batch_field="meta.data Batch_info (Rep1/Rep2...: experiment replicate)", batch_kind="biological (separate replicate experiments) - verify",
                modality="CRISPRi (dCas9-KRAB-MeCP2)", cell_calling="authors (Parse split-pool)",
                guide_assignment="authors' guide / gene (single guide per cell retained in object)",
                matrix="Seurat counts (int32 extract, verified integral by VCC2026)", gene_ids="symbols only (Parse reference)",
                flags=["sample typo 'BXCP3_<pathway>' (IFNG, TNFA) holds BxPC-3 cells; VCC2026 dropped them",
                       "'unknown_unknown' sample cells in IFNB", "no unstimulated arm: every cell stimulated",
                       "repo jiang24 is the perturbench re-export (fewer cells, 8,290 genes)"])
    return Source("jiang25", obs, meta, genes, None, vals)


def mcfaline(cfg):
    """McFaline-Figueroa 2024 sci-Plex-GxE (GSM7056149), VCC2026 flat extracts, all treatments."""
    d = Path(cfg["vcc_data"]) / "raw/mcfaline2023/extract"
    parts = []
    for line in ("A172", "T98G", "U87MG"):
        m = pd.read_csv(d / line / "meta.csv", dtype=str, usecols=["cell", "treatment", "dose", "replicate", "gene_id",
                                                                    "guide_number", "gRNA_id", "n.umi", "PCR_plate", "hash_plate"])
        g = m.gene_id.fillna("").to_numpy().astype(object)
        one = (m.guide_number == "1").to_numpy()
        is_ctrl = one & np.isin(g, ["NTC", "random"])
        cond = (m.treatment.fillna("NA") + "_" + m.dose.fillna("NA")).to_numpy()
        parts.append(frame(len(m), context=np.array([f"mcfaline23__{line}__{c}" for c in cond], object), cell_type=line,
                           condition=cond, donor=line, batch=m.replicate.astype(str).to_numpy(),
                           perturbation=np.where(one & ~is_ctrl, g, ""), is_control=is_ctrl,
                           control_kind=np.where(is_ctrl, np.where(g == "NTC", "non_targeting", "random_sequence"), None),
                           guide=m.gRNA_id.to_numpy(), n_guides=pd.to_numeric(m.guide_number, errors="coerce").to_numpy(),
                           total_counts=pd.to_numeric(m["n.umi"], errors="coerce").to_numpy()))
        parts[-1]["pcr_plate"] = m.PCR_plate.to_numpy()
    obs = pd.concat(parts, ignore_index=True)
    vals = np.fromfile(d / "A172/x.bin", np.int32, count=cfg["integer_sample_values"]).astype(float)
    meta = dict(deposit="GEO GSM7056149 (GSE225775) sciPlexGxE_2 preprocessed cds RDS + UMI.count.matrix.gz",
                files=[str(d.parent), str(Path(cfg["repo_data"]) / "mcfaline23/mcfaline23_gxe_processed.h5ad")],
                batch_field="colData replicate (replicate_1/2...); PCR_plate also present",
                batch_kind="biological (replicate wells/plates) - verify", modality="CRISPRi (dCas9-KRAB), kinome library",
                cell_calling="authors (sci-RNA-seq3)", guide_assignment="authors' gene_id / gRNA_id / guide_number",
                matrix="cds counts (int32 extract)", gene_ids="Ensembl ids + symbols",
                flags=["drug x dose arms make many small contexts", "'random' guides kept as a second control kind"])
    return Source("mcfaline23", obs, meta, np.array(open(d / "A172/genes.txt").read().splitlines()),
                  np.array(open(d / "A172/ensembl.txt").read().splitlines()), vals)


def viperturb(cfg):
    """Bradu 2026 VIPerturb-seq K562 genome-wide (Zenodo 18460279); meta dumped from the RDS by rds_meta.R."""
    out = Path(cfg["out_dir"]) / "_viperturb_meta"
    parts = []
    for b in ("A", "B", "C"):
        f = out / f"bin{b}.meta.csv"
        if not f.exists():
            subprocess.run([cfg["vcc_rscript"], str(Path(__file__).with_name("rds_meta.R")),
                            str(Path(cfg["vcc_data"]) / f"raw/viperturb/genome_wide_bin{b}.RDS"), str(out), f"bin{b}"], check=True)
        m = pd.read_csv(f, low_memory=False)
        m["bin"] = b
        parts.append(m)
    m = pd.concat(parts, ignore_index=True)
    m = m[~((m.gene == "NO-TARGET") & (m.bin != "A"))].reset_index(drop=True)
    is_ctrl = (m.gene == "NO-TARGET").to_numpy()
    tc = next((c for c in ("nCount_RNA", "nCount_Flex") if c in m), None)
    obs = frame(len(m), context="viperturb__K562", cell_type="K562", condition="unstimulated", donor="K562",
                batch=m["sample"].astype(str).to_numpy(), perturbation=np.where(is_ctrl, "", m.gene.fillna("").astype(str)),
                is_control=is_ctrl, control_kind=np.where(is_ctrl, "non_targeting", None),
                guide=m["guide"].to_numpy() if "guide" in m else None,
                total_counts=m[tc].to_numpy() if tc else None)
    info = json.loads((out / "binA.info.json").read_text())
    meta = dict(deposit="Zenodo 18460279 genome_wide_bin{A,B,C}.RDS (Seurat)", files=[str(Path(cfg["vcc_data"]) / "raw/viperturb")],
                batch_field="meta.data sample (10x lane)", batch_kind="technical", modality="CRISPRi (10x Flex, fixed cells)",
                cell_calling="authors", guide_assignment="authors' meta.data gene (assigned single-target cells only)",
                matrix=f"Seurat counts: {info}", gene_ids="symbols only",
                flags=["the three bins share the same NO-TARGET cells (deduplicated)", f"meta columns: {list(m.columns)[:40]}"])
    vals = np.array(info.get("sample_values", []), float)
    return Source("viperturb", obs, meta, None, None, vals)


def xatlas_repo(cfg):
    """Repo xatlas_orion_processed.h5ad (log-normalized; for version comparison only)."""
    p = Path(cfg["repo_data"]) / "xatlas_orion/xatlas_orion_processed.h5ad"
    with h5py.File(p, "r") as h:
        cond = col(h, "obs/condition").astype(str)
        ct = col(h, "obs/cell_type").astype(str)
        is_ctrl = np.isin(cond, ["ctrl", "control", "Non-Targeting", "non-targeting"])
        obs = frame(len(cond), context=np.array(["orion__" + c for c in ct], object), cell_type=ct, condition="unstimulated",
                    batch=col(h, "obs/sample").astype(str), perturbation=np.where(is_ctrl, "", cond), is_control=is_ctrl)
        vals = matrix_values(h, "X", 200000)
    meta = dict(deposit="repo processed", files=[str(p)], batch_field="obs/sample", batch_kind="technical",
                modality="CRISPRi", cell_calling="", guide_assignment="", matrix="X log-normalized (not raw)",
                gene_ids="symbols", flags=["normalized only: not admissible"])
    return Source("xatlas_orion_repo", obs, meta, None, None, vals)


SOURCES = {f.__name__: f for f in [scperturb_adamson, scperturb_frangieh, scperturb_sunshine, scperturb_norman, scperturb_wessels,
                                   replogle22_k562, replogle22_rpe1, replogle22_gwps, nadig25_hepg2, nadig25_jurkat, replogle20,
                                   orion_hct116, orion_hek293t, kolf, vcc2025_h1, feng, dld1, hesc, tian21_neurons_crispri, tian21_neurons_crispra,
                                   cd4t, jiang, mcfaline, viperturb]}
