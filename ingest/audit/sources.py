"""One read-only loader per deposit: standardized per-cell obs plus deposit metadata."""
import glob
import gzip
import io
import itertools
import json
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from contextlib import contextmanager
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


def tar_table(tar, member, **kw):
    """A gzipped csv/tsv member of a tar archive as a DataFrame."""
    with tarfile.open(tar) as t:
        return pd.read_csv(t.extractfile(member), compression="gzip", **kw)


def mtx_values(tar, member, n):
    """The first n stored values of a gzipped matrix.mtx member."""
    with tarfile.open(tar) as t, gzip.open(t.extractfile(member), "rt") as fh:
        body = (line for line in fh if not line.startswith("%"))
        return np.array([line.split()[2] for line in itertools.islice(body, 1, n + 1)], float)


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


ADAMSON_RUNS = {"GSM2406675_10X001": (1, "day 8", {"1": ("adamson16__K562__pilot", "unstimulated")}),
                "GSM2406677_10X005": (3, "day 5", {"1": ("adamson16__K562__epistasis__tunicamycin_6h", "tunicamycin_6h"),
                                                   "2": ("adamson16__K562__epistasis__thapsigargin_4h", "thapsigargin_4h"),
                                                   "3": ("adamson16__K562__epistasis__DMSO_6h", "DMSO_6h")})}


def adamson16_other_lanes(cfg):
    """GEO GSE90546_RAW.tar: the pilot (10X001, one-guide vectors) and UPR epistasis (10X005, three-guide vectors, one GEM group per drug) experiments."""
    tar = Path(cfg["root"]) / "raw/adamson16_other_lanes/GSE90546_RAW.tar"
    parts = []
    for run, (guides_per_vector, day, contexts) in ADAMSON_RUNS.items():
        bc = tar_table(tar, f"{run}_barcodes.tsv.gz", sep="\t", header=None)[0]
        ci = tar_table(tar, f"{run}_cell_identities.csv.gz").set_index("cell BC").reindex(bc)
        ident = ci["guide identity"].fillna("*")
        one = ((ci["number of cells"] == 1) & ci["good coverage"].eq(True) & (ident != "*")).to_numpy()
        is_ctrl = one & (ident.str.contains("(", regex=False) | ident.str.startswith("3x_neg_ctrl")).to_numpy()
        targets = [join_targets(t for t in g.rsplit("_", 1)[0].split("_") if t != "only") for g in ident]
        gem = bc.str.split("-").str[1]
        parts.append(frame(len(bc), context=gem.map(lambda g: contexts[g][0]).to_numpy(), cell_type="K562", condition=gem.map(lambda g: contexts[g][1]).to_numpy(),
                           timepoint=day, donor="K562", batch=gem.to_numpy(), perturbation=np.where(one & ~is_ctrl, targets, ""), is_control=is_ctrl,
                           control_kind=np.where(is_ctrl, "non_targeting", None), guide=ci["guide identity"].to_numpy(),
                           n_guides=ci["number of cells"].fillna(0).to_numpy() * guides_per_vector))
        parts[-1]["guide_umi"], parts[-1]["barcode"], parts[-1]["gsm"] = ci["UMI count"].to_numpy(), bc.to_numpy(), run
    obs = pd.concat(parts, ignore_index=True)
    g = tar_table(tar, "GSM2406677_10X005_genes.tsv.gz", sep="\t", header=None, names=["id", "name"])
    meta = dict(deposit="GEO GSE90546_RAW.tar: GSM2406675 (10X001) and GSM2406677 (10X005) matrix.mtx.txt + barcodes + genes + cell_identities.csv",
                files=[str(tar)], batch_field="barcode suffix = GEM group (10X001 one; 10X005 three, one per drug treatment, so one batch per context)",
                batch_kind="technical", modality="CRISPRi (dCas9-KRAB; 10X005 three-guide vectors)", cell_calling="authors (Cell Ranger 1.0/1.1 barcodes)",
                guide_assignment="authors' cell_identities: good coverage and number of cells == 1 assigned, >= 2 multiple, 0 or '*' unidentifiable",
                matrix="matrix.mtx.txt.gz (10X001 integer, 10X005 real-typed, integer-valued)",
                gene_ids="genes.tsv Ensembl ids + symbols (10X001 35,635 features, Cell Ranger 1.0; 10X005 32,738 as 10X010)",
                flags=["GSM2406681 (10X010) in the same tar is the reference's scperturb_adamson context",
                       "22 10X005 identity rows name guides of other vectors (PSMD12_pDS009 etc.), none with good coverage, so all are unidentifiable",
                       "44 10X005 identity rows have no matrix barcode", "the 10X001 matrix has no counts on its 13 MT- features (pct_mito 0 for every cell)"])
    return Source("adamson16_other_lanes", obs, meta, g.name.to_numpy(), g.id.to_numpy(), mtx_values(tar, "GSM2406677_10X005_matrix.mtx.txt.gz", cfg["integer_sample_values"]))


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


WESSELS_TARS = dict(carpool="GSE213957_HEX293FT_and_SpeciesMixing_CaRPool-seq.tar.gz", thp1="GSE213957_THP1-CaRPool-seq_and_GSE213957_HEX293FTstabRNA.tar.gz",
                    perturbseq="GSE213957_Perturb-seq.tar.gz")


def wessels23_hek293ft(cfg):
    """GEO GSE213957 HEK293FT CaRPool-seq (29 crRNA arrays, one lane shared with species mixing) and stabRNA (stabilized crRNAs, HTO-54 in the four THP-1 lanes)."""
    d = Path(cfg["root"]) / "raw/wessels23_hek293ft"
    m = pd.read_csv(d / "GSE213957_HEX293FT_CaRPool-seq.metadata.tsv.gz", sep="\t", index_col=0)
    s = pd.read_csv(d / "GSE213957_HEX293FTstabRNA.metadata.tsv.gz", sep="\t", index_col=0).sort_values("Lane", kind="stable")
    nt = (m.TargetGenes == "NT").to_numpy()
    arrays = frame(len(m), context="wessels23__HEK293FT__CaRPool-seq", cell_type="HEK293FT", condition="unstimulated", donor="HEK293FT", batch="1",
                   perturbation=np.where(nt, "", [join_targets(x.split("_")) for x in m.TargetGenes]), is_control=nt,
                   control_kind=np.where(nt, "non_targeting", None), guide=m.Guides.to_numpy(), n_guides=(m.Array.str.count("_") + 1).to_numpy(),
                   total_counts=m.nCount_RNA.to_numpy(), n_genes=m.nFeature_RNA.to_numpy(), pct_mito=m["percent.mt"].to_numpy())
    arrays["array"], arrays["hto"], arrays["barcode"] = m.Array.to_numpy(), m["hash.ID"].to_numpy(), m.index.to_numpy()
    nt = (s.TargetGene == "NT").to_numpy()
    stab = frame(len(s), context="wessels23__HEK293FT__stabRNA", cell_type="HEK293FT", condition="unstimulated", donor="HEK293FT", batch=s.Lane.to_numpy(),
                 perturbation=np.where(nt, "", s.TargetGene), is_control=nt, control_kind=np.where(nt, "non_targeting", None), guide=s.bcgRNA.to_numpy(),
                 n_guides=1.0, total_counts=s.nCount_RNA.to_numpy(), n_genes=s.nFeature_RNA.to_numpy(), pct_mito=s["percent.mt"].to_numpy())
    stab["array"], stab["guide_umi"], stab["barcode"] = s.Array.to_numpy(), s["UMI.g1"].to_numpy(), s.index.str.split("_").str[1].to_numpy()
    obs = pd.concat([arrays, stab], ignore_index=True)
    f = tar_table(d / WESSELS_TARS["carpool"], "HEK293FT_and_SpeciesMixing_CaRPool-seq.GEXGDO.features.tsv.gz", sep="\t", header=None, names=["id", "name", "type"])
    hg = f[f.id.str.startswith("hg38_")]
    meta = dict(deposit="GEO GSE213957 HEK293FT_and_SpeciesMixing_CaRPool-seq GEXGDO mtx + HEX293FT_CaRPool-seq.metadata; "
                        "THP1-CaRPool-seq_and_HEK293FTstabRNA GEXGDO1-4 mtx + HEX293FTstabRNA.metadata",
                files=[str(d / WESSELS_TARS["carpool"]), str(d / "GSE213957_HEX293FT_CaRPool-seq.metadata.tsv.gz"), str(d / WESSELS_TARS["thp1"]),
                       str(d / "GSE213957_HEX293FTstabRNA.metadata.tsv.gz")],
                batch_field="CaRPool-seq: one 10x lane (barcode suffix 1; hash.ID HTO-01..08 kept as obs hto); stabRNA: metadata Lane L1-L4 (the four THP-1 lanes)",
                batch_kind="technical", modality="Cas13 (CaRPool-seq, RfxCas13d)",
                cell_calling="authors (Cell Ranger, < 2,500 UMI removed, HTODemux; the metadata lists assigned cells only)",
                guide_assignment="authors: CaRPool-seq MultiSeqDemux single bcgRNA (Guides = array id, Array = gRNA composition); stabRNA g1/g2 bcgRNA UMI rule",
                matrix="matrix.mtx.gz integer, features x barcodes", gene_ids="Cell Ranger hg38 (Ensembl 97) ids + symbols; the CaRPool-seq lane also carries mm10 features",
                flags=["species-mixing cells (mouse, BBO hashtags) and unassigned barcodes of the CaRPool-seq lane are absent from the metadata and excluded",
                       "stabRNA lane matrices carry 8 transgene features (eGFP, Blast, Cas9, Puro, Cas13d, AsCas12a, MeCP2, KRAB) absent from the CaRPool-seq lane",
                       "the meaning of hash.ID HTO-01..08 within the CaRPool-seq lane is not stated in the paper; every array is in every HTO"])
    return Source("wessels23_hek293ft", obs, meta, hg.name.str[5:].to_numpy(), hg.id.str[5:].to_numpy(),
                  mtx_values(d / WESSELS_TARS["carpool"], "HEK293FT_and_SpeciesMixing_CaRPool-seq.GEXGDO.matrix.mtx.gz", cfg["integer_sample_values"]))


def wessels23_perturbseq(cfg):
    """GEO GSE213957 Perturb-seq arm: HEK293FT Cas9-nuclease, KRAB-dCas9 and KRAB-dCas9-MeCP2 lines with 1-3 sgRNA vectors, one hashed lane."""
    tar = Path(cfg["root"]) / "raw/wessels23_hek293ft" / WESSELS_TARS["perturbseq"]
    p = tar_table(tar, "Perturb-seq.metadata.tsv.gz", sep="\t", index_col=0)
    p = p[p.Experiment == "Perturb-seq"]
    ref = tar_table(tar, "Perturb-seq.GDO.feature_ref.csv.gz")
    target_id = {i.split("-")[0]: g for i, g in zip(ref.id, ref.target_gene_id) if g.startswith("ENSG")}
    nt = (p.TargetGenes == "NT").to_numpy()
    lab = np.where(nt, "", [join_targets(x.split("/")) for x in p.TargetGenes])
    obs = frame(len(p), context=("wessels23__HEK293FT__" + p.Type).to_numpy(), cell_type="HEK293FT", condition="unstimulated", timepoint="day 12",
                donor="HEK293FT", batch=p.Pool.to_numpy(), perturbation=lab, target_ensembl=[target_id.get(x) for x in lab], is_control=nt,
                control_kind=np.where(nt, "non_targeting", None), guide=p["TargetGuideCombination.detailed"].to_numpy(),
                n_guides=p.Pool.str[-1].astype(int).to_numpy())
    obs["barcode"] = p.index.str.split("_").str[0].to_numpy()
    f = tar_table(tar, "Perturb-seq.GEXGDO.features.tsv.gz", sep="\t", header=None, names=["id", "name", "type"])
    gex = f[f.type == "Gene Expression"]
    meta = dict(deposit="GEO GSE213957 Perturb-seq GEXGDO mtx + Perturb-seq.metadata (in the tar) + GDO.feature_ref", files=[str(tar)],
                batch_field="metadata Pool = hashtag of the cell line x sgRNA-vector pool (n=1, n=2, n=3 vectors), all in one 10x lane",
                batch_kind="technical (separately transduced pools within one lane)",
                modality="Cas9 Perturb-seq: CRISPRko (Cas9 nuclease), CRISPRi (KRAB-dCas9), CRISPRi (KRAB-dCas9-MeCP2), one per context",
                cell_calling="authors (Cell Ranger, HTODemux doublets removed)",
                guide_assignment="authors (HTODemux on sgRNA features); only cells with the sgRNA number and combination expected for their pool are in the metadata",
                matrix="matrix.mtx.gz integer, features x barcodes", gene_ids="Cell Ranger hg38 (Ensembl 97) ids + symbols, 8 transgene features",
                flags=["the metadata carries no per-cell counts (total_counts, n_genes, pct_mito NA)",
                       "target Ensembl ids from GDO.feature_ref (CD71 = TFRC ENSG00000072274)"])
    return Source("wessels23_perturbseq", obs, meta, gex.name.to_numpy(), gex.id.to_numpy(), mtx_values(tar, "Perturb-seq.GEXGDO.matrix.mtx.gz", cfg["integer_sample_values"]))


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
                           condition=cond, donor=line, batch=(m.replicate.astype(str) + "_" + m.PCR_plate.astype(str)).to_numpy(),
                           perturbation=np.where(one & ~is_ctrl, g, ""), is_control=is_ctrl,
                           control_kind=np.where(is_ctrl, np.where(g == "NTC", "non_targeting", "random_sequence"), None),
                           guide=m.gRNA_id.to_numpy(), n_guides=pd.to_numeric(m.guide_number, errors="coerce").to_numpy(),
                           total_counts=pd.to_numeric(m["n.umi"], errors="coerce").to_numpy()))
        parts[-1]["replicate"] = m.replicate.to_numpy()
    obs = pd.concat(parts, ignore_index=True)
    vals = np.fromfile(d / "A172/x.bin", np.int32, count=cfg["integer_sample_values"]).astype(float)
    meta = dict(deposit="GEO GSM7056149 (GSE225775) sciPlexGxE_2 preprocessed cds RDS + UMI.count.matrix.gz",
                files=[str(d.parent), str(Path(cfg["repo_data"]) / "mcfaline23/mcfaline23_gxe_processed.h5ad")],
                batch_field="colData replicate and PCR_plate joined (replicate_1_A ...); replicate kept as an obs column",
                batch_kind="technical (RT/PCR plate) nested in the transduction replicate", modality="CRISPRi (dCas9-KRAB), kinome library",
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


def mtx_values(fh, n):
    """The first n stored values of a MatrixMarket byte stream."""
    line = fh.readline()
    while line.startswith(b"%"):
        line = fh.readline()
    return np.array([float(fh.readline().split()[2]) for _ in range(n)])


SHIFRUT_SAMPLES = {"D1N": ("D1", "NoStim"), "D1S": ("D1", "Stim"), "D2N": ("D2", "NoStim"), "D2S": ("D2", "Stim")}


def shifrut_sample(tar, sample):
    """The nested Cell Ranger archive (barcodes.tsv, genes.tsv, matrix.mtx) of one Shifrut sample, opened for random access."""
    outer = tarfile.open(tar)
    m = next(x for x in outer.getmembers() if x.name.endswith(f"_{sample}_matrix.tar.gz"))
    return tarfile.open(fileobj=outer.extractfile(m), mode="r:gz")


def shifrut_guide_calls(tar, sample):
    """The authors' CellBC_sgRNA table of one Shifrut sample, indexed by cell barcode."""
    with tarfile.open(tar) as outer:
        m = next(x for x in outer.getmembers() if x.name.endswith(f"_{sample}_CellBC_sgRNA.csv.gz"))
        return pd.read_csv(gzip.open(outer.extractfile(m), "rt")).set_index("Cell.BC")


def shifrut18_t_crisprko(cfg):
    """Shifrut 2018 SLICE CROP-seq in primary CD8+ T cells (GEO GSE119450_RAW.tar): four Cell Ranger samples + the authors' guide calls."""
    tar = Path(cfg["root"]) / "raw/shifrut18_t_crisprko/GSE119450_RAW.tar"
    parts, genes, vals = [], None, None
    for sample, (donor, cond) in SHIFRUT_SAMPLES.items():
        inner = shifrut_sample(tar, sample)
        bc = pd.read_csv(inner.extractfile("barcodes.tsv"), header=None)[0].str.split("-").str[0]
        if genes is None:
            genes = pd.read_csv(inner.extractfile("genes.tsv"), sep="\t", header=None, names=["id", "name"])
            vals = mtx_values(inner.extractfile("matrix.mtx"), cfg["integer_sample_values"])
        calls = shifrut_guide_calls(tar, sample).reindex(bc)
        target = calls["gRNA.ID"].astype(str).str.split(".").str[2].to_numpy()
        assigned = calls["gRNA.ID"].notna().to_numpy()
        is_ctrl = assigned & (target == "NonTarget")
        parts.append(frame(len(bc), context=ctx("shifrut18_t_crisprko", "CD8T", donor, cond), cell_type="primary CD8+ T", condition=cond, timepoint="36h",
                           donor=donor, batch=f"{donor}_{cond}", perturbation=np.where(assigned & ~is_ctrl, target, ""), is_control=is_ctrl,
                           control_kind=np.where(is_ctrl, "non_targeting", None), guide=calls["gRNA.ID"].to_numpy(), n_guides=assigned.astype(int)))
        parts[-1]["guide_umi"] = calls["UMI.count"].to_numpy()
    obs = pd.concat(parts, ignore_index=True)
    meta = dict(deposit="GEO GSE119450_RAW.tar: GSM3375483-86 {D1N,D1S,D2N,D2S}_matrix.tar.gz (Cell Ranger 2.0 filtered) + GSM3375487-90 CellBC_sgRNA.csv.gz",
                files=[str(tar)], batch_field="GEO sample = donor x condition (one 10x lane each; sample.name of the guide table)",
                batch_kind="technical (lane), one lane per donor x condition; the donor is part of the context",
                modality="CRISPRko (Cas9 RNP electroporation, SLICE; CROP-seq sgRNA capture)", cell_calling="authors (Cell Ranger 2.0 default filter)",
                guide_assignment="authors' CellBC_sgRNA tables (U6 reamplification reads; cells with more than one assigned guide removed by the authors): "
                                 "one guide per listed barcode, barcodes without a row are unassigned",
                matrix="matrix.mtx integer per sample", gene_ids="Cell Ranger GRCh38 (Ensembl ids + symbols, 33,694 genes)",
                flags=["guide UMI.count is 1 for about two thirds of the assigned cells",
                       "the paper also keeps only cells with more than 500 genes (qc_genes_500 flag, not applied)",
                       "about a third of the guide-table barcodes are absent from the Cell Ranger filtered matrices"])
    return Source("shifrut18_t_crisprko", obs, meta, genes.name.to_numpy(), genes.id.to_numpy(), vals)
MCCUTCHEON_LANES = {"crispri": ["GSM6761464_CRISPRi_D1", "GSM6761465_CRISPRi_D2", "GSM6761466_CRISPRi_D3"],
                    "crispra": ["GSM6761467_CRISPRa_D1", "GSM6761468_CRISPRa_D2", "GSM6761469_CRISPRa_D3"]}


def mtx_entries(fh, chunk=20000000):
    """(n_features, n_cells, chunks) of a gzipped MatrixMarket coordinate file; chunks are 0-based int32 (feature, cell, count) arrays."""
    g = gzip.GzipFile(fileobj=fh)
    line = g.readline()
    while line.startswith(b"%"):
        line = g.readline()
    nf, nc, _ = (int(v) for v in line.split())
    chunks = ((t.r.to_numpy() - 1, t.c.to_numpy() - 1, t.v.to_numpy())
              for t in pd.read_csv(g, sep=" ", header=None, names=["r", "c", "v"], dtype=np.int32, chunksize=chunk))
    return nf, nc, chunks


def mccutcheon_lanes(root, arm):
    """Yield (lane, barcodes, features, mtx entries) per donor lane of one arm, read from the nested GEO tar without extraction."""
    with tarfile.open(Path(root) / "raw/mccutcheon23_cd8t/GSE218988_RAW.tar") as outer:
        for lane in MCCUTCHEON_LANES[arm]:
            with tarfile.open(fileobj=outer.extractfile(f"{lane}.tar.gz"), mode="r|gz") as inner:
                files = {Path(m.name).name: inner.extractfile(m).read() for m in inner}
            barcodes = np.array(gzip.decompress(files["barcodes.tsv.gz"]).decode().split(), dtype=object)
            features = pd.read_csv(io.BytesIO(files["features.tsv.gz"]), sep="\t", header=None, names=["id", "name", "type"], compression="gzip")
            yield lane, barcodes, features, mtx_entries(io.BytesIO(files["matrix.mtx.gz"]))


def mccutcheon(cfg, arm):
    """McCutcheon 2023 CD8 T mini-TF Perturb-seq, one arm (GSE218988): Cell Ranger lanes per donor; gRNA calls by the paper's rule (no calls deposited)."""
    name = f"mccutcheon23_cd8t_{arm}"
    min_umi = cfg["guide_call_min_umi"]["mccutcheon23_cd8t_" + arm]
    parts, vals = [], None
    for lane, barcodes, features, (nf, nc, chunks) in mccutcheon_lanes(cfg["root"], arm):
        is_g = (features.type == "CRISPR Guide Capture").to_numpy()
        gcol = np.full(nf, -1)
        gcol[is_g] = np.arange(is_g.sum())
        umi = np.zeros((nc, int(is_g.sum())), np.int64)
        tot, ngen = np.zeros(nc, np.int64), np.zeros(nc, np.int64)
        for r, c, v in chunks:
            if vals is None:
                vals = v[: cfg["integer_sample_values"]].astype(float)
            g = gcol[r]
            sel = g >= 0
            np.add.at(umi, (c[sel], g[sel]), v[sel])
            tot += np.bincount(c[~sel], weights=v[~sel], minlength=nc).astype(np.int64)
            ngen += np.bincount(c[~sel], minlength=nc)
        gnames = features.name[is_g].to_numpy()
        tg = np.array(["non-targeting" if x.startswith("NT_") else x.rsplit("_", 1)[0] for x in gnames], object)
        called = umi >= min_umi
        nguide = called.sum(1)
        ok = nguide == 1
        best = np.where(ok, tg[umi.argmax(1)], "")
        is_ctrl = ok & (best == "non-targeting")
        donor = lane.rsplit("_", 1)[1]
        parts.append(frame(nc, context=ctx(name, donor), cell_type="primary CD8+ T", condition="CD3_CD28_activated", timepoint="day 10",
                           donor=donor, batch=lane, perturbation=np.where(ok & ~is_ctrl, best, ""), is_control=is_ctrl,
                           control_kind=np.where(is_ctrl, "non_targeting", None),
                           guide=np.array([";".join(gnames[row]) if row.any() else None for row in called], object),
                           n_guides=nguide, total_counts=tot, n_genes=ngen))
        parts[-1]["guide_umi"] = umi.max(1)
        parts[-1]["barcode"] = barcodes
        parts[-1]["assigned_by"] = f"paper rule computed here: gRNA UMI >= {min_umi}"
        if len(parts) == 1:
            genes, ens = features.name[~is_g].to_numpy(), features.id[~is_g].to_numpy()
    obs = pd.concat(parts, ignore_index=True)
    cas = {"crispri": "CRISPRi (dSaCas9-KRAB, all-in-one lentivirus, Thy1.1-sorted)", "crispra": "CRISPRa (VP64-dSaCas9-VP64, all-in-one lentivirus, Thy1.1-sorted)"}
    meta = dict(deposit=f"GEO GSE218988 (SubSeries GSE218985) GSE218988_RAW.tar: {', '.join(MCCUTCHEON_LANES[arm])}.tar.gz, Cell Ranger 6.0.1 filtered barcodes/features/matrix.mtx",
                files=[str(Path(cfg["root"]) / "raw/mccutcheon23_cd8t")],
                batch_field="GEO sample = one 10x 5' HT v2 lane per donor and arm; donors are separate contexts", batch_kind="technical (one lane per donor)",
                modality=cas[arm], cell_calling="Cell Ranger 6.0.1 filtered barcodes (authors); the paper's QC (<200 genes, >20% mito, <5% ribosomal, DoubletFinder) not applied",
                guide_assignment=f"NOT deposited as calls: computed per gRNA from the 40 CRISPR Guide Capture features as in the paper's methods (gRNA UMI > 4, i.e. >= {min_umi}); "
                                 "cells with exactly one called gRNA are assigned; controls are cells whose single called gRNA is NT",
                matrix="matrix.mtx.gz integer, 36,601 Gene Expression + 40 CRISPR Guide Capture features", gene_ids="Cell Ranger GRCh38 Ensembl ids + symbols",
                flags=[f"gRNA capture is shallow: {int((obs.n_guides == 0).sum())} of {len(obs)} cells have no gRNA at >= {min_umi} UMI, {int((obs.n_guides > 1).sum())} have several",
                       "the paper pools donors per arm and compares each gRNA's cells with cells carrying only NT gRNAs; here each donor is a context",
                       "32 targeting gRNAs over 24 genes (several gRNAs per gene) + 8 NT gRNAs"])
    return Source(name, obs, meta, genes, ens, vals)


def mccutcheon23_cd8t_crispri(cfg):
    return mccutcheon(cfg, "crispri")


def mccutcheon23_cd8t_crispra(cfg):
    return mccutcheon(cfg, "crispra")
PAPALEXI_CDNA = "GSM4633614_ECCITE_cDNA_counts.tsv.gz"


@contextmanager
def tar_text(path, member):
    """Text handle on a gzip member of a tar, read in place."""
    with tarfile.open(path) as tar, gzip.open(tar.extractfile(member), "rt") as fh:
        yield fh


def papalexi21_thp1(cfg):
    """Papalexi 2021 ECCITE-seq THP-1 (GSE153056): the pooled 111-guide experiment, stimulated cells only, from the authors' metadata."""
    d = Path(cfg["root"]) / "raw/papalexi21_thp1"
    m = pd.read_csv(d / "GSE153056_ECCITE_metadata.tsv.gz", sep="\t", index_col=0)
    is_ctrl = (m.gene == "NT").to_numpy()
    obs = frame(len(m), context=ctx("papalexi21_thp1", "THP1", "IFNG_DAC_TGFB1"), cell_type="THP-1", condition="IFNG_DAC_TGFB1", timepoint="72h",
                donor="THP-1", batch=m["orig.ident"].to_numpy(), perturbation=np.where(is_ctrl, "", m.gene), is_control=is_ctrl,
                control_kind=np.where(is_ctrl, "non_targeting", None), guide=m.guide_ID.to_numpy(), n_guides=1.0,
                total_counts=m.nCount_RNA.to_numpy(), n_genes=m.nFeature_RNA.to_numpy(), pct_mito=m["percent.mito"].to_numpy())
    obs["barcode"] = m.index.to_numpy()
    obs["transduction_replicate"] = m.replicate.to_numpy()
    genes, vals = [], []
    with tar_text(d / "GSE153056_RAW.tar", PAPALEXI_CDNA) as fh:
        fh.readline()
        for line in fh:
            name, rest = line.split("\t", 1)
            genes.append(name.strip('"'))
            if len(vals) < cfg["integer_sample_values"]:
                v = np.fromstring(rest, dtype=np.float64, sep="\t")
                vals.extend(v[v != 0])
    meta = dict(deposit="GEO GSE153056 GSE153056_ECCITE_metadata.tsv.gz + GSM4633614_ECCITE_cDNA_counts.tsv.gz (in GSE153056_RAW.tar)",
                files=[str(d / "GSE153056_RAW.tar"), str(d / "GSE153056_ECCITE_metadata.tsv.gz")],
                batch_field="metadata orig.ident (10x 5' lane, 8); replicate = transduction replicate hashed into the lanes (rep1 in lanes 1-4, rep2 and rep3 in lanes 5-8)",
                batch_kind="technical (lane) nested in transduction replicate", modality="CRISPRko (doxycycline-inducible Cas9, lentiviral CROP-seq sgRNA)",
                cell_calling="authors (Cell Ranger 2.1.1 hg19, > 10% mito and < 100 genes removed, HTO doublets and negatives removed, replicate 4 removed)",
                guide_assignment="authors' guide_ID: gRNA with the most GDO counts; < 5 counts for every gRNA = negative, several high gRNAs = doublet; only singlets deposited",
                matrix="dense genes x cells tsv.gz, integer; column sums equal nCount_RNA", gene_ids="symbols only (hg19 Cell Ranger reference)",
                flags=["only the 90% stimulated (DAC + TGFB1 + IFNG) cells are deposited; the 10% unstimulated cells are not",
                       "MULTI_ID / HTO_classification name the replicates rep1/rep3/rep4 and rep1/rep3/rep2; the replicate column (rep1-3) is used",
                       "PDL1g1-3 and eGFPg1 are in the GDO feature list but assigned to no cell",
                       "the arrayed ECCITE experiment (GSM4633608-13) and the CITE-seq experiment are not ingested"])
    return Source("papalexi21_thp1", obs, meta, np.array(genes, dtype=object), None, np.array(vals[: cfg["integer_sample_values"]]))


HSPC_CONTROLS = {"NT": "non_targeting", "AAVS1": "safe_targeting"}


def martinrufino25_hspc(cfg):
    """Martin-Rufino 2025 Perturb-multiome HSPC (GSE274113): authors' annotated metadata, counts from the per-replicate CROPseq h5."""
    d = Path(cfg["root"]) / "raw/hspc25_multiome"
    m = pd.read_csv(d / "GSE274113_annotated_metadata.csv.gz", dtype=str, keep_default_na=False)
    m = m[m.iloc[:, 0] != ""].reset_index(drop=True)
    is_ctrl = (m.target == "NT").to_numpy()
    kind = m.perturbation_name.str.rsplit("_", n=1).str[0].map(HSPC_CONTROLS)
    obs = frame(len(m), context=("martinrufino25_hspc__HSPC__" + m.Timepoint.str.replace(" ", "", regex=False)).to_numpy(),
                cell_type="primary HSPC (erythroid differentiation)", condition="erythroid differentiation", timepoint=m.Timepoint.to_numpy(),
                donor="CD34+ HSPC donor", batch=m.replicate.to_numpy(), perturbation=np.where(is_ctrl, "", m.target), is_control=is_ctrl,
                control_kind=np.where(is_ctrl, kind, None), guide=m.perturbation_name.to_numpy(), n_guides=1.0,
                total_counts=pd.to_numeric(m.nCount_RNA).to_numpy(), n_genes=pd.to_numeric(m.nFeature_RNA).to_numpy())
    obs["barcode"] = m.iloc[:, 0].str.split("_").str[1].to_numpy()
    obs["author_cell_type"] = m.new_CellType.to_numpy()
    first = sorted(d.glob("GSE274113_filtered_feature_bc_matrix_*.h5"))[0]
    ids, names, ft, _ = tenx_features(first)
    gex = ft == "Gene Expression"
    meta = dict(deposit="GEO GSE274113 GSE274113_annotated_metadata.csv.gz + GSE274113_filtered_feature_bc_matrix_<k>.h5 (Cellranger v7 GEX + CRISPR Guide Capture, one per replicate)",
                files=[str(d)], batch_field="metadata replicate (rep<k>) = one 10x Multiome channel; 4 technical replicates per time point on one chip",
                batch_kind="technical", modality="CRISPRko (Cas9 RNP electroporation, lentiviral CROP-seq sgRNA)",
                cell_calling="authors (Cellranger ARC barcodes with a detected sgRNA: 137,604 cells)",
                guide_assignment="authors' perturbation_name: one sgRNA per cell, equal to the CRISPR Guide Capture feature with the most UMIs in every replicate; cells without a detected sgRNA are not deposited",
                matrix="10x h5 CSC integer; GEX counts identical to the rep<k> Multiome (ATAC + GEX) h5 for every annotated cell",
                gene_ids="Cellranger GRCh38 features (Ensembl ids + symbols)",
                flags=["controls: NT_1-3 non-targeting and AAVS1_1-3 safe-targeting, both labelled target NT by the authors",
                       "rep11 and rep15 are not on GEO; the Multiome h5 of rep9 is on GEO but not on disk, its CROPseq h5 is",
                       "the metadata csv has an empty row after every cell row",
                       "cells of one context span many differentiation states (author_cell_type); the authors compare each cell with nearest-neighbour controls",
                       "Peaks and CRISPR Guide Capture features skipped"])
    return Source("martinrufino25_hspc", obs, meta, names[gex], ids[gex], tenx_values(first, cfg["integer_sample_values"]))
JOUNG_ATLAS = "GSE217460_210322_TFAtlas_raw.h5ad"


def joung_atlas(cfg):
    """Decompressed TF Atlas h5ad under raw/, written from the deposit's .gz when absent, with a receipt derived from the .gz."""
    gz = Path(cfg["root"]) / "raw" / "joung23_tf_atlas" / (JOUNG_ATLAS + ".gz")
    out = gz.with_suffix("")
    if not out.exists():
        part = out.with_name(f".{out.name}.{os.getpid()}.part")
        with gzip.open(gz, "rb") as src, open(part, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 24)
        os.replace(part, out)
    from receipts import raw_receipt
    raw_receipt(dict(cfg, root=cfg["root"]), "joung23_tf_atlas", out, derived_from=str(gz))
    return out


def joung23_tf_atlas(cfg):
    """Joung 2023 TF Atlas (GEO GSE217460): H1 hESC + barcoded TF ORF library, 7 days in STEMdiff APEL, SHARE-seq RNA."""
    with h5py.File(joung_atlas(cfg), "r") as h:
        orf = col(h, "obs/TF").astype(str)
        n = len(orf)
        sym = np.array([x.split("-", 1)[1] for x in orf], object)
        is_ctrl = np.isin(sym, ["GFP", "mCherry"])
        experiment = pd.Series(col(h, "obs/batch").astype(str))
        sublibrary = pd.Series(index(h, "obs")).str.extract(r"(P1\.\d+)")[0]
        obs = frame(n, context="joung23_tf_atlas__H1", cell_type="H1 hESC", condition="STEMdiff_APEL", timepoint="day 7", donor="H1",
                    batch=(experiment + "_" + sublibrary).to_numpy(), perturbation=np.where(is_ctrl, "", sym),
                    is_control=is_ctrl, control_kind=np.where(is_ctrl, "non_targeting", None), guide=orf, n_guides=1.0,
                    total_counts=col(h, "obs/n_counts"), n_genes=col(h, "obs/n_genes"), pct_mito=100 * col(h, "obs/percent_mito"))
        obs["experiment"] = experiment.to_numpy()
        genes = index(h, "var")
        vals = matrix_values(h, "X", cfg["integer_sample_values"])
        shape = h["X"].shape
    meta = dict(deposit="GEO GSE217460 GSE217460_210322_TFAtlas_raw.h5ad.gz; the _differentiated_raw.h5ad.gz file is the clusters 6-8 subset of the same cells and is not loaded",
                files=[str(Path(cfg["root"]) / "raw" / "joung23_tf_atlas" / (JOUNG_ATLAS + ".gz"))],
                batch_field="obs/batch (experiment 0/1) + P1 index of the cell name (SHARE-seq PCR sub-library, one per anndata concatenation key)",
                batch_kind="technical (sub-library) within experiment; GFP/mCherry controls only in experiment 1",
                modality="ORF overexpression (MORF library of 3,548 TF isoforms, lentiviral, MOI <0.3)",
                cell_calling="authors (SHARE-seq, >700 and <7,000 genes, <5% mitochondrial reads)",
                guide_assignment="authors' TF barcode call: top barcode with >10 reads and >50% more reads than the second; cells without a mapped ORF excluded, so one ORF per cell",
                matrix=f"X dense float32 {shape}, integer-valued",
                gene_ids="symbols only (var index); mitochondrial genes under old symbols (ATP6, ATP8, COX3, CYTB, ND2-ND6; COX1, COX2, ND1 ambiguous)",
                flags=["GFP and mCherry control ORFs recorded as non_targeting; guide keeps the TFORF id",
                       "several isoforms of one TF are several guides of one perturbation label",
                       "percent_mito is a fraction in the deposit (x100 here); the shard's pct_mito misses COX1, COX2 and ND1 and is a lower bound"])
    return Source("joung23_tf_atlas", obs, meta, genes, None, vals)


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











SOURCES = {f.__name__: f for f in [scperturb_adamson, scperturb_frangieh, scperturb_sunshine, scperturb_norman,
                                   scperturb_wessels, replogle22_k562, replogle22_rpe1, replogle22_gwps, nadig25_hepg2,
                                   nadig25_jurkat, replogle20, orion_hct116, orion_hek293t, kolf, vcc2025_h1, feng, dld1, hesc,
                                   cd4t, jiang, mcfaline, viperturb, shifrut18_t_crisprko, mccutcheon23_cd8t_crispri,
                                   mccutcheon23_cd8t_crispra, tian21_neurons_crispri, tian21_neurons_crispra, papalexi21_thp1,
                                   martinrufino25_hspc, adamson16_other_lanes, wessels23_hek293ft, wessels23_perturbseq,
                                   joung23_tf_atlas]}
