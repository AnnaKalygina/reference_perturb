"""Global gene axis from the Ensembl 112 GTF and per-dataset mapping of native features onto it."""
import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "audit"))
from config import CONFIG, load, path  # noqa: E402

PRIMARY = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}
ATTR = re.compile(r'(\w+) "([^"]*)"')
TABLE = CONFIG / "genes_ensembl112.tsv.gz"


def build_table(cfg):
    """Write config/genes_ensembl112.tsv.gz from the GTF: one row per gene on the primary assembly, with TSS."""
    rows = []
    with gzip.open(path(cfg, cfg["genes"]["gtf"]), "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] != "gene" or (cfg["genes"]["primary_assembly_only"] and f[0] not in PRIMARY):
                continue
            a = dict(ATTR.findall(f[8]))
            start, end, strand = int(f[3]), int(f[4]), f[6]
            rows.append((a["gene_id"], a.get("gene_name", ""), a.get("gene_biotype", ""), f[0], start, end, strand,
                         start if strand == "+" else end, a.get("gene_version", "")))
    t = pd.DataFrame(rows, columns=["ensembl_id", "symbol", "biotype", "chromosome", "start", "end", "strand", "tss", "version"])
    if t.ensembl_id.duplicated().any():
        raise SystemExit("duplicate gene ids in the GTF")
    t.to_csv(TABLE, sep="\t", index=False, compression="gzip")
    return t


def axis():
    """The global gene table in axis order."""
    return pd.read_csv(TABLE, sep="\t", dtype={"chromosome": str, "version": str})


def strip_version(ids):
    """Versionless Ensembl ids."""
    return pd.Series(ids, dtype="object").astype(str).str.split(".").str[0]


def map_features(native_ids, native_symbols, hgnc, axis_ids):
    """Map native features to axis ids: id first, then HGNC names; duplicates on either side are dropped, every row logged."""
    pos = pd.Series(np.arange(len(axis_ids)), index=axis_ids)
    m = pd.DataFrame(dict(native_id=pd.Series(native_ids, dtype="object") if native_ids is not None else None,
                          native_symbol=pd.Series(native_symbols, dtype="object") if native_symbols is not None else None))
    m["ensembl"], m["method"] = None, "unmapped"
    if native_ids is not None:
        e = strip_version(m.native_id)
        hit = e.isin(pos.index) & e.str.startswith("ENSG")
        m.loc[hit, "ensembl"], m.loc[hit, "method"] = e[hit], "ensembl_id"
    if native_symbols is not None:
        todo = m.ensembl.isna() & m.native_symbol.notna()
        for i in np.flatnonzero(todo):
            cls, e = hgnc.symbol(str(m.native_symbol.iat[i]))
            if e in pos.index and cls in ("approved", "previous", "alias"):
                m.iat[i, m.columns.get_loc("ensembl")], m.iat[i, m.columns.get_loc("method")] = e, f"hgnc_{cls}"
            elif cls != "unmapped":
                m.iat[i, m.columns.get_loc("method")] = cls
    dup_native = m.native_id.duplicated(keep=False) if native_ids is not None else pd.Series(False, index=m.index)
    dup_axis = m.ensembl.notna() & m.ensembl.duplicated(keep=False)
    m.loc[dup_native, "method"] = "duplicate_native_id"
    m.loc[dup_axis, "method"] = "several_native_features_for_one_gene"
    m.loc[dup_native | dup_axis, "ensembl"] = None
    m["axis_index"] = m.ensembl.map(pos).astype("Int64")
    return m


def mapping_summary(m):
    """Counts per mapping method."""
    return {k: int(v) for k, v in m.method.value_counts().items()}


if __name__ == "__main__":
    t = build_table(load())
    print(f"{len(t)} genes, {t.biotype.value_counts().get('protein_coding', 0)} protein coding")
