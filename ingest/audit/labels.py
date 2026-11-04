"""Gate for perturbation labels: map each single-target label to one gene or exclude it with a reason."""
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from audit import CFG  # noqa: E402
from genes import HGNC  # noqa: E402

OUT = Path(CFG["out_dir"])
NOT_A_GENE = re.compile(r"^(nan|none|na|null|)$|[^A-Za-z0-9._-]", re.I)
ENSEMBL = re.compile(r"^ENSG\d{11}$")


@lru_cache(maxsize=1)
def hgnc():
    """Cached HGNC lookups."""
    return HGNC(CFG["hgnc"])


def by_name(s):
    """(class, Ensembl id) for one symbol from HGNC names; a unique non-pseudogene candidate resolves an ambiguous old name."""
    h = hgnc()
    if s in h.approved:
        return ("approved", h.approved[s]) if isinstance(h.approved[s], str) else ("approved_no_ensembl", None)
    for cls, d in (("previous", h.prev), ("alias", h.alias)):
        if s in d:
            if len(d[s]) == 1:
                return cls, next(iter(d[s]))
            genes = d[s] - h.pseudogene
            return (f"{cls}_unique_non_pseudogene", next(iter(genes))) if len(genes) == 1 else ("ambiguous", None)
    return "unmapped", None


def classify(label, deposit_ensembl):
    """(class, symbol, Ensembl id, kept) for one label: the deposit's target Ensembl id if HGNC knows it, else HGNC names, else the deposit id."""
    h = hgnc()
    if NOT_A_GENE.search(label):
        return "not_a_gene", None, None, False
    dep = deposit_ensembl if isinstance(deposit_ensembl, str) and ENSEMBL.match(deposit_ensembl) else None
    if dep in h.ens_symbol:
        named = h.approved.get(label)
        if isinstance(named, str) and named != dep:
            return "deposit_id_and_name_disagree", None, dep, False
        return "deposit_ensembl", h.ens_symbol[dep], dep, True
    if ENSEMBL.match(label):
        return ("ensembl_id", h.ens_symbol[label], label, True) if label in h.ens_symbol else ("ensembl_id_not_in_hgnc", None, label, False)
    for s, suffix in ((label, ""), (label.replace("_", "-"), "_after_underscore_to_hyphen")):
        cls, e = by_name(s)
        if cls == "approved_no_ensembl" and dep:
            return "approved_no_ensembl_with_deposit_ensembl", s, dep, True
        if cls not in ("ambiguous", "unmapped"):
            return cls + suffix, h.ens_symbol.get(e, s) if e else s, e, True
    if dep:
        return "deposit_ensembl_not_in_hgnc", label, dep, True
    cls, _ = by_name(label)
    return cls, None, None, False


def gate(counts, deposit):
    """Single-target counts with labels mapped to one gene each, and the per-label gate log."""
    dep = deposit.set_index("perturbation").deposit_ensembl if len(deposit) else pd.Series(dtype=object)
    labs = counts.perturbation.unique()
    m = pd.DataFrame([(x, *classify(x, dep.get(x))) for x in labs], columns=["perturbation", "label_class", "symbol", "ensembl", "kept"])
    m["class_by_name_only"] = [by_name(x)[0] for x in m.perturbation]
    log = counts.merge(m, on="perturbation").groupby(
        ["context", "perturbation", "label_class", "class_by_name_only", "symbol", "ensembl", "kept"], dropna=False).n_cells.sum().reset_index()
    gene = log.ensembl.fillna(log.symbol)
    shared = log.kept & (log.groupby([log.context, gene]).perturbation.transform("size") > 1)
    drop = shared & (log.perturbation != log.symbol)
    log.loc[drop, "label_class"] = "same_gene_as_another_label"
    log.loc[drop, "kept"] = False
    keep = log[log.kept][["context", "perturbation", "symbol"]]
    kept = counts.merge(keep, on=["context", "perturbation"])[["context", "batch", "symbol", "n_cells"]].rename(columns={"symbol": "perturbation"})
    return kept, log
