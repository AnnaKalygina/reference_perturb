"""HGNC-based mapping of gene symbols and Ensembl ids, for audit statistics."""
import numpy as np
import pandas as pd


class HGNC:
    """Symbol and Ensembl lookups from an HGNC complete set."""

    def __init__(self, path):
        h = pd.read_csv(path, sep="\t", dtype=str, usecols=["symbol", "status", "alias_symbol", "prev_symbol", "ensembl_gene_id", "locus_group"])
        h = h[h.status == "Approved"]
        self.pseudogene = set(h.ensembl_gene_id[h.locus_group == "pseudogene"].dropna())
        self.approved = dict(zip(h.symbol, h.ensembl_gene_id))
        self.ens = set(h.ensembl_gene_id.dropna())
        self.ens_symbol = dict(zip(h.ensembl_gene_id, h.symbol))
        self.prev, self.alias = {}, {}
        for d, c in ((self.prev, "prev_symbol"), (self.alias, "alias_symbol")):
            for sym, e in zip(h[c], h.ensembl_gene_id):
                if isinstance(sym, str) and isinstance(e, str):
                    for s in sym.split("|"):
                        d.setdefault(s, set()).add(e)

    def symbol(self, s):
        """(class, ensembl or None) for one symbol."""
        if s in self.approved:
            e = self.approved[s]
            return ("approved", e) if isinstance(e, str) else ("approved_no_ensembl", None)
        for cls, d in (("previous", self.prev), ("alias", self.alias)):
            if s in d:
                return (cls, next(iter(d[s]))) if len(d[s]) == 1 else ("ambiguous", None)
        return ("unmapped", None)

    def summary(self, genes, ensembl):
        """Mapping statistics for one feature axis."""
        out = {}
        if ensembl is not None:
            e = pd.Series(ensembl, dtype=str).str.split(".").str[0]
            out["n_features"] = len(e)
            out["id_type"] = "ensembl"
            out["ensembl_in_hgnc"] = int(e.isin(self.ens).sum())
            out["ensembl_not_in_hgnc"] = int((~e.isin(self.ens)).sum())
            out["duplicate_ensembl_ids"] = int(e.duplicated().sum())
            if genes is not None:
                g = pd.Series(genes, dtype=str)
                agree = [self.ens_symbol.get(a) == b for a, b in zip(e, g)]
                out["symbol_differs_from_hgnc_for_ensembl"] = int(len(agree) - sum(agree))
        elif genes is not None:
            g = pd.Series(genes, dtype=str)
            cls = [self.symbol(s) for s in g]
            out["n_features"] = len(g)
            out["id_type"] = "symbol"
            out.update({f"symbols_{k}": int(v) for k, v in pd.Series([c for c, _ in cls]).value_counts().items()})
            mapped = pd.Series([e for _, e in cls]).dropna()
            out["features_sharing_an_ensembl_id"] = int(mapped.duplicated(keep=False).sum())
        return out

    def targets(self, labels):
        """Mapping classes of perturbation target symbols."""
        cls = pd.Series([self.symbol(s)[0] for s in labels])
        return {k: int(v) for k, v in cls.value_counts().items()}
