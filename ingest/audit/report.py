"""Markdown tables from the per-source audit summaries."""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from audit import CFG  # noqa: E402

OUT = Path(CFG["out_dir"])


def f(x, d=0):
    """Number formatting."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "-"
    return f"{x:,.{d}f}"


def main():
    S = [json.load(open(p)) for p in sorted(glob.glob(str(OUT / "*/summary.json")))]
    a = ["| source | cells | contexts | controls | perturbed | unassigned | batches | batch field | kind | int check | genes | targets HGNC |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        ic, g = s["integer_check"], s["genes"]
        gs = (f"{g.get('n_features', '-')} {g.get('id_type', '-')}"
              + (f", {g['ensembl_not_in_hgnc']} not in HGNC" if "ensembl_not_in_hgnc" in g else "")
              + (f", {g.get('symbols_unmapped', 0)} unmapped / {g.get('symbols_ambiguous', 0)} ambiguous" if g.get("id_type") == "symbol" else ""))
        t = s["targets"]
        a.append(f"| {s['source']} | {f(s['n_cells'])} | {s['n_contexts']} | {f(s['n_controls'])} | {f(s['n_perturbed'])} | "
                 f"{f(s['n_unassigned'])} | {s['n_batches']} | {s['meta']['batch_field']} | {s['meta']['batch_kind']} | "
                 f"{ic['non_integer']}/{f(ic['n_nonzero'])} non-int, max {f(ic['max'])} | {gs} | "
                 f"{sum(t.values())}: {t.get('previous', 0) + t.get('alias', 0)} prev/alias, {t.get('unmapped', 0)} unm, {t.get('ambiguous', 0)} amb |")
    b = ["| context | cells | controls | perts (+combos) | batches | ctrl/batch min-med-max | batches <20/<50/<100 ctrl | pert cells in <20/<50/<100 | ctrl-free | batches/pert med | cells/pert med | K562-ess overlap |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        for c in s["contexts"]:
            q, bp, cp = c["ctrl_per_batch"], c["batches_per_pert"], c["cells_per_pert"]
            sh = "/".join(f(100 * (c[f"share_pert_cells_in_batches_ctrl_lt_{t}"] or 0), 1) for t in (20, 50, 100))
            b.append(f"| {c['context']} | {f(c['n_cells'])} | {f(c['n_controls'])} | {c['n_perts']} (+{c['n_combo_perts']}) | {c['n_batches']} | "
                     f"{f(q['min'])}-{f(q['median'])}-{f(q['max'])} | {c['batches_ctrl_lt_20']}/{c['batches_ctrl_lt_50']}/{c['batches_ctrl_lt_100']} | "
                     f"{sh} % | {c['control_free_batches']} | {f(bp['median'])} | {f(cp['median'])} | {c['k562_essential_overlap']} |")
    rows = []
    for s in S:
        for t in s["thresholds"]:
            rows.append(dict(source=s["source"], **t))
    th = pd.DataFrame(rows)
    tot = th.groupby("min_controls")[["perturbed_cells_dropped", "perts_lost_all", "perts_below_min_cells", "n_perts"]].sum()
    c = ["| min controls per batch | perturbed cells dropped | perturbations losing all cells | perturbations left with < 10 cells | sources affected (cells dropped > 1 %) |",
         "|---|---|---|---|---|"]
    for t, r in tot.iterrows():
        hit = th[(th.min_controls == t) & (th.share_cells_dropped > 0.01)]
        c.append(f"| {t} | {f(r.perturbed_cells_dropped)} | {f(r.perts_lost_all)} | {f(r.perts_below_min_cells)} of {f(r.n_perts)} | "
                 + (", ".join(f"{x.source} {100 * x.share_cells_dropped:.0f}%" for x in hit.itertuples()) or "none") + " |")
    groups = 0
    for p in glob.glob(str(OUT / "*/perturbations.csv.gz")):
        groups += int(pd.read_csv(p, usecols=["n_batches"]).n_batches.sum())
    batches = sum(s["n_batches"] for s in S)
    perts = sum(len(pd.read_csv(p, usecols=["n_cells"])) for p in glob.glob(str(OUT / "*/perturbations.csv.gz")))
    sizes = dict(context_perturbation_rows=perts, context_batch_perturbation_rows=groups, context_batch_rows=batches)
    (OUT / "tables.md").write_text("\n".join(["## sources", *a, "", "## contexts", *b, "", "## thresholds", *c, "", "## suffstat rows",
                                              json.dumps(sizes)]) + "\n")
    print(json.dumps(sizes))


if __name__ == "__main__":
    main()
