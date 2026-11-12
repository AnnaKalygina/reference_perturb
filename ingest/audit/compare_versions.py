"""Compare the repo processed h5ad and the VCC2026 summaries with the raw-deposit audit, per context."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from audit import CFG  # noqa: E402
from h5 import col, index  # noqa: E402

OUT = Path(CFG["out_dir"])
REPO = Path(CFG["repo_data"])
VCC = Path(CFG["vcc_data"])
JPW = {"IFNB": "ifnb", "IFNG": "ifng", "INS": "ins", "TGFB1": "tgfb", "TNFA": "tnfa"}

PAIRS = [
    ("adamson16", "adamson16", lambda o: "adamson16__K562", lambda c: "adamson__k562" if c == "adamson16__K562" else None),
    ("scperturb_frangieh", "frangieh21",
     lambda o: "frangieh21__melanoma__" + o["perturbation_2"].map({"Control": "control", "IFNγ": "IFNG", "Co-culture": "TIL_coculture"}),
     lambda c: {"control": "frangieh__control", "IFNG": "frangieh__ifng", "TIL_coculture": "frangieh__coculture"}[c.split("__")[-1]]),
    ("scperturb_sunshine", "sunshine23", lambda o: "sunshine23__Calu3", lambda c: "sunshine__calu3"),
    ("scperturb_norman", "norman19", lambda o: "norman19__K562", lambda c: None),
    ("scperturb_wessels", "wessels23", lambda o: "wessels23__THP1", lambda c: None),
    ("replogle20", "replogle20", lambda o: "replogle20__K562", lambda c: "replogle2020__k562"),
    ("replogle22_k562", "replogle22", lambda o: "replogle22_essential__" + o["cell_type"], lambda c: "replogle_nadig__k562"),
    ("replogle22_rpe1", "replogle22", lambda o: "replogle22_essential__" + o["cell_type"], lambda c: "replogle_nadig__rpe1"),
    ("jiang", "jiang24", lambda o: "jiang25__" + o["cell_type"].str.upper() + "__" + o["pathway"].astype(str),
     lambda c: f"jiang2025_{JPW.get(c.split('__')[2], '?')}__{c.split('__')[1].lower()}"),
    ("mcfaline", "mcfaline23", lambda o: "mcfaline23__" + o["cell_type"].str.upper() + "__" + o["treatment"].astype(str).replace("none", "vehicle") + "_"
     + o["dose"].astype(float).astype(int).astype(str),
     lambda c: f"mcfaline2023__{c.split('__')[1].lower()}" if "vehicle" in c else None),
]


def repo_obs(name):
    """Repo processed obs columns needed for comparison."""
    with h5py.File(REPO / name / f"{name}_processed.h5ad", "r") as h:
        o = pd.DataFrame({c: col(h, f"obs/{c}") for c in ["perturbation", "cell_type", "batch", "perturbation_2", "pathway",
                                                            "treatment", "dose"] if f"obs/{c}" in h})
        n_var = len(index(h, "var"))
    return o, n_var


def vcc_ctx(name):
    """VCC2026 summary of one context."""
    p = VCC / f"{name}.h5"
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        t = pd.Series(f["pert/n_cells"][:], index=[x.decode() if isinstance(x, bytes) else x for x in f["pert/targets"][:]])
        return dict(n_controls=int(f["control/n_cells"][()]), targets=t, measured=int(f["measured"][:].sum()),
                    batch_key=str(f.attrs.get("batch_key")))


def per_pert_agreement(a, b):
    """Shared targets, share with identical counts, median ratio b/a."""
    s = a.index.intersection(b.index)
    if not len(s):
        return dict(shared=0)
    r = b[s] / a[s]
    return dict(shared=int(len(s)), only_raw=int(len(a.index.difference(b.index))), only_other=int(len(b.index.difference(a.index))),
                identical_counts=float((a[s] == b[s]).mean()), median_ratio=float(r.median()), min_ratio=float(r.min()))


def main():
    rows = []
    for src, repo, rkey, vkey in PAIRS:
        s = json.loads((OUT / src / "summary.json").read_text())
        pt = pd.read_csv(OUT / src / "perturbations.csv.gz")
        bt = pd.read_csv(OUT / src / "batches.csv", dtype={"batch": str})
        o, n_var = repo_obs(repo)
        o["ctx"] = rkey(o)
        for c in s["contexts"]:
            ctx = c["context"]
            raw = pt[(pt.context == ctx) & (~pt.combo)].set_index("perturbation").n_cells
            ro = o[o.ctx == ctx]
            rp = ro[ro.perturbation != "control"].perturbation.value_counts()
            rp = rp[~rp.index.str.contains(r"\+")]
            raw_b = set(bt[bt.context == ctx].batch.astype(str))
            rep_b = set(ro.batch.astype(str))
            row = dict(source=src, context=ctx, raw_cells=c["n_cells"], raw_controls=c["n_controls"], raw_perts=c["n_perts"],
                       raw_batches=len(raw_b), repo_cells=len(ro), repo_controls=int((ro.perturbation == "control").sum()),
                       repo_perts=int(len(rp)), repo_genes=n_var, repo_batch_labels=len(rep_b),
                       repo_batches_match_raw=bool(rep_b == raw_b) if len(ro) else None,
                       repo_vs_raw=per_pert_agreement(raw, rp) if len(ro) else None)
            v = vcc_ctx(vkey(ctx)) if vkey(ctx) else None
            if v:
                row.update(vcc_context=vkey(ctx), vcc_controls=v["n_controls"], vcc_perts=int(len(v["targets"])),
                           vcc_measured_genes=v["measured"], vcc_batch_key=v["batch_key"], vcc_vs_raw=per_pert_agreement(raw, v["targets"]))
            rows.append(row)
    (OUT / "compare_versions.json").write_text(json.dumps(rows, indent=1, default=str))
    df = pd.json_normalize(rows)
    df.to_csv(OUT / "compare_versions.csv", index=False)
    print(df[["context", "raw_controls", "repo_controls", "vcc_controls", "raw_perts", "repo_perts", "vcc_perts",
              "repo_batches_match_raw", "repo_vs_raw.median_ratio", "vcc_vs_raw.identical_counts", "vcc_vs_raw.median_ratio"]].to_string())


if __name__ == "__main__":
    main()
