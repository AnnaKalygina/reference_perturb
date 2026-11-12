"""CLI: python -m ingest <command> ...; units, status and submission derive everything from the files on disk."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from config import load, path  # noqa: E402
from receipts import write_json  # noqa: E402

STAGES = ["shard", "moments", "deltas", "validate", "scores"]
UPSTREAM = {"moments": "shard", "deltas": "moments", "validate": "deltas", "scores": "shard"}
WORKERS = {"scores": 8}


def contexts_of(cfg):
    """(dataset, context) pairs that have a shard."""
    return sorted((p.parent.name, p.stem) for p in path(cfg, "shards").glob("*/*.h5ad"))


def receipt(cfg, stage, unit):
    p = path(cfg, "receipts", "outputs", stage, unit.replace("/", "__") + ".json")
    return json.loads(p.read_text()) if p.exists() else None


def state(cfg, stage, unit, outputs):
    """missing / done / stale from outputs and receipts."""
    rec = receipt(cfg, stage, unit)
    if rec is None:
        return "partial" if any(Path(o).exists() for o in outputs) else "missing"
    if rec["config_hash"] != cfg["hash"]:
        return "stale"
    return "done" if all(Path(o).exists() for o in outputs) else "partial"


def stage_outputs(cfg, stage, dataset, context):
    """Expected outputs of a stage for one context."""
    if stage == "shard":
        return [path(cfg, "shards", dataset, f"{context}.h5ad")]
    if stage == "moments":
        return [path(cfg, "suffstats", dataset, context, "pert_moments.h5")]
    if stage == "deltas":
        return [path(cfg, "deltas", dataset, context, "deltas.h5")]
    if stage == "validate":
        return [path(cfg, "validation", f"{context}.json")]
    return [path(cfg, "scores", f"{context}.parquet")]


def running(cfg):
    """Job names of the user's queued or running ingest jobs."""
    out = subprocess.run(["squeue", "--me", "-h", "-o", "%j|%T"], capture_output=True, text=True).stdout
    return {line.split("|")[0]: line.split("|")[1] for line in out.splitlines() if line.startswith("ingest:")}


def shard_units(cfg):
    """context -> shard unit, from the shard receipts."""
    out = {}
    for p in path(cfg, "receipts", "outputs", "shard").glob("*.json") if path(cfg, "receipts", "outputs", "shard").exists() else []:
        rec = json.loads(p.read_text())
        for c in rec.get("contexts", []):
            out[c] = rec["unit"]
    return out


def status(cfg):
    """One row per (context, stage)."""
    jobs = running(cfg)
    units = shard_units(cfg)
    rows = []
    for dataset, context in contexts_of(cfg):
        for stage in STAGES:
            unit = units.get(context, dataset) if stage == "shard" else context
            rec = receipt(cfg, stage, unit)
            rows.append(dict(dataset=dataset, context=context, stage=stage, state=state(cfg, stage, unit, stage_outputs(cfg, stage, dataset, context)),
                             job=jobs.get(f"ingest:{stage}:{unit}"), seconds=rec and rec.get("seconds"), peak_rss_gb=rec and rec.get("peak_rss_gb"),
                             commit=rec and rec.get("code_commit"), finished=rec and rec.get("finished")))
    t = pd.DataFrame(rows)
    t.to_parquet(path(cfg, "status.parquet"))
    return t


def peak_rss(cfg):
    """(stage, dataset) -> largest peak_rss_gb over the output receipts; the dataset of a context is its shard directory."""
    dataset_of = {context: dataset for dataset, context in contexts_of(cfg)}
    peaks = {}
    for stage in STAGES:
        for p in path(cfg, "receipts", "outputs", stage).glob("*.json"):
            rec = json.loads(p.read_text())
            key = (stage, Path(rec["outputs"][0]).parent.name if stage == "shard" else dataset_of.get(rec["unit"]))
            peaks[key] = max(peaks.get(key, 0), rec.get("peak_rss_gb", 0))
    return peaks


def memory_gb(cfg, stage, dataset, peak):
    """Per-worker memory: the config entry of (stage, dataset) or the stage default, raised to rerun_memory_factor x the observed peak."""
    m = cfg["slurm"]["memory_gb"][stage]
    return int(max(m.get(dataset, m["default"]), cfg["slurm"]["rerun_memory_factor"] * peak + 1))


def memory_table(cfg):
    """Observed peaks per stage and dataset, and the proposed slurm.memory_gb: rerun_memory_factor x the peak, rounded up to a multiple of 8 GB."""
    peaks = pd.Series(peak_rss(cfg)).unstack(0)[STAGES].sort_index()
    proposed = np.ceil(peaks * cfg["slurm"]["rerun_memory_factor"] / 8) * 8
    table = {}
    for stage in STAGES:
        default = int(proposed[stage].mode().iloc[0])
        table[stage] = dict(default=default, **{d: int(v) for d, v in proposed[stage].dropna().items() if v != default})
    return peaks, table


def rehash(cfg, old):
    """Rewrite every output receipt carrying the old config hash to the current one, keeping the history; returns the count."""
    if cfg["dirty"]:
        raise SystemExit("rehash needs a clean checkout")
    if old == cfg["hash"]:
        raise SystemExit(f"{old} is the current hash")
    n = 0
    for p in path(cfg, "receipts", "outputs").glob("*/*.json"):
        rec = json.loads(p.read_text())
        if rec["config_hash"] != old:
            continue
        rec["config_hash"] = cfg["hash"]
        rec.setdefault("config_hash_history", []).append({"from": old, "to": cfg["hash"], "commit": cfg["commit"], "date": time.strftime("%Y-%m-%d")})
        write_json(p, rec)
        n += 1
    return n


def submit(cfg, stage, datasets, dependency=None):
    """Submit the missing units of a stage; returns {unit: job id}."""
    jobs = running(cfg)
    peaks = peak_rss(cfg)
    submitted = {}
    units = []
    if stage == "shard":
        from adapters import ADAPTERS
        import adapters_more  # noqa: F401
        import adapters_cd4t  # noqa: F401
        import adapters_tcells  # noqa: F401
        import adapters_tar  # noqa: F401
        for d in datasets or ADAPTERS:
            if d in ADAPTERS:
                units += [(d, u) for u in ADAPTERS[d](cfg).units()]
    else:
        units = [(d, c) for d, c in contexts_of(cfg) if not datasets or d in datasets]
    for dataset, unit in units:
        name = f"ingest:{stage}:{unit}"
        if name in jobs or (stage != "shard" and state(cfg, stage, unit, stage_outputs(cfg, stage, dataset, unit)) == "done"):
            continue
        if stage == "shard" and receipt(cfg, "shard", unit) and receipt(cfg, "shard", unit)["config_hash"] == cfg["hash"]:
            continue
        large = stage == "shard" and dataset in cfg["slurm"]["large_shards"]
        t = cfg["slurm"]["time"]["shard_large" if large else "default"]
        workers = WORKERS.get(stage, 1)
        cmd = ["sbatch", "--parsable", f"--job-name={name}", f"--mem={memory_gb(cfg, stage, dataset, peaks.get((stage, dataset), 0)) * workers}G", f"--cpus-per-task={max(2, workers)}", f"--time={t}", f"--partition={cfg['slurm']['partition']}"]
        dep = dependency.get(unit) if dependency else None
        if dep:
            cmd.append(f"--dependency=afterok:{dep}")
        cmd += ["slurm/stage.sbatch", stage] + ([dataset, "--unit", unit] if stage == "shard" else [unit])
        jid = subprocess.run(cmd, capture_output=True, text=True, cwd=cfg["root"], check=True).stdout.strip()
        submitted[unit] = jid
        print(f"{name} -> {jid}", flush=True)
    return submitted


def main():
    ap = argparse.ArgumentParser(prog="python -m ingest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    s = sub.add_parser("submit")
    s.add_argument("--stage", choices=STAGES + ["all"], default="all")
    s.add_argument("--dataset", nargs="*")
    m = sub.add_parser("manifest")
    m.add_argument("--schema", action="store_true")
    sub.add_parser("memory")
    r = sub.add_parser("rehash")
    r.add_argument("--from", dest="old", required=True)
    a = ap.parse_args()
    cfg = load()
    if a.cmd == "status":
        t = status(cfg)
        print(t.groupby(["stage", "state"]).size().unstack(fill_value=0).to_string())
        print(t[t.state != "done"].to_string(index=False) if (t.state != "done").any() else "all done")
    elif a.cmd == "submit":
        stages = STAGES if a.stage == "all" else [a.stage]
        dependency = None
        for stage in stages:
            dependency = submit(cfg, stage, a.dataset, dependency)
    elif a.cmd == "memory":
        peaks, table = memory_table(cfg)
        print("observed peak_rss_gb, max over units", peaks.to_string(float_format="{:.2f}".format), sep="\n")
        print(yaml.safe_dump(dict(slurm=dict(memory_gb=table)), default_flow_style=None, sort_keys=False, width=1000), end="")
    elif a.cmd == "rehash":
        print(f"{rehash(cfg, a.old)} receipts rehashed from {a.old} to {cfg['hash']}")
    elif a.cmd == "manifest":
        from receipts import write_json
        from shard import deposit_fields
        rows = []
        for p in sorted(path(cfg, "manifest").glob("*.json")):
            r = json.loads(p.read_text())
            fields, sources = deposit_fields(cfg, r["dataset"], r["context"])
            r.update(fields)
            r["sources"].update(sources)
            rows.append((p, r))
        if a.schema:
            print(json.dumps(sorted({k for _, r in rows for k in r}), indent=1))
        else:
            for p, r in rows:
                write_json(p, r)
            t = pd.DataFrame([r for _, r in rows])
            t.to_parquet(path(cfg, "manifest.parquet"))
            print(f"{len(t)} manifests re-rendered and written to manifest.parquet")


if __name__ == "__main__":
    main()
