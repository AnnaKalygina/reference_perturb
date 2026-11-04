"""CD4 T (Marson 2026): one unit per donor x condition; cells come from the authors' obs tables, counts from the GEO lane matrices."""
import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "audit"))
import sources  # noqa: E402
from adapters import ADAPTERS, chunks_of  # noqa: E402
from adapters_more import TenxLanes, UnitAdapter  # noqa: E402
from h5 import dec  # noqa: E402


class CD4T(UnitAdapter, TenxLanes):
    dataset, loader = "cd4t", staticmethod(sources.cd4t)

    def dir(self):
        return Path(self.cfg["paths"]["vcc_data"]) / "raw/cd4t_marson2025"

    def units(self):
        return sorted(p.name.split(".")[0] for p in (self.dir() / "obs").glob("*.obs.parquet"))

    def select(self, obs, unit):
        donor, cond = unit.split("_", 1)
        return (obs.context == f"cd4t_marson25__{donor}__{cond}").to_numpy()

    def lane_file(self, lane, unit):
        donor, cond = unit.split("_", 1)
        s3 = self.dir() / "s3_lanes" / f"CD4i_{lane}_{donor}_{cond}.h5"
        if s3.exists():
            return s3
        hits = glob.glob(str(self.dir() / f"GSM*_CD4i_{lane}_{donor}_{cond}_sample_filtered_feature_bc_matrix.h5"))
        if len(hits) != 1:
            raise SystemExit(f"cd4t {unit} lane {lane}: {len(hits)} GEO matrices")
        return Path(hits[0])

    def obs(self, unit):
        o = super().obs(unit)
        cells = pd.read_parquet(self.dir() / "obs" / f"{unit}.obs.parquet", columns=["lane_id"])
        if len(cells) != len(o):
            raise SystemExit(f"cd4t {unit}: loader obs has {len(o)} rows, the obs table {len(cells)}")
        o["barcode"] = cells.index.str.split("_").str[0].to_numpy()
        o["lane"] = cells.lane_id.astype(str).str.replace("CD4i_", "", regex=False).to_numpy()
        present = np.zeros(len(o), bool)
        self.lane_cols = {}
        for lane in sorted(o.lane.unique()):
            with h5py.File(self.lane_file(lane, unit), "r") as h:
                bc = pd.Index(dec(h["matrix/barcodes"][:]))
            rows = np.flatnonzero((o.lane == lane).to_numpy())
            pos = bc.get_indexer(o.barcode.to_numpy()[rows])
            present[rows[pos >= 0]] = True
            self.lane_cols[lane] = dict(zip(o.barcode.to_numpy()[rows[pos >= 0]], pos[pos >= 0]))
        self.missing_from_geo = int((~present).sum())
        self.meta = dict(self.meta, annotated_cells_missing_from_geo={unit: self.missing_from_geo})
        if self.missing_from_geo > 0.05 * len(o):
            raise SystemExit(f"cd4t {unit}: {self.missing_from_geo} of {len(o)} annotated cells are absent from the GEO matrices")
        o = o[present].copy()
        o = o.iloc[np.argsort(o.lane.to_numpy(), kind="stable")].reset_index(drop=True)
        o["deposit_row"] = np.arange(len(o))
        self._obs = o
        return o

    def features(self, unit):
        o = self._obs
        with h5py.File(self.lane_file(sorted(o.lane.unique())[0], unit), "r") as h:
            m = h["matrix"]
            gex = dec(m["features/feature_type"][:]) == "Gene Expression"
            return dec(m["features/id"][:])[gex], dec(m["features/name"][:])[gex]

    def chunks(self, unit):
        o = self._obs
        ref = {g: i for i, g in enumerate(self.features(unit)[0])}
        start = 0
        for lane in sorted(o.lane.unique()):
            rows = np.flatnonzero((o.lane == lane).to_numpy())
            x = self.lane_matrix(self.lane_file(lane, unit), ref)
            cols = np.array([self.lane_cols[lane][b] for b in o.barcode.to_numpy()[rows]])
            x = x[cols]
            for lo, piece in chunks_of(x):
                yield start + lo, piece
            start += x.shape[0]


ADAPTERS["cd4t"] = CD4T
