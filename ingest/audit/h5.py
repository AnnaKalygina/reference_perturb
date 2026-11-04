"""Read-only h5py helpers for AnnData and 10x files."""
import h5py
import numpy as np


def dec(a):
    """Byte or object array to str array."""
    a = np.asarray(a)
    if a.dtype.kind in "SO":
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a], dtype=object)
    return a.astype(str).astype(object)


def col(h, key):
    """One AnnData obs/var column as an object array, for both categorical layouts."""
    node = h[key]
    if isinstance(node, h5py.Group):
        cats = dec(node["categories"][:])
        codes = node["codes"][:]
        out = np.empty(len(codes), dtype=object)
        out[codes >= 0] = cats[codes[codes >= 0]]
        out[codes < 0] = None
        return out
    grp, name = key.rsplit("/", 1)
    legacy = f"{grp}/__categories/{name}"
    if legacy in h:
        cats = dec(h[legacy][:])
        codes = node[:]
        out = np.empty(len(codes), dtype=object)
        out[codes >= 0] = cats[codes[codes >= 0]]
        out[codes < 0] = None
        return out
    v = node[:]
    return dec(v) if v.dtype.kind in "SO" else v


def index(h, grp):
    """obs or var index."""
    k = h[grp].attrs.get("_index", "_index")
    return col(h, f"{grp}/{k.decode() if isinstance(k, bytes) else k}")


def has(h, key):
    """Whether an AnnData column exists."""
    return key in h


def matrix_values(h, key, n):
    """Up to n stored values of an AnnData matrix (dense rows or sparse data)."""
    node = h[key]
    if isinstance(node, h5py.Dataset):
        rows = max(1, n // max(1, node.shape[1]))
        return np.asarray(node[:rows]).ravel()
    return np.asarray(node["data"][:n])


def tenx_values(path, n):
    """Up to n stored values of a 10x h5 matrix."""
    with h5py.File(path, "r") as h:
        return np.asarray(h["matrix/data"][:n])


def tenx_features(path):
    """(ids, names, feature_type, genome) of a 10x h5."""
    with h5py.File(path, "r") as h:
        f = h["matrix/features"]
        g = dec(f["genome"][:]) if "genome" in f else None
        return dec(f["id"][:]), dec(f["name"][:]), dec(f["feature_type"][:]), g
