import h5py
import numpy as np


def _walk_hdf(file_or_group: h5py.File | h5py.Group) -> dict:
    out = {}
    for key in file_or_group.keys():
        match type(file_or_group[key]):
            case h5py.Dataset:
                out[key] = np.asarray(file_or_group[key])
            case h5py.Group:
                out[key] = _walk_hdf(file_or_group[key])
    return out


def _read_hdf(path: str) -> dict:
    out = {"attributes": {}, "datasets": {}}
    file = h5py.File(path, "r")
    for atr in file.attrs.keys():
        out["attributes"][atr] = file.attrs[atr]
    out["datasets"] = _walk_hdf(file)
    file.close()
    return out


class File:
    def __init__(self, path: str) -> None:
        self.path = path
        self._file_dict = _read_hdf(path)

    def __repr__(self) -> str:
        pass
