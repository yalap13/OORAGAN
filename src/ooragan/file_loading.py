import os
import h5py
import numpy as np
from glob import glob
from pathlib import Path


def _walk_hdf(file_or_group: h5py.File | h5py.Group) -> dict:
    """Walks an HDF file hierarchy and converts it into dictionary."""
    out = {}
    for key in file_or_group.keys():
        match type(file_or_group[key]):
            case h5py.Dataset:
                out[key] = np.asarray(file_or_group[key])
            case h5py.Group:
                out[key] = _walk_hdf(file_or_group[key])
            case _:
                raise TypeError("Invalid type")
    return out


def _read_hdf(path: str) -> dict:
    """Reads an HDF file from its path."""
    out = {"attributes": {}, "datasets": {}}
    file = h5py.File(path, "r")
    for atr in file.attrs.keys():
        out["attributes"][atr] = file.attrs[atr]
    out["datasets"] = _walk_hdf(file)
    file.close()
    return out


class _File:
    """
    Defines a loaded HDF file and implements methods to get data from the file.

    Parameters
    ----------
    path : str
        Path to the HDF file.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._file_dict = _read_hdf(path)


def _load_files_from_path(path: str) -> list[_File]:
    """Loads multiple files from a directory, walking through it."""
    files = []
    for paths, _, _ in os.walk(path):
        for file in glob(os.path.join(paths, "*.hdf5")):
            file_obj = _File(file)
            files.append(file_obj)
    if not files:
        raise RuntimeError("No HDF5 files were found in this directory")
    return files


class Dataset:
    """
    Data container. Contains data and information on measurements saved in HDF5
    files.

    Note
    ----
    Supports only data from HDF5 files as of now. For txt files, use the old
    Dataset class located in ooragan.old.Dataset.

    Parameters
    ----------
    path : str
        Path of the folder for multiple data files or for a single data file.
    attenuation_cryostat : float
        Total attenuation present on the cryostat. Must be a negative number.
    """

    def __init__(self, path: str, attenuation_cryostat: float) -> None:
        if attenuation_cryostat > 0:
            raise ValueError("Attenuation must be negative")
        if Path(path).suffix == "":
            self._files = _load_files_from_path(path)
        else:
            self._files = [_File(path)]
