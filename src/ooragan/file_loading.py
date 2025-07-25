import os
import h5py
import re
import numpy as np
from glob import glob
from pathlib import Path
from typing import Any, Self, Optional

from numpy.typing import NDArray

from .parameters import NullParameter, Parameter
from .util import convert_complex_to_magphase, convert_magphase_to_complex

# TODO:
# - Fetch the description and unit of parameters from the dataset's attributes
#   in the hdf5 file
# - Ultimately, change the way the data is loaded for fitting so that it is
#   possible to fit the data for any parameter

KNOWN_PARAMETERS = [
    "VNA",
    "VNA Average",
    "VNA Power",
    "VNA Bandwidth",
    "Variable Attenuator",
    "VNA Frequency",
    "s21_real",
    "s21_imag",
    "s21_mag",
    "s21_phase",
    "Index",
    "Magnet",
]


def _walk_hdf(
    file_or_group: Any,
    additional_params: list[str],
) -> dict[str, NDArray]:
    """Walks an HDF file hierarchy and converts it into dictionary."""
    out = {}
    for key in file_or_group.keys():
        if key in KNOWN_PARAMETERS or key in additional_params:
            match type(file_or_group[key]):
                case h5py.Dataset:
                    out[key] = np.asarray(file_or_group[key])
                case h5py.Group:
                    out[key] = _walk_hdf(file_or_group[key], additional_params)
                case _:
                    raise TypeError("Invalid type")
    return out


def _read_hdf(path: str, additional_params: list[str]) -> dict:
    """Reads an HDF file from its path."""
    out = {"attributes": {}, "datasets": {}}
    file = h5py.File(path, "r")
    for atr in file.attrs.keys():
        out["attributes"][atr] = file.attrs[atr]
    out["datasets"] = _walk_hdf(file, additional_params)
    file.close()
    return out


class File:
    """
    Defines a loaded HDF file and implements methods to get data from the file.

    Note
    ----
    The ``File`` objects are created automatically when creating a
    :class:`Dataset` from a path. **They are generally not be used directly.**

    Parameters
    ----------
    path : str
        Path to the HDF file.
    additional_params : list of str, optional
        list of additional parameter names to extract from the files.

        .. note::

            If left to ``None`` only those parameters will be extracted:

            - VNA
            - VNA Average
            - VNA Power
            - VNA Bandwidth
            - VNA Frequency
            - Variable Attenuator
            - s21_real
            - s21_imag
            - s21_mag
            - s21_phase
            - Index
            - Magnet
    """

    def __init__(
        self, path: str, additional_params: Optional[list[str]] = None
    ) -> None:
        self.path = path
        self._additional_params = (
            additional_params if additional_params is not None else []
        )
        self._file_dict = _read_hdf(path, self._additional_params)
        self.infos = self._file_dict["attributes"]

        # Declare all possible parameters and s21_* parameters.
        self.vna_average = NullParameter()
        self.vna_bandwidth = NullParameter()
        self.vna_frequency = NullParameter()
        self.vna_power = NullParameter()
        self.digital_attenuation = NullParameter()
        self.magnetic_field = NullParameter()
        self.index = NullParameter()
        self.voltage_bias = NullParameter()
        self.s21_mag = NullParameter()
        self.s21_phase = NullParameter()
        self.s21_real = NullParameter()
        self.s21_imag = NullParameter()

        self._populate_params()

    def _populate_params(self) -> None:
        """
        Replaces the NullParameters for Parameters when they exist in the file
        """
        for key, value in self._file_dict["datasets"].items():
            if isinstance(value, dict):
                if len(value.keys()) == 1:
                    attribute = key.lower().replace(" ", "_")
                    parameter = Parameter(value[key], key)
                    self.with_param(attribute, parameter)
                elif key == "VNA":
                    self.with_param(
                        "vna_frequency",
                        Parameter(value["VNA Frequency"], "VNA Frequency"),
                    )
                    if "s21_real" in value.keys():
                        self.with_param(
                            "s21_real", Parameter(value["s21_real"], "s21_real")
                        )
                        self.with_param(
                            "s21_imag", Parameter(value["s21_imag"], "s21_imag")
                        )
                        mag, phase = convert_complex_to_magphase(
                            value["s21_real"], value["s21_imag"]
                        )
                        self.with_param("s21_mag", Parameter(mag, "s21_mag"))
                        self.with_param("s21_phase", Parameter(phase, "s21_phase"))
                    elif "s21_mag" in value.keys():
                        self.with_param(
                            "s21_mag", Parameter(value["s21_mag"], "s21_mag")
                        )
                        self.with_param(
                            "s21_phase", Parameter(value["s21_phase"], "s21_phase")
                        )
                        real, imag = convert_magphase_to_complex(
                            value["s21_mag"], value["s21_phase"]
                        )
                        self.with_param("s21_real", Parameter(real, "s21_real"))
                        self.with_param("s21_imag", Parameter(imag, "s21_imag"))
                    else:
                        raise NotImplementedError()
            else:
                attribute = key.lower().replace(" ", "_")
                self.with_param(attribute, Parameter(value, key))

    def with_param(self, attribute: str, parameter: Parameter) -> Self:
        """
        Adds a :class:`Parameter` to a file.

        Parameters
        ----------
        attribute : str
            Attribute name. The Parameter will be called using the
            ``File.attribute`` syntax.
        parameter : :class:`Parameter`
            The parameter to add.
        """
        self.__dict__.update({attribute: parameter})
        return self

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if not name.startswith("__") and isinstance(value, NullParameter):
            print(
                f"UserWarning: The attribute {name} is not defined for this file! Will return a NullParameter."
            )
        return value


def _load_files_from_path(
    path: str,
    additional_params: list[str],
) -> list[File]:
    """Loads multiple files from a directory, walking through it."""
    files = []
    for paths, _, _ in os.walk(path):
        for file in glob(os.path.join(paths, "*.hdf5")):
            file_obj = File(file, additional_params)
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
        Total attenuation present in the cryostat. Must be a negative number.
    additional_params : list of str, optional
        list of additional parameter names to extract from the files.

        .. note::

            If left to ``None`` only those parameters will be extracted:

            - VNA
            - VNA Average
            - VNA Power
            - VNA Bandwidth
            - VNA Frequency
            - Variable Attenuator
            - s21_real
            - s21_imag
            - s21_mag
            - s21_phase
            - Index
            - Magnet

    Attributes
    ----------
    f<i> : :class:`File`
        *Dynamically created* attributes allowing to retrieve specific files
        with the syntax ``myDataset.f0`` for the 0th file.
    files : dict
        Dictionary of the contained files with the keys being of the format
        ``"f<i>"`` where `<i>` represents the integer index of the file
        starting with 0.
    """

    files: dict[str, File] = {}

    def __init__(
        self,
        path: str,
        attenuation_cryostat: float,
        additional_params: Optional[list[str]] = None,
    ) -> None:
        if attenuation_cryostat > 0:
            raise ValueError("Attenuation must be negative")
        if Path(path).suffix == "":
            additional_params = (
                additional_params if additional_params is not None else []
            )
            self._files_list = _load_files_from_path(path, additional_params)
        else:
            self._files_list = [File(path, additional_params)]

        for i, file in enumerate(self._files_list):
            self.files.update({str(i): file})

    def __getattribute__(self, name: str) -> Any:
        if not name.startswith("__") and re.fullmatch(r"f\d+", name):
            return self.FILES[name.removeprefix("f")]
        return super().__getattribute__(name)
