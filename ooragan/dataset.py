import os
import numpy as np
import h5py
import re

from typing import Optional, Union, Self
from os import PathLike
from glob import glob
from pathlib import Path
from tabulate import tabulate
from numpy.typing import NDArray, ArrayLike
from datetime import datetime
from copy import deepcopy
from warnings import warn

from .util import (
    strtime,
    convert_complex_to_magphase,
    convert_magphase_to_complex,
)


class Dataset:
    """
    General data container extracting data and information on measurements
    from .hdf5 or .txt files.

    Parameters
    ----------
    path : str
        Path of the folder for multiple data files or for a single data file.
    attenuation_cryostat : float
        Total attenuation present on the cryostat. Must be a negative number.
    print_out : bool, optional
        If ``True`` the dataset information will be printed. Defaults to ``True``.
    file_extension : str, optional
        Optional parameter to specify the file extension in the case where there is
        "hdf5" and "txt" files in the same directory.
    format : str, optional
        Specifies the format in which to load the data. Can be ``"complex"`` or
        ``"magphase"``. Defaults to ``"complex"``.
    comments : str, optional
        Character indicating a commented line in txt files. Defaults to "#".
    delimiter : str, optional
        Delimiter for the txt file columns. If ``None``, considers any whitespaces as
        delimiter. Defaults to ``None``.

    Attributes
    ----------
    cryostat_info : dict of dict
        Dictionnary in which the keys are the file paths and the values are a dictionnary of
        the cryostat temperature data.
    data : dict of list of NDArray or list of NDArray
        Dictionnary in which the keys are the file paths and the values are the list of data
        arrays from this file.
    end_time : dict of time.struct_time or time.struct_time
        Dictionnary in which the keys are the file paths and the values are the end time of the
        measurement.
    files : list of str or str
        List of the files path included in the dataset.
    frequency_range : dict of dict
        Dictionnary in which the keys are the file paths and the values are a dictionnary containing
        the "start" and the "end" of the frequency range.
    mixing_temp : dict of float or float
        Dictionnary in which the keys are the file paths and the values are the temperature of the
        mixing stage in Kelvins.
    power : dict of NDArray or NDArray
        Dictionnary in which the keys are the file paths and the values are an array of the values for
        the total power in dB.
    start_time : dict of time.struct_time or time.struct_time
        Dictionnary in which the keys are the file paths and the values are the start time of the
        measurement.
    variable_attenuator : dict of NDArray or NDArray
        Dictionnary in which the keys are the file paths and the values are an array of the values of
        attenuation on the variable attenuator in dB.
    vna_average : dict of NDArray or NDArray
        Dictionnary in which the keys are the file paths and the values are an array of the values for
        the VNA averaging number.
    vna_bandwidth : dict of NDArray or NDArray
        Dictionnary in which the keys are the file paths and the values are an array of the values for
        the VNA bandwidth in Hz.
    vna_power : dict of NDArray or NDArray
        Dictionnary in which the keys are the file paths and the values are an array of the values for
        the VNA output power in dB.

    Examples
    --------
    You can create a ``Dataset`` from a hdf5 or txt file simply by passing your file path as

    >>> from resonatoranalysis import Dataset
    >>> dataset = Dataset("path/to/your/file.hdf5", -80)
    Files :
      1. path/to/your/file.hdf5
    File infos :
      File no.  Start time             Start freq. (GHz)    Stop freq. (GHz)  Power (dB)                     Mixing temp. (K)
    ----------  -------------------  -------------------  ------------------  ---------------------------  ------------------
             1  2023-08-29 22:14:04              5.35653             5.36653  -100.0, -90.0, -80.0, -70.0           0.0154368

    You can also create a ``Dataset`` from a directory containing multiple hdf5 or txt files

    >>> dataset = Dataset("path/to/data/foder", -80)
    Found 4 files
    Files :
      1. path/to/data/folder/file_1.hdf5
      2. path/to/data/folder/file_2.hdf5
      3. path/to/data/folder/file_3.hdf5
      4. path/to/data/folder/file_4.hdf5
    File infos :
      File no.  Start time             Start freq. (GHz)    Stop freq. (GHz)  Power (dB)                             Mixing temp. (K)
    ----------  -------------------  -------------------  ------------------  -----------------------------------  ------------------
             1  2023-08-29 22:14:04              5.35653             5.36653  -100.0, -90.0, -80.0, -70.0                   0.0154368
             2  2023-08-31 01:48:44              4.89003             4.89053  -100.0, -90.0, -80.0, -70.0                   0.0136144
             3  2023-10-02 09:12:53              2                  18        -110.0, -100.0, -90.0, -80.0, -70.0
             4  2023-11-05 03:04:30              5.95844             5.96844  -110.0, -100.0, -90.0, -80.0                  0.0142297
    """

    def __init__(
        self,
        path: Union[str, PathLike],
        attenuation_cryostat: float,
        print_out: bool = True,
        file_extension: Optional[str] = None,
        format: str = "complex",
        comments: str = "#",
        delimiter: Optional[str] = None,
    ) -> None:
        if attenuation_cryostat > 0:
            raise ValueError("Attenuation value must be negative")

        if Path(path).suffix == "":
            hdf5_files = []
            txt_files = []
            for paths, _, _ in os.walk(path):
                for file in glob(os.path.join(paths, "*.hdf5")):
                    hdf5_files.append(file)
                for file in glob(os.path.join(paths, "*.txt")):
                    if "_results.txt" not in file:
                        txt_files.append(file)
            if hdf5_files and txt_files:
                if file_extension is not None:
                    if file_extension == "hdf5":
                        self._data_container = HDF5Data(
                            hdf5_files, attenuation_cryostat, format=format
                        )
                        print(f"Found {len(hdf5_files)} files")
                    elif file_extension == "txt":
                        self._data_container = TXTData(
                            txt_files,
                            attenuation_cryostat,
                            format=format,
                            comments=comments,
                            delimiter=delimiter,
                        )
                        print(f"Found {len(txt_files)} files")
                    else:
                        raise ValueError(
                            "Invalid file extension. Make sure you did not put a '.' before the extension"
                        )
                else:
                    raise RuntimeError(
                        "Both 'hdf5' and 'txt' files have been found in the provided path."
                        + "Optional parameter 'file_extension' must be specified"
                    )
            elif hdf5_files:
                self._data_container = HDF5Data(
                    hdf5_files, attenuation_cryostat, format=format
                )
                print(f"Found {len(hdf5_files)} files")
            elif txt_files:
                self._data_container = TXTData(
                    txt_files,
                    attenuation_cryostat,
                    format=format,
                    comments=comments,
                    delimiter=delimiter,
                )
                print(f"Found {len(txt_files)} files")
            else:
                raise FileNotFoundError("No '.hdf5' or '.txt' files were found")
        elif Path(path).suffix == ".hdf5":
            self._data_container = HDF5Data([path], attenuation_cryostat, format=format)
        elif Path(path).suffix == ".txt":
            self._data_container = TXTData(
                [path],
                attenuation_cryostat,
                format=format,
                comments=comments,
                delimiter=delimiter,
            )
        else:
            raise RuntimeError(
                f"Extension '.{Path(path).suffix}' not supported. Supported file types are '.hdf5' and '.txt'"
            )
        if print_out:
            print(self)

    @property
    def data(self) -> dict | list:
        """Raw data"""
        if len(self._data_container.files) == 1:
            return self._data_container.data[self.files]
        return self._data_container.data

    @property
    def format(self) -> str:
        """Data format, either ``"complex"`` or ``"magphase"``"""
        return self._data_container.format

    @property
    def cryostat_info(self) -> dict:
        """
        Various informations about the cryostat

        Warning
        -------
        Available only for Dataset created from hdf5 files.
        """
        if isinstance(self._data_container, HDF5Data):
            if len(self._data_container.files) == 1:
                return self._data_container.cryostat_info[self.files]
            return self._data_container.cryostat_info
        else:
            raise AttributeError(
                "Attribute 'cryostat_info' not defined for Dataset from txt files"
            )

    @property
    def files(self) -> list:
        """Files in the Dataset"""
        if len(self._data_container.files) == 1:
            return self._data_container.files[0]
        return self._data_container.files

    @property
    def vna_average(self) -> dict | ArrayLike:
        """VNA averaging count"""
        if len(self._data_container.files) == 1:
            return self._data_container.vna_average[self.files]
        return self._data_container.vna_average

    @property
    def vna_bandwidth(self) -> dict | ArrayLike:
        """VNA bandwith in Hz"""
        if len(self._data_container.files) == 1:
            return self._data_container.vna_bandwidth[self.files]
        return self._data_container.vna_bandwidth

    @property
    def vna_power(self) -> dict | ArrayLike:
        """VNA output power in dBm"""
        if len(self._data_container.files) == 1:
            return self._data_container.vna_power[self.files]
        return self._data_container.vna_power

    @property
    def variable_attenuator(self) -> dict | ArrayLike:
        """
        Attenuation value of the variable attenuator

        Warning
        -------
        Available only for Dataset created from hdf5 files.
        """
        if isinstance(self._data_container, HDF5Data):
            if len(self._data_container.files) == 1:
                return self._data_container.variable_attenuator[self.files]
            return self._data_container.variable_attenuator
        else:
            raise AttributeError(
                "Attribute 'variable_attenuator' not defined for Dataset from txt files"
            )

    @property
    def start_time(self) -> dict | ArrayLike:
        """Start time of the measurement"""
        if len(self._data_container.files) == 1:
            return self._data_container.start_time[self.files]
        return self._data_container.start_time

    @property
    def end_time(self) -> dict | ArrayLike:
        """
        End time of the measurement

        Warning
        -------
        Available only for Dataset created from hdf5 files.
        """
        if isinstance(self._data_container, HDF5Data):
            if len(self._data_container.files) == 1:
                return self._data_container.end_time[self.files]
            return self._data_container.end_time
        else:
            raise AttributeError(
                "Attribute 'end_time' not defined for Dataset from txt files"
            )

    @property
    def mixing_temp(self) -> dict | ArrayLike:
        """
        Mixing stage temperature

        Warning
        -------
        Available only for Dataset created from hdf5 files.
        """
        if isinstance(self._data_container, HDF5Data):
            if len(self._data_container.files) == 1:
                return self._data_container.mixing_temp[self.files]
            return self._data_container.mixing_temp
        else:
            raise AttributeError(
                "Attribute 'mixing_temp' not defined for Dataset from txt files"
            )

    @property
    def power(self) -> dict | ArrayLike:
        """
        Total power including VNA output power, physical attenuation in the setup and
        variable attenuator, if present. Given in dBm.
        """
        if len(self._data_container.files) == 1:
            return self._data_container.power[self.files]
        return self._data_container.power

    @property
    def frequency_range(self) -> dict | ArrayLike:
        """Start and stop frequency of the measurement"""
        if len(self._data_container.files) == 1:
            return self._data_container.frequency_range[self.files]
        return self._data_container.frequency_range

    def __str__(self) -> None:
        return self._data_container.__str__()

    def _is_empty(self) -> bool:
        """
        Returns ``True`` if the Dataset contains no data, ``False`` otherwise.
        """
        return self._data_container._is_empty()

    def __add__(self, other_dataset: Self) -> Self:
        self.convert_magphase_to_complex()
        other_dataset.convert_magphase_to_complex()
        data = {**self._data_container.data, **other_dataset._data_container.data}
        files = self._data_container.files + other_dataset._data_container.files
        vna_average = {
            **self._data_container.vna_average,
            **other_dataset._data_container.vna_average,
        }
        vna_bandwidth = {
            **self._data_container.vna_bandwidth,
            **other_dataset._data_container.vna_bandwidth,
        }
        vna_power = {
            **self._data_container.vna_power,
            **other_dataset._data_container.vna_power,
        }
        start_time = {
            **self._data_container.start_time,
            **other_dataset._data_container.start_time,
        }
        frequency_range = {
            **self._data_container.frequency_range,
            **other_dataset._data_container.frequency_range,
        }
        power = {**self._data_container.power, **other_dataset._data_container.power}
        variable_attenuator = {}
        cryostat_info = {}
        end_time = {}
        mixing_temp = {}
        try:
            variable_attenuator.update(self._data_container.variable_attenuator)
            cryostat_info.update(self._data_container.cryostat_info)
            end_time.update(self._data_container.end_time)
            mixing_temp.update(self._data_container.mixing_temp)
        except AttributeError:
            for file in self._data_container.files:
                variable_attenuator[file] = None
                cryostat_info[file] = None
                end_time[file] = None
                mixing_temp[file] = None
        try:
            variable_attenuator.update(
                other_dataset._data_container.variable_attenuator
            )
            cryostat_info.update(other_dataset._data_container.cryostat_info)
            end_time.update(other_dataset._data_container.end_time)
            mixing_temp.update(other_dataset._data_container.mixing_temp)
        except AttributeError:
            for file in other_dataset._data_container.files:
                variable_attenuator[file] = None
                cryostat_info[file] = None
                end_time[file] = None
                mixing_temp[file] = None
        new_dataset = deepcopy(self)
        new_dataset._data_container = AbstractData(
            data,
            files,
            "complex",
            vna_average,
            vna_bandwidth,
            vna_power,
            variable_attenuator,
            cryostat_info,
            start_time,
            end_time,
            mixing_temp,
            power,
            frequency_range,
        )
        return new_dataset

    def slice(
        self,
        file_index: int | list[int] = [],
        power: float | list[float] = [],
    ) -> Self:
        """
        Extracts a slice of the Dataset as a new Dataset.

        Parameters
        ----------
        file_index : int | list[int], optional
            Index or list of indices (as displayed in the Dataset table) of files to
            get data from. Defaults to ``[]``.
        power_index : int | list[int], optional
            If specified, will fetch data for those power values. Defaults to ``[]``.
        """
        if file_index == [] and power == []:
            return self
        new_dataset = deepcopy(self)
        new_dataset._data_container = self._data_container.slice(
            file_index=file_index, power=power
        )
        return new_dataset

    def convert_magphase_to_complex(self, deg: bool = False, dBm: bool = False) -> None:
        """
        Converts the Dataset's data from magnitude and phase to complex.

        Parameters
        ----------
        deg : bool, optional
            Set to ``True`` if the phase is in degrees. Defaults to ``False``.
        dBm : bool, optional
            Set to ``True`` if the magnitude is in dBm. Defaults to ``False``.
        """
        self._data_container.convert_magphase_to_complex(deg=deg, dBm=dBm)

    def convert_complex_to_magphase(self, deg: bool = False) -> None:
        """
        Converts the Dataset's data from complex to magnitude and phase.

        Parameters
        ----------
        deg : bool
            If ``True`` the phase is returned in degrees, else in radians.
            Defaults to ``False``.
        """
        self._data_container.convert_complex_to_magphase(deg=deg)


class HDF5Data:
    """
    Data container for hdf5 files.

    Parameters
    ----------
    files_list : list
        List of files.
    attenuation_cryostat : float
        Total attenuation present on the cryostat. Must be a negative number.
    format : str
        Specifies the format in which to load the data. Can be ``"complex"`` or
        ``"magphase"``. Defaults to ``"complex"``.
    """

    def __init__(
        self,
        files_list: list,
        attenuation_cryostat: float,
        format: str,
    ) -> None:
        self.format = format
        self.files = files_list
        self._file_index_dict = {str(i + 1): file for i, file in enumerate(self.files)}
        self.data = self._get_data_from_hdf5()
        if self.format == "magphase":
            self.convert_complex_to_magphase()
        info = self._get_info_from_hdf5()
        self.vna_average = {
            key: (
                info[key]["vna_info"]["VNA Average"]
                if "VNA Average" in info[key]["vna_info"]
                else None
            )
            for key in self.files
        }
        self.vna_bandwidth = {
            key: info[key]["vna_info"]["VNA Bandwidth"] for key in self.files
        }
        self.vna_power = {key: info[key]["vna_info"]["VNA Power"] for key in self.files}
        self.variable_attenuator = {
            key: (
                info[key]["vna_info"]["Variable Attenuator"]
                if "Variable Attenuator" in info[key]["vna_info"]
                else 0
            )
            for key in self.files
        }
        self.cryostat_info = {key: info[key]["temps"] for key in self.files}
        self.start_time = {
            key: strtime(info[key]["temps"]["Started"]) for key in self.files
        }
        self.end_time = {
            key: (
                strtime(info[key]["temps"]["Ended"])
                if "Ended" in info[key]["temps"]
                else None
            )
            for key in self.files
        }
        self.mixing_temp = {
            key: (
                info[key]["temps"]["Temperature mixing LT End (Kelvin)"]
                if "Temperature mixing LT End (Kelvin)" in info[key]["temps"]
                else None
            )
            for key in self.files
        }
        self.power = self._calculate_power(attenuation_cryostat)
        self.frequency_range = self._get_freq_range()

    def __str__(self) -> str:
        """
        Customized printing function.
        """
        output = "Files :\n"
        for k, v in self._file_index_dict.items():
            output += f"  {k}. {v}\n"
        output += "File infos :\n"
        table = self._make_table_array()
        headers = [
            "File no.",
            "Start time",
            "Start freq. (GHz)",
            "Stop freq. (GHz)",
            "Power (dB)",
            "Mixing temp. (K)",
        ]
        output += tabulate(table, headers)
        return output

    def _is_empty(self) -> bool:
        """
        Checks if the Dataset is empty.
        """
        return self.data == {}

    def _make_table_array(self) -> NDArray:
        """
        Utilitary function used to generate the table for the customized
        printing function.
        """
        file_no_arr = np.array([int(i) for i in self._file_index_dict.keys()])
        start_arr = np.array(
            [
                datetime.fromtimestamp(self.start_time[file]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                for file in self.files
            ]
        )
        freq_start_arr = np.array(
            [self.frequency_range[file]["start"] for file in self.files]
        )
        freq_stop_arr = np.array(
            [self.frequency_range[file]["stop"] for file in self.files]
        )
        mxc_temp_arr = np.array([self.mixing_temp[file] for file in self.files])
        power_arr = np.array(
            [
                (
                    str(list(self.power[file])).lstrip("[").rstrip("]")
                    if self.power[file].ndim == 1
                    else str(list(self.power[file][0])).lstrip("[").rstrip("]")
                )
                for file in self.files
            ]
        )
        table = np.array(
            [
                file_no_arr,
                start_arr,
                freq_start_arr / 1e9,
                freq_stop_arr / 1e9,
                power_arr,
                mxc_temp_arr,
            ]
        )
        return table.T

    def _get_info_from_hdf5(self) -> dict[str, dict]:
        """
        Utilitary function to get the metadata from the HDF5 files and store it into
        a dictionnary of the format {file path: info dictionnary}.
        """
        global_dict = {}
        for file in self.files:
            try:
                with h5py.File(file, "r") as f:
                    keylst = [key for key in f.keys()]
                    atrlst = [atr for atr in f.attrs.keys()]
                    keylst.remove("VNA")
                    info_dict = {element: None for element in keylst}
                    atr_dict = {element: None for element in atrlst}
                    for key in keylst:
                        try:
                            info_dict[key] = f[key][:]
                        except TypeError:
                            info_dict[key] = f[key][key][0]
                        except Exception as err:
                            print("Unexpected error : ", err)
                    for atr in atrlst:
                        try:
                            atr_dict[atr] = f.attrs[atr]
                        except TypeError:
                            try:
                                atr_dict[atr] = f.attrs[atr][:]
                            except TypeError:
                                atr_dict[atr] = f.attrs[atr][atr][0]
                        except Exception as err:
                            print("Unexpected error : ", err)
                global_dict[file] = {"vna_info": info_dict, "temps": atr_dict}
            except OSError:
                continue
        return global_dict

    def _get_freq_range(self) -> dict[str, dict]:
        """
        Utilitary function used to get the frequency range for the measurement.
        """
        freq_info = {}
        for file in self.files:
            start = self.data[file][0][0, :][0]
            stop = self.data[file][0][0, :][-1]
            freq_info[file] = {"start": start, "stop": stop}
        return freq_info

    def _get_data_from_hdf5(self) -> dict | list:
        """
        Utilitary function to get data from multiple hdf5 files at once.
        """
        data_dict = {}
        for file in self.files:
            data = []
            try:
                with h5py.File(file, "r") as hdf5_data:
                    freq = hdf5_data["VNA"]["VNA Frequency"][:]
                    try:
                        real = np.squeeze(hdf5_data["VNA"]["s21_real"][:])
                        imag = np.squeeze(hdf5_data["VNA"]["s21_imag"][:])
                        if real.ndim > 1:
                            for i in range(len(real)):
                                arr = np.stack((freq.T, real[i].T, imag[i].T))
                                data.append(arr)
                        else:
                            arr = np.stack((freq.T, real.T, imag.T))
                            data.append(arr)
                    except KeyError:
                        mag = np.squeeze(hdf5_data["VNA"]["s21_mag"][:])
                        phase = np.squeeze(hdf5_data["VNA"]["s21_phase"][:])
                        if mag.ndim > 1:
                            for i in range(len(mag)):
                                real, imag = convert_magphase_to_complex(
                                    mag[i], phase[i]
                                )
                                arr = np.stack((freq.T, real[i].T, imag[i].T))
                                data.append(arr)
                        else:
                            real, imag = convert_magphase_to_complex(mag, phase)
                            arr = np.stack((freq.T, real.T, imag.T))
                            data.append(arr)
                    hdf5_data.close()
            except OSError:
                warn(
                    f"File {file} could not be opened as it was not closed properly."
                    + "To resolve, see the 'h5clear' command from the HDF5 library."
                )
            data_dict[file] = data
        return data_dict

    def _calculate_power(self, att_cryo: float):
        powers = {}
        for file in self.files:
            powers[file] = (
                self.vna_power[file] + att_cryo - self.variable_attenuator[file]
            )
        return powers

    def slice(
        self,
        file_index: int | list[int] = [],
        power: int | list[int] = [],
    ) -> Self:
        """
        Extracts a slice of the Dataset as a new Dataset.
        """
        if not isinstance(file_index, list):
            file_index = [file_index]
        if not isinstance(power, list):
            power = [power]
        slice = deepcopy(self)
        if file_index != []:
            try:
                for fi in file_index:
                    self._file_index_dict[str(fi)]
            except KeyError as err:
                raise IndexError(f"Invalid file_index: {err}")
            inverse_files = [
                self._file_index_dict[i]
                for i in self._file_index_dict.keys()
                if int(i) not in file_index
            ]
        else:
            inverse_files = []
        inverted_file_dict = {val: key for key, val in self._file_index_dict.items()}
        for file in self.files:
            to_remove = (
                list(self.power[file])
                if np.squeeze(self.power[file]).shape == ()
                else list(np.squeeze(self.power[file]))
            )
            any_found = False
            if power != []:
                for p in power:
                    if p in self.power[file]:
                        to_remove.remove(p)
                        any_found = True
            else:
                to_remove = []
                any_found = True
            if not any_found or file in inverse_files:
                slice.files.remove(file)
                slice._file_index_dict.pop(inverted_file_dict[file])
                slice.data.pop(file)
                slice.vna_average.pop(file)
                slice.vna_bandwidth.pop(file)
                slice.vna_power.pop(file)
                slice.variable_attenuator.pop(file)
                slice.cryostat_info.pop(file)
                slice.start_time.pop(file)
                slice.end_time.pop(file)
                slice.mixing_temp.pop(file)
                slice.power.pop(file)
                slice.frequency_range.pop(file)
            else:
                idx_to_remove = []
                for p in sorted(to_remove, reverse=True):
                    idx = np.where(np.squeeze(self.power[file]) == p)[0][0]
                    slice.data[file].pop(idx)
                    idx_to_remove.append(idx)
                try:
                    slice.vna_average[file] = np.delete(
                        slice.vna_average[file], idx_to_remove
                    )
                except IndexError:
                    pass
                try:
                    slice.vna_bandwidth[file] = np.delete(
                        slice.vna_bandwidth[file], idx_to_remove
                    )
                except IndexError:
                    pass
                try:
                    slice.vna_power[file] = np.delete(
                        slice.vna_power[file], idx_to_remove
                    )
                except IndexError:
                    pass
                try:
                    slice.variable_attenuator[file] = np.delete(
                        slice.variable_attenuator[file], idx_to_remove
                    )
                except IndexError:
                    pass
                try:
                    slice.power[file] = np.delete(slice.power[file], idx_to_remove)
                except IndexError:
                    pass
        if slice._is_empty():
            raise ValueError(f"No file contains the specified power values {power}")
        return slice

    def convert_complex_to_magphase(self, deg: bool = False) -> None:
        """
        Converts the Dataset's data from complex to magnitude and phase.

        Parameters
        ----------
        deg : bool
            If ``True`` the phase is returned in degrees, else in radians.
            Defaults to ``False``.
        """
        if self.format == "magphase":
            return
        for file in self.files:
            for arr in self.data[file]:
                arr[1, :], arr[2, :] = convert_complex_to_magphase(
                    arr[1, :], arr[2, :], deg=deg
                )
        self.format = "magphase"

    def convert_magphase_to_complex(self, deg: bool = False, dBm: bool = False) -> None:
        """
        Converts the Dataset's data from magnitude and phase to complex.

        Parameters
        ----------
        deg : bool, optional
            Set to ``True`` if the phase is in degrees. Defaults to ``False``.
        dBm : bool, optional
            Set to ``True`` if the magnitude is in dBm. Defaults to ``False``.
        """
        if self.format == "complex":
            return
        for file in self.files:
            for arr in self.data[file]:
                arr[1, :], arr[2, :] = convert_magphase_to_complex(
                    arr[1, :], arr[2, :], deg=deg, dBm=dBm
                )
        self.format = "complex"


class TXTData:
    """
    Data container for txt files.

    Parameters
    ----------
    files_list : list
        List of txt files.
    attenuation_cryostat : float
        Total attenuation present on the cryostat. Must be a negative number.
    format : str
        Specifies the format in which to load the data. Can be ``"complex"`` or
        ``"magphase"``. Defaults to ``"complex"``.
    comments : str
        Character indicating a commented line in txt files. Defaults to "#".
    delimiter : str
        Delimiter for the txt file columns. If ``None``, considers any whitespaces as
        delimiter. Defaults to ``None``.
    """

    def __init__(
        self,
        files_list: list,
        attenuation_cryostat: float,
        format: str,
        comments: str,
        delimiter: str,
    ) -> None:
        self.format = format
        self._all_files = files_list
        self._sweep_info_files = []
        for file in self._all_files:
            if not "readval" in file:
                with open(file, "r") as f:
                    for line in f:
                        if line.startswith("#"):
                            if line.endswith("time\n"):
                                self._sweep_info_files.append(file)
                                break
                        else:
                            break
                    f.close()
        self.data, info = self._get_data_info_from_txt(
            comments=comments, delimiter=delimiter
        )
        self.files = self._sweep_info_files + self._standalone_files
        if self.format == "magphase":
            self.convert_complex_to_magphase()
        self.vna_average = info["vna_average"]
        self.vna_bandwidth = info["vna_bandwidth"]
        self.vna_power = info["vna_power"]
        self.start_time = info["start_time"]
        self.frequency_range = {
            key: {"start": info["start_freq"][key], "stop": info["stop_freq"][key]}
            for key in self.files
        }
        self.power = {
            key: (
                info["vna_power"][key] + attenuation_cryostat
                if info["vna_power"][key] is not None
                else None
            )
            for key in self.files
        }
        self._file_index_dict = {str(i + 1): file for i, file in enumerate(self.files)}

    def _is_empty(self) -> bool:
        """
        Checks if the Dataset is empty.
        """
        return self.data == {}

    def __str__(self) -> str:
        """
        Customized printing function.
        """
        output = "Files :\n"
        for k, v in self._file_index_dict.items():
            output += f"  {k}. {v}\n"
        output += "File infos :\n"
        table = self._make_table_array()
        headers = [
            "File no.",
            "Start time",
            "Start freq. (GHz)",
            "Stop freq. (GHz)",
            "Power (dB)",
        ]
        output += tabulate(table, headers)
        return output

    def _make_table_array(self) -> NDArray:
        """
        Utilitary function used to generate the table for the customized
        printing function.
        """
        file_no_arr = np.array([int(i) for i in self._file_index_dict.keys()])
        start_arr = np.array(
            [
                (
                    datetime.fromtimestamp(self.start_time[file]).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if self.start_time[file] is not None
                    else None
                )
                for file in self.files
            ]
        )
        freq_start_arr = np.array(
            [self.frequency_range[file]["start"] for file in self.files]
        )
        freq_stop_arr = np.array(
            [self.frequency_range[file]["stop"] for file in self.files]
        )
        power_arr = np.array(
            [
                (
                    str(list(self.power[file])).lstrip("[").rstrip("]")
                    if self.power[file] is not None
                    else None
                )
                for file in self.files
            ]
        )
        table = np.array(
            [
                file_no_arr,
                start_arr,
                freq_start_arr / 1e9,
                freq_stop_arr / 1e9,
                power_arr,
            ]
        )
        return table.T

    def _get_data_info_from_txt(
        self, comments: str, delimiter: str
    ) -> tuple[dict, dict]:
        """
        Utilitary function extracting the data and VNA parameters.
        """
        data = {}
        info = {
            "start_time": {},
            "vna_average": {},
            "vna_bandwidth": {},
            "vna_power": {},
            "start_freq": {},
            "stop_freq": {},
        }
        standalone_files = deepcopy(self._all_files)
        for file in self._sweep_info_files:
            standalone_files.remove(file)
            sweep_files = [
                f
                for f in self._all_files
                if file.rstrip(".txt") in f and "readval" in f
            ]
            sweep_params = np.loadtxt(file, comments=comments, delimiter=delimiter)
            sweep_data = []
            sweep_powers = []
            sweep_averages = []
            sweep_bandwidths = []
            for sf in sweep_files:
                standalone_files.remove(sf)
                sweep_file_info = self._parse_parameters(sf)
                try:
                    sweep_powers.append(
                        float(sweep_file_info["port_power_level_dBm"][0])
                        if "port_power_level_dBm" in sweep_file_info
                        else float(sweep_file_info["power_dbm_port1"])
                    )
                except KeyError:
                    pass
                sweep_averages.append(
                    sweep_file_info["average_count"]
                    if "average_count" in sweep_file_info
                    else sweep_file_info["sweep_average_count"]
                )
                sweep_bandwidths.append(sweep_file_info["bandwidth"])
                sweep_start_freq = sweep_file_info["freq_start"]
                sweep_stop_freq = sweep_file_info["freq_stop"]
                arr = np.loadtxt(sf, comments=comments, delimiter=delimiter).T
                try:
                    if sweep_file_info["options"]["unit"] == "db_deg":
                        arr[1, :], arr[2, :] = convert_magphase_to_complex(
                            arr[1, :],
                            arr[2, :],
                            deg=True,
                            dBm=True,
                        )
                except KeyError:
                    continue
                sweep_data.append(arr)
            data[file] = sweep_data
            info["start_time"][file] = (
                sweep_params[0, 2] if sweep_params.shape != (0,) else None
            )
            info["vna_average"][file] = np.array(sweep_averages)
            info["vna_bandwidth"][file] = np.array(sweep_bandwidths)
            info["vna_power"][file] = (
                np.array(sweep_powers) if sweep_powers != [] else None
            )
            info["start_freq"][file] = sweep_start_freq
            info["stop_freq"][file] = sweep_stop_freq
        self._standalone_files = standalone_files
        for file in standalone_files:
            file_info = self._parse_parameters(file)
            info["start_time"][file] = None
            info["vna_average"][file] = (
                np.array([file_info["average_count"]])
                if "average_count" in file_info
                else np.array([file_info["sweep_average_count"]])
            )
            info["vna_bandwidth"][file] = np.array([file_info["bandwidth"]])
            try:
                info["vna_power"][file] = (
                    np.array([file_info["power_dbm_port1"]])
                    if "power_dbm_port1" in file_info
                    else np.array([file_info["port_power_level_dBm"]])
                )
            except KeyError:
                info["vna_power"][file] = None
            info["start_freq"][file] = file_info["freq_start"]
            info["stop_freq"][file] = file_info["freq_stop"]
            arr = np.loadtxt(file, comments=comments, delimiter=delimiter).T
            try:
                if file_info["options"]["unit"] == "db_deg":
                    arr[1, :], arr[2, :] = convert_magphase_to_complex(
                        arr[1, :],
                        arr[2, :],
                        deg=True,
                        dBm=True,
                    )
            except KeyError:
                pass
            data[file] = [arr]
        return data, info

    def slice(
        self,
        file_index: int | list[int] = [],
        power: int | list[int] = [],
    ) -> Self:
        """
        Extracts a slice of the Dataset as a new Dataset.
        """
        if not isinstance(file_index, list):
            file_index = [file_index]
        if not isinstance(power, list):
            power = [power]
        slice = deepcopy(self)
        if file_index != []:
            try:
                for fi in file_index:
                    self._file_index_dict[str(fi)]
            except KeyError as err:
                raise IndexError(f"Invalid file_index: {err}")
            inverse_files = [
                self._file_index_dict[i]
                for i in self._file_index_dict.keys()
                if int(i) not in file_index
            ]
        else:
            inverse_files = []
        inverted_file_dict = {val: key for key, val in self._file_index_dict.items()}
        for file in self.files:
            any_found = False
            if self.power[file] is not None:
                to_remove = (
                    list(self.power[file])
                    if np.squeeze(self.power[file]).shape == ()
                    else list(np.squeeze(self.power[file]))
                )
                if power != []:
                    for p in power:
                        if p in self.power[file]:
                            to_remove.remove(p)
                            any_found = True
                else:
                    to_remove = []
                    any_found = True
            elif self.power[file] is None and power == []:
                any_found = True
            if not any_found or file in inverse_files:
                slice.files = [f for f in slice.files if file[:-4] not in f]
                try:
                    slice._sweep_info_files.remove(file)
                except:
                    slice._standalone_files.remove(file)
                slice._file_index_dict.pop(inverted_file_dict[file])
                slice.data.pop(file)
                slice.vna_average.pop(file)
                slice.vna_bandwidth.pop(file)
                slice.vna_power.pop(file)
                slice.start_time.pop(file)
                slice.power.pop(file)
                slice.frequency_range.pop(file)
            else:
                idx_to_remove = []
                for p in sorted(to_remove, reverse=True):
                    idx = np.where(np.squeeze(self.power[file]) == p)[0][0]
                    slice.data[file].pop(idx)
                    idx_to_remove.append(idx)
                try:
                    slice.vna_average[file] = np.delete(
                        slice.vna_average[file], idx_to_remove
                    )
                except IndexError:
                    pass
                try:
                    slice.vna_bandwidth[file] = np.delete(
                        slice.vna_bandwidth[file], idx_to_remove
                    )
                except IndexError:
                    pass
                try:
                    slice.vna_power[file] = np.delete(
                        slice.vna_power[file], idx_to_remove
                    )
                except IndexError:
                    pass
                try:
                    slice.power[file] = (
                        np.delete(slice.power[file], idx_to_remove)
                        if slice.power[file] is not None
                        else None
                    )
                except IndexError:
                    pass
        if slice._is_empty():
            raise ValueError(f"No file contains the specified power values {power}")
        return slice

    def convert_complex_to_magphase(self, deg: bool = False) -> None:
        """
        Converts the Dataset's data from complex to magnitude and phase.

        Parameters
        ----------
        deg : bool
            If ``True`` the phase is returned in degrees, else in radians.
            Defaults to ``False``.
        """
        if self.format == "magphase":
            return
        files = self._sweep_info_files + self._standalone_files
        for file in files:
            for arr in self.data[file]:
                arr[1, :], arr[2, :] = convert_complex_to_magphase(
                    arr[1, :], arr[2, :], deg=deg
                )
        self.format = "magphase"

    def convert_magphase_to_complex(self, deg: bool = False, dBm: bool = False) -> None:
        """
        Converts the Dataset's data from magnitude and phase to complex.

        Parameters
        ----------
        deg : bool, optional
            Set to ``True`` if the phase is in degrees. Defaults to ``False``.
        dBm : bool, optional
            Set to ``True`` if the magnitude is in dBm. Defaults to ``False``.
        """
        if self.format == "complex":
            return
        files = self._sweep_info_files + self._standalone_files
        for file in files:
            for arr in self.data[file]:
                arr[1, :], arr[2, :] = convert_magphase_to_complex(
                    arr[1, :], arr[2, :], deg=deg, dBm=dBm
                )
        self.format = "complex"

    def _parse_parameters(self, file_path: str) -> dict:
        """
        Utilitary function to parse the information given in commented lines of txt files.
        """
        parameters = {}

        with open(file_path, "r") as file:
            lines = file.readlines()
            file.close()

        # Regular expression to match the parameter lines
        pattern = re.compile(r"^#(?P<key>[^=]+)=(?P<value>.+)$")

        for line in lines:
            match = pattern.match(line.strip())
            if match:
                key = match.group("key").strip()
                value = match.group("value").strip()

                # Convert the value to the appropriate type
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif re.match(r"^-?\d+\.?\d*$", value):  # match integers and floats
                    value = float(value) if "." in value else int(value)
                elif value.startswith("[") and value.endswith("]"):
                    value = value[1:-1].split(",")
                    value = [v.strip() for v in value]
                elif value.startswith("{") and value.endswith("}"):
                    value = eval(value)  # Evaluate dictionaries, assuming trusted input
                else:
                    value = value.strip('"')

                parameters[key] = value

        return parameters


class AbstractData:
    """
    Data container for a mix of different file formats.

    Note
    ----
    This data container class does not extract data from file as the other data
    container classes do. It is mearly intended to be used when merging two
    previously created Datasets from different file formats.

    Parameters
    ----------
    data : dict
        Data for each file.
    files : list
        Files contained in the dataset.
    format : str
        Data format. Can be ``"complex"`` or ``"magphase"``.
    vna_average : dict
        VNA averaging for each file.
    vna_bandwidth : dict
        VNA bandwidth for each file.
    vna_power : dict
        VNA output power for each file.
    variable_attenuator : dict
        Variable attenuator value for each file.
    cryostat_info : dict
        Crystat information for each file.
    start_time : dict
        Measurement start time for each file.
    end_time : dict
        Measurement end time for each file.
    mixing_temp : dict
        Temperature of the mixing stage for each file.
    power : dict
        Total input power in the device for each file.
    frequency_range : dict
        Frequency range for each file.
    """

    def __init__(
        self,
        data: dict,
        files: list,
        format: str,
        vna_average: dict,
        vna_bandwidth: dict,
        vna_power: dict,
        variable_attenuator: dict,
        cryostat_info: dict,
        start_time: dict,
        end_time: dict,
        mixing_temp: dict,
        power: dict,
        frequency_range: dict,
    ) -> None:
        self.data = data
        self.files = files
        self.format = format
        self.vna_average = vna_average
        self.vna_bandwidth = vna_bandwidth
        self.vna_power = vna_power
        self.variable_attenuator = variable_attenuator
        self.cryostat_info = cryostat_info
        self.start_time = start_time
        self.end_time = end_time
        self.mixing_temp = mixing_temp
        self.power = power
        self.frequency_range = frequency_range
        self._file_index_dict = {str(i + 1): file for i, file in enumerate(self.files)}

    def __str__(self):
        """
        Customized printing method
        """
        output = "Files :\n"
        for k, v in self._file_index_dict.items():
            output += f"  {k}. {v}\n"
        output += "File infos :\n"
        table = self._make_table_array()
        headers = [
            "File no.",
            "Start time",
            "Start freq. (GHz)",
            "Stop freq. (GHz)",
            "Power (dBm)",
        ]
        output += tabulate(table, headers)
        return output

    def _make_table_array(self):
        """
        Utilitary function used to generate the table for the customized
        printing function.
        """
        file_no_arr = np.array([int(i) for i in self._file_index_dict.keys()])
        start_arr = np.array(
            [
                (
                    datetime.fromtimestamp(self.start_time[file]).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if self.start_time[file] is not None
                    else None
                )
                for file in self.files
            ]
        )
        freq_start_arr = np.array(
            [self.frequency_range[file]["start"] for file in self.files]
        )
        freq_stop_arr = np.array(
            [self.frequency_range[file]["stop"] for file in self.files]
        )
        power_arr = []
        for file in self.files:
            if self.power[file] is not None:
                if self.power[file].ndim == 1:
                    power_arr.append(
                        str(list(self.power[file])).lstrip("[").rstrip("]")
                    )
                else:
                    power_arr.append(
                        str(list(self.power[file][0])).lstrip("[").rstrip("]")
                    )
            else:
                power_arr.append(None)
        power_arr = np.array(power_arr)
        table = np.array(
            [
                file_no_arr,
                start_arr,
                freq_start_arr / 1e9,
                freq_stop_arr / 1e9,
                power_arr,
            ]
        )
        return table.T

    def _is_empty(self) -> bool:
        """
        Checks if the Dataset is empty.
        """
        return self.data == {}

    def slice(self, file_index: int | list[int], power: float | list[float]) -> Self:
        """
        Extracts a slice of the Dataset as a new Dataset.
        """
        if not isinstance(file_index, list):
            file_index = [file_index]
        if not isinstance(power, list):
            power = [power]
        slice = deepcopy(self)
        if file_index != []:
            try:
                for fi in file_index:
                    self._file_index_dict[str(fi)]
            except KeyError as err:
                raise IndexError(f"Invalid file_index: {err}")
            inverse_files = [
                self._file_index_dict[i]
                for i in self._file_index_dict.keys()
                if int(i) not in file_index
            ]
        else:
            inverse_files = []
        inverted_file_dict = {val: key for key, val in self._file_index_dict.items()}
        for file in self.files:
            any_found = False
            if self.power[file] is not None:
                to_remove = (
                    list(self.power[file])
                    if np.squeeze(self.power[file]).shape == ()
                    else list(np.squeeze(self.power[file]))
                )
                if power != []:
                    for p in power:
                        if p in self.power[file]:
                            to_remove.remove(p)
                            any_found = True
                else:
                    to_remove = []
                    any_found = True
            elif self.power[file] is None and power == []:
                any_found = True
            if not any_found or file in inverse_files:
                slice.files = [f for f in slice.files if file[:-4] not in f]
                slice._file_index_dict.pop(inverted_file_dict[file])
                slice.data.pop(file)
                slice.vna_average.pop(file)
                slice.vna_bandwidth.pop(file)
                slice.vna_power.pop(file)
                slice.start_time.pop(file)
                slice.power.pop(file)
                slice.frequency_range.pop(file)
            else:
                idx_to_remove = []
                for p in sorted(to_remove, reverse=True):
                    idx = np.where(np.squeeze(self.power[file]) == p)[0][0]
                    slice.data[file].pop(idx)
                    idx_to_remove.append(idx)
                try:
                    slice.vna_average[file] = np.delete(
                        slice.vna_average[file], idx_to_remove
                    )
                except IndexError:
                    pass
                try:
                    slice.vna_bandwidth[file] = np.delete(
                        slice.vna_bandwidth[file], idx_to_remove
                    )
                except IndexError:
                    pass
                try:
                    slice.vna_power[file] = np.delete(
                        slice.vna_power[file], idx_to_remove
                    )
                except IndexError:
                    pass
                try:
                    slice.power[file] = (
                        np.delete(slice.power[file], idx_to_remove)
                        if slice.power[file] is not None
                        else None
                    )
                except IndexError:
                    pass
        if slice._is_empty():
            raise ValueError(f"No file contains the specified power values {power}")
        return slice

    def convert_complex_to_magphase(self, deg: bool = False) -> None:
        """
        Converts the Dataset's data from complex to magnitude and phase.

        Parameters
        ----------
        deg : bool
            If ``True`` the phase is returned in degrees, else in radians.
            Defaults to ``False``.
        """
        if self.format == "magphase":
            return
        for file in self.files:
            for arr in self.data[file]:
                arr[1, :], arr[2, :] = convert_complex_to_magphase(
                    arr[1, :], arr[2, :], deg=deg
                )
        self.format = "magphase"

    def convert_magphase_to_complex(self, deg: bool = False, dBm: bool = False) -> None:
        """
        Converts the Dataset's data from magnitude and phase to complex.

        Parameters
        ----------
        deg : bool, optional
            Set to ``True`` if the phase is in degrees. Defaults to ``False``.
        dBm : bool, optional
            Set to ``True`` if the magnitude is in dBm. Defaults to ``False``.
        """
        if self.format == "complex":
            return
        for file in self.files:
            for arr in self.data[file]:
                arr[1, :], arr[2, :] = convert_magphase_to_complex(
                    arr[1, :], arr[2, :], deg=deg, dBm=dBm
                )
        self.format = "complex"
