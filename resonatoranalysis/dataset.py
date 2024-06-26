import os
import numpy as np
import h5py
import re

from typing import Optional, Union
from os import PathLike
from glob import glob
from pathlib import Path
from tabulate import tabulate
from numpy.typing import NDArray, ArrayLike
from datetime import datetime
from copy import deepcopy

from .util import (
    strtime,
    convert_complex_to_dB,
    convert_magang_to_complex,
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
                    txt_files.append(file)
            if hdf5_files and txt_files:
                if file_extension is not None:
                    if file_extension == "hdf5":
                        self._data_container = HDF5Data(
                            hdf5_files, attenuation_cryostat
                        )
                        print(f"Found {len(hdf5_files)} files")
                    elif file_extension == "txt":
                        self._data_container = TXTData(
                            txt_files,
                            attenuation_cryostat,
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
                self._data_container = HDF5Data(hdf5_files, attenuation_cryostat)
                print(f"Found {len(hdf5_files)} files")
            elif txt_files:
                self._data_container = TXTData(
                    txt_files,
                    attenuation_cryostat,
                    comments=comments,
                    delimiter=delimiter,
                )
                print(f"Found {len(txt_files)} files")
            else:
                raise FileNotFoundError("No '.hdf5' or '.txt' files were found")
        elif Path(path).suffix == ".hdf5":
            self._data_container = HDF5Data([path], attenuation_cryostat)
        elif Path(path).suffix == ".txt":
            self._data_container = TXTData(
                [path], attenuation_cryostat, comments=comments, delimiter=delimiter
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

    def get_data(
        self,
        file_index: Optional[int | list[int]] = None,
        power: Optional[float | list[float]] = None,
    ) -> dict | ArrayLike:
        """
        Extracts either all data or, if indices are specified, a part of the data.

        Parameters
        ----------
        file_index : int | list[int], optional
            Index or list of indices (as displayed in the Dataset table) of files to
            get data from. Defaults to ``None``.
        power_index : int | list[int], optional
            If specified, will fetch data for those power values. Defaults to ``None``.
        """
        if file_index is None and power is None:
            return self.data
        return self._data_container.get_data(file_index=file_index, power=power)

    def convert_magang_to_complex(self) -> None:
        """
        Converts the Dataset's data from magnitude-angle to complex format.
        """
        self._data_container.convert_magang_to_complex()

    def convert_complex_to_dB(self, deg: bool = False) -> None:
        """
        Converts the Dataset's data from complex to dB.

        Parameters
        ----------
        deg : bool, optional
            If ``True`` the angle array will be in degrees. Defaults to ``False``,
            making the angles in radians.
        """
        self._data_container.convert_complex_to_dB(deg=deg)


class HDF5Data:
    """
    Data container for hdf5 files.

    Parameters
    ----------
    files_list : list
        List of files.
    attenuation_cryostat : float
        Total attenuation present on the cryostat. Must be a negative number.
    """

    def __init__(
        self,
        files_list: list,
        attenuation_cryostat: float,
    ) -> None:
        self.files = files_list
        self._file_index_dict = {str(i + 1): file for i, file in enumerate(self.files)}
        self.data = self._get_data_from_hdf5()
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
        self.start_time = {key: info[key]["temps"]["Started"] for key in self.files}
        self.end_time = {
            key: (
                info[key]["temps"]["Ended"] if "Ended" in info[key]["temps"] else None
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
        for i, file in enumerate(self.files):
            output += f"  {i+1}. {file}\n"
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

    def _make_table_array(self) -> NDArray:
        """
        Utilitary function used to generate the table for the customized
        printing function.
        """
        file_no_arr = np.array([i + 1 for i in range(len(self.files))])
        start_arr = np.array(
            [
                datetime.fromtimestamp(strtime(self.start_time[file])).strftime(
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
                    if len(self.power[file].shape) == 1
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
                            arr = np.stack((freq.T, mag[i].T, phase[i].T))
                            data.append(arr)
                    else:
                        arr = np.stack((freq.T, mag.T, phase.T))
                        data.append(arr)
                hdf5_data.close()
            data_dict[file] = data
        return data_dict

    def _calculate_power(self, att_cryo: float):
        powers = {}
        for file in self.files:
            powers[file] = (
                self.vna_power[file] + att_cryo - self.variable_attenuator[file]
            )
        return powers

    def get_data(
        self, file_index: int | list[int] = None, power: int | list[int] = None
    ) -> ArrayLike:
        """
        Extracts either all data or, if indices are specified, a part of the data.
        """
        output = {}
        if file_index is not None and power is None:
            try:
                if isinstance(file_index, list):
                    for fi in file_index:
                        self._file_index_dict[str(fi)]
                else:
                    self._file_index_dict[str(file_index)]
            except KeyError as err:
                print("Invalid file_index", err)
                return None
            if isinstance(file_index, list):
                for fi in file_index:
                    output[self._file_index_dict[str(fi)]] = self.data[
                        self._file_index_dict[str(fi)]
                    ]
                return output
            else:
                return self.data[self._file_index_dict[str(file_index)]]
        elif power is not None and file_index is None:
            for file in self.files:
                if isinstance(power, list):
                    file_data_temp = []
                    for p in power:
                        try:
                            file_data_temp.append(
                                self.data[file][np.where(self.power[file] == p)[0][0]]
                            )
                        except IndexError:
                            continue
                    if len(file_data_temp) > 0:
                        output[file] = file_data_temp
                else:
                    try:
                        output[file] = self.data[file][
                            np.where(self.power[file] == power)[0][0]
                        ]
                    except IndexError:
                        continue
            if output == {}:
                raise ValueError(f"No power value(s) {power} in any file")
            return output
        else:
            try:
                if isinstance(file_index, list):
                    for fi in file_index:
                        self._file_index_dict[str(fi)]
                else:
                    self._file_index_dict[str(file_index)]
            except KeyError as err:
                print("Invalid file_index", err)
                return None
            if isinstance(file_index, list):
                for fi in file_index:
                    if isinstance(power, list):
                        file_data_temp = []
                        for p in power:
                            try:
                                file_data_temp.append(
                                    self.data[self._file_index_dict[str(fi)]][
                                        np.where(
                                            self.power[self._file_index_dict[str(fi)]]
                                            == p
                                        )[0][0]
                                    ]
                                )
                            except IndexError:
                                continue
                        if len(file_data_temp) > 0:
                            output[self._file_index_dict[str(fi)]] = file_data_temp
                    else:
                        try:
                            output[self._file_index_dict[str(fi)]] = self.data[
                                self._file_index_dict[str(fi)]
                            ][
                                np.where(
                                    self.power[self._file_index_dict[str(fi)]] == power
                                )[0][0]
                            ]
                        except IndexError:
                            continue
            else:
                if isinstance(power, list):
                    file_data_temp = []
                    for p in power:
                        try:
                            file_data_temp.append(
                                self.data[self._file_index_dict[str(file_index)]][
                                    np.where(
                                        self.power[
                                            self._file_index_dict[str(file_index)]
                                        ]
                                        == p
                                    )[0][0]
                                ]
                            )
                        except IndexError:
                            continue
                    if len(file_data_temp) > 0:
                        output[self._file_index_dict[str(file_index)]] = file_data_temp
                else:
                    try:
                        output[self._file_index_dict[str(file_index)]] = self.data[
                            self._file_index_dict[str(file_index)]
                        ][
                            np.where(
                                self.power[self._file_index_dict[str(file_index)]]
                                == power
                            )[0][0]
                        ]
                    except IndexError:
                        pass
            if output == {}:
                raise ValueError(
                    f"No power value(s) {power} in any of the specified file"
                )
            return output

    def convert_complex_to_dB(self, deg: bool = False) -> None:
        """
        Converts the Dataset's data from complex to power in dB.
        """
        for file in self.files:
            for arr in self.data[file]:
                arr[1, :], arr[2, :] = convert_complex_to_dB(
                    arr[1, :], arr[2, :], deg=deg
                )

    def convert_magang_to_complex(self) -> None:
        """
        Converts the Dataset's data from magnitude and angle to complex.
        """
        for file in self.files:
            for arr in self.data[file]:
                complex = convert_magang_to_complex(arr)
                arr[1, :], arr[2, :] = np.real(complex), np.imag(complex)


class TXTData:
    """
    Data container for txt files.

    Parameters
    ----------
    files_list : list
        List of txt files.
    attenuation_cryostat : float
        Total attenuation present on the cryostat. Must be a negative number.
    comments : str, optional
        Character indicating a commented line in txt files. Defaults to "#".
    delimiter : str, optional
        Delimiter for the txt file columns. If ``None``, considers any whitespaces as
        delimiter. Defaults to ``None``.
    """

    def __init__(
        self,
        files_list: list,
        attenuation_cryostat: float,
        comments: str,
        delimiter: str,
    ) -> None:
        self.files = files_list
        self._sweep_info_files = []
        for file in self.files:
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
        self.vna_average = info["vna_average"]
        self.vna_bandwidth = info["vna_bandwidth"]
        self.vna_power = info["vna_power"]
        self.start_time = info["start_time"]
        files = self._sweep_info_files + self._standalone_files
        self.frequency_range = {
            key: {"start": info["start_freq"][key], "stop": info["stop_freq"][key]}
            for key in files
        }
        self.power = {
            key: (
                info["vna_power"][key] + attenuation_cryostat
                if info["vna_power"][key] is not None
                else None
            )
            for key in files
        }
        self._file_index_dict = {str(i + 1): file for i, file in enumerate(files)}

    def __str__(self) -> str:
        """
        Customized printing function.
        """
        output = "Files :\n"
        for i, file in enumerate(self._sweep_info_files):
            output += f"  {i+1}. {file}\n"
        sifnbr = len(self._sweep_info_files)
        for i, file in enumerate(self._standalone_files):
            output += f"  {i+sifnbr+1}. {file}\n"
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
        files = self._sweep_info_files + self._standalone_files
        file_no_arr = np.array([i + 1 for i in range(len(files))])
        start_arr = np.array(
            [
                (
                    datetime.fromtimestamp(self.start_time[file][0]).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if self.start_time[file] is not None
                    else None
                )
                for file in files
            ]
        )
        freq_start_arr = np.array(
            [self.frequency_range[file]["start"] for file in files]
        )
        freq_stop_arr = np.array([self.frequency_range[file]["stop"] for file in files])
        power_arr = np.array(
            [
                (
                    str(list(self.power[file])).lstrip("[").rstrip("]")
                    if self.power[file] is not None
                    else None
                )
                for file in files
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

    def _get_data_info_from_txt(self, comments: str, delimiter: str) -> dict:
        """
        Utilitary function extracting the data and VNA parameters.
        """
        data = {}
        info = {
            "start_time": {},
            "duration": {},
            "vna_average": {},
            "vna_bandwidth": {},
            "vna_power": {},
            "start_freq": {},
            "stop_freq": {},
        }
        standalone_files = deepcopy(self.files)
        for file in self._sweep_info_files:
            standalone_files.remove(file)
            sweep_files = [
                f for f in self.files if file.rstrip(".txt") in f and "readval" in f
            ]
            power_indices_present = []
            for sf in sweep_files:
                match = re.search(r"readval_(\d+)", sf)
                if match:
                    power_indices_present.append(match.group(1))
            sweep_params = np.loadtxt(file, comments=comments, delimiter=delimiter)
            sweep_data = []
            sweep_durations = []
            sweep_averages = []
            sweep_bandwidths = []
            for sf in sweep_files:
                standalone_files.remove(sf)
                arr = np.loadtxt(sf, comments=comments, delimiter=delimiter).T
                sweep_data.append(arr)
                sweep_file_info = self._parse_parameters(sf)
                sweep_durations.append(sweep_file_info["sweep_time"])
                sweep_averages.append(
                    sweep_file_info["average_count"]
                    if "average_count" in sweep_file_info
                    else sweep_file_info["sweep_average_count"]
                )
                sweep_bandwidths.append(sweep_file_info["bandwidth"])
                sweep_start_freq = sweep_file_info["freq_start"]
                sweep_stop_freq = sweep_file_info["freq_stop"]
            data[file] = sweep_data
            info["start_time"][file] = sweep_params[:, 2].T
            info["duration"][file] = np.array(sweep_durations)
            info["vna_average"][file] = np.array(sweep_averages)
            info["vna_bandwidth"][file] = np.array(sweep_bandwidths)
            info["vna_power"][file] = np.array(
                [sweep_params[int(i), 0] for i in power_indices_present]
            )
            info["start_freq"][file] = sweep_start_freq
            info["stop_freq"][file] = sweep_stop_freq
        self._standalone_files = standalone_files
        for file in standalone_files:
            data[file] = [np.loadtxt(file, comments=comments, delimiter=delimiter).T]
            file_info = self._parse_parameters(file)
            info["start_time"][file] = None
            info["duration"][file] = np.array([file_info["sweep_time"]])
            info["vna_average"][file] = (
                np.array([file_info["average_count"]])
                if "average_count" in file_info
                else np.array([file_info["sweep_average_count"]])
            )
            info["vna_bandwidth"][file] = np.array([file_info["bandwidth"]])
            # TODO: Verify if 'port_attenuation' really is the same as 'power_dbm_port1'
            info["vna_power"][file] = (
                np.array([file_info["power_dbm_port1"]])
                if "power_dbm_port1" in file_info
                else None
            )
            info["start_freq"][file] = file_info["freq_start"]
            info["stop_freq"][file] = file_info["freq_stop"]
        return data, info

    def get_data(
        self, file_index: int | list[int] = None, power: int | list[int] = None
    ) -> ArrayLike:
        """
        Extracts either all data or, if indices are specified, a part of the data.
        """
        output = {}
        files = self._sweep_info_files + self._standalone_files
        if file_index is not None and power is None:
            try:
                if isinstance(file_index, list):
                    for fi in file_index:
                        self._file_index_dict[str(fi)]
                else:
                    self._file_index_dict[str(file_index)]
            except KeyError as err:
                print("Invalid file_index", err)
                return None
            if isinstance(file_index, list):
                for fi in file_index:
                    output[self._file_index_dict[str(fi)]] = self.data[
                        self._file_index_dict[str(fi)]
                    ]
                return output
            else:
                return self.data[self._file_index_dict[str(file_index)]]
        elif power is not None and file_index is None:
            for file in files:
                if isinstance(power, list):
                    file_data_temp = []
                    for p in power:
                        try:
                            file_data_temp.append(
                                self.data[file][np.where(self.power[file] == p)[0][0]]
                            )
                        except IndexError as err:
                            print(err)
                            continue
                    if len(file_data_temp) > 0:
                        output[file] = file_data_temp
                else:
                    try:
                        output[file] = self.data[file][
                            np.where(self.power[file] == power)[0][0]
                        ]
                    except IndexError:
                        continue
            if output == {}:
                raise ValueError(f"No power value(s) {power} in any file")
            return output
        else:
            try:
                if isinstance(file_index, list):
                    for fi in file_index:
                        self._file_index_dict[str(fi)]
                else:
                    self._file_index_dict[str(file_index)]
            except KeyError as err:
                print("Invalid file_index", err)
                return None
            if isinstance(file_index, list):
                for fi in file_index:
                    if isinstance(power, list):
                        file_data_temp = []
                        for p in power:
                            try:
                                file_data_temp.append(
                                    self.data[self._file_index_dict[str(fi)]][
                                        np.where(
                                            self.power[self._file_index_dict[str(fi)]]
                                            == p
                                        )[0][0]
                                    ]
                                )
                            except IndexError:
                                continue
                        if len(file_data_temp) > 0:
                            output[self._file_index_dict[str(fi)]] = file_data_temp
                    else:
                        try:
                            output[self._file_index_dict[str(fi)]] = self.data[
                                self._file_index_dict[str(fi)]
                            ][
                                np.where(
                                    self.power[self._file_index_dict[str(fi)]] == power
                                )[0][0]
                            ]
                        except IndexError:
                            continue
            else:
                if isinstance(power, list):
                    file_data_temp = []
                    for p in power:
                        try:
                            file_data_temp.append(
                                self.data[self._file_index_dict[str(file_index)]][
                                    np.where(
                                        self.power[
                                            self._file_index_dict[str(file_index)]
                                        ]
                                        == p
                                    )[0][0]
                                ]
                            )
                        except IndexError:
                            continue
                    if len(file_data_temp) > 0:
                        output[self._file_index_dict[str(file_index)]] = file_data_temp
                else:
                    try:
                        output[self._file_index_dict[str(file_index)]] = self.data[
                            self._file_index_dict[str(file_index)]
                        ][
                            np.where(
                                self.power[self._file_index_dict[str(file_index)]]
                                == power
                            )[0][0]
                        ]
                    except IndexError:
                        pass
            if output == {}:
                raise ValueError(
                    f"No power value(s) {power} in any of the specified file"
                )
            return output

    def convert_complex_to_dB(self, deg: bool = False) -> None:
        """
        Converts the Dataset's data from complex to power in dB.
        """
        files = self._sweep_info_files + self._standalone_files
        for file in files:
            for arr in self.data[file]:
                arr[1, :], arr[2, :] = convert_complex_to_dB(
                    arr[1, :], arr[2, :], deg=deg
                )

    def convert_magang_to_complex(self) -> None:
        """
        Converts the Dataset's data from magnitude and angle to complex.
        """
        files = self._sweep_info_files + self._standalone_files
        for file in files:
            for arr in self.data[file]:
                complex = convert_magang_to_complex(arr)
                arr[1, :], arr[2, :] = np.real(complex), np.imag(complex)

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
