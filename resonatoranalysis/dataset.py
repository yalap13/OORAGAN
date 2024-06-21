import os
import numpy as np
import h5py

from typing import Optional, Union
from os import PathLike
from glob import glob
from pathlib import Path
from tabulate import tabulate
from numpy.typing import NDArray, ArrayLike
from datetime import datetime

from .util import (
    calculate_power,
    strtime,
    convert_complex_to_dB,
    convert_magang_to_complex,
)


class Dataset:
    """
    General data set container extracting data and information on measurements
    from .hdf5 or .txt files.

    Parameters
    ----------
    path : str
        Path of the folder for multiple data files or for a single data file.
    attenuation_cryostat : float
        Total attenuation present on the cryostat. Must be a negative number.
    """

    def __init__(
        self,
        path: Union[str, PathLike],
        attenuation_cryostat: float,
    ) -> None:
        """
        General data set container extracting data and information on measurements
        from .hdf5 or .txt files.

        Parameters
        ----------
        path : str
            Path of the folder for multiple data files or for a single data file.
        attenuation_cryostat : float
            Total attenuation present on the cryostat. Must be a negative number.
        """
        if attenuation_cryostat > 0:
            raise ValueError("Attenuation value must be negative")
        self._data_container = HDF5Data(path, attenuation_cryostat)

    @property
    def data(self) -> dict | list:
        if len(self._data_container.files) == 1:
            return self._data_container.data[self.files]
        return self._data_container.data

    @property
    def cryostat_info(self) -> dict:
        if len(self._data_container.files) == 1:
            return self._data_container.cryostat_info[self.files]
        return self._data_container.cryostat_info

    @property
    def files(self) -> list:
        if len(self._data_container.files) == 1:
            return self._data_container.files[0]
        return self._data_container.files

    @property
    def vna_average(self) -> dict | ArrayLike:
        if len(self._data_container.files) == 1:
            return self._data_container.vna_average[self.files]
        return self._data_container.vna_average

    @property
    def vna_bandwidth(self) -> dict | ArrayLike:
        if len(self._data_container.files) == 1:
            return self._data_container.vna_bandwidth[self.files]
        return self._data_container.vna_bandwidth

    @property
    def vna_power(self) -> dict | ArrayLike:
        if len(self._data_container.files) == 1:
            return self._data_container.vna_power[self.files]
        return self._data_container.vna_power

    @property
    def variable_attenuator(self) -> dict | ArrayLike:
        if len(self._data_container.files) == 1:
            return self._data_container.variable_attenuator[self.files]
        return self._data_container.variable_attenuator

    @property
    def start_time(self) -> dict | ArrayLike:
        if len(self._data_container.files) == 1:
            return self._data_container.start_time[self.files]
        return self._data_container.start_time

    @property
    def end_time(self) -> dict | ArrayLike:
        if len(self._data_container.files) == 1:
            return self._data_container.end_time[self.files]
        return self._data_container.end_time

    @property
    def mixing_temp(self) -> dict | ArrayLike:
        if len(self._data_container.files) == 1:
            return self._data_container.mixing_temp[self.files]
        return self._data_container.mixing_temp

    @property
    def power(self) -> dict | ArrayLike:
        if len(self._data_container.files) == 1:
            return self._data_container.power[self.files]
        return self._data_container.power

    @property
    def frequency_range(self) -> dict | ArrayLike:
        if len(self._data_container.files) == 1:
            return self._data_container.frequency_range[self.files]
        return self._data_container.frequency_range

    def __str__(self) -> None:
        return self._data_container.__str__()

    def convert_magang_to_complex(self) -> None:
        self._data_container.convert_magang_to_complex()

    def convert_complex_to_dB(self) -> None:
        self._data_container.convert_complex_to_dB()


class HDF5Data:
    """
    Class representing a complete dataset extracted from a folder. Properties are automatically
    extracted from the data files.

    Parameters
    ----------
    path : str
        Path to folder containing data files or full path to specific data file.
    attenuation_cryostat : float
        Total attenuation present on the cryostat. Must be a negative number.
    file_extension : str
        File extension to search for in given path. Supports "hdf5", "txt" and "csv".
        Defaults to "hdf5".
    comments : str
        Defines the comment symbol in the data file. Defaults to "#".
    delimiter : str, optional
        Defines the .txt data file delimiter.

    Attributes
    ----------
    cryostat_info : dict[str, dict]
        Dictionnary in which the keys are the file paths and the values are a dictionnary of
        the cryostat temperature data.
    data : dict[str, list[NDArray]]
        Dictionnary in which the keys are the file paths and the values are the list of data
        arrays from this file.
    end_time : dict[str, time.struct_time]
        Dictionnary in which the keys are the file paths and the values are the end time of the
        measurement.
    files : list[str]
        List of the files path included in the dataset.
    frequency_range : dict[str, dict]
        Dictionnary in which the keys are the file paths and the values are a dictionnary containing
        the "start" and the "end" of the frequency range.
    mixing_temp : dict[str, float]
        Dictionnary in which the keys are the file paths and the values are the temperature of the
        mixing stage in Kelvins.
    power : dict[str, NDArray]
        Dictionnary in which the keys are the file paths and the values are an array of the values for
        the total power in dB.
    start_time : dict[str, time.struct_time]
        Dictionnary in which the keys are the file paths and the values are the start time of the
        measurement.
    variable_attenuator : dict[str, NDArray]
        Dictionnary in which the keys are the file paths and the values are an array of the values of
        attenuation on the variable attenuator in dB.
    vna_average : dict[str, NDArray]
        Dictionnary in which the keys are the file paths and the values are an array of the values for
        the VNA averaging number.
    vna_bandwidth : dict[str, NDArray]
        Dictionnary in which the keys are the file paths and the values are an array of the values for
        the VNA bandwidth in Hz.
    vna_power : dict[str, NDArray]
        Dictionnary in which the keys are the file paths and the values are an array of the values for
        the VNA output power in dB.
    """

    def __init__(
        self,
        path: str,
        attenuation_cryostat: float,
    ) -> None:
        """
        Class representing a complete dataset extracted from a folder. Properties are automatically
        extracted from the data files.

        Parameters
        ----------
        path : str
            Path to folder containing data files or full path to specific data file.
        attenuation_cryostat : float
            Total attenuation present on the cryostat. Must be a negative number.
        file_extension : str
            File extension to search for in given path. Supports "hdf5", "txt" and "csv".
            Defaults to "hdf5".
        comments : str
            Defines the comment symbol in the data file. Defaults to "#".
        delimiter : str, optional
            Defines the .txt data file delimiter.

        Attributes
        ----------
        cryostat_info : dict[str, dict]
            Dictionnary in which the keys are the file paths and the values are a dictionnary of
            the cryostat temperature data.
        data : dict[str, list[NDArray]]
            Dictionnary in which the keys are the file paths and the values are the list of data
            arrays from this file.
        end_time : dict[str, time.struct_time]
            Dictionnary in which the keys are the file paths and the values are the end time of the
            measurement.
        files : list[str]
            List of the files path included in the dataset.
        frequency_range : dict[str, dict]
            Dictionnary in which the keys are the file paths and the values are a dictionnary containing
            the "start" and the "end" of the frequency range.
        mixing_temp : dict[str, float]
            Dictionnary in which the keys are the file paths and the values are the temperature of the
            mixing stage in Kelvins.
        power : dict[str, NDArray]
            Dictionnary in which the keys are the file paths and the values are an array of the values for
            the total power in dB.
        start_time : dict[str, time.struct_time]
            Dictionnary in which the keys are the file paths and the values are the start time of the
            measurement.
        variable_attenuator : dict[str, NDArray]
            Dictionnary in which the keys are the file paths and the values are an array of the values of
            attenuation on the variable attenuator in dB.
        vna_average : dict[str, NDArray]
            Dictionnary in which the keys are the file paths and the values are an array of the values for
            the VNA averaging number.
        vna_bandwidth : dict[str, NDArray]
            Dictionnary in which the keys are the file paths and the values are an array of the values for
            the VNA bandwidth in Hz.
        vna_power : dict[str, NDArray]
            Dictionnary in which the keys are the file paths and the values are an array of the values for
            the VNA output power in dB.
        """
        self.data, self.files = self._get_data_from_hdf5(path)
        info = self._get_info_from_hdf5()
        self.vna_average = {
            key: info[key]["vna_info"]["VNA Average"] for key in self.files
        }
        self.vna_bandwidth = {
            key: info[key]["vna_info"]["VNA Bandwidth"] for key in self.files
        }
        self.vna_power = {key: info[key]["vna_info"]["VNA Power"] for key in self.files}
        self.variable_attenuator = {
            key: info[key]["vna_info"]["Variable Attenuator"] for key in self.files
        }
        self.cryostat_info = {key: info[key]["temps"] for key in self.files}
        self.start_time = {key: info[key]["temps"]["Started"] for key in self.files}
        self.end_time = {key: info[key]["temps"]["Ended"] for key in self.files}
        self.mixing_temp = {
            key: info[key]["temps"]["Temperature mixing LT End (Kelvin)"]
            for key in self.files
        }
        self.power = calculate_power(attenuation_cryostat, info)
        self.frequency_range = self._get_freq_range()

    def __str__(self) -> str:
        """
        Customized printing function.
        """
        output = "Files :\n"
        i = 1
        for file in self.files:
            output += f"  {i}. {file}\n"
            i += 1
        output += "File infos :\n"
        table = self._make_table_array()
        headers = [
            "File no.",
            "Start time",
            "End time",
            "Start freq. (GHz)",
            "Stop freq. (GHz)",
            "Power min (dB)",
            "Power max (dB)",
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
        end_arr = np.array(
            [
                datetime.fromtimestamp(strtime(self.end_time[file])).strftime(
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
        min_power_arr = np.array([np.min(self.power[file]) for file in self.files])
        max_power_arr = np.array([np.max(self.power[file]) for file in self.files])
        table = np.array(
            [
                file_no_arr,
                start_arr,
                end_arr,
                freq_start_arr / 1e9,
                freq_stop_arr / 1e9,
                min_power_arr,
                max_power_arr,
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

    def _get_data_from_hdf5(self, path: str) -> dict | list:
        """
        Utilitary function to get data from multiple hdf5 files at once.
        """
        if Path(path).suffix == "":
            files_list = []
            for paths, _, _ in os.walk(path):
                for file in glob(os.path.join(paths, "*.hdf5")):
                    files_list.append(file)
        else:
            files_list = [path]

        if len(files_list) == 0:
            raise FileNotFoundError("No files were found")
        elif len(files_list) > 1:
            print(f"Found {len(files_list)} files")

        data_dict = {}
        for file in files_list:
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
        return data_dict, files_list

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
