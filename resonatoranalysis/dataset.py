import os
import numpy as np

from typing import Optional, Union
from os import PathLike
from glob import glob
from pathlib import Path
from resonator import background
from tabulate import tabulate
from numpy.typing import NDArray
from datetime import datetime

from .file_handler import datapicker, gethdf5info
from .util import calculate_power, strtime
from .analysis import fit_resonator_test


class Dataset:
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
        path: Union[str, PathLike],
        attenuation_cryostat: float,
        file_extension: str = "hdf5",
        comments: str = "#",
        delimiter: Optional[str] = None,
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
        """
        self.data, self.files = datapicker(path, file_extension, comments, delimiter)
        hdf5info = self._get_info_from_hdf5s(path)
        self.vna_average = {
            key: hdf5info[key]["vna_info"]["VNA Average"] for key in self.files
        }
        self.vna_bandwidth = {
            key: hdf5info[key]["vna_info"]["VNA Bandwidth"] for key in self.files
        }
        self.vna_power = {
            key: hdf5info[key]["vna_info"]["VNA Power"] for key in self.files
        }
        self.variable_attenuator = {
            key: hdf5info[key]["vna_info"]["Variable Attenuator"] for key in self.files
        }
        self.cryostat_info = {key: hdf5info[key]["temps"] for key in self.files}
        self.start_time = {key: hdf5info[key]["temps"]["Started"] for key in self.files}
        self.end_time = {key: hdf5info[key]["temps"]["Ended"] for key in self.files}
        self.mixing_temp = {
            key: hdf5info[key]["temps"]["Temperature mixing LT End (Kelvin)"]
            for key in self.files
        }
        self.power = calculate_power(attenuation_cryostat, hdf5info)
        self.frequency_range = self._get_freq_info()

    def __str__(self) -> str:
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

    def _get_info_from_hdf5s(self, path) -> dict[str, dict]:
        """
        Get all datasets info about VNA measurement apart from the VNA S21 data itself.
        Returns the info in a dictionary.

        Parameters
        ----------
        fname : str
            Full path to the HDF5 file.
        show : bool, optional
            Prints the keys found in the HDF5 file. The default is False.

        Returns
        -------
        {info : value, ...}

        """
        if Path(path).suffix == "":
            files_list = []
            for paths, _, _ in os.walk(path):
                for file in glob(os.path.join(paths, f"*.hdf5")):
                    files_list.append(file)
        else:
            files_list = [path]

        files_list.sort()

        if len(files_list) == 0:
            raise FileNotFoundError("No files were found")
        elif len(files_list) > 1:
            print(f"Found {len(files_list)} files")

        global_dict = {}
        for file in files_list:
            info_dict, atr_dict = gethdf5info(file)
            global_dict[file] = {"vna_info": info_dict, "temps": atr_dict}
        return global_dict

    def _get_freq_info(self):
        freq_info = {}
        for file in self.files:
            start = self.data[file][0][0, :][0]
            stop = self.data[file][0][0, :][-1]
            freq_info[file] = {"start": start, "stop": stop}
        return freq_info

    def fit_resonators(
        self,
        f_r=None,
        couploss=1e-6,
        intloss=1e-6,
        bg=background.MagnitudePhaseDelay(),
        savepic=False,
        savepath="",
        write=False,
        basepath=os.getcwd(),
        threshold=0.5,
        start=0,
        jump=10,
        nodialog=False,
    ):
        raise NotImplementedError("Not yet implemented")
