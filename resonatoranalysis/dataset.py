from typing import Optional, Union
from os import PathLike

from .file_handler import datapicker, gethdf5info
from .util import calculate_power, get_freq_info


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
        self.data = datapicker(path, file_extension, comments, delimiter)
        hdf5info = gethdf5info(path)
        self.files = hdf5info.keys()
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
        self.frequency_range = get_freq_info(self.data)
