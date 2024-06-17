"""
Much of the contents of this file has been written by Gabriel Ouellet, 
fellow master's student. All his code is available on the JosePh Gitlab.
"""

import glob
import os
import numpy as np
import h5py

from pathlib import Path
from typing import Optional, Union
from os import PathLike
from numpy.typing import NDArray

from .util import choice


def datapicker(
    path: Union[str, PathLike],
    comments: str = "#",
    delimiter: Optional[str] = None,
    multidir: bool = False,
) -> NDArray | list:
    """
    Used to create numpy array from a data file. Inspired by readfile function from pyHegel.
    Can be multiple files.

    Parameters
    ----------
    path : str, optional
        Full path to data file. Can be a glob parameter for multiple files. The default is None.
    comments : str, optional
        Defines the comment symbol in the data file. The default is "#".
    delimiter : str or None, optional
        Defines the .txt data file delimiter. The default is None.
    multidir : bool, optional
        I don't think I ever tested this. Supposed to be an option for which it searches for your datafiles in other
        directories. The default is False
    """
    extention = Path(path).suffix
    if multidir:
        listo = []
        for paths, subdir, files in os.walk(path):
            for file in glob.glob(os.path.join(paths, f"*.{extention}")):
                listo.append(file)

    else:
        listo = glob.glob(path)

    listo.sort()
    filelist = []
    filelist.extend(listo)
    data = []

    if len(filelist) == 0:

        print("No file found")
        return

    elif len(filelist) > 1:

        print(f"Found {len(listo)} files")
        multidir = True

    for fn in listo:

        if extention == ".txt":
            arr = np.loadtxt(fn, comments=comments, delimiter=delimiter).T

        elif extention == ".csv":
            arr = np.genfromtxt(fn, delimiter=delimiter, skip_header=3).T

        elif extention == ".hdf5":
            with h5py.File(fn, "r") as hdf5_data:
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
                return data, filelist

        elif extention == ".dat":
            arr = None
            with open(fn) as f:
                for num, line in enumerate(f):
                    if "[Data]" in line:
                        arr = np.genfromtxt(
                            path,
                            delimiter=delimiter,
                            skip_header=num + 2,
                            missing_values="",
                            filling_values=None,
                        ).T
                        break

        else:
            raise TypeError(
                "Unrecognized file extension. Program currently support .csv, .txt, .dat and .hdf5 files."
            )

        data.append(arr)

    if multidir:
        return data, filelist

    if len(filelist) == 1:
        return np.array(data)[0], filelist

    return np.array(data), filelist


def writer(
    data: dict[str, float],
    path: Union[str, PathLike],
    name: str = "%T",
    nodialog: bool = False,
) -> None:
    """
    Writes data results in a txt file.

    Parameters
    ----------
    data : dict
        Dictionary of data results from resonator fitter.
    path : str
        Path to .txt file.
    name : str
        Name of txt file.
    nodialog : bool
        Set to True if you don't want the pop-up window for overwrite. Defaults to False.
        Warning : if the file already exists, will overwrite.
    """
    full_path = os.path.join(path, name + "_results.txt")

    if os.path.exists(full_path) and not nodialog:
        decision = choice("Overwrite warning", "Text file already exists, overwrite?")

        if not decision:
            return

    with open(full_path, "w") as f:
        f.write(name + "\n\n" + "------------------------------------------" + "\n\n")

        for key, value in data.items():
            f.write(str(key) + " : " + str(value) + "\n\n")

        f.close()


def getter(path: Union[str, PathLike]) -> dict[str, float]:
    """
    If for some reason you want to recreate the dictionary with the fit results, you can do that.

    Parameters
    ----------
    path : str
        Path to text file with fit results
    """

    with open(path, "r") as f:

        for line in f:

            if not line.find("coupling loss : "):
                coupling_loss = float(line.split(": ")[1])
            elif not line.find("internal loss : "):
                internal_loss = float(line.split(": ")[1])
            elif not line.find("total loss : "):
                total_loss = float(line.split(": ")[1])
            elif not line.find("coupling loss error : "):
                coupling_loss_error = float(line.split(": ")[1])
            elif not line.find("internal loss error : "):
                internal_loss_error = float(line.split(": ")[1])
            elif not line.find("total loss error : "):
                total_loss_error = float(line.split(": ")[1])
            elif not line.find("coupling quality factor : "):
                Q_c = float(line.split(": ")[1])
            elif not line.find("coupling quality factor error : "):
                Q_c_error = float(line.split(": ")[1])
            elif not line.find("internal quality factor : "):
                Q_i = float(line.split(": ")[1])
            elif not line.find("internal quality factor error : "):
                Q_i_error = float(line.split(": ")[1])
            elif not line.find("total quality factor : "):
                Q_t = float(line.split(": ")[1])
            elif not line.find("total quality factor error : "):
                Q_t_error = float(line.split(": ")[1])
            elif not line.find("background model : "):
                background_model_name = line.split(": ")[1].strip()
            elif not line.find("resonance frequency : "):
                f_r = float(line.split(": ")[1])
            elif not line.find("resonance frequency error : "):
                f_r_error = float(line.split(": ")[1])
            elif not line.find("photon number : "):
                photnum = float(line.split(": ")[1])
            else:
                pass

        f.close()

    result = {
        "coupling loss": coupling_loss,
        "internal loss": internal_loss,
        "total loss": total_loss,
        "coupling loss error": coupling_loss_error,
        "internal loss error": internal_loss_error,
        "total loss error": total_loss_error,
        "coupling quality factor": Q_c,
        "internal quality factor": Q_i,
        "total quality factor": Q_t,
        "coupling quality factor error": Q_c_error,
        "internal quality factor error": Q_i_error,
        "total quality factor error": Q_t_error,
        "background model": background_model_name,
        "resonance frequency": f_r,
        "resonance frequency error": f_r_error,
        "photon number": photnum,
    }

    return result


def gethdf5info(filename: Union[str, PathLike], show: bool = False) -> dict:
    """
    Get all datasets info about VNA measurement apart from the VNA S21 data itself.
    Returns the info in a dictionary.

    Parameters
    ----------
    filename : str
        Full path to the HDF5 file.
    show : bool, optional
        Prints the keys found in the HDF5 file. The default is False.

    Returns
    -------
    {info : value, ...}

    """
    with h5py.File(filename, "r") as file:
        keylst = [key for key in file.keys()]
        atrlst = [atr for atr in file.attrs.keys()]
        keylst.remove("VNA")
        info_dict = {element: None for element in keylst}
        atr_dict = {element: None for element in atrlst}
        for key in keylst:
            try:
                info_dict[key] = file[key][:]
            except TypeError:
                info_dict[key] = file[key][key][0]
            except Exception as err:
                print("Unexpected error : ", err)
        for atr in atrlst:
            try:
                atr_dict[atr] = file.attrs[atr]
            except TypeError:
                try:
                    atr_dict[atr] = file.attrs[atr][:]
                except TypeError:
                    atr_dict[atr] = file.attrs[atr][atr][0]
            except Exception as err:
                print("Unexpected error : ", err)
        if show:
            print("All hdf5 keys : ", keylst)
    return info_dict, atr_dict
