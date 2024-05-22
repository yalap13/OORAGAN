"""
Much of the contents of this file has been written by Gabriel Ouellet, 
fellow master's student. All his code is available on the JosePh Gitlab.
"""

import glob
import os
import numpy as np
import h5py

from pathlib import Path
from typing import Optional
from .util import choice


def datapicker(
    path: str,
    file_extension: str = "hdf5",
    comments: str = "#",
    delimiter: Optional[str] = None,
):
    """
    Used to extract data from raw files. Returns a dictionnary in which the keys are the filenames
    and the values are the NDArrays of the data.

    Parameters
    ----------
    path : str
        Path to folder containing data files or full path to specific data file.
    file_extension : str
        File extension to search for in given path. Supports "hdf5", "txt" and "csv".
        Defaults to "hdf5".
    comments : str
        Defines the comment symbol in the data file. Defaults to "#".
    delimiter : str, optional
        Defines the .txt data file delimiter.

    Returns
    -------
    Dictionnary in which the keys are the filenames and the values are the NDArrays of the data.

    """
    if Path(path).suffix == "":
        files_list = []
        for paths, _, _ in os.walk(path):
            for file in glob.glob(os.path.join(paths, f"*.{file_extension}")):
                files_list.append(file)
    else:
        files_list = [path]

    files_list.sort()

    if len(files_list) == 0:
        raise FileNotFoundError("No files were found")
    elif len(files_list) > 1:
        print(f"Found {len(files_list)} files")

    data = {}
    for file in files_list:
        if file_extension == "txt":
            arr = np.loadtxt(file, comments=comments, delimiter=delimiter).T
            data[file] = arr
        elif file_extension == "csv":
            arr = np.genfromtxt(file, delimiter=delimiter, skip_header=3).T
            data[file] = arr
        elif file_extension == "hdf5":
            with h5py.File(file, "r") as hdf5_data:
                freq = hdf5_data["VNA"]["VNA Frequency"][:]
                try:
                    real = np.squeeze(hdf5_data["VNA"]["s21_real"][:])
                    imag = np.squeeze(hdf5_data["VNA"]["s21_imag"][:])
                    if real.ndim > 1:
                        for i in range(len(real)):
                            arr = np.stack((freq.T, real[i].T, imag[i].T))
                            data[file] = arr
                    else:
                        arr = np.stack((freq.T, real.T, imag.T))
                        data[file] = arr
                except KeyError:
                    mag = np.squeeze(hdf5_data["VNA"]["s21_mag"][:])
                    phase = np.squeeze(hdf5_data["VNA"]["s21_phase"][:])
                    if mag.ndim > 1:
                        for i in range(len(mag)):
                            arr = np.stack((freq.T, mag[i].T, phase[i].T))
                            data[file] = arr
                    else:
                        arr = np.stack((freq.T, mag.T, phase.T))
                        data[file] = arr
                hdf5_data.close()
        else:
            raise ValueError(
                "Unrecognized file extension. Program currently support .csv, .txt and .hdf5 files."
            )

    return data


def writer(data: dict, path: str, name: str = "%T", nodialog: bool = False):
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

    Returns
    -------
    None.

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


def getter(path):
    """
    If for some reason you want to recreate the dictionary with the fit results, you can do that.

    Parameters
    ----------
    path : str
        Path to text file with fit results

    Returns
    -------
    None.

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


def gethdf5info(path, show=False):
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
            for file in glob.glob(os.path.join(paths, f"*.hdf5")):
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
            if show:
                print("All hdf5 keys : ", keylst)
        global_dict[file] = {"vna_info": info_dict, "temps": atr_dict}
    return global_dict
