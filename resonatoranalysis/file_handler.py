"""
The contents of this file has been written by Gabriel Ouellet, fellow master's
student.
"""

import glob
import os
import time
import numpy as np
import h5py
import easygui as eg

from pathlib import Path


def choice(title=None, msg=None):
    """
    Function that opens a choice box for the user to choose between yes or no.
    Returns True if yes, False if no.

    """
    if title is None:
        title = "Overwrite warning"
    if msg is None:
        msg = (
            "Do you really want to save the figure? It could delete an existing figure."
        )
    user_choice = eg.ynbox(msg=msg, title=title)
    return user_choice


def automater(att_frigo, fn=None):
    """
    Automating initializing method for hdf5 files. Includes an optional file selector.

    Parameters
    ----------
    att_frigo : int
        Attenuation from VNA to sample. Will be added to powers found in hdf5 file.
    fn : str, optional
        Full path to data file. Can be a glob parameter for multiple files.

    Returns
    -------
    numpy.array object

    """
    if fn is None:
        fn = eg.fileopenbox(
            "Veuillez sélectionner un ou plusieurs fichiers à analyser",
            "Automater",
            multiple=True,
        )
        if fn is None:
            return None

    dico = {"data": [], "power": np.array([]), "info": [], "temps": [], "file": []}

    for f in fn:
        data, file = datapicker(f)
        info, temps = gethdf5info(f)
        try:
            power = info["VNA Power"] + att_frigo - info["Variable Attenuator"]
        except KeyError:
            power = info["VNA Power"] + att_frigo

        dico["data"] += data
        dico["power"] = np.append(dico["power"], power)
        dico["info"] += [info]
        dico["temps"] += [temps]
        dico["file"] += file

    return dico


def datapicker(path, comments="#", delimiter=None, multidir=False):
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

    Returns
    -------
    numpy.array object

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


def writer(data, path, name="%T", nodialog=False):
    """
    Writes some data results in a txt file.

    Parameters
    ----------
    data : dict
        Dictionary of data results from resonator fitter.
    path : str
        Path to .txt file.
    name : str, optional
        Name of txt file
    nodialog : bool, optional
        Set to True if you don't want the pop-up window for overwrite. Warning : if the file already exists, will
        overwrite.

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


def convert_magang_to_complex(data, deg=False, dBm=False):
    """
    Converts a numpy array that has a magnitude array and an angle array to
    an array of complex values.

    Parameters
    ----------
    data : numpy.array
        Raw data array to convert.
    deg : bool, optional
        If data angle array is in degrees. The default is False.
    dBm : bool, optional
        If data magnitude array is in dBm. The default is False.

    Returns
    -------
    np.array(complex_values)

    """
    mag = data[1]

    if deg:
        ang = np.deg2rad(data[2])

    else:
        ang = data[2]

    if dBm:
        s21_complex = 10 ** (mag / 20) * np.exp(1j * ang)

    else:
        s21_complex = mag * np.exp(1j * ang)

    return s21_complex


def convert_magang_to_dB(data, deg=False, dBm=False):
    """
    Converts a numpy array that has a magnitude array and an angle array to
    an array of power values in dBm.

    Parameters
    ----------
    data : numpy.array
        Raw data array to convert.
    deg : bool, optional
        If data angle array is in degrees. The default is False.
    dBm : bool, optional
        If data magnitude array is in dBm. The default is False.

    Returns
    -------
    np.array(complex_values)

    """
    mag = data[1]

    if deg:
        ang = np.deg2rad(data[2])

    else:
        ang = data[2]

    if dBm:
        s21 = np.abs(10 ** (mag / 20) * np.exp(1j * ang))

    else:
        s21 = 20 * np.log10(np.abs(mag * np.exp(1j * ang)))

    return s21


def convert_complex_to_dB(real, imag, deg=False):
    """
    Converts a numpy array that has a magnitude array and an angle array to
    an array of power values in dBm.

    Parameters
    ----------
    real : numpy.array
        Raw data array of real values to convert.
    imag : numpy.array
        Raw data array of imaginary values to convert.
    deg : bool, optional
        If you want the phase to be returned in degrees. The default is False.

    Returns
    -------
    magnitude_array, phase_array

    """

    phase = np.angle(real + 1j * imag, deg=deg)
    mag = 20 * np.log10(np.sqrt(real**2 + imag**2))

    return mag, phase


def gethdf5info(f, show=False):
    """
    Get all datasets info about VNA measurement apart from the VNA S21 data itself. Returns the info in a dictionary.

    Parameters
    ----------
    f : str
        Full path to the HDF5 file.
    show : bool, optional
        Prints the keys found in the HDF5 file. The default is False.

    Returns
    -------
    {info : value, ...}

    """
    with h5py.File(f, "r") as file:
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


def strtime(s):
    """
    Converts the string from time.struct_time that MeaVis outputs in the hdf5 metadata to an understandable format.

    Parameters
    ----------
    s : str
        String containing time.struct_time

    Returns
    -------
    Time since epoch

    """
    beg_year = s.find("tm_year")
    end_year = s.find(",", beg_year)
    year = int(s[beg_year + 8 : end_year])

    beg_month = s.find("tm_mon")
    end_month = s.find(",", beg_month)
    month = int(s[beg_month + 7 : end_month])

    beg_mday = s.find("tm_mday")
    end_mday = s.find(",", beg_mday)
    mday = int(s[beg_mday + 8 : end_mday])

    beg_hour = s.find("tm_hour")
    end_hour = s.find(",", beg_hour)
    hour = int(s[beg_hour + 8 : end_hour])

    beg_min = s.find("tm_min")
    end_min = s.find(",", beg_min)
    mins = int(s[beg_min + 7 : end_min])

    beg_sec = s.find("tm_sec")
    end_sec = s.find(",", beg_sec)
    sec = int(s[beg_sec + 7 : end_sec])

    beg_wday = s.find("tm_wday")
    end_wday = s.find(",", beg_wday)
    wday = int(s[beg_wday + 8 : end_wday])

    beg_yday = s.find("tm_yday")
    end_yday = s.find(",", beg_yday)
    yday = int(s[beg_yday + 8 : end_yday])

    beg_dst = s.find("tm_isdst")
    end_dst = s.find(")", beg_dst)
    dst = int(s[beg_dst + 9 : end_dst])

    return time.mktime((year, month, mday, hour, mins, sec, wday, yday, dst))
