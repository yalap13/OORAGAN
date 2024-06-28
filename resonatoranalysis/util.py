import time
import easygui as eg
import numpy as np

from numpy.typing import NDArray


def choice(title: str = None, msg: str = None) -> bool:
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


def strtime(s: str) -> float:
    """
    Converts the string from time.struct_time that MeaVis outputs in the hdf5 metadata to an
    understandable format.

    Parameters
    ----------
    s : str
        String containing time.struct_time

    Returns
    -------
    Float representing the time since epoch.

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


def convert_magphase_to_complex(
    mag: NDArray, phase: NDArray, deg: bool = False, dBm: bool = False
) -> NDArray:
    """
    Converts magnitude and phase data into real and imaginary.

    Parameters
    ----------
    mag : NDArray
        Magnitude array.
    phase : NDArray
        Phase array
    deg : bool, optional
        Set to ``True`` if the phase is in degrees. Defaults to ``False``.
    dBm : bool, optional
        Set to ``True`` if the magnitude is in dBm. Defaults to ``False``.
    """
    if deg:
        phase = np.deg2rad(phase)
    if dBm:
        s21_complex = 10 ** (mag / 20) * np.exp(1j * phase)
    else:
        s21_complex = mag * np.exp(1j * phase)

    return s21_complex.real, s21_complex.imag


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


def convert_complex_to_magphase(
    real: NDArray, imag: NDArray, deg: bool = False
) -> NDArray:
    """
    Converts real and imaginary data into magnitude (dB) and phase.

    Parameters
    ----------
    real : NDArray
        Real data array.
    imag : NDArray
        Imaginary data array.
    deg : bool, optional
        If ``True`` the phase is returned in degrees. Defaults to ``False``.
    """

    phase = np.angle(real + 1j * imag, deg=deg)
    mag = 20 * np.log10(np.sqrt(real**2 + imag**2))

    return mag, phase


def calculate_power(att_cryo: float, info: dict):
    """
    Computes the power for each data file from their info and the attenuation on the cryostat.

    Parameters
    ----------
    att_cryo : float
        Attenuation on the cryostat.
    info : dict
        Dictionnary of info extracted from the hdf5 files.

    Returns
    -------
    Dictionnary with filenames as keys and array of powers as values.
    """
    powers = {}
    for file in info.keys():
        powers[file] = (
            info[file]["vna_info"]["VNA Power"]
            + att_cryo
            - info[file]["vna_info"]["Variable Attenuator"]
        )
    return powers
