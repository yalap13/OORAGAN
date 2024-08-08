import time
import easygui as eg
import numpy as np

from numpy.typing import NDArray
from IPython import get_ipython
from typing import Optional


def choice(title: Optional[str] = None, msg: Optional[str] = None) -> bool:
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
    mag: NDArray, phase: NDArray, deg: bool = True, dBm: bool = True
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
        Set to ``True`` if the phase is in degrees. Defaults to ``True``.
    dBm : bool, optional
        Set to ``True`` if the magnitude is in dBm. Defaults to ``True``.
    """
    if deg:
        phase = np.deg2rad(phase)
    if dBm:
        s21_complex = 10 ** (mag / 20) * np.exp(1j * phase)
    else:
        s21_complex = mag * np.exp(1j * phase)

    return s21_complex.real, s21_complex.imag


def convert_complex_to_magphase(
    real: NDArray, imag: NDArray, deg: bool = True
) -> NDArray:
    """
    Converts real and imaginary data into magnitude (dBm) and phase.

    Parameters
    ----------
    real : NDArray
        Real data array.
    imag : NDArray
        Imaginary data array.
    deg : bool, optional
        If ``True`` the phase is returned in degrees. Defaults to ``True``.
    """

    phase = np.angle(real + 1j * imag, deg=deg)
    mag = 20 * np.log10(np.sqrt(real**2 + imag**2))

    return mag, phase


def is_interactive() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
