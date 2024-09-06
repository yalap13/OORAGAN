import time
import easygui as eg
import numpy as np
import graphinglib as gl
import pandas as pd

from numpy.typing import NDArray, ArrayLike
from resonator import base
from IPython import get_ipython
from typing import Optional, Literal
from graphinglib import MultiFigure


FREQ_UNIT_CONVERSION = {"GHz": 1e9, "MHz": 1e6, "kHz": 1e3}


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
    r"""
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

    Notes
    -----
    This conversion is defined as

    .. math::

        S_{21}^\text{complex} = 10^{\frac{|S_{21}|}{20}}e^{i\phi}

    where the magnitude :math:`|S_{21}|` is in dB and the phase :math:`\phi` is in degrees.
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
    r"""
    Converts real and imaginary data into magnitude (dBm) and phase.

    Parameters
    ----------
    real : NDArray
        Real data array.
    imag : NDArray
        Imaginary data array.
    deg : bool, optional
        If ``True`` the phase is returned in degrees. Defaults to ``True``.

    Notes
    -----
    This conversion is defined as

    .. math::

        |S_{21}|=20\cdot\log_{10}\sqrt{\mathrm{Re}(S_{21})^2+\mathrm{Im}(S_{21})^2}

        \phi=\arctan\left(\frac{\mathrm{Im}(S_{21})}{\mathrm{Re}(S_{21})}\right)
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


def load_graph_data(path: str) -> dict[str, NDArray]:
    """
    Loads the data saved in csv files by the Grapher objects and returns it
    as a dictionnary with label as key and NDArrays of the data.

    Parameters
    ----------
    path : str
        Complete file file path of the file containing the data.
    """
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    loaded_data = {}
    for label in df.columns.get_level_values(0).unique():
        loaded_data[label] = np.array(df[label]).T
    return loaded_data


def plot_triptych(
    freq: ArrayLike,
    mag: ArrayLike,
    phase: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    fit_result: Optional[base.ResonatorFitter] = None,
    freq_unit: Literal["GHz", "MHz", "kHz"] = "GHz",
    title: Optional[str] = None,
    three_ticks: bool = False,
    figure_style: str = "default",
) -> MultiFigure:
    """
    Plots the magnitude vs frequency, the phase vs frequency and the complex data in a
    single figure.

    Parameters
    ----------
    freq : ArrayLike
        Frequency array.
    mag : ArrayLike
        Magnitude array.
    phase : ArrayLike
        Phase array.
    real : ArrayLike
        Real values array.
    imag : ArrayLike
        Imaginary values array.
    fit_result : ResonatorFitter, optional
        Fit result from the resonator library. Defaults to ``None``.
    freq_unit : {"GHz", "MHz", "kHz"}, optional
        Unit in which the frequency is given. Defaults to ``"GHz"``.
    title : str, optional
        Title of the figure. Defaults to ``None``.
    three_ticks : bool, optional
        If ``True``, only three ticks will be displayed on the x axis: the minimum
        frequency, the maximum and the frequency. Defaults to ``False``.
    figure_style : str, optional
        GraphingLib figure style to apply to the plot. See
        [here](https://www.graphinglib.org/doc-1.5.0/handbook/figure_style_file.html#graphinglib-styles-showcase)
        for more info.
    """
    freq = np.asarray(freq) / FREQ_UNIT_CONVERSION[freq_unit]
    fig_mag_vs_freq = gl.Figure(
        f"Frequency ({freq_unit})", "Magnitude (dBm)", figure_style=figure_style
    )
    mag_vs_freq = gl.Scatter(freq, mag, marker_style=".")
    fig_mag_vs_freq.add_elements(mag_vs_freq)
    fig_phase_vs_freq = gl.Figure(
        f"Frequency ({freq_unit})", "Phase (deg)", figure_style=figure_style
    )
    phase_vs_freq = gl.Scatter(freq, phase, marker_style=".")
    fig_phase_vs_freq.add_elements(phase_vs_freq)
    fig_complex = gl.Figure(
        "real", "imag", figure_style=figure_style, aspect_ratio="equal"
    )
    complex = gl.Scatter(real, imag, marker_style=".", label="Data")
    hline = gl.Hlines([0], line_styles=":", line_widths=1, colors="silver")
    vline = gl.Vlines([0], line_styles=":", line_widths=1, colors="silver")
    fig_complex.add_elements(hline, vline, complex)

    if fit_result is not None:
        fit = fit_result.evaluate_fit(fit_result.frequency)
        fr = fit_result.evaluate_fit(fit_result.resonance_frequency)
        mag_fit = gl.Curve(
            fit_result.frequency / FREQ_UNIT_CONVERSION[freq_unit],
            20 * np.log10(np.abs(fit)),
            color="k",
            line_width=1,
        )
        mag_point = gl.Scatter(
            fit_result.resonance_frequency / FREQ_UNIT_CONVERSION[freq_unit],
            20 * np.log10(np.abs(fr)),
            face_color="k",
        )
        phase_fit = gl.Curve(
            fit_result.frequency / FREQ_UNIT_CONVERSION[freq_unit],
            np.degrees(np.angle(fit)),
            color="k",
            line_width=1,
        )
        phase_point = gl.Scatter(
            fit_result.resonance_frequency / FREQ_UNIT_CONVERSION[freq_unit],
            np.degrees(np.angle(fr)),
            face_color="k",
        )
        complex_fit = gl.Curve(
            fit.real, fit.imag, color="k", line_width=1, label="Best fit"
        )
        complex_point = gl.Scatter(fr.real, fr.imag, face_color="k", label="Resonance")
        fig_mag_vs_freq.add_elements(mag_fit, mag_point)
        fig_phase_vs_freq.add_elements(phase_fit, phase_point)
        fig_complex.add_elements(complex_fit, complex_point)

    if three_ticks:
        fig_mag_vs_freq.set_ticks(xticks=[np.min(freq), np.mean(freq), np.max(freq)])
        fig_phase_vs_freq.set_ticks(xticks=[np.min(freq), np.mean(freq), np.max(freq)])
        fig_complex.set_ticks(xticks=[np.min(real), 0, np.max(real)])
    triptych = gl.MultiFigure(
        2, 2, (10, 6), title=title, reference_labels=False, figure_style=figure_style
    )
    triptych.add_figure(fig_mag_vs_freq, 0, 0, 1, 1)
    triptych.add_figure(fig_phase_vs_freq, 1, 0, 1, 1)
    triptych.add_figure(fig_complex, 0, 1, 2, 1)
    return triptych


def level_phase(phase: ArrayLike, deg: bool = False) -> ArrayLike:
    """
    Levels the phase by substracting the slope.
    """
    unwrapped_phase = np.unwrap(phase, 180) if deg else np.unwrap(phase)
    pointA = unwrapped_phase[0]
    pointB = unwrapped_phase[-1]
    slope = np.linspace(pointA, pointB, len(unwrapped_phase))
    return unwrapped_phase - slope
