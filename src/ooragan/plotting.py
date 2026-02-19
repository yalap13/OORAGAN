from numpy.typing import ArrayLike
from typing import Optional, Literal
from resonator import base
from graphinglib import SmartFigure
import graphinglib as gl
import numpy as np

from .util import convert_complex_to_magphase


FREQ_UNIT_CONVERSION = {"GHz": 1e9, "MHz": 1e6, "kHz": 1e3}


def triptych(
    freq: ArrayLike,
    complex_data: ArrayLike,
    fit_result: Optional[base.ResonatorFitter] = None,
    freq_unit: Literal["GHz", "MHz", "kHz"] = "GHz",
    title: Optional[str] = None,
    three_ticks: bool = False,
    figure_style: str = "default",
) -> SmartFigure:
    """
    Plots the magnitude vs frequency, the phase vs frequency and the complex data in a
    single figure.

    Parameters
    ----------
    freq : ArrayLike
        Frequency array.
    complex : ArrayLike
        Complex data array.
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
    real = np.real(complex_data)
    imag = np.imag(complex_data)
    mag, phase = convert_complex_to_magphase(real, imag)
    mag_vs_freq = gl.Scatter(freq, mag, marker_style=".")
    phase_vs_freq = gl.Scatter(freq, phase, marker_style=".")
    col1 = gl.SmartFigure(
        2,
        1,
        sub_x_labels=[None, "Fréquence (GHz)"],
        sub_y_labels=["$|S_{21}|$ (dB)", "arg($S_{21}$) (deg)"],
        share_x=True,
        reference_labels=False,
        height_ratios=[1, 1.3],
        elements=[mag_vs_freq, phase_vs_freq],
        figure_style=figure_style,
    )
    complex = gl.Scatter(real, imag, marker_style=".")
    hline = gl.Hlines([0], line_styles=":", line_widths=1, colors="silver")
    vline = gl.Vlines([0], line_styles=":", line_widths=1, colors="silver")
    fig_complex = gl.SmartFigure(
        x_label=r"$\mathrm{Re}(S_{21})$",
        y_label=r"$\mathrm{Im}(S_{21})$",
        figure_style=figure_style,
        aspect_ratio="equal",
        reference_labels=False,
        elements=[complex, hline, vline],
    )
    triptych = gl.SmartFigure(
        1,
        2,
        size=(10, 6),
        reference_labels=False,
        figure_style=figure_style,
        elements=[col1, fig_complex],
        title=title,
    )

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
        triptych[0, 0][0, 0] += [mag_fit, mag_point]
        triptych[0, 0][1, 0] += [phase_fit, phase_point]
        triptych[0, 1].add_elements(complex_fit, complex_point)

    if three_ticks:
        triptych[0, 0].set_ticks(x_ticks=[np.min(freq), np.mean(freq), np.max(freq)])
        # triptych[0, 0][1, 0].set_ticks(
        #     x_ticks=[np.min(freq), np.mean(freq), np.max(freq)]
        # )
        triptych[0, 1].set_ticks(
            x_ticks=[np.min(real), 0, np.max(real)],
            y_ticks=[np.min(imag), 0, np.max(imag)],
        )

    return triptych
