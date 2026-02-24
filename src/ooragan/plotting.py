from numpy.typing import ArrayLike, NDArray
from typing import Optional, Literal, Iterable
from resonator import base
from graphinglib import SmartFigure, Curve
import graphinglib as gl
import numpy as np
from scipy.constants import e, hbar, k

from .util import convert_complex_to_magphase
from .typing import _FitResult


FREQ_UNIT_CONVERSION = {"GHz": 1e9, "MHz": 1e6, "kHz": 1e3}


def triptych(
    freq: ArrayLike,
    complex_data: ArrayLike,
    resonator_fitter: Optional[base.ResonatorFitter] = None,
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
    resonator_fitter : ResonatorFitter, optional
        Fit result from the resonator library. Defaults to ``None``.

        .. note::

           If a ``ResonatorFitter`` is provided, the fit and the resonance frequency obtained from the fit are displayed.

    freq_unit : {"GHz", "MHz", "kHz"}, optional
        Unit in which the frequency is given. Defaults to ``"GHz"``.
    title : str, optional
        Title of the figure. Defaults to ``None``.
    three_ticks : bool, optional
        If ``True``, only three ticks will be displayed on the x axis: the minimum
        frequency, the maximum and the frequency. Defaults to ``False``.
    figure_style : str, optional
        GraphingLib figure style to apply to the plot. See
        `here <https://www.graphinglib.org/doc-1.5.0/handbook/figure_style_file.html#graphinglib-styles-showcase>`_
        for more info.


    Examples
    --------
    .. plot::

       import numpy as np
       from ooragan import triptych

       # Generate data from theoretical model
       def S21(f, Q, Qc, f0):
           return 1 - (Q / Qc) / (1 + 2j * Q * (f - f0) / f0)


       Q = 85000
       Qc = 100000
       f0 = 6.5e9
       tau = 0.0000001
       a = 1
       alpha = 0
       phi = 0
       frequency = np.linspace(6.4995e9, 6.5005e9, 5001)

       fig = triptych(
           frequency,
           S21(frequency, Q, Qc, f0),
       )
       fig.show()
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

    if resonator_fitter is not None:
        fit = resonator_fitter.evaluate_fit(resonator_fitter.frequency)
        fr = resonator_fitter.evaluate_fit(resonator_fitter.resonance_frequency)
        mag_fit = gl.Curve(
            resonator_fitter.frequency / FREQ_UNIT_CONVERSION[freq_unit],
            20 * np.log10(np.abs(fit)),
            color="k",
            line_width=1,
        )
        mag_point = gl.Scatter(
            resonator_fitter.resonance_frequency / FREQ_UNIT_CONVERSION[freq_unit],
            20 * np.log10(np.abs(fr)),
            face_color="k",
        )
        phase_fit = gl.Curve(
            resonator_fitter.frequency / FREQ_UNIT_CONVERSION[freq_unit],
            np.degrees(np.angle(fit)),
            color="k",
            line_width=1,
        )
        phase_point = gl.Scatter(
            resonator_fitter.resonance_frequency / FREQ_UNIT_CONVERSION[freq_unit],
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
        triptych[0, 1].set_ticks(
            x_ticks=[np.min(real), 0, np.max(real)],
            y_ticks=[np.min(imag), 0, np.max(imag)],
        )

    return triptych


def _as_function_of_photon_nbr(
    elements: Iterable[Curve],
    y_label: str,
    figure_style: str,
    title: Optional[str],
) -> SmartFigure:
    fig = SmartFigure(
        x_label=r"$\tilde n$",
        y_label=y_label,
        elements=elements,
        figure_style=figure_style,
        title=title,
        log_scale_x=True,
        log_scale_y=True,
        legend_loc="outside center right",
    )
    return fig


def quality_factors(
    fit_results: _FitResult | list[_FitResult],
    freq_unit: Literal["GHz", "MHz", "kHz"] = "GHz",
    title: Optional[str] = None,
    figure_style: str = "default",
) -> SmartFigure:
    r"""
    Plots the quality factors :math:`Q_i` and :math:`Q_c` as a function of average photon number in the
    resonator :math:`(\tilde n)`.

    Parameters
    ----------
    fit_results : FitResult or list of FitResult
        Single or list of FitResult from a Fitter.
    freq_unit : {"GHz", "MHz", "kHz"}, optional
        Unit in which the frequency is given. Defaults to ``"GHz"``.
    title : str, optional
        Title of the figure. Defaults to ``None``.
    figure_style : str, optional
        GraphingLib figure style to apply to the plot. See
        `here <https://www.graphinglib.org/doc-1.5.0/handbook/figure_style_file.html#graphinglib-styles-showcase>`_
        for more info.
    """
    if isinstance(fit_results, _FitResult):
        fit_results = [fit_results]
    elif isinstance(fit_results, list):
        if not all(isinstance(i, _FitResult) for i in fit_results):
            raise TypeError("can only accept a FitResult or a list of FitResults")
    else:
        raise TypeError("can only accept a FitResult or a list of FitResults")
    elements = []
    if figure_style == "default":
        figure_style = gl.get_default_style()

    for i, fr in enumerate(fit_results):
        qi = Curve(
            fr.photon_nbr,
            fr.Q_i,
            label="{:.3f} {}".format(
                np.mean(fr.f_r) / FREQ_UNIT_CONVERSION[freq_unit], freq_unit
            ),
            color=gl.get_color(figure_style, i),
        )
        qc = Curve(
            fr.photon_nbr, fr.Q_c, line_style="--", color=gl.get_color(figure_style, i)
        )
        elements.append(qi)
        elements.append(qc)

    fig = _as_function_of_photon_nbr(
        elements=elements, y_label="$Q_i, Q_c$", figure_style=figure_style, title=title
    )
    return fig


def losses(
    fit_results: _FitResult | list[_FitResult],
    freq_unit: Literal["GHz", "MHz", "kHz"] = "GHz",
    title: Optional[str] = None,
    figure_style: str = "default",
) -> SmartFigure:
    r"""
    Plots the losses :math:`\delta_i` and :math:`\delta_c` as a function of average photon number in the
    resonator :math:`(\tilde n)`.

    Parameters
    ----------
    fit_results : FitResult or list of FitResult
        Single or list of FitResult from a Fitter.
    freq_unit : {"GHz", "MHz", "kHz"}, optional
        Unit in which the frequency is given. Defaults to ``"GHz"``.
    title : str, optional
        Title of the figure. Defaults to ``None``.
    figure_style : str, optional
        GraphingLib figure style to apply to the plot. See
        `here <https://www.graphinglib.org/doc-1.5.0/handbook/figure_style_file.html#graphinglib-styles-showcase>`_
        for more info.
    """
    if isinstance(fit_results, _FitResult):
        fit_results = [fit_results]
    elif isinstance(fit_results, list):
        if not all(isinstance(i, _FitResult) for i in fit_results):
            raise TypeError("can only accept a FitResult or a list of FitResults")
    else:
        raise TypeError("can only accept a FitResult or a list of FitResults")
    elements = []
    if figure_style == "default":
        figure_style = gl.get_default_style()

    for i, fr in enumerate(fit_results):
        di = Curve(
            fr.photon_nbr,
            fr.internal_loss,
            label="{:.3f} {}".format(
                np.mean(fr.f_r) / FREQ_UNIT_CONVERSION[freq_unit], freq_unit
            ),
            color=gl.get_color(figure_style, i),
        )
        dc = Curve(
            fr.photon_nbr,
            fr.coupling_loss,
            line_style="--",
            color=gl.get_color(figure_style, i),
        )
        elements.append(di)
        elements.append(dc)

    fig = _as_function_of_photon_nbr(
        elements=elements,
        y_label=r"$\delta_i, \delta_c$",
        figure_style=figure_style,
        title=title,
    )
    return fig


def _frequency_ratio(Bll: NDArray, thetaB: float) -> NDArray:
    """
    Fitting function for variation of resonant frequency as a function
    of parallel magnetic field.
    """
    t = 100e-9
    w = 25e-6
    Tc = 12.5
    D = 2e-5
    return (
        -np.pi
        / 48
        * (e**2 * t**2)
        / (hbar * k * Tc)
        * D
        * (1 + thetaB**2 * w**2 / t**2)
        * Bll**2
    )


def magnetic_field(
    fit_result: _FitResult,
    two_way_sweep: bool = True,
    show_frequency: bool = True,
    fit_frequency: bool = False,
    title: Optional[str] = None,
    figure_style: str = "default",
) -> SmartFigure:
    r"""
    Plots the internal quality factor (:math:`Q_i`) as a function of the parallel magnetic field (:math:`B_\parallel`).
    Also can plot and fit the frequency shift induced by the magnetic field.

    Parameters
    ----------
    fit_result : FitResult or list of FitResult
        Single or list of FitResult from a Fitter.
    two_way_sweep : bool, optional
        Whether or not the data was taken while the field was going up **and** down. Defaults to ``True``.
    show_frequency : bool, optional
        Whether or not to show the frequency variation on the plot. Defaults to ``True``.
    fit_frequency : bool, optional
        Whether or not to fit the frequency variation. Defaults to ``False``.
    title : str, optional
        Title of the figure. Defaults to ``None``.
    figure_style : str, optional
        GraphingLib figure style to apply to the plot. See
        `here <https://www.graphinglib.org/doc-1.5.0/handbook/figure_style_file.html#graphinglib-styles-showcase>`_
        for more info.

    Notes
    -----
    The frequency is fitted using the model

    .. math:: \frac{\Delta f}{f_r} = -\frac{\pi}{48}\frac{e^2t^2}{\hbar k_B T_c}D\left(1+\theta_B^2\frac{w^2}{t^2}\right)B_\parallel^2

    from C. Roy, S. Frasca and P. Scarlino, *Magnetic-field-resilient high-impedance high-kinetic-inductance superconducting
    resonators*, Phys. Rev. Appl. **25**, 014069 (2026).
    """
    if not isinstance(fit_result, _FitResult):
        raise TypeError("can only accept a single FitResult")
    if figure_style == "default":
        figure_style = gl.get_default_style()

    deltaf_over_f0 = (fit_result.f_r - fit_result.f_r[0]) / fit_result.f_r[0]
    qi_curves = []
    if two_way_sweep:
        max_idx = np.where(fit_result.magnet_field == np.max(fit_result.magnet_field))[
            0
        ][0]
        qi_to = Curve(
            fit_result.magnet_field[: max_idx + 1],
            fit_result.Q_i[: max_idx + 1],
            color="tab:blue",
            label=r"${}\to{}$T".format(
                fit_result.magnet_field[0], fit_result.magnet_field[max_idx]
            ),
        )
        qi_back = Curve(
            fit_result.magnet_field[max_idx:],
            fit_result.Q_i[max_idx:],
            color="tab:blue",
            line_style="--",
            label=r"${}\to{}$T".format(
                fit_result.magnet_field[max_idx], fit_result.magnet_field[-1]
            ),
        )
        qi_curves.append(qi_to)
        qi_curves.append(qi_back)
        deltaf = Curve(
            fit_result.magnet_field[: max_idx + 1],
            deltaf_over_f0[: max_idx + 1] * 100,
            color="firebrick",
        )
    else:
        qi = Curve(fit_result.magnet_field, fit_result.Q_i, color="tab:blue")
        qi_curves.append(qi)
        deltaf = Curve(fit_result.magnet_field, deltaf_over_f0 * 100, color="firebrick")
    fig = SmartFigure(
        x_label=r"$B_\parallel$",
        y_label="$Q_i$",
        log_scale_y=True,
        title=title,
        figure_style=figure_style,
        elements=qi_curves,
    )

    if show_frequency:
        fig.set_tick_params(
            axis="y", which="both", label_color="tab:blue", color="tab:blue"
        )
        # TODO: Set color of the y_label
        twin_y = fig.create_twin_axis(
            is_y=True, label=r"$\Delta f / f_r$ (%)", elements=[deltaf]
        )
        twin_y.set_tick_params(which="both", color="firebrick", label_color="firebrick")
        twin_y.set_visual_params(label_color="firebrick")

    if show_frequency and fit_frequency:
        fit = gl.FitFromFunction(
            _frequency_ratio, deltaf, color="firebrick", line_style=":"
        )
        twin_y.add_elements(fit)

    return fig
