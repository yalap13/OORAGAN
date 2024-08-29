import os
import numpy as np
import graphinglib as gl
import pandas as pd

from typing import overload, Optional, Literal
from datetime import datetime
from graphinglib import Figure
from copy import deepcopy

from .dataset import Dataset
from .util import FREQ_UNIT_CONVERSION, plot_triptych
from .resonator_fitter import ResonatorFitter


class DatasetGrapher:
    """
    Grapher for a Dataset object containing raw data.

    Parameters
    ----------
    dataset : Dataset
        Dataset containing the data to graph.
    savepath : str, optional
        Path where to create the ``data_plots`` folder to save the generated plots.
        Defaults to the current working directory.
    image_type : str, optional
        Image file type to save the figures. Defaults to ``"svg"``.
    """

    def __init__(
        self,
        dataset: Dataset,
        savepath: Optional[str] = None,
        image_type: str = "svg",
    ) -> None:
        self._savepath = savepath
        if self._savepath is None:
            self._savepath = os.path.join(os.getcwd(), "data_plots")
        self._dataset = dataset
        self._image_type = image_type

    def plot_mag_vs_freq(
        self,
        file_index: int | list[int] = [],
        power: float | list[float] = [],
        freq_unit: Literal["GHz", "MHz", "kHz"] = "GHz",
        size: tuple | Literal["default"] = "default",
        three_ticks: bool = False,
        title: Optional[str] = None,
        figure_style: str = "default",
        save: bool = True,
    ) -> None:
        """
        Plots the magnitude in dBm as a function of frequency in GHz.

        Parameters
        ----------
        file_index : int or list of int, optional
            Index or list of indices (as displayed in the Dataset table) of files to
            get data from. Defaults to ``[]``.
        power : float or list of float, optional
            If specified, will fetch data for those power values. Defaults to ``[]``.
        freq_unit : {"GHz", "MHz", "kHz"}, optional
            Units of frequency to use. Defaults to ``"GHz"``.
        size : tuple, optional
            Figure size. Default depends on the ``figure_style`` configuration.
        three_ticks : bool, optional
            If ``True``, only three ticks will be displayed on the x axis: the minimum
            frequency, the maximum and the frequency. Defaults to ``False``.
        title : str, optional
            Figure title applied to all figures and appended with the frequency range.
            Defaults to ``None``.
        figure_style : str, optional
            GraphingLib figure style to apply to the plot. See
            [here](https://www.graphinglib.org/doc-1.5.0/handbook/figure_style_file.html#graphinglib-styles-showcase)
            for more info.
        save : bool, optional
            If ``True``, saves the plot at the location specified for the class.
            Defaults to ``True``.
        """
        dataset = self._dataset.slice(file_index=file_index, power=power)
        dataset.convert_complex_to_magphase()
        _power = dataset._data_container.power
        data = dataset._data_container.data
        for file in dataset._data_container.files:
            squeezed_power = (
                np.squeeze(_power[file]) if _power[file].shape != (1,) else _power[file]
            )
            for i, d in enumerate(data[file]):
                if title is not None:
                    new_title = title + "_"
                    new_title += str(
                        np.mean(d[0, :]) / FREQ_UNIT_CONVERSION[freq_unit]
                    )[:4].replace(".", "_")
                    new_title += f"{freq_unit}_{squeezed_power[i]}dBm"
                else:
                    new_title = title
                figure = gl.Figure(
                    f"Frequency ({freq_unit})",
                    "Magnitude (dBm)",
                    title=new_title,
                    size=size,
                    figure_style=figure_style,
                )
                scatter = gl.Scatter(
                    d[0, :] / FREQ_UNIT_CONVERSION[freq_unit], d[1, :], marker_style="."
                )
                figure.add_elements(scatter)
                if three_ticks:
                    figure.set_ticks(
                        xticks=[
                            np.min(d[0, :]) / FREQ_UNIT_CONVERSION[freq_unit],
                            np.mean(d[0, :]) / FREQ_UNIT_CONVERSION[freq_unit],
                            np.max(d[0, :]) / FREQ_UNIT_CONVERSION[freq_unit],
                        ]
                    )
                if save:
                    if not os.path.exists(self._savepath):
                        os.mkdir(self._savepath)
                    if dataset._data_container.start_time[file] is not None:
                        time = datetime.fromtimestamp(
                            dataset._data_container.start_time[file]
                        ).strftime("%Y-%m-%d_%H-%M-%S")
                    else:
                        time = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
                    fname = (
                        "mag_vs_freq_"
                        + str(np.mean(d[0, :]) / FREQ_UNIT_CONVERSION[freq_unit])[
                            :5
                        ].replace(".", "_")
                        + f"{freq_unit}_{squeezed_power[i]}dBm_{time}."
                        + self._image_type
                    )
                    figure.save(os.path.join(self._savepath, fname))
                else:
                    figure.show()

    def plot_phase_vs_freq(
        self,
        file_index: int | list[int] = [],
        power: float | list[float] = [],
        freq_unit: Literal["GHz", "MHz", "kHz"] = "GHz",
        size: tuple | Literal["default"] = "default",
        three_ticks: bool = False,
        title: Optional[str] = None,
        figure_style: str = "default",
        save: bool = True,
    ) -> None:
        """
        Plots the phase in degrees as a function of frequency in GHz.

        Parameters
        ----------
        file_index : int or list of int, optional
            Index or list of indices (as displayed in the Dataset table) of files to
            get data from. Defaults to ``[]``.
        power : float or list of float, optional
            If specified, will fetch data for those power values. Defaults to ``[]``.
        freq_unit : {"GHz", "MHz", "kHz"}, optional
            Units of frequency to use. Defaults to ``"GHz"``.
        size : tuple, optional
            Figure size. Default depends on the ``figure_style`` configuration.
        three_ticks : bool, optional
            If ``True``, only three ticks will be displayed on the x axis: the minimum
            frequency, the maximum and the mean frequency. Defaults to ``False``.
        title : str, optional
            Figure title applied to all figures and appended with the frequency range.
            Defaults to ``None``.
        figure_style : str, optional
            GraphingLib figure style to apply to the plot. See
            [here](https://www.graphinglib.org/doc-1.5.0/handbook/figure_style_file.html#graphinglib-styles-showcase)
            for more info.
        save : bool, optional
            If ``True``, saves the plot at the location specified for the class.
            Defaults to ``True``.
        """
        dataset = self._dataset.slice(file_index=file_index, power=power)
        dataset.convert_complex_to_magphase()
        _power = dataset._data_container.power
        data = dataset._data_container.data
        for file in dataset._data_container.files:
            squeezed_power = (
                np.squeeze(_power[file]) if _power[file].shape != (1,) else _power[file]
            )
            for i, d in enumerate(data[file]):
                if title is not None:
                    new_title = title + "_"
                    new_title += str(
                        np.mean(d[0, :]) / FREQ_UNIT_CONVERSION[freq_unit]
                    )[:4].replace(".", "_")
                    new_title += f"{freq_unit}_{squeezed_power[i]}dBm"
                else:
                    new_title = title
                figure = gl.Figure(
                    f"Frequency ({freq_unit})",
                    "Phase (rad)",
                    title=new_title,
                    size=size,
                    figure_style=figure_style,
                )
                scatter = gl.Scatter(
                    d[0, :] / FREQ_UNIT_CONVERSION[freq_unit], d[2, :], marker_style="."
                )
                figure.add_elements(scatter)
                if three_ticks:
                    figure.set_ticks(
                        xticks=[
                            np.min(d[0, :]) / FREQ_UNIT_CONVERSION[freq_unit],
                            np.mean(d[0, :]) / FREQ_UNIT_CONVERSION[freq_unit],
                            np.max(d[0, :]) / FREQ_UNIT_CONVERSION[freq_unit],
                        ]
                    )
                if save:
                    if not os.path.exists(self._savepath):
                        os.mkdir(self._savepath)
                    if dataset._data_container.start_time[file] is not None:
                        time = datetime.fromtimestamp(
                            dataset._data_container.start_time[file]
                        ).strftime("%Y-%m-%d_%H-%M-%S")
                    else:
                        time = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
                    fname = (
                        "phase_vs_freq_"
                        + str(np.mean(d[0, :]) / FREQ_UNIT_CONVERSION[freq_unit])[
                            :5
                        ].replace(".", "_")
                        + f"{freq_unit}_{squeezed_power[i]}dBm_{time}."
                        + self._image_type
                    )
                    figure.save(os.path.join(self._savepath, fname))
                else:
                    figure.show()

    def plot_complex(
        self,
        file_index: int | list[int] = [],
        power: float | list[float] = [],
        freq_unit: Literal["GHz", "MHz", "kHz"] = "GHz",
        size: tuple | Literal["default"] = "default",
        three_ticks: bool = False,
        title: Optional[str] = None,
        figure_style: str = "default",
        save: bool = True,
    ) -> None:
        """
        Plots the signal in the complex plane.

        Parameters
        ----------
        file_index : int or list of int, optional
            Index or list of indices (as displayed in the Dataset table) of files to
            get data from. Defaults to ``[]``.
        power : float or list of float, optional
            If specified, will fetch data for those power values. Defaults to ``[]``.
        freq_unit : {"GHz", "MHz", "kHz"}, optional
            Units of frequency to use. Defaults to ``"GHz"``.
        size : tuple, optional
            Figure size. Default depends on the ``figure_style`` configuration.
        three_ticks : bool, optional
            If ``True``, only three ticks will be displayed on the x axis: the minimum
            frequency, the maximum and the mean frequency. Defaults to ``False``.
        title : str, optional
            Figure title applied to all figures and appended with the frequency range.
            Defaults to ``None``.
        figure_style : str, optional
            GraphingLib figure style to apply to the plot. See
            [here](https://www.graphinglib.org/doc-1.5.0/handbook/figure_style_file.html#graphinglib-styles-showcase)
            for more info.
        save : bool, optional
            If ``True``, saves the plot at the location specified for the class.
            Defaults to ``True``.
        """
        dataset = self._dataset.slice(file_index=file_index, power=power)
        dataset.convert_magphase_to_complex()
        _power = dataset._data_container.power
        data = dataset._data_container.data
        for file in dataset._data_container.files:
            squeezed_power = (
                np.squeeze(_power[file]) if _power[file].shape != (1,) else _power[file]
            )
            for i, d in enumerate(data[file]):
                if title is not None:
                    new_title = title + "_"
                    new_title += str(
                        np.mean(d[0, :]) / FREQ_UNIT_CONVERSION[freq_unit]
                    )[:4].replace(".", "_")
                    new_title += f"{freq_unit}_{squeezed_power[i]}dBm"
                else:
                    new_title = title
                figure = gl.Figure(
                    "real",
                    "imag",
                    size=size,
                    title=title,
                    figure_style=figure_style,
                )
                scatter = gl.Scatter(d[1, :], d[2, :], marker_style=".")
                hline = gl.Hlines(0, line_styles="--", line_widths=1, colors="silver")
                vline = gl.Vlines(0, line_styles="--", line_widths=1, colors="silver")
                figure.add_elements(scatter, hline, vline)
                if three_ticks:
                    figure.set_ticks(xticks=[np.min(d[1, :]), 0, np.max(d[1, :])])
                if save:
                    if not os.path.exists(self._savepath):
                        os.mkdir(self._savepath)
                    if dataset._data_container.start_time[file] is not None:
                        time = datetime.fromtimestamp(
                            dataset._data_container.start_time[file]
                        ).strftime("%Y-%m-%d_%H-%M-%S")
                    else:
                        time = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
                    fname = (
                        "complex_"
                        + str(np.mean(d[0, :]) / FREQ_UNIT_CONVERSION[freq_unit])[
                            :5
                        ].replace(".", "_")
                        + f"{freq_unit}_{squeezed_power[i]}dBm_{time}."
                        + self._image_type
                    )
                    figure.save(os.path.join(self._savepath, fname))
                else:
                    figure.show()

    def plot_triptych(
        self,
        file_index: int | list[int] = [],
        power: float | list[float] = [],
        title: Optional[str] = None,
        freq_unit: Literal["GHz", "MHz", "kHz"] = "GHz",
        three_ticks: bool = False,
        figure_style: str = "default",
        save: bool = True,
    ) -> None:
        """
        Plots the magnitude vs frequency, the phase vs frequency and the complex data in a
        single figure.

        Parameters
        ----------
        file_index : int or list of int, optional
            Index or list of indices (as displayed in the Dataset table) of files to
            get data from. Defaults to ``[]``.
        power : float or list of float, optional
            If specified, will fetch data for those power values. Defaults to ``[]``.
        title : str, optional
            Figure title applied to all figures and appended with the frequency range.
            Defaults to ``None``.
        freq_unit : {"GHz", "MHz", "kHz"}, optional
            Units of frequency to use. Defaults to ``"GHz"``.
        three_ticks : bool, optional
            If ``True``, only three ticks will be displayed on the x axis: the minimum
            frequency, the maximum and the mean frequency. Defaults to ``False``.
        figure_style : str, optional
            GraphingLib figure style to apply to the plot. See
            [here](https://www.graphinglib.org/doc-1.5.0/handbook/figure_style_file.html#graphinglib-styles-showcase)
            for more info.
        save : bool, optional
            If ``True``, saves the plot at the location specified for the class.
            Defaults to ``True``.
        """
        dataset = self._dataset.slice(file_index=file_index, power=power)
        _power = dataset._data_container.power
        if dataset.format == "complex":
            data_complex = deepcopy(dataset._data_container.data)
            dataset.convert_complex_to_magphase()
            data_magphase = deepcopy(dataset._data_container.data)
        else:
            data_magphase = deepcopy(dataset._data_container.data)
            dataset.convert_magphase_to_complex()
            data_complex = deepcopy(dataset._data_container.data)
        for file in dataset._data_container.files:
            squeezed_power = (
                np.squeeze(_power[file]) if _power[file].shape != (1,) else _power[file]
            )
            for i in range(len(data_complex[file])):
                if title is not None:
                    new_title = title + "_"
                    new_title += str(
                        np.mean(data_complex[i][0, :]) / FREQ_UNIT_CONVERSION[freq_unit]
                    )[:4].replace(".", "_")
                    new_title += f"{freq_unit}_{squeezed_power[i]}dBm"
                else:
                    new_title = title
                triptych = plot_triptych(
                    data_complex[file][i][0, :],
                    data_magphase[file][i][1, :],
                    data_magphase[file][i][2, :],
                    data_complex[file][i][1, :],
                    data_complex[file][i][2, :],
                    freq_unit=freq_unit,
                    title=new_title,
                    three_ticks=three_ticks,
                    figure_style=figure_style,
                )
                if save:
                    if not os.path.exists(self._savepath):
                        os.mkdir(self._savepath)
                    if dataset._data_container.start_time[file] is not None:
                        time = datetime.fromtimestamp(
                            dataset._data_container.start_time[file]
                        ).strftime("%Y-%m-%d_%H-%M-%S")
                    else:
                        time = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
                    fname = (
                        "triptych_"
                        + str(
                            np.mean(data_complex[i][0, :])
                            / FREQ_UNIT_CONVERSION[freq_unit]
                        )[:4].replace(".", "_")
                        + f"{freq_unit}_{squeezed_power[i]}dBm_{time}."
                        + self._image_type
                    )
                    triptych.save(os.path.join(self._savepath, fname))
                else:
                    triptych.show()


class ResonatorFitterGrapher:
    """
    Grapher for a ResonatorFitter object containing fit results.

    Parameters
    ----------
    res_fitter : ResonatorFitter
        ResonatorFitter containing the data to graph.
    savepath : str, optional
        Path where to create the ``fit_results_plots`` folder to save the generated
        plots. Defaults to the current working directory.
    name : str, optional
        Name given to the saved plots. If left to ``None``, ``name`` will be the date.
    image_type : str, optional
        Image file type to save the figures. Defaults to ``"svg"``.
    match_pattern : dict
        Dictionnary of files associated to a resonance name.
    save_graph_data : bool, optional
        If ``True``, saves the graph's data in a csv file. Defaults to ``False``.
    """

    def __init__(
        self,
        res_fitter: ResonatorFitter,
        savepath: str = None,
        name: Optional[str] = None,
        image_type: str = "svg",
        match_pattern: dict[str, tuple] = None,
        save_graph_data: bool = False,
    ) -> None:
        self._res_fitter = res_fitter
        self._savepath = savepath
        self._name = name if name is not None else datetime.today().strftime("%Y-%m-%d")
        self._image_type = image_type
        self._match_pattern = match_pattern
        self._file_index_dict = (
            self._res_fitter.dataset._data_container._file_index_dict
        )
        self._save_graph_data = save_graph_data
        if self._savepath is None:
            self._savepath = os.getcwd()

    def plot_Qi_vs_power(
        self,
        photon: bool = False,
        freq_unit: Literal["GHz", "MHz", "kHz"] = "GHz",
        x_lim: Optional[tuple] = None,
        y_lim: Optional[tuple] = None,
        size: tuple | Literal["default"] = "default",
        title: Optional[str] = None,
        legend_loc: str = "best",
        legend_cols: int = 1,
        figure_style: str = "default",
        save: bool = False,
    ) -> Figure:
        """
        Plots the internal quality factor as a function of input power or photon number.

        Parameters
        ----------
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        freq_unit : {"GHz", "MHz", "kHz"}, optional
            Units of frequency to use. Defaults to ``"GHz"``.
        x_lim : tuple, optional
            Limits for the x-axis.
        y_lim : tuple, optional
            Limits for the y-axis.
        size : tuple, optional
            Figure size. Default depends on the ``figure_style`` configuration.
        title : str, optional
            Figure title.
        legend_loc : str, optional
            Positionning of the legend. Can be one of {"best", "upper right", "upper left",
            "lower left", "lower right", "right", "center left", "center right", "lower center",
            "upper center", "center"} or {"outside upper center", "outside center right",
            "outside lower center"}. Defaults to "best".
        legend_cols : int, optional
            Number of columns in the legend. Defaults to 1.
        figure_style : str, optional
            GraphingLib figure style to apply to the plot. See
            [here](https://www.graphinglib.org/doc-1.5.0/handbook/figure_style_file.html#graphinglib-styles-showcase)
            for more info.
        save : bool, optional
            If ``True``, saves the plot at the location specified for the class.
            Defaults to ``False``.
        """
        files = list(self._res_fitter.Q_i.keys())
        if photon:
            x_label = "Photon number"
        else:
            x_label = "Input power (dBm)"
        figure = gl.Figure(
            x_label,
            r"$Q_i$",
            log_scale_y=True,
            log_scale_x=photon,
            x_lim=x_lim,
            y_lim=y_lim,
            size=size,
            figure_style=figure_style,
            title=title,
        )
        if self._match_pattern is not None:
            for label, indices in self._match_pattern.items():
                power = []
                Qi = []
                Qi_err = []
                for i in indices:
                    if photon:
                        power.extend(
                            self._res_fitter.photon_number[
                                self._file_index_dict[str(i)]
                            ]
                        )
                    else:
                        power.extend(
                            self._res_fitter.input_power[self._file_index_dict[str(i)]]
                        )
                    Qi.extend(self._res_fitter.Q_i[self._file_index_dict[str(i)]])
                    Qi_err.extend(
                        self._res_fitter.Q_i_err[self._file_index_dict[str(i)]]
                    )
                    files.remove(self._file_index_dict[str(i)])
                Qi = [x for _, x in sorted(zip(power, Qi), key=lambda pair: pair[0])]
                Qi_err = [
                    x for _, x in sorted(zip(power, Qi_err), key=lambda pair: pair[0])
                ]
                power.sort()
                curve = gl.Curve(power, Qi, label=label)
                curve.add_errorbars(y_error=Qi_err)
                figure.add_elements(curve)
        for file in files:
            label = f"{self._res_fitter.f_r[file][0]/FREQ_UNIT_CONVERSION[freq_unit]:.3f} {freq_unit}"
            power = (
                self._res_fitter.photon_number[file]
                if photon
                else self._res_fitter.input_power[file]
            )
            Qi = self._res_fitter.Q_i[file]
            Qi_err = self._res_fitter.Q_i_err[file]
            power.sort()
            curve = gl.Curve(power, Qi, label=label)
            curve.add_errorbars(y_error=Qi_err)
            figure.add_elements(curve)
        if save:
            if not os.path.exists(os.path.join(self._savepath, "fit_results_plots")):
                os.mkdir(os.path.join(self._savepath, "fit_results_plots"))
            name = f"Qi_vs_power_{self._name}." + self._image_type
            path = os.path.join(self._savepath, "fit_results_plots", name)
            figure.save(path, legend_loc=legend_loc, legend_cols=legend_cols)
        else:
            figure.show(legend_loc=legend_loc, legend_cols=legend_cols)
        if self._save_graph_data:
            name = f"Qi_vs_power_{self._name}.csv"
            path = os.path.join(self._savepath, "fit_results_plots", name)
            temp = {}
            for element in figure._elements:
                temp[element._label] = {
                    "power": element._x_data,
                    "Qi": element._y_data,
                    "Qi_err": element._y_error,
                }
            df = pd.DataFrame(
                {
                    (key, sub_key): pd.Series(val)
                    for key, d in temp.items()
                    for sub_key, val in d.items()
                }
            )
            df.to_csv(path)
        return figure

    def plot_Qc_vs_power(
        self,
        photon: bool = False,
        freq_unit: Literal["GHz", "MHz", "kHz"] = "GHz",
        x_lim: Optional[tuple] = None,
        y_lim: Optional[tuple] = None,
        size: tuple | Literal["default"] = "default",
        title: Optional[str] = None,
        legend_loc: str = "best",
        legend_cols: int = 1,
        figure_style: str = "default",
        save: bool = False,
    ) -> Figure:
        """
        Plots the coupling quality factor as a function of input power or photon number.

        Parameters
        ----------
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        freq_unit : {"GHz", "MHz", "kHz"}, optional
            Units of frequency to use. Defaults to ``"GHz"``.
        x_lim : tuple, optional
            Limits for the x-axis.
        y_lim : tuple, optional
            Limits for the y-axis.
        size : tuple, optional
            Figure size. Default depends on the ``figure_style`` configuration.
        title : str, optional
            Figure title.
        legend_loc : str, optional
            Positionning of the legend. Can be one of {"best", "upper right", "upper left",
            "lower left", "lower right", "right", "center left", "center right", "lower center",
            "upper center", "center"} or {"outside upper center", "outside center right",
            "outside lower center"}. Defaults to "best".
        legend_cols : int, optional
            Number of columns in the legend. Defaults to 1.
        figure_style : str, optional
            GraphingLib figure style to apply to the plot. See
            [here](https://www.graphinglib.org/doc-1.5.0/handbook/figure_style_file.html#graphinglib-styles-showcase)
            for more info.
        save : bool, optional
            If ``True``, saves the plot at the location specified for the class.
            Defaults to ``False``.
        """
        files = list(self._res_fitter.Q_c.keys())
        if photon:
            x_label = "Photon number"
        else:
            x_label = "Input power (dBm)"
        figure = gl.Figure(
            x_label,
            r"$Q_c$",
            log_scale_y=True,
            log_scale_x=photon,
            x_lim=x_lim,
            y_lim=y_lim,
            size=size,
            figure_style=figure_style,
            title=title,
        )
        if self._match_pattern is not None:
            for label, indices in self._match_pattern.items():
                power = []
                Qc = []
                Qc_err = []
                for i in indices:
                    if photon:
                        power.extend(
                            self._res_fitter.photon_number[
                                self._file_index_dict[str(i)]
                            ]
                        )
                    else:
                        power.extend(
                            self._res_fitter.input_power[self._file_index_dict[str(i)]]
                        )
                    Qc.extend(self._res_fitter.Q_c[self._file_index_dict[str(i)]])
                    Qc_err.extend(
                        self._res_fitter.Q_c_err[self._file_index_dict[str(i)]]
                    )
                    files.remove(self._file_index_dict[str(i)])
                Qc = [x for _, x in sorted(zip(power, Qc), key=lambda pair: pair[0])]
                Qc_err = [
                    x for _, x in sorted(zip(power, Qc_err), key=lambda pair: pair[0])
                ]
                power.sort()
                curve = gl.Curve(power, Qc, label=label)
                curve.add_errorbars(y_error=Qc_err)
                figure.add_elements(curve)
        for file in files:
            label = f"{self._res_fitter.f_r[file][0]/FREQ_UNIT_CONVERSION[freq_unit]:.3f} {freq_unit}"
            power = (
                self._res_fitter.photon_number[file]
                if photon
                else self._res_fitter.input_power[file]
            )
            Qc = self._res_fitter.Q_c[file]
            Qc_err = self._res_fitter.Q_c_err[file]
            curve = gl.Curve(power, Qc, label=label)
            curve.add_errorbars(y_error=Qc_err)
            figure.add_elements(curve)
        if save:
            if not os.path.exists(os.path.join(self._savepath, "fit_results_plots")):
                os.mkdir(os.path.join(self._savepath, "fit_results_plots"))
            name = f"Qc_vs_power_{self._name}." + self._image_type
            path = os.path.join(self._savepath, "fit_results_plots", name)
            figure.save(path, legend_loc=legend_loc, legend_cols=legend_cols)
        else:
            figure.show(legend_loc=legend_loc, legend_cols=legend_cols)
        if self._save_graph_data:
            name = f"Qc_vs_power_{self._name}.csv"
            path = os.path.join(self._savepath, "fit_results_plots", name)
            temp = {}
            for element in figure._elements:
                temp[element._label] = {
                    "power": element._x_data,
                    "Qc": element._y_data,
                    "Qc_err": element._y_error,
                }
            df = pd.DataFrame(
                {
                    (key, sub_key): pd.Series(val)
                    for key, d in temp.items()
                    for sub_key, val in d.items()
                }
            )
            df.to_csv(path)
        return figure

    def plot_Qt_vs_power(
        self,
        photon: bool = False,
        freq_unit: Literal["GHz", "MHz", "kHz"] = "GHz",
        x_lim: Optional[tuple] = None,
        y_lim: Optional[tuple] = None,
        size: tuple | Literal["default"] = "default",
        title: Optional[str] = None,
        legend_loc: str = "best",
        legend_cols: int = 1,
        figure_style: str = "default",
        save: bool = False,
    ) -> Figure:
        """
        Plots the total quality factor as a function of input power or photon number.

        Parameters
        ----------
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        freq_unit : {"GHz", "MHz", "kHz"}, optional
            Units of frequency to use. Defaults to ``"GHz"``.
        x_lim : tuple, optional
            Limits for the x-axis.
        y_lim : tuple, optional
            Limits for the y-axis.
        size : tuple, optional
            Figure size. Default depends on the ``figure_style`` configuration.
        title : str, optional
            Figure title.
        legend_loc : str, optional
            Positionning of the legend. Can be one of {"best", "upper right", "upper left",
            "lower left", "lower right", "right", "center left", "center right", "lower center",
            "upper center", "center"} or {"outside upper center", "outside center right",
            "outside lower center"}. Defaults to "best".
        legend_cols : int, optional
            Number of columns in the legend. Defaults to 1.
        figure_style : str, optional
            GraphingLib figure style to apply to the plot. See
            [here](https://www.graphinglib.org/doc-1.5.0/handbook/figure_style_file.html#graphinglib-styles-showcase)
            for more info.
        save : bool, optional
            If ``True``, saves the plot at the location specified for the class.
            Defaults to ``False``.
        """
        files = list(self._res_fitter.Q_t.keys())
        if photon:
            x_label = "Photon number"
        else:
            x_label = "Input power (dBm)"
        figure = gl.Figure(
            x_label,
            r"$Q_t$",
            log_scale_y=True,
            log_scale_x=photon,
            x_lim=x_lim,
            y_lim=y_lim,
            size=size,
            figure_style=figure_style,
            title=title,
        )
        if self._match_pattern is not None:
            for label, indices in self._match_pattern.items():
                power = []
                Qt = []
                Qt_err = []
                for i in indices:
                    if photon:
                        power.extend(
                            self._res_fitter.photon_number[
                                self._file_index_dict[str(i)]
                            ]
                        )
                    else:
                        power.extend(
                            self._res_fitter.input_power[self._file_index_dict[str(i)]]
                        )
                    Qt.extend(self._res_fitter.Q_t[self._file_index_dict[str(i)]])
                    Qt_err.extend(
                        self._res_fitter.Q_t_err[self._file_index_dict[str(i)]]
                    )
                    files.remove(self._file_index_dict[str(i)])
                Qt = [x for _, x in sorted(zip(power, Qt), key=lambda pair: pair[0])]
                Qt_err = [
                    x for _, x in sorted(zip(power, Qt_err), key=lambda pair: pair[0])
                ]
                power.sort()
                curve = gl.Curve(power, Qt, label=label)
                curve.add_errorbars(y_error=Qt_err)
                figure.add_elements(curve)
        for file in files:
            label = f"{self._res_fitter.f_r[file][0]/FREQ_UNIT_CONVERSION[freq_unit]:.3f} {freq_unit}"
            power = (
                self._res_fitter.photon_number[file]
                if photon
                else self._res_fitter.input_power[file]
            )
            Qt = self._res_fitter.Q_t[file]
            Qt_err = self._res_fitter.Q_t_err[file]
            curve = gl.Curve(power, Qt, label=label)
            curve.add_errorbars(y_error=Qt_err)
            figure.add_elements(curve)
        if save:
            if not os.path.exists(os.path.join(self._savepath, "fit_results_plots")):
                os.mkdir(os.path.join(self._savepath, "fit_results_plots"))
            name = f"Qt_vs_power_{self._name}." + self._image_type
            path = os.path.join(self._savepath, "fit_results_plots", name)
            figure.save(path, legend_loc=legend_loc, legend_cols=legend_cols)
        else:
            figure.show(legend_loc=legend_loc, legend_cols=legend_cols)
        if self._save_graph_data:
            name = f"Qt_vs_power_{self._name}.csv"
            path = os.path.join(self._savepath, "fit_results_plots", name)
            temp = {}
            for element in figure._elements:
                temp[element._label] = {
                    "power": element._x_data,
                    "Qt": element._y_data,
                    "Qt_err": element._y_error,
                }
            df = pd.DataFrame(
                {
                    (key, sub_key): pd.Series(val)
                    for key, d in temp.items()
                    for sub_key, val in d.items()
                }
            )
            df.to_csv(path)
        return figure

    def plot_Fshift_vs_power(
        self,
        f_design: dict[str, float],
        photon: bool = False,
        freq_unit: Literal["GHz", "MHz", "kHz"] = "GHz",
        x_lim: Optional[tuple] = None,
        y_lim: Optional[tuple] = None,
        size: tuple | Literal["default"] = "default",
        title: Optional[str] = None,
        legend_loc: str = "best",
        legend_cols: int = 1,
        figure_style: str = "default",
        save: bool = False,
    ) -> Figure:
        """
        Plots the frequency shift as a function of input power or photon number.

        Parameters
        ----------
        f_design : dict of float
            Designed frequency of the attributed resonators formatted as
            ``{"<file_index>": <frequency in Hz>}`` with the ``file_index`` being the
            same index as used in the Dataset and ResonatorFitter classes.
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        freq_unit : {"GHz", "MHz", "kHz"}, optional
            Units of frequency to use. Defaults to ``"GHz"``.
        x_lim : tuple, optional
            Limits for the x-axis.
        y_lim : tuple, optional
            Limits for the y-axis.
        size : tuple, optional
            Figure size. Default depends on the ``figure_style`` configuration.
        title : str, optional
            Figure title.
        legend_loc : str, optional
            Positionning of the legend. Can be one of {"best", "upper right", "upper left",
            "lower left", "lower right", "right", "center left", "center right", "lower center",
            "upper center", "center"} or {"outside upper center", "outside center right",
            "outside lower center"}. Defaults to "best".
        legend_cols : int, optional
            Number of columns in the legend. Defaults to 1.
        figure_style : str, optional
            GraphingLib figure style to apply to the plot. See
            [here](https://www.graphinglib.org/doc-1.5.0/handbook/figure_style_file.html#graphinglib-styles-showcase)
            for more info.
        save : bool, optional
            If ``True``, saves the plot at the location specified for the class.
            Defaults to ``False``.
        """
        files = list(self._res_fitter.f_r.keys())
        F_diff = {
            self._file_index_dict[i]: self._res_fitter.f_r[self._file_index_dict[i]]
            - f_design[i]
            for i in f_design.keys()
        }
        F_diff_err = {
            self._file_index_dict[i]: self._res_fitter.f_r_err[self._file_index_dict[i]]
            for i in f_design.keys()
        }
        if photon:
            x_label = "Photon number"
        else:
            x_label = "Input power (dBm)"
        figure = gl.Figure(
            x_label,
            rf"$f_r-f_0$ ({freq_unit})",
            log_scale_x=photon,
            x_lim=x_lim,
            y_lim=y_lim,
            size=size,
            figure_style=figure_style,
            title=title,
        )
        if self._match_pattern is not None:
            for label, indices in self._match_pattern.items():
                power = []
                fshift = []
                fshift_err = []
                for i in indices:
                    if photon:
                        power.extend(
                            self._res_fitter.photon_number[
                                self._file_index_dict[str(i)]
                            ]
                        )
                    else:
                        power.extend(
                            self._res_fitter.input_power[self._file_index_dict[str(i)]]
                        )
                    fshift.extend(
                        F_diff[self._file_index_dict[str(i)]]
                        / FREQ_UNIT_CONVERSION[freq_unit]
                    )
                    fshift_err.extend(
                        F_diff_err[self._file_index_dict[str(i)]]
                        / FREQ_UNIT_CONVERSION[freq_unit]
                    )
                    files.remove(self._file_index_dict[str(i)])
                fshift = [
                    x for _, x in sorted(zip(power, fshift), key=lambda pair: pair[0])
                ]
                fshift_err = [
                    x
                    for _, x in sorted(zip(power, fshift_err), key=lambda pair: pair[0])
                ]
                power.sort()
                curve = gl.Curve(power, fshift, label=label)
                curve.add_errorbars(y_error=fshift_err)
                figure.add_elements(curve)
        for file in files:
            label = f"{self._res_fitter.f_r[file][0]/FREQ_UNIT_CONVERSION[freq_unit]:.3f} {freq_unit}"
            power = (
                self._res_fitter.photon_number[file]
                if photon
                else self._res_fitter.input_power[file]
            )
            fshift = F_diff[file] / FREQ_UNIT_CONVERSION[freq_unit]
            fshift_err = F_diff_err[file] / FREQ_UNIT_CONVERSION[freq_unit]
            curve = gl.Curve(power, fshift, label=label)
            curve.add_errorbars(y_error=fshift_err)
            figure.add_elements(curve)
        if save:
            if not os.path.exists(os.path.join(self._savepath, "fit_results_plots")):
                os.mkdir(os.path.join(self._savepath, "fit_results_plots"))
            name = f"Fshift_vs_power_{self._name}." + self._image_type
            path = os.path.join(self._savepath, "fit_restults_plots", name)
            figure.save(path, legend_loc=legend_loc, legend_cols=legend_cols)
        else:
            figure.show(legend_loc=legend_loc, legend_cols=legend_cols)
        if self._save_graph_data:
            name = f"Fshift_vs_power_{self._name}.csv"
            path = os.path.join(self._savepath, "fit_results_plots", name)
            temp = {}
            for element in figure._elements:
                temp[element._label] = {
                    "power": element._x_data,
                    "Fshift": element._y_data,
                    "Fshift_err": element._y_error,
                }
            df = pd.DataFrame(
                {
                    (key, sub_key): pd.Series(val)
                    for key, d in temp.items()
                    for sub_key, val in d.items()
                }
            )
            df.to_csv(path)
        return figure

    def plot_Fr_vs_power(
        self,
        photon: bool = False,
        freq_unit: Literal["GHz", "MHz", "kHz"] = "GHz",
        x_lim: Optional[tuple] = None,
        y_lim: Optional[tuple] = None,
        size: tuple | Literal["default"] = "default",
        title: Optional[str] = None,
        legend_loc: str = "best",
        legend_cols: int = 1,
        figure_style: str = "default",
        save: bool = False,
    ) -> Figure:
        """
        Plots the resonance frequency as a function of input power or photon number.

        Parameters
        ----------
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        freq_unit : {"GHz", "MHz", "kHz"}, optional
            Units of frequency to use. Defaults to ``"GHz"``.
        x_lim : tuple, optional
            Limits for the x-axis.
        y_lim : tuple, optional
            Limits for the y-axis.
        size : tuple, optional
            Figure size. Default depends on the ``figure_style`` configuration.
        title : str, optional
            Figure title.
        legend_loc : str, optional
            Positionning of the legend. Can be one of {"best", "upper right", "upper left",
            "lower left", "lower right", "right", "center left", "center right", "lower center",
            "upper center", "center"} or {"outside upper center", "outside center right",
            "outside lower center"}. Defaults to "best".
        legend_cols : int, optional
            Number of columns in the legend. Defaults to 1.
        figure_style : str, optional
            GraphingLib figure style to apply to the plot. See
            [here](https://www.graphinglib.org/doc-1.5.0/handbook/figure_style_file.html#graphinglib-styles-showcase)
            for more info.
        save : bool, optional
            If ``True``, saves the plot at the location specified for the class.
            Defaults to ``False``.
        """
        files = list(self._res_fitter.f_r.keys())
        if photon:
            x_label = "Photon number"
        else:
            x_label = "Input power (dBm)"
        figure = gl.Figure(
            x_label,
            rf"$f_r$ ({freq_unit})",
            log_scale_x=photon,
            x_lim=x_lim,
            y_lim=y_lim,
            size=size,
            figure_style=figure_style,
            title=title,
        )
        if self._match_pattern is not None:
            for label, indices in self._match_pattern.items():
                power = []
                fr = []
                fr_err = []
                for i in indices:
                    if photon:
                        power.extend(
                            self._res_fitter.photon_number[
                                self._file_index_dict[str(i)]
                            ]
                        )
                    else:
                        power.extend(
                            self._res_fitter.input_power[self._file_index_dict[str(i)]]
                        )
                    fr.extend(
                        self._res_fitter.f_r[self._file_index_dict[str(i)]]
                        / FREQ_UNIT_CONVERSION[freq_unit]
                    )
                    fr_err.extend(
                        self._res_fitter.f_r_err[self._file_index_dict[str(i)]]
                        / FREQ_UNIT_CONVERSION[freq_unit]
                    )
                    files.remove(self._file_index_dict[str(i)])
                fr = [x for _, x in sorted(zip(power, fr), key=lambda pair: pair[0])]
                fr_err = [
                    x for _, x in sorted(zip(power, fr_err), key=lambda pair: pair[0])
                ]
                power.sort()
                curve = gl.Curve(power, fr, label=label)
                curve.add_errorbars(y_error=fr_err)
                figure.add_elements(curve)
        for file in files:
            label = f"{self._res_fitter.f_r[file][0]/FREQ_UNIT_CONVERSION[freq_unit]:.3f} {freq_unit}"
            power = (
                self._res_fitter.photon_number[file]
                if photon
                else self._res_fitter.input_power[file]
            )
            fr = self._res_fitter.f_r[file] / FREQ_UNIT_CONVERSION[freq_unit]
            fr_err = self._res_fitter.f_r_err[file] / FREQ_UNIT_CONVERSION[freq_unit]
            curve = gl.Curve(power, fr, label=label)
            curve.add_errorbars(y_error=fr_err)
            figure.add_elements(curve)
        if save:
            if not os.path.exists(os.path.join(self._savepath, "fit_results_plots")):
                os.mkdir(os.path.join(self._savepath, "fit_results_plots"))
            name = f"Fr_vs_power_{self._name}." + self._image_type
            path = os.path.join(self._savepath, "fit_results_plots", name)
            figure.save(path, legend_loc=legend_loc, legend_cols=legend_cols)
        else:
            figure.show(legend_loc=legend_loc, legend_cols=legend_cols)
        if self._save_graph_data:
            name = f"Fr_vs_power_{self._name}.csv"
            path = os.path.join(self._savepath, "fit_results_plots", name)
            temp = {}
            for element in figure._elements:
                temp[element._label] = {
                    "power": element._x_data,
                    "Fr": element._y_data,
                    "Fr_err": element._y_error,
                }
            df = pd.DataFrame(
                {
                    (key, sub_key): pd.Series(val)
                    for key, d in temp.items()
                    for sub_key, val in d.items()
                }
            )
            df.to_csv(path)
        return figure

    def plot_internal_loss_vs_power(
        self,
        photon: bool = False,
        freq_unit: Literal["GHz", "MHz", "kHz"] = "GHz",
        x_lim: Optional[tuple] = None,
        y_lim: Optional[tuple] = None,
        size: tuple | Literal["default"] = "default",
        title: Optional[str] = None,
        legend_loc: str = "best",
        legend_cols: int = 1,
        figure_style: str = "default",
        save: bool = False,
    ) -> Figure:
        """
        Plots the internal losses as a function of input power or photon number.

        Parameters
        ----------
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        freq_unit : {"GHz", "MHz", "kHz"}, optional
            Units of frequency to use. Defaults to ``"GHz"``.
        x_lim : tuple, optional
            Limits for the x-axis.
        y_lim : tuple, optional
            Limits for the y-axis.
        size : tuple, optional
            Figure size. Default depends on the ``figure_style`` configuration.
        title : str, optional
            Figure title.
        legend_loc : str, optional
            Positionning of the legend. Can be one of {"best", "upper right", "upper left",
            "lower left", "lower right", "right", "center left", "center right", "lower center",
            "upper center", "center"} or {"outside upper center", "outside center right",
            "outside lower center"}. Defaults to "best".
        legend_cols : int, optional
            Number of columns in the legend. Defaults to 1.
        figure_style : str, optional
            GraphingLib figure style to apply to the plot. See
            [here](https://www.graphinglib.org/doc-1.5.0/handbook/figure_style_file.html#graphinglib-styles-showcase)
            for more info.
        save : bool, optional
            If ``True``, saves the plot at the location specified for the class.
            Defaults to ``False``.
        """
        files = list(self._res_fitter.L_i.keys())
        if photon:
            x_label = "Photon number"
        else:
            x_label = "Input power (dBm)"
        figure = gl.Figure(
            x_label,
            r"$L_i$",
            log_scale_y=True,
            log_scale_x=photon,
            x_lim=x_lim,
            y_lim=y_lim,
            size=size,
            figure_style=figure_style,
            title=title,
        )
        if self._match_pattern is not None:
            for label, indices in self._match_pattern.items():
                power = []
                Li = []
                Li_err = []
                for i in indices:
                    if photon:
                        power.extend(
                            self._res_fitter.photon_number[
                                self._file_index_dict[str(i)]
                            ]
                        )
                    else:
                        power.extend(
                            self._res_fitter.input_power[self._file_index_dict[str(i)]]
                        )
                    Li.extend(self._res_fitter.L_i[self._file_index_dict[str(i)]])
                    Li_err.extend(
                        self._res_fitter.L_i_err[self._file_index_dict[str(i)]]
                    )
                    files.remove(self._file_index_dict[str(i)])
                Li = [x for _, x in sorted(zip(power, Li), key=lambda pair: pair[0])]
                Li_err = [
                    x for _, x in sorted(zip(power, Li_err), key=lambda pair: pair[0])
                ]
                power.sort()
                curve = gl.Curve(power, Li, label=label)
                curve.add_errorbars(y_error=Li_err)
                figure.add_elements(curve)
        for file in files:
            label = f"{self._res_fitter.f_r[file][0]/FREQ_UNIT_CONVERSION[freq_unit]:.3f} {freq_unit}"
            power = (
                self._res_fitter.photon_number[file]
                if photon
                else self._res_fitter.input_power[file]
            )
            Li = self._res_fitter.L_i[file]
            Li_err = self._res_fitter.L_i_err[file]
            curve = gl.Curve(power, Li, label=label)
            curve.add_errorbars(y_error=Li_err)
            figure.add_elements(curve)
        if save:
            if not os.path.exists(os.path.join(self._savepath, "fit_results_plots")):
                os.mkdir(os.path.join(self._savepath, "fit_results_plots"))
            name = f"Li_vs_power_{self._name}." + self._image_type
            path = os.path.join(self._savepath, "fit_results_plots", name)
            figure.save(path, legend_loc=legend_loc, legend_cols=legend_cols)
        else:
            figure.show(legend_loc=legend_loc, legend_cols=legend_cols)
        if self._save_graph_data:
            name = f"Li_vs_power_{self._name}.csv"
            path = os.path.join(self._savepath, "fit_results_plots", name)
            temp = {}
            for element in figure._elements:
                temp[element._label] = {
                    "power": element._x_data,
                    "Li": element._y_data,
                    "Li_err": element._y_error,
                }
            df = pd.DataFrame(
                {
                    (key, sub_key): pd.Series(val)
                    for key, d in temp.items()
                    for sub_key, val in d.items()
                }
            )
            df.to_csv(path)
        return figure


@overload
def grapher(
    data_object: Dataset,
    savepath: Optional[str] = None,
    name: Optional[str] = None,
    file_type: str = "svg",
    match_pattern: Optional[dict] = None,
    save_graph_data: bool = False,
) -> DatasetGrapher: ...


@overload
def grapher(
    data_object: ResonatorFitter,
    savepath: Optional[str] = None,
    name: Optional[str] = None,
    file_type: str = "svg",
    match_pattern: Optional[dict] = None,
    save_graph_data: bool = False,
) -> ResonatorFitterGrapher: ...


def grapher(
    data_object: Dataset | ResonatorFitter,
    savepath: str = None,
    name: Optional[str] = None,
    image_type: str = "svg",
    match_pattern: Optional[dict] = None,
    save_graph_data: bool = False,
) -> DatasetGrapher | ResonatorFitterGrapher:
    """
    Factory function to create a Grapher according to the type of the data_object parameter.

    .. note:: To see the available plots, see :class:`DatasetGrapher <ooragan.graphing.DatasetGrapher>` or :class:`ResonatorFitterGrapher <ooragan.graphing.ResonatorFitterGrapher>`.

    .. seealso:: The Graphers use the `GraphingLib library <https://graphinglib.org/>`_ to generate plots.

    Parameters
    ----------
    data_object : Dataset or ResonatorFitter
        Object from which to create the Grapher.
    savepath : str, optional
        Path where to create the ``plots`` folder to save the generated plots. Defaults
        to the current working directory.
    name : str, optional
        Name given to the saved plots. If left to ``None``, ``name`` will be the date.
    image_type : str, optional
        Image file type to save the figures. Defaults to ``"svg"``.
    match_pattern : dict, optional
        Dictionnary of files associated to a resonance name.
    save_graph_data : bool, optional
        If ``True``, saves the graph's data in a csv file. Defaults to ``False``.
    """
    if isinstance(data_object, Dataset):
        return DatasetGrapher(
            dataset=data_object,
            savepath=savepath,
            image_type=image_type,
        )
    elif isinstance(data_object, ResonatorFitter):
        return ResonatorFitterGrapher(
            res_fitter=data_object,
            savepath=savepath,
            name=name,
            image_type=image_type,
            match_pattern=match_pattern,
            save_graph_data=save_graph_data,
        )
    else:
        raise TypeError(
            "A Grapher can only be initiated from a Dataset or ResonatorFitter object"
        )
