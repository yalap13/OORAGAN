import os
import numpy as np
import graphinglib as gl
import pandas as pd

from typing import overload, Optional, Literal
from datetime import datetime
from copy import deepcopy

from .dataset import Dataset, HDF5Data, TXTData
from .resonator_fitter import ResonatorFitter


class DatasetGrapher:
    """
    Grapher for a Dataset object containing raw data.

    Parameters
    ----------
    dataset : Dataset
        Dataset containing the data to graph.
    """

    def __init__(self, dataset: Dataset) -> None:
        self._data = dataset.data
        self._power = dataset.power

    def plot_complex(self):
        raise NotImplementedError

    def plot_mag_vs_phase(self):
        raise NotImplementedError

    def plot_S21_vs_freq(self):
        raise NotImplementedError

    def plot_phase_vs_freq(self):
        raise NotImplementedError

    def plot_complex_circle(self):
        raise NotImplementedError


class ResonatorFitterGrapher:
    """
    Grapher for a ResonatorFitter object containing fit results.

    Parameters
    ----------
    res_fitter : ResonatorFitter
        ResonatorFitter containing the data to graph.
    savepath : str, optional
        Path where to create the ``plots`` folder to save the generated plots. Defaults
        to the current working directory.
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
        savepath: Optional[str] = None,
        name: Optional[str] = None,
        image_type: str = "svg",
        match_pattern: dict[str, tuple] = None,
        save_graph_data: bool = False,
    ) -> None:
        self._res_fitter = res_fitter
        self._savepath = savepath if savepath is not None else os.getcwd()
        self._name = name if name is not None else datetime.today().strftime("%Y-%m-%d")
        self._image_type = image_type
        self._match_pattern = match_pattern
        self._file_index_dict = (
            self._res_fitter.dataset._data_container._file_index_dict
        )
        self._save_graph_data = save_graph_data
        if not os.path.exists(os.path.join(self._savepath, "plots")):
            os.mkdir(os.path.join(self._savepath, "plots"))

    def plot_Qi_vs_power(
        self,
        photon: bool = True,
        x_lim: Optional[tuple] = None,
        y_lim: Optional[tuple] = None,
        size: tuple | Literal["default"] = "default",
        title: Optional[str] = None,
        show_grid: bool | Literal["default"] = "default",
        legend_loc: str = "best",
        legend_cols: int = 1,
        figure_style: str = "default",
        save: bool = True,
    ) -> None:
        """
        Plots the internal quality factor as a function of input power or photon number.

        Parameters
        ----------
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        x_lim : tuple, optional
            Limits for the x-axis.
        y_lim : tuple, optional
            Limits for the y-axis.
        size : tuple, optional
            Figure size. Default depends on the ``figure_style`` configuration.
        title : str, optional
            Figure title.
        show_grid : bool, optional
            Display the grid. Default depends on the ``figure_style`` configuration.
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
            Defaults to ``True``.
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
            show_grid=show_grid,
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
                scatter = gl.Scatter(power, Qi, label=label)
                scatter.add_errorbars(y_error=Qi_err)
                figure.add_elements(scatter)
        for file in files:
            label = f"{self._res_fitter.f_r:.3f} GHz"
            power = (
                self._res_fitter.photon_number[file]
                if photon
                else self._res_fitter.input_power[file]
            )
            Qi = self._res_fitter.Q_i[file]
            Qi_err = self._res_fitter.Q_i_err[file]
            scatter = gl.Scatter(power, Qi, label=label)
            scatter.add_errorbars(y_error=Qi_err)
            figure.add_elements(scatter)
        if save:
            name = f"Qi_vs_power_{self._name}." + self._image_type
            path = os.path.join(self._savepath, "plots", name)
            figure.save(path, legend_loc=legend_loc, legend_cols=legend_cols)
        else:
            figure.show(legend_loc=legend_loc, legend_cols=legend_cols)
        if self._save_graph_data:
            name = f"Qi_vs_power_{self._name}.csv"
            path = os.path.join(self._savepath, "plots", name)
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

    def plot_Qc_vs_power(
        self,
        photon: bool = False,
        x_lim: Optional[tuple] = None,
        y_lim: Optional[tuple] = None,
        size: tuple | Literal["default"] = "default",
        title: Optional[str] = None,
        show_grid: bool | Literal["default"] = "default",
        legend_loc: str = "best",
        legend_cols: int = 1,
        figure_style: str = "default",
        save: bool = True,
    ) -> None:
        """
        Plots the coupling quality factor as a function of input power or photon number.

        Parameters
        ----------
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        x_lim : tuple, optional
            Limits for the x-axis.
        y_lim : tuple, optional
            Limits for the y-axis.
        size : tuple, optional
            Figure size. Default depends on the ``figure_style`` configuration.
        title : str, optional
            Figure title.
        show_grid : bool, optional
            Display the grid. Default depends on the ``figure_style`` configuration.
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
            Defaults to ``True``.
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
            show_grid=show_grid,
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
                scatter = gl.Scatter(power, Qc, label=label)
                scatter.add_errorbars(y_error=Qc_err)
                figure.add_elements(scatter)
        for file in files:
            label = f"{self._res_fitter.f_r/1e9:.3f} GHz"
            power = (
                self._res_fitter.photon_number[file]
                if photon
                else self._res_fitter.input_power[file]
            )
            Qc = self._res_fitter.Q_c[file]
            Qc_err = self._res_fitter.Q_c_err[file]
            scatter = gl.Scatter(power, Qc, label=label)
            scatter.add_errorbars(y_error=Qc_err)
            figure.add_elements(scatter)
        if save:
            name = f"Qc_vs_power_{self._name}." + self._image_type
            path = os.path.join(self._savepath, "plots", name)
            figure.save(path, legend_loc=legend_loc, legend_cols=legend_cols)
        else:
            figure.show(legend_loc=legend_loc, legend_cols=legend_cols)
        if self._save_graph_data:
            name = f"Qc_vs_power_{self._name}.csv"
            path = os.path.join(self._savepath, "plots", name)
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

    def plot_Q_vs_power(
        self,
        photon: bool = False,
        x_lim: Optional[tuple] = None,
        y_lim: Optional[tuple] = None,
        size: tuple | Literal["default"] = "default",
        title: Optional[str] = None,
        show_grid: bool | Literal["default"] = "default",
        legend_loc: str = "best",
        legend_cols: int = 1,
        figure_style: str = "default",
        save: bool = False,
    ) -> None:
        """
        Plots the total quality factor as a function of input power or photon number.

        Parameters
        ----------
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        x_lim : tuple, optional
            Limits for the x-axis.
        y_lim : tuple, optional
            Limits for the y-axis.
        size : tuple, optional
            Figure size. Default depends on the ``figure_style`` configuration.
        title : str, optional
            Figure title.
        show_grid : bool, optional
            Display the grid. Default depends on the ``figure_style`` configuration.
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
            show_grid=show_grid,
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
                scatter = gl.Scatter(power, Qt, label=label)
                scatter.add_errorbars(y_error=Qt_err)
                figure.add_elements(scatter)
        for file in files:
            label = f"{self._res_fitter.f_r/1e9:.3f} GHz"
            power = (
                self._res_fitter.photon_number[file]
                if photon
                else self._res_fitter.input_power[file]
            )
            Qt = self._res_fitter.Q_t[file]
            Qt_err = self._res_fitter.Q_t_err[file]
            scatter = gl.Scatter(power, Qt, label=label)
            scatter.add_errorbars(y_error=Qt_err)
            figure.add_elements(scatter)
        if save:
            name = f"Qt_vs_power_{self._name}." + self._image_type
            path = os.path.join(self._savepath, "plots", name)
            figure.save(path, legend_loc=legend_loc, legend_cols=legend_cols)
        else:
            figure.show(legend_loc=legend_loc, legend_cols=legend_cols)
        if self._save_graph_data:
            name = f"Qt_vs_power_{self._name}.csv"
            path = os.path.join(self._savepath, "plots", name)
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

    def plot_Fshift_vs_power(
        self,
        f_design: dict,
        photon: bool = False,
        x_lim: Optional[tuple] = None,
        y_lim: Optional[tuple] = None,
        size: tuple | Literal["default"] = "default",
        title: Optional[str] = None,
        show_grid: bool | Literal["default"] = "default",
        legend_loc: str = "best",
        legend_cols: int = 1,
        figure_style: str = "default",
        save: bool = False,
    ):
        """
        Plots the frequency shift as a function of input power or photon number.

        Parameters
        ----------
        f_design : dict
            Designed frequency of the attributed resonators.
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        x_lim : tuple, optional
            Limits for the x-axis.
        y_lim : tuple, optional
            Limits for the y-axis.
        size : tuple, optional
            Figure size. Default depends on the ``figure_style`` configuration.
        title : str, optional
            Figure title.
        show_grid : bool, optional
            Display the grid. Default depends on the ``figure_style`` configuration.
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
        f_r = {}
        raise NotImplementedError("Not yet implemented")

    def plot_Fr_vs_power(
        self,
        photon: bool = False,
        x_lim: Optional[tuple] = None,
        y_lim: Optional[tuple] = None,
        size: tuple | Literal["default"] = "default",
        title: Optional[str] = None,
        show_grid: bool | Literal["default"] = "default",
        legend_loc: str = "best",
        legend_cols: int = 1,
        figure_style: str = "default",
        save: bool = False,
    ):
        """
        Plots the resonance frequency as a function of input power or photon number.

        Parameters
        ----------
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        x_lim : tuple, optional
            Limits for the x-axis.
        y_lim : tuple, optional
            Limits for the y-axis.
        size : tuple, optional
            Figure size. Default depends on the ``figure_style`` configuration.
        title : str, optional
            Figure title.
        show_grid : bool, optional
            Display the grid. Default depends on the ``figure_style`` configuration.
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
        f_r = {}
        raise NotImplementedError("Not yet implemented")

    def plot_internal_loss_vs_power(
        self,
        photon: bool = False,
        x_lim: Optional[tuple] = None,
        y_lim: Optional[tuple] = None,
        size: tuple | Literal["default"] = "default",
        title: Optional[str] = None,
        show_grid: bool | Literal["default"] = "default",
        legend_loc: str = "best",
        legend_cols: int = 1,
        figure_style: str = "default",
        save: bool = False,
    ):
        """
        Plots the internal losses as a function of input power or photon number.

        Parameters
        ----------
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        x_lim : tuple, optional
            Limits for the x-axis.
        y_lim : tuple, optional
            Limits for the y-axis.
        size : tuple, optional
            Figure size. Default depends on the ``figure_style`` configuration.
        title : str, optional
            Figure title.
        show_grid : bool, optional
            Display the grid. Default depends on the ``figure_style`` configuration.
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
            show_grid=show_grid,
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
                scatter = gl.Scatter(power, Li, label=label)
                scatter.add_errorbars(y_error=Li_err)
                figure.add_elements(scatter)
        for file in files:
            label = f"{self._res_fitter.f_r/1e9:.3f} GHz"
            power = (
                self._res_fitter.photon_number[file]
                if photon
                else self._res_fitter.input_power[file]
            )
            Li = self._res_fitter.L_i[file]
            Li_err = self._res_fitter.L_i_err[file]
            scatter = gl.Scatter(power, Li, label=label)
            scatter.add_errorbars(y_error=Li_err)
            figure.add_elements(scatter)
        if save:
            name = f"Li_vs_power_{self._name}." + self._image_type
            path = os.path.join(self._savepath, "plots", name)
            figure.save(path, legend_loc=legend_loc, legend_cols=legend_cols)
        else:
            figure.show(legend_loc=legend_loc, legend_cols=legend_cols)
        if self._save_graph_data:
            name = f"Li_vs_power_{self._name}.csv"
            path = os.path.join(self._savepath, "plots", name)
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


@overload
def grapher(
    data_object: Dataset,
    savepath: Optional[str] = None,
    name: Optional[str] = None,
    file_type: str = "svg",
    match_pattern: Optional[dict] = None,
) -> DatasetGrapher: ...


@overload
def grapher(
    data_object: ResonatorFitter,
    savepath: Optional[str] = None,
    name: Optional[str] = None,
    file_type: str = "svg",
    match_pattern: Optional[dict] = None,
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

    Parameters
    ----------
    data_object : Dataset | ResonatorFitter
        Object from which to create the Grapher.
    savepath : str, optional
        Path where to create the ``plots`` folder to save the generated plots. Defaults
        to the current working directory.
    name : str, optional
        Name given to the saved plots. If left to ``None``, ``name`` will be the date.
    image_type : str, optional
        Image file type to save the figures. Defaults to ``"svg"``.
    match_pattern : dict
        Dictionnary of files associated to a resonance name.
    save_graph_data : bool, optional
        If ``True``, saves the graph's data in a csv file. Defaults to ``False``.
    """
    if isinstance(data_object, Dataset):
        return DatasetGrapher(data_object, savepath)
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
