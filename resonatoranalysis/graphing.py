import os
import numpy as np
import graphinglib as gl

from typing import overload, Optional, Literal
from datetime import datetime

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
    file_type : str, optional
        Image file type to save the figures. Defaults to ``"svg"``.
    """

    def __init__(
        self,
        res_fitter: ResonatorFitter,
        savepath: Optional[str] = None,
        name: Optional[str] = None,
        file_type: str = "svg",
    ) -> None:
        self._res_fitter = res_fitter
        self._savepath = savepath if savepath is not None else os.getcwd()
        self._name = name if name is not None else datetime.today().strftime("%Y-%m-%d")
        self._file_type = file_type
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
        Qi = {}
        Qi_err = {}
        power = {}
        f_r = self._res_fitter.f_r
        files = list(self._res_fitter.Q_i.keys())
        labels = [
            str(np.mean(f_r[file][0] / 1e9))[:5].replace(".", "_") for file in files
        ]
        for i, label in enumerate(labels):
            if label in Qi:
                Qi[label] = np.append(Qi[label], self._res_fitter.Q_i[files[i]])
                Qi_err[label] = np.append(
                    Qi_err[label], self._res_fitter.Q_i_err[files[i]]
                )
                if photon:
                    power[label] = np.append(
                        power[label], self._res_fitter.photon_number[files[i]]
                    )
                else:
                    power[label] = np.append(
                        power[label], self._res_fitter.input_power[files[i]]
                    )
            else:
                Qi[label] = self._res_fitter.Q_i[files[i]]
                Qi_err[label] = self._res_fitter.Q_i_err[files[i]]
                if photon:
                    power[label] = self._res_fitter.photon_number[files[i]]
                else:
                    power[label] = self._res_fitter.input_power[files[i]]
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
        for label in Qi.keys():
            scatter = gl.Scatter(power[label], Qi[label], label=label + " GHz")
            scatter.add_errorbars(y_error=Qi_err[label])
            figure.add_elements(scatter)
        if save:
            name = f"Qi_vs_power_{self._name}." + self._file_type
            path = os.path.join(self._savepath, "plots", name)
            figure.save(path, legend_loc=legend_loc, legend_cols=legend_cols)
        else:
            figure.show(legend_loc=legend_loc, legend_cols=legend_cols)

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
        Qc = {}
        Qc_err = {}
        power = {}
        f_r = self._res_fitter.f_r
        files = list(self._res_fitter.Q_c.keys())
        labels = [
            str(np.mean(f_r[file][0] / 1e9))[:5].replace(".", "_") for file in files
        ]
        for i, label in enumerate(labels):
            if label in Qc:
                Qc[label] = np.append(Qc[label], self._res_fitter.Q_c[files[i]])
                Qc_err[label] = np.append(
                    Qc_err[label], self._res_fitter.Q_c_err[files[i]]
                )
                if photon:
                    power[label] = np.append(
                        power[label], self._res_fitter.photon_number[files[i]]
                    )
                else:
                    power[label] = np.append(
                        power[label], self._res_fitter.input_power[files[i]]
                    )
            else:
                Qc[label] = self._res_fitter.Q_c[files[i]]
                Qc_err[label] = self._res_fitter.Q_c_err[files[i]]
                if photon:
                    power[label] = self._res_fitter.photon_number[files[i]]
                else:
                    power[label] = self._res_fitter.input_power[files[i]]
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
        for label in Qc.keys():
            scatter = gl.Scatter(power[label], Qc[label], label=label + " GHz")
            scatter.add_errorbars(y_error=Qc_err[label])
            figure.add_elements(scatter)
        if save:
            name = f"Qc_vs_power_{self._name}." + self._file_type
            path = os.path.join(self._savepath, "plots", name)
            figure.save(path, legend_loc=legend_loc, legend_cols=legend_cols)
        else:
            figure.show(legend_loc=legend_loc, legend_cols=legend_cols)

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
        Qt = {}
        Qt_err = {}
        power = {}
        f_r = self._res_fitter.f_r
        files = list(self._res_fitter.Q_t.keys())
        labels = [
            str(np.mean(f_r[file][0] / 1e9))[:5].replace(".", "_") for file in files
        ]
        for i, label in enumerate(labels):
            if label in Qt:
                Qt[label] = np.append(Qt[label], self._res_fitter.Q_t[files[i]])
                Qt_err[label] = np.append(
                    Qt_err[label], self._res_fitter.Q_t_err[files[i]]
                )
                if photon:
                    power[label] = np.append(
                        power[label], self._res_fitter.photon_number[files[i]]
                    )
                else:
                    power[label] = np.append(
                        power[label], self._res_fitter.input_power[files[i]]
                    )
            else:
                Qt[label] = self._res_fitter.Q_t[files[i]]
                Qt_err[label] = self._res_fitter.Q_t_err[files[i]]
                if photon:
                    power[label] = self._res_fitter.photon_number[files[i]]
                else:
                    power[label] = self._res_fitter.input_power[files[i]]
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
        for label in Qt.keys():
            scatter = gl.Scatter(power[label], Qt[label], label=label + " GHz")
            scatter.add_errorbars(y_error=Qt_err[label])
            figure.add_elements(scatter)
        if save:
            name = f"Q_vs_power_{self._name}." + self._file_type
            path = os.path.join(self._savepath, "plots", name)
            figure.save(path, legend_loc=legend_loc, legend_cols=legend_cols)
        else:
            figure.show(legend_loc=legend_loc, legend_cols=legend_cols)

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
        Li = {}
        Li_err = {}
        power = {}
        f_r = self._res_fitter.f_r
        files = list(self._res_fitter.L_i.keys())
        labels = [
            str(np.mean(f_r[file][0] / 1e9))[:5].replace(".", "_") for file in files
        ]
        for i, label in enumerate(labels):
            if label in Li:
                Li[label] = np.append(Li[label], self._res_fitter.L_i[files[i]])
                Li_err[label] = np.append(
                    Li_err[label], self._res_fitter.L_i_err[files[i]]
                )
                if photon:
                    power[label] = np.append(
                        power[label], self._res_fitter.photon_number[files[i]]
                    )
                else:
                    power[label] = np.append(
                        power[label], self._res_fitter.input_power[files[i]]
                    )
            else:
                Li[label] = self._res_fitter.L_i[files[i]]
                Li_err[label] = self._res_fitter.L_i_err[files[i]]
                if photon:
                    power[label] = self._res_fitter.photon_number[files[i]]
                else:
                    power[label] = self._res_fitter.input_power[files[i]]
        if photon:
            x_label = "Photon number"
        else:
            x_label = "Input power (dBm)"
        figure = gl.Figure(
            x_label,
            "Internal losses",
            log_scale_y=True,
            log_scale_x=photon,
            x_lim=x_lim,
            y_lim=y_lim,
            size=size,
            show_grid=show_grid,
            figure_style=figure_style,
            title=title,
        )
        for label in Li.keys():
            scatter = gl.Scatter(power[label], Li[label], label=label + " GHz")
            scatter.add_errorbars(y_error=Li_err[label])
            figure.add_elements(scatter)
        if save:
            name = f"intloss_vs_power_{self._name}." + self._file_type
            path = os.path.join(self._savepath, "plots", name)
            figure.save(path, legend_loc=legend_loc, legend_cols=legend_cols)
        else:
            figure.show(legend_loc=legend_loc, legend_cols=legend_cols)


@overload
def grapher(data_object: Dataset, savepath: str) -> DatasetGrapher: ...


@overload
def grapher(data_object: ResonatorFitter, savepath: str) -> ResonatorFitterGrapher: ...


def grapher(
    data_object: Dataset | ResonatorFitter,
    savepath: str,
) -> DatasetGrapher | ResonatorFitterGrapher:
    """
    Factory function to create a Grapher according to the type of the data_object parameter.

    Parameters
    ----------
    data_object : Dataset | ResonatorFitter
        Object from which to create the Grapher.
    """
    if isinstance(data_object, Dataset):
        return DatasetGrapher(data_object, savepath)
    elif isinstance(data_object, ResonatorFitter):
        return ResonatorFitterGrapher(data_object, savepath)
    else:
        raise TypeError(
            "A Grapher can only be initiated from a Dataset or ResonatorFitter object"
        )
