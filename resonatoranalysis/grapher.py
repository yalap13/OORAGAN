import numpy as np
import graphinglib as gl

from typing import overload

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
        """
        Grapher for a Dataset object containing raw data.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing the data to graph.
        """
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
    """

    def __init__(self, res_fitter: ResonatorFitter) -> None:
        """
        Grapher for a ResonatorFitter object containing fit results.

        Parameters
        ----------
        res_fitter : ResonatorFitter
            ResonatorFitter containing the data to graph.
        """
        self._res_fitter = res_fitter

    def plot_Qi_vs_power(self, photon: bool = False):
        """
        Plots the internal quality factor as a function of input power or photon number.

        Parameters
        ----------
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        """
        Qi = self._res_fitter.Q_i
        Qi_err = self._res_fitter.Q_i_err
        if photon:
            power = self._res_fitter.photon_number
        else:
            power = self._res_fitter.dataset._data_container.power

    def plot_Qc_vs_power(self, photon: bool = False):
        """
        Plots the coupling quality factor as a function of input power or photon number.

        Parameters
        ----------
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        """
        Qc = self._res_fitter.Q_c
        Qc_err = self._res_fitter.Q_c_err
        if photon:
            power = self._res_fitter.photon_number
        else:
            power = self._res_fitter.dataset._data_container.power

    def plot_Q_vs_power(self, photon: bool = False):
        """
        Plots the total quality factor as a function of input power or photon number.

        Parameters
        ----------
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        """
        Qt = self._res_fitter.Q_t
        Qt_err = self._res_fitter.Q_t_err
        if photon:
            power = self._res_fitter.photon_number
        else:
            power = self._res_fitter.dataset._data_container.power

    def plot_Fshift_vs_power(self, f_expected: dict, photon: bool = False):
        """
        Plots the frequency shift as a function of input power or photon number.

        Parameters
        ----------
        f_expected : dict
            Expected resonance frequency for each fitted file in the ResonatorFitter.
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        """
        f_fit = self._res_fitter.f_r
        f_fit_err = self._res_fitter.f_r_err
        if photon:
            power = self._res_fitter.photon_number
        else:
            power = self._res_fitter.dataset._data_container.power

    def plot_Fr_vs_power(self, photon: bool = False):
        """
        Plots the fitted resonance frequency as a function of input power
        or photon number.

        Parameters
        ----------
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        """
        f_fit = self._res_fitter.f_r
        f_fit_err = self._res_fitter.f_r_err
        if photon:
            power = self._res_fitter.photon_number
        else:
            power = self._res_fitter.dataset._data_container.power

    def plot_internal_loss_vs_power(self, photon: bool = False):
        """
        Plots the internal loss as a function of input power or photon number.

        Parameters
        ----------
        photon : bool
            If ``True``, plots as a function of photon number. Defaults to ``False``.
        """
        int_loss = self._res_fitter.L_i
        int_loss_err = self._res_fitter.L_i_err
        if photon:
            power = self._res_fitter.photon_number
        else:
            power = self._res_fitter.dataset._data_container.power


@overload
def grapher(data_object: Dataset) -> DatasetGrapher: ...


@overload
def grapher(data_object: ResonatorFitter) -> ResonatorFitterGrapher: ...


def grapher(
    data_object: Dataset | ResonatorFitter,
) -> DatasetGrapher | ResonatorFitterGrapher:
    """
    Factory function to create a Grapher according to the type of the data_object parameter.

    Parameters
    ----------
    data_object : Dataset | ResonatorFitter
        Object from which to create the Grapher.
    """
    if isinstance(data_object, Dataset):
        return DatasetGrapher(data_object)
    elif isinstance(data_object, ResonatorFitter):
        return ResonatorFitterGrapher(data_object)
    else:
        raise TypeError(
            "A Grapher can only be initiated from a Dataset or ResonatorFitter object"
        )
