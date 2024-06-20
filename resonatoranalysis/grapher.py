from typing import overload

from .dataset import Dataset
from .resonator_fitter import ResonatorFitter


class DatasetGrapher:
    """
    Grapher for a Dataset object containing raw data.

    Parameters
    ----------
    dataset : Dataset
        Dataset containing the data to graph.
    """

    def __init__(self, dataset: any) -> None:
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

    def plot_all_S_params(self):
        raise NotImplementedError

    def plot_triptique(self):
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
        raise NotImplementedError

    def plot_Qc_vs_power(self, photon: bool = False):
        raise NotImplementedError

    def plot_triptique(self):
        raise NotImplementedError

    def plot_Fshift_vs_power(self, photon: bool = False):
        raise NotImplementedError

    def plot_Fr_vs_power(self, photon: bool = False):
        raise NotImplementedError

    def plot_internal_loss_vs_power(self, photon: bool = False):
        raise NotImplementedError

    def plot_photon_number_vs_power(self):
        raise NotImplementedError


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
