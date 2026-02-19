import os
import re
import lmfit
from typing import Optional, Literal, Any, overload
from numpy import float64, floating, ndarray, ndindex, arange, mean, array
from resonator import background, base, shunt, reflection
from numpy.typing import ArrayLike, NDArray
from graphinglib import SmartFigure

from .file_loading import Dataset, File
from .plotting import triptych
from .util import choice


class FitResult:
    """
    Fit results container for a single file.

    Parameters
    ----------
    results : list of resonator.base.ResonatorFitter
        List of ResonatorFitter objects from the :class:`Fitter`.
    photon_nbr : list of float
        List of the computed photon numbers.
    """

    def __init__(
        self, results: list[base.ResonatorFitter], photon_nbr: list[float]
    ) -> None:
        if all(isinstance(fitter, base.ResonatorFitter) for fitter in results):
            self._results = results
        else:
            raise TypeError(
                "Must provide a list of only resonator.base.ResonatorFitter instances"
            )
        self._photon_number = photon_nbr

    @property
    def photon_nbr(self) -> NDArray:
        """
        The computed photon numbers.
        """
        return array(self._photon_number)

    @property
    def Q_c(self) -> NDArray:
        """
        The coupling quality factors.
        """
        return self._get_res("Q_c")

    @property
    def Q_c_error(self) -> NDArray:
        """
        The standard error of the coupling quality factor.
        """
        return self._get_res("Q_c_error")

    @property
    def Q_i(self) -> NDArray:
        """
        The internal quality factors.
        """
        return self._get_res("Q_i")

    @property
    def Q_i_error(self) -> NDArray:
        """
        The standard error of the internal quality factor.
        """
        return self._get_res("Q_i_error")

    @property
    def Q_t(self) -> NDArray:
        """
        The total quality factors.
        """
        return self._get_res("Q_t")

    @property
    def Q_t_error(self) -> NDArray:
        """
        The standard error of the total quality factor.
        """
        return self._get_res("Q_t_error")

    @property
    def f_r(self) -> NDArray:
        """
        The resonance frequency.
        """
        return self._get_res("f_r")

    @property
    def f_r_error(self) -> NDArray:
        """
        The standard error of the resonance frequency.
        """
        return self._get_res("f_r_error")

    @property
    def omega_r(self) -> NDArray:
        """
        The resonance angular frequency.
        """
        return self._get_res("omega_r")

    @property
    def omega_r_error(self) -> NDArray:
        """
        The standard error of the resonance angular frequency.
        """
        return self._get_res("omega_r_error")

    @property
    def internal_loss(self) -> NDArray:
        """
        The internal loss.
        """
        return self._get_res("internal_loss")

    @property
    def internal_loss_error(self) -> NDArray:
        """
        The internal loss error.
        """
        return self._get_res("internal_loss_error")

    def _get_res(self, name: str) -> NDArray:
        """
        Gets the requested value from each ResonatorFitter and returns them into a
        numpy.array.
        """
        out = []
        for fitter in self._results:
            try:
                out.append(fitter.__getattribute__(name))
            except AttributeError:
                out.append(fitter.__getattr__(name))
        return array(out)

    def append(
        self, results: list[base.ResonatorFitter], photon_nbr: list[float]
    ) -> None:
        """
        Append new results to existing FitResult instance.

        Parameters
        ----------
        results : list of resonator.base.ResonatorFitter
            List of ResonatorFitter objects from the :class:`Fitter`.
        photon_nbr : list of float
            List of the computed photon numbers.
        """
        if all(isinstance(fitter, base.ResonatorFitter) for fitter in results):
            for res in results:
                self._results.append(res)
        else:
            raise TypeError(
                "Must provide a list of only resonator.base.ResonatorFitter instances"
            )
        for pn in photon_nbr:
            self._photon_number.append(pn)


class Fitter:
    """
    Fitting object. Takes the raw data as input and acts as a container for the
    fit results.

    .. seealso::

        This object is a wrapper of `Daniel Flanigan's resonator
        library <https://github.com/danielflanigan/resonator>`_.

    Parameters
    ----------
    data : :class:`Dataset` or :class:`File`
        Data containing objects.
    savepath : str, optional
        Where to create the "fit_results" and "fit_images" directories to save
        the fit results and images. The default is the current working
        directory.
    """

    def __init__(
        self,
        data: Dataset | File,
        savepath: Optional[str] = None,
    ) -> None:
        self._files: dict[str, File] = {}
        self._fit_results: dict[str, FitResult] = {}
        if isinstance(data, File):
            self._files.update({"0": data})
        else:
            self._files.update(data.files)
        self._savepath = savepath if savepath is not None else os.getcwd()

    def _assert_all_files_same_shape(self) -> bool:
        """Utilitary method to check all files have the same data shape."""
        if len(self._files) == 1:
            return True
        else:
            first_shape = self._files["0"].shape
            return all(file.shape == first_shape for file in self._files.values())

    def _test_fit(
        self,
        result: Any,
        threshold: Any,
        verbose: bool = False,
    ) -> bool:
        """
        Function taken from Nicolas Bourlet (thanks), available on Gitlab of
        JosePh group. Check whether the lmfit result is correct or not.
        """
        tag = True
        if not result.errorbars:
            tag = False
            if verbose:
                print("No errorbars were calculated")
            return tag

        for param in result.params.values():
            if threshold * abs(param.value) < abs(param.stderr):
                tag = False
                if verbose:
                    print(
                        "Parameter:{} has its error larger than its value".format(
                            param.name
                        )
                    )
        return tag

    def _resonator_fitter(
        self,
        data: NDArray,
        freq: NDArray,
        power: Optional[float] = None,
        params: Optional[lmfit.Parameter] = None,
        fit_method: Literal["shunt", "reflection", "kerr_shunt"] = "shunt",
        background: base.BackgroundModel = background.MagnitudePhaseDelay(),
    ) -> tuple[base.ResonatorFitter, ArrayLike]:
        """
        Wrapper around resonator library.
        """
        if fit_method == "shunt":
            result = shunt.LinearShuntFitter(
                frequency=freq,
                data=data,
                params=params,
                background_model=background,
            )
        elif fit_method == "reflection":
            result = reflection.LinearReflectionFitter(
                frequency=freq,
                data=data,
                params=params,
                background_model=background,
            )
        elif fit_method == "kerr_shunt":
            result = shunt.KerrShuntFitter(
                frequency=freq,
                data=data,
                params=params,
                background_model=background,
            )
        else:
            raise NotImplementedError("Fit method unrecognized")
        if power is not None:
            photon = result.photon_number_from_power(result.f_r, power)
        else:
            photon = 0
        return result, photon

    def _file_naming_util(
        self,
        file: File,
        idx: tuple,
        frequency: floating,
    ) -> str:
        """Utilitary to generate file name given the files parameters."""
        out = f"{frequency:.3f}GHz_"
        for param in file.list_params():
            if not param.startswith("s21_") and not param == "VNA Frequency":
                attr = param.lower().replace(" ", "_")
                value = file.__dict__[attr].range[idx]
                unit = file.__dict__[attr].unit
                out += f"{attr}{value}{unit}_"
        return out

    def fit(
        self,
        files: list[int] = [],
        background: base.BackgroundModel = background.MagnitudePhaseDelay(),
        fit_method: Literal["shunt", "reflection", "kerr_shunt"] = "shunt",
        threshold: float | NDArray[float64] = 0.5,
        trim_start: int = 0,
        trim_jump: int = 10,
        save_fig: bool = False,
        overwrite_warn: bool = False,
    ) -> None:
        r"""
        Fitting specified resonator data.

        Parameters
        ----------
        files : list of int, optional
            List of indices of files for which to fit the data. Defaults to ``[]``
            which fits data for all files.
        background : resonator.base.BackgroundModel, optional
            Background model from the resonator library.
            The default is ``background.MagnitudePhaseDelay()``.
        fit_method : str, optional
            Fitting method, either ``"shunt"``, ``"reflection"`` or ``"kerr"``.
            Defaults to ``"shunt"``.
        write : bool, optional
            If ``True``, saves the fit results in a .txt file in the "fit_results"
            folder in the directory specified by ``savepath``. The default is ``False``.
        threshold : float, optional
            A value greater than 0 which determine the error tolerance on fit values.
            The default is ``0.5`` (50%).
        trim_start : int, optional
            Where to start trimming the data. The default is ``0``.
        trim_jump : int, optional
            Step between the trimming of the data to fit better. The default is ``10``.
        save_fig : bool, optional
            If ``True``, saves the fit plots generated by the function in the "images"
            folder in the directory specified by ``savepath``. The default is ``False``.
        overwrite_warn : bool, optional
            If ``True``, does not display the overwriting warning popup.
            The default is ``False``.

        Notes
        -----
        In the **shunt** mode, the resonance is fitted using

        .. math:: S_{21}(f) = 1-\frac{\frac{Q}{Q_c}}{1+2iQ\frac{f-f_0}{f_0}}

        In the **reflection** mode, the resonance is fitted using

        .. math:: S_{21}(f) = -1+\frac{\frac{Q}{Q_c}}{1+2iQ\frac{f-f_0}{f_0}}

        The fit itself is performed by the lmfit library.
        The `fit` method uses an auxilary method `_test_fit` to verify the maximum error tolerance
        on the fit results is respected. If not, the data is trimmed by a specified amount on both
        sides and the fits is tried again.

        .. seealso:: See the `lmfit library <https://lmfit.github.io/lmfit-py/>`_.

        .. note::

            Once the data has been fitted using :py:meth:`fit <ResonatorFitter.fit>`
            method, the fit result figures (triptychs) are saved in the ``_fit_figures``
            dictionnary of the `ResonatorFitter` instance.
        """
        # Verify the saving path exist
        if save_fig and not os.path.exists(
            os.path.join(self._savepath, "results", "fit_images")
        ):
            os.makedirs(os.path.join(self._savepath, "results", "fit_images"))

        if not files:
            files = list(map(int, self._files.keys()))

        # Verify the shape of the threshold matches the shape of the files' data
        if isinstance(threshold, ndarray):
            if self._assert_all_files_same_shape():
                dim = self._files["0"].shape[:-1]
                if dim != threshold.shape:
                    raise ValueError(
                        "Shape of array-like threshold must match data "
                        + f"shape, in this case {dim}"
                    )
            else:
                raise ValueError(
                    "All files must be of the same shape to provide a threshold"
                    + "array"
                )
        fail_count = 0
        for file in files:
            if str(file) in self._fit_results.keys():
                continue
            file_obj = self._files[str(file)]
            frequency = file_obj.vna_frequency.range
            temp = []
            temp_photon = []
            for idx in ndindex(file_obj.shape[:-1]):
                real = file_obj.s21_real.range[idx]
                imag = file_obj.s21_imag.range[idx]
                complex = real + 1j * imag
                input_power = (
                    file_obj.vna_power.range[idx] + file_obj.cryostat_attenuation
                )
                if "Variable Attenuator" in file_obj.list_params():
                    input_power -= file_obj.variable_attenuator.range[idx]
                succeeded = False
                for ti in arange(trim_start, len(frequency) // 2, trim_jump):
                    if ti == 0:
                        complex_trim = complex
                        frequency_trim = frequency
                    else:
                        complex_trim = complex[ti:-ti]
                        frequency_trim = frequency[ti:-ti]
                    try:
                        fitter, photon = self._resonator_fitter(
                            complex_trim,
                            frequency_trim,
                            power=input_power,
                            background=background,
                            fit_method=fit_method,
                        )
                    except ValueError:
                        continue
                    if self._test_fit(
                        fitter.result,
                        verbose=False,
                        threshold=threshold[idx]
                        if isinstance(threshold, ndarray)
                        else threshold,
                    ):
                        succeeded = True
                        temp.append(fitter)
                        temp_photon.append(photon)
                        if save_fig:
                            _ = self._plot_fit(
                                fitter,
                                complex_trim,
                                frequency_trim,
                                name=self._file_naming_util(
                                    self._files[str(file)], idx, mean(frequency)
                                ),
                                savepath=os.path.join(
                                    self._savepath, "results", "fit_images"
                                ),
                                nodialog=not overwrite_warn,
                            )
                        break
                if not succeeded:
                    fail_count += 1

            if str(file) in self._fit_results.keys():
                self._fit_results[str(file)].append(temp, temp_photon)
            else:
                self._fit_results.update({str(file): FitResult(temp, temp_photon)})
        print("{} fit failures".format(fail_count))

    def _plot_fit(
        self,
        result: base.ResonatorFitter,
        complex_data: NDArray,
        frequency: NDArray,
        name: str = "",
        savepath: str = "",
        nodialog: bool = False,
    ) -> SmartFigure:
        """
        Utilitary function to plot the fit result as a triptych (resonator.see.triptych).
        """
        fig = triptych(
            frequency,
            complex_data,
            fit_result=result,
            three_ticks=True,
        )
        filename = os.path.join(savepath, name + ".svg")
        if os.path.exists(filename) and not nodialog:
            overwrite = choice()
            if overwrite:
                fig.save(filename)
        else:
            fig.save(filename)
        return fig

    def __getattribute__(self, name: str) -> FitResult | Any:
        if not name.startswith("__") and re.fullmatch(r"f\d+", name):
            try:
                return self._fit_results[name.removeprefix("f")]
            except KeyError:
                raise IndexError(f"No fit result with index {name.removeprefix('f')}")
        return super().__getattribute__(name)

    @overload
    def __getitem__(self, index: int) -> FitResult: ...

    @overload
    def __getitem__(self, index: slice) -> list[FitResult]: ...

    def __getitem__(self, index: int | slice) -> FitResult | list[FitResult]:
        """
        Returns the FitResult(s) with the given index (or indices).

        Parameters
        ----------
        index : int or slice
            The index or indices (as a slice) of FitResults to get from this Fitter.
            If the start or stop of the slice are left empty, will return all FitResults
            with indices inside the given bounds. This means for returning all FitResults,
            use slice ``[:]``.

        Returns
        -------
        FitResults | list[FitResults]
            If a single index is specified, a single FitResult is returned. A list otherwise.
        """
        total_files = len(self._files.keys())
        if isinstance(index, int):
            if not -total_files <= index <= total_files - 1:
                raise IndexError(
                    "index {} out of bounds for number of files {}".format(
                        index, total_files
                    )
                )
            try:
                if index < 0:
                    index = total_files + index
                return self._fit_results[str(index)]
            except KeyError:
                raise IndexError("no results for file with index {}".format(index))
        if isinstance(index, slice):
            if index.start and not -total_files <= index.start <= total_files - 1:
                raise IndexError(
                    "index [{}, {}] out of bounds for number of files {}".format(
                        index.start, index.stop, total_files
                    )
                )
            if index.stop and not -total_files <= index.stop <= total_files - 1:
                raise IndexError(
                    "index [{}, {}] out of bounds for number of files {}".format(
                        index.start, index.stop, total_files
                    )
                )
            idx_list = list(range(*index.indices(total_files)))
            if index.start is None or index.stop is None:
                idx_list = set(idx_list).intersection(
                    map(int, list(self._fit_results.keys()))
                )
            try:
                return [self._fit_results[str(i)] for i in idx_list]
            except KeyError as e:
                raise IndexError("no results for file with index {}".format(e.args[0]))
        raise TypeError("indices must be int or slice, not {}".format(type(index)))
