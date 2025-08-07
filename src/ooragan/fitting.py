import os
import lmfit
from typing import Optional, Literal, Any
from numpy import float64, floating, ndarray, ndindex, arange, mean
from resonator import background, base, shunt, reflection
from numpy.typing import ArrayLike, NDArray
from graphinglib import MultiFigure

from .file_loading import Dataset, File
from .util import plot_triptych, choice


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

    _files: dict[str, File] = {}
    _fit_results: dict[str, base.ResonatorFitter | None] = {}

    def __init__(
        self,
        data: Dataset | File,
        savepath: Optional[str] = None,
    ) -> None:
        if isinstance(data, File):
            self._files.update({"0": data})
            self._fit_results.update({"0": None})
        else:
            self._files.update(data.files)
            for key in data.files.keys():
                self._fit_results.update({key: None})
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
        trim_index: int,
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
                frequency=freq[trim_index:-trim_index],
                data=data[trim_index:-trim_index],
                params=params,
                background_model=background,
            )
        elif fit_method == "reflection":
            result = reflection.LinearReflectionFitter(
                frequency=freq[trim_index:-trim_index],
                data=data[trim_index:-trim_index],
                params=params,
                background_model=background,
            )
        elif fit_method == "kerr_shunt":
            result = shunt.KerrShuntFitter(
                frequency=freq[trim_index:-trim_index],
                data=data[trim_index:-trim_index],
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
            if not param.startswith("s21_") or not param == "VNA Frequency":
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
        save_results: bool = False,
        overwrite_warn: bool = False,
    ) -> None:
        # Verify the saving paths exist
        if save_fig and not os.path.exists(
            os.path.join(self._savepath, "results", "fit_images")
        ):
            os.makedirs(os.path.join(self._savepath, "results", "fit_images"))
        if save_results and not os.path.exists(
            os.path.join(self._savepath, "results", "fit_results")
        ):
            os.mkdir(os.path.join(self._savepath, "results", "fit_results"))

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
        for file in files:
            file_obj = self._files[str(file)]
            frequency = file_obj.vna_frequency.range
            for idx in ndindex(file_obj.shape[:-1]):
                real = file_obj.s21_real.range[idx]
                imag = file_obj.s21_imag.range[idx]
                complex = real + 1j * imag
                input_power = file_obj.vna_power.range[idx]
                for ti in arange(trim_start, len(frequency) // 2, trim_jump):
                    if ti == 0:
                        complex_trim = complex
                        frequency_trim = frequency
                    else:
                        complex_trim = complex[ti:-ti]
                        frequency_trim = frequency[ti:-ti]
                    fitter, photon = self._resonator_fitter(
                        complex_trim,
                        frequency_trim,
                        ti,
                        power=input_power,
                        background=background,
                        fit_method=fit_method,
                    )
                    if self._test_fit(
                        fitter.result,
                        verbose=False,
                        threshold=threshold[idx]
                        if isinstance(threshold, ndarray)
                        else threshold,
                    ):
                        self._fit_results.update({str(file): fitter})
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
                                nodialog=overwrite_warn,
                            )
                        break

    def _plot_fit(
        self,
        result: base.ResonatorFitter,
        complex_data: NDArray,
        frequency: NDArray,
        name: str = "",
        savepath: str = "",
        nodialog: bool = False,
    ) -> MultiFigure:
        """
        Utilitary function to plot the fit result as a triptych (resonator.see.triptych).
        """
        triptych = plot_triptych(
            frequency,
            complex_data,
            fit_result=result,
            three_ticks=True,
        )
        filename = os.path.join(savepath, name + ".svg")
        if os.path.exists(filename) and not nodialog:
            overwrite = choice()
            if overwrite:
                triptych.save(filename)
        else:
            triptych.save(filename)
        return triptych

    # def __getattribute__(self, name: str, /) -> Any:
    #     pass
