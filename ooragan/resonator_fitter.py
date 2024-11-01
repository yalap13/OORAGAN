import os
import sys
import lmfit
import numpy as np

from resonator import background, shunt, reflection, base
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Literal
from lmfit import Parameter
from graphinglib import MultiFigure

from .dataset import Dataset
from .util import choice, convert_complex_to_magphase, is_interactive, plot_triptych

if is_interactive():
    from IPython.display import display, clear_output


class ResonatorFitter:
    """
    Resonator fitting object

    .. seealso:: This object is a wrapper of the `resonator <https://github.com/danielflanigan/resonator/>`_ library.

    Parameters
    ----------
    dataset : Dataset
        Dataset of the data to fit.
    savepath : str, optional
        Where to create the "fit_results" and "fit_images" directories to save the fit
        results and images. The default is the current working directory.
    """

    def __init__(self, dataset: Dataset, savepath: Optional[str] = None):
        self.dataset = dataset
        if self.dataset.format == "magphase":
            self.dataset.convert_magphase_to_complex()
        self._fit_results = {}
        self._savepath = savepath if savepath is not None else os.getcwd()

    @property
    def Q_c(self) -> dict:
        """Coupling quality factor"""
        return {
            file: np.array(values["Q_c"]) for file, values in self._fit_results.items()
        }

    @property
    def Q_c_err(self) -> dict:
        """Coupling quality factor error"""
        return {
            file: np.array(values["Q_c_err"])
            for file, values in self._fit_results.items()
        }

    @property
    def Q_i(self) -> dict:
        """Internal quality factor"""
        return {
            file: np.array(values["Q_i"]) for file, values in self._fit_results.items()
        }

    @property
    def Q_i_err(self) -> dict:
        """Internal quality factor error"""
        return {
            file: np.array(values["Q_i_err"])
            for file, values in self._fit_results.items()
        }

    @property
    def Q_t(self) -> dict:
        """Total quality factor"""
        return {
            file: np.array(values["Q_t"]) for file, values in self._fit_results.items()
        }

    @property
    def Q_t_err(self) -> dict:
        """Total quality factor error"""
        return {
            file: np.array(values["Q_t_err"])
            for file, values in self._fit_results.items()
        }

    @property
    def L_c(self) -> dict:
        """Coupling losses"""
        return {
            file: np.array(values["L_c"]) for file, values in self._fit_results.items()
        }

    @property
    def L_c_err(self) -> dict:
        """Coupling losses error"""
        return {
            file: np.array(values["L_c_err"])
            for file, values in self._fit_results.items()
        }

    @property
    def L_i(self) -> dict:
        """Internal losses"""
        return {
            file: np.array(values["L_i"]) for file, values in self._fit_results.items()
        }

    @property
    def L_i_err(self) -> dict:
        """Internal losses error"""
        return {
            file: np.array(values["L_i_err"])
            for file, values in self._fit_results.items()
        }

    @property
    def L_t(self) -> dict:
        """Total losses"""
        return {
            file: np.array(values["L_t"]) for file, values in self._fit_results.items()
        }

    @property
    def L_t_err(self) -> dict:
        """Total losses error"""
        return {
            file: np.array(values["L_t_err"])
            for file, values in self._fit_results.items()
        }

    @property
    def f_r(self) -> dict:
        """Resonance frequency"""
        return {
            file: np.array(values["f_r"]) for file, values in self._fit_results.items()
        }

    @property
    def f_r_err(self) -> dict:
        """Resonance frequency error"""
        return {
            file: np.array(values["f_r_err"])
            for file, values in self._fit_results.items()
        }

    @property
    def photon_number(self) -> dict:
        """Photon number"""
        return {
            file: np.array(values["photon_number"])
            for file, values in self._fit_results.items()
        }

    @property
    def input_power(self) -> dict:
        """Input power"""
        return {
            file: np.array(values["input_power"])
            for file, values in self._fit_results.items()
        }

    def fit(
        self,
        file_index: int | list[int] = [],
        power: float | list[float] = [],
        f_r: float = None,
        couploss: float = 1e-6,
        intloss: float = 1e-6,
        bg: base.BackgroundModel = background.MagnitudePhaseDelay(),
        fit_method: Literal["shunt", "reflection"] = "shunt",
        savepic: bool = False,
        showpic: bool = False,
        write: bool = False,
        threshold: float = 0.5,
        start: int = 0,
        jump: int = 10,
        nodialog: bool = False,
    ) -> None:
        r"""
        Fitting specified resonator data.

        Parameters
        ----------
        file_index : int or list of int, optional
            Index or list of indices of files (as displayed in the Dataset table) for
            which to fit the data. Defaults to ``[]`` which fits data for all files.
        power : float or list of float, optional
            If specified, will fit data for those power values.
            Defaults to ``[]`` which fits for all power values.
        f_r : float, optional
            Resonance frequency, adds to fit parameters. Default is ``None``.
        couploss : int, optional
            Coupling loss of resonator, adds to fit parameters. The default is ``1e-6``.
        intloss : int, optional
            Internal loss of resonator, adds to fit parameters. The default is ``1e-6``.
        bg : resonator.base.BackgroundModel, optional
            Background model from the resonator library.
            The default is ``background.MagnitudePhaseDelay()``.
        fit_method : str, optional
            Fitting method, either ``"shunt"`` or ``"reflection"``. Defaults to ``"shunt"``.
        savepic : bool, optional
            If ``True``, saves the fit plots generated by the function in the "images"
            folder in the directory specified by ``savepath``. The default is ``False``.
        showpic : bool, optional
            If ``True``, the fit pictures will be displayed. Defaults to ``False``.
        write : bool, optional
            If ``True``, saves the fit results in a .txt file in the "fit_results" folder
            in the directory specified by ``savepath``. The default is ``False``.
        threshold : float, optional
            A value greater than 0 which determine the error tolerance on fit values.
            The default is ``1`` (100%).
        start : int, optional
            Where to start trimming the data. The default is ``0``.
        jump : int, optional
            Step between the trimming of the data to fit better. The default is ``10``.
        nodialog : bool, optional
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
        sides and the fit is tried again.

        .. seealso:: See the `lmfit library <https://lmfit.github.io/lmfit-py/>`_.
        """
        if savepic and not os.path.exists(os.path.join(self._savepath, "fit_images")):
            os.mkdir(os.path.join(self._savepath, "fit_images"))
        if write and not os.path.exists(os.path.join(self._savepath, "fit_results")):
            os.mkdir(os.path.join(self._savepath, "fit_results"))

        params = lmfit.Parameters()
        params.add(name="internal_loss", value=intloss)
        params.add(name="coupling_loss", value=couploss)
        params.add(name="resonance_frequency", value=f_r, min=1e9, max=1e10)

        sliced_dataset = self.dataset.slice(file_index=file_index, power=power)
        files = sliced_dataset._data_container.files
        power_points = []
        data = []
        for file in files:
            if sliced_dataset._data_container.power[file] is not None:
                power_points.append(
                    list(
                        np.squeeze(sliced_dataset._data_container.power[file])
                        if sliced_dataset._data_container.power[file].ndim == 2
                        else sliced_dataset._data_container.power[file]
                    )
                )
            else:
                power_points.append(
                    [None] * len(sliced_dataset._data_container.data[file])
                )
            data.append(sliced_dataset._data_container.data[file])

        status = {}
        print("Fitting progress:")
        for file in files:
            status[file] = "  pending..."
            print(file + status[file])
        if is_interactive():
            display("")
        else:
            sys.stdout.write(f"\033[{len(files)}A")

        for i, file_power_points in enumerate(power_points):
            failed = []
            status[files[i]] = "  \033[34mfitting...\033[00m"
            if is_interactive():
                clear_output(wait=True)
                for file in files:
                    print(file + status[file])
                display("")
            else:
                sys.stdout.write("\33[2K")
                print(files[i] + status[files[i]], end="\r")
            data_temp = {
                "Q_c": [],
                "Q_c_err": [],
                "Q_i": [],
                "Q_i_err": [],
                "Q_t": [],
                "Q_t_err": [],
                "L_c": [],
                "L_c_err": [],
                "L_i": [],
                "L_i_err": [],
                "L_t": [],
                "L_t_err": [],
                "f_r": [],
                "f_r_err": [],
                "photon_number": [],
                "input_power": [],
            }
            for j, p in enumerate(file_power_points):
                frequency = data[i][j][0, :]
                s21_complex = data[i][j][1, :] + 1j * data[i][j][2, :]
                mag, phase = convert_complex_to_magphase(
                    data[i][j][1, :], data[i][j][2, :]
                )
                succeeded = False
                for t in np.arange(start, len(frequency) // 2, jump):
                    # Trim data, unwrap and S21 complex creation
                    if t == 0:
                        freq_cut = frequency
                        s21_complex_cut = s21_complex
                    else:
                        freq_cut = frequency[t:-t]
                        s21_complex_cut = s21_complex[t:-t]

                    try:
                        result, photon = self._resonator_fitter(
                            s21_complex_cut,
                            freq_cut,
                            power=p,
                            bg=bg,
                            fit_method=fit_method,
                        )
                    except:
                        continue

                    # Filter out bad fits
                    if self._test_fit(
                        result.result, verbose=False, threshold=threshold
                    ):
                        succeeded = True
                        data_temp["Q_c"].append(result.coupling_quality_factor)
                        data_temp["Q_c_err"].append(
                            result.coupling_quality_factor_error
                        )
                        data_temp["Q_i"].append(result.internal_quality_factor)
                        data_temp["Q_i_err"].append(
                            result.internal_quality_factor_error
                        )
                        data_temp["Q_t"].append(result.total_quality_factor)
                        data_temp["Q_t_err"].append(result.total_quality_factor_error)
                        data_temp["L_c"].append(result.coupling_loss)
                        data_temp["L_c_err"].append(result.coupling_loss_error)
                        data_temp["L_i"].append(result.internal_loss)
                        data_temp["L_i_err"].append(result.internal_loss_error)
                        data_temp["L_t"].append(result.total_loss)
                        data_temp["L_t_err"].append(result.total_loss_error)
                        data_temp["f_r"].append(result.f_r)
                        data_temp["f_r_err"].append(result.f_r_error)
                        data_temp["photon_number"].append(photon)
                        data_temp["input_power"].append(p)
                        break
                if not succeeded:
                    failed.append(p)
                else:
                    a = str(np.mean(frequency / 1e9))[:5].replace(".", "_")
                    self._plot_fit(
                        result,
                        save=savepic,
                        savepath=os.path.join(self._savepath, "fit_images"),
                        show=showpic,
                        name=f"{a}GHz_{p}_dBm",
                        nodialog=nodialog,
                        trimmed_data={
                            "real": np.real(s21_complex),
                            "imag": np.imag(s21_complex),
                            "phase": phase,
                            "mag": mag,
                            "freq": frequency,
                        },
                    )
                    if write:
                        a = str(np.mean(frequency / 1e9))[:5].replace(".", "_")
                        self._write_fit(
                            data_temp,
                            os.path.join(self._savepath, "fit_results"),
                            filename=f"{a}GHz_{p}_dBm",
                            nodialog=nodialog,
                        )
            if failed != []:
                status[files[i]] = (
                    f"  \033[33mCompleted with error\n\33[2K\033[31mFailed for power values {failed}\033[00m"
                )
            else:
                status[files[i]] = "  \033[32mCompleted\033[00m"
            if is_interactive():
                clear_output(wait=True)
                for file in files:
                    print(file + status[file])
                display("")
            else:
                sys.stdout.write("\33[2K")
                print(files[i] + status[files[i]])
            if data_temp["f_r"] != []:
                self._fit_results[files[i]] = data_temp

    def _resonator_fitter(
        self,
        data: NDArray,
        freq: NDArray,
        power: Optional[float] = None,
        params: Optional[Parameter] = None,
        fit_method: str = "shunt",
        bg: base.BackgroundModel = background.MagnitudePhaseDelay(),
        trim_indices: tuple = None,
    ) -> tuple[base.ResonatorFitter, ArrayLike]:
        """
        Wrapper around resonator library.

        Parameters
        ----------
        data : NDArray
            Data array of complex values to be fitted.
        freq : NDArray
            Array of frequencies.
        power : float, optional
            Power input in the resonator used to calculate number of photons.
            The default is ``None``.
        params : lmfit.Parameters, optional
            Fit parameters, passed as a lmfit.Parameter object.
        fit_method : str, optional
            Fit method to be used, can be "reflection" or "shunt". The default is "shunt".
        bg : resonator.base.BackgroundModel, optional
            Background model as defined in the resonator library. The default is
            background.MagnitudePhaseDelay().
        trim_indices : tuple, optional
            Tuple of the form ``(start, end)`` of where to trim the data to fit.
            The default is ``None``.
        """

        if fit_method == "shunt":

            if trim_indices:
                result = shunt.LinearShuntFitter(
                    frequency=freq[trim_indices[0] : -trim_indices[1]],
                    data=data[trim_indices[0] : -trim_indices[1]],
                    params=params,
                    background_model=bg,
                )
            else:
                result = shunt.LinearShuntFitter(
                    frequency=freq, data=data, params=params, background_model=bg
                )

        elif fit_method == "reflection":

            if trim_indices:
                result = reflection.LinearReflectionFitter(
                    frequency=freq[trim_indices[0] : -trim_indices[1]],
                    data=data[trim_indices[0] : -trim_indices[1]],
                    params=params,
                    background_model=bg,
                )
            else:
                result = reflection.LinearReflectionFitter(
                    frequency=freq, data=data, params=params, background_model=bg
                )

        else:
            print("Fit method not recognized")
            return

        if power is not None:
            photon = result.photon_number_from_power(result.f_r, power)
        else:
            photon = 0

        return result, photon

    def _test_fit(
        self, result: base.ResonatorFitter, threshold: float = 1, verbose: bool = False
    ) -> bool:
        """
        Function taken from Nicolas Bourlet (thanks), available on Gitlab of JosePh
        group. Check whether the lmfit result is correct or not.

        Parameters
        ----------
        result : ResonatorFitter class object
            Output from lmfit, is the result of a minimization made by lmfit fit functions.
        threshold : float, optional
            A value greater than 0 which determine the error tolerance on fit values.
            The default is ``1`` (100%).
        verbose : bool, optional
            If True, displays if the threshold has been reached or if there were no errorbars found.
            The default is ``False``.
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

    def _write_fit(
        self, data: dict, path: str, filename: str, nodialog: bool = True
    ) -> None:
        """
        Utility function to write the fit results to txt files.
        """
        full_path = os.path.join(path, filename + "_results.txt")

        if os.path.exists(full_path) and not nodialog:
            decision = choice(
                "Overwrite warning", "Text file already exists, overwrite?"
            )

            if not decision:
                return

        with open(full_path, "w") as f:
            f.write(
                filename
                + "\n\n"
                + "------------------------------------------"
                + "\n\n"
            )

            for key, value in data.items():
                f.write(str(key) + " : " + str(value[-1]) + "\n\n")

            f.close()

    def _plot_fit(
        self,
        result: base.ResonatorFitter,
        trimmed_data: Optional[dict] = None,
        save: bool = False,
        savepath: str = "",
        show: bool = False,
        name: str = "",
        nodialog: bool = False,
    ) -> MultiFigure:
        """
        Utilitary function to plot the fit result as a triptych (resonator.see.triptych).
        """
        triptych = plot_triptych(
            trimmed_data["freq"],
            trimmed_data["mag"],
            trimmed_data["phase"],
            trimmed_data["real"],
            trimmed_data["imag"],
            fit_result=result,
            three_ticks=True,
        )

        if save:
            filename = os.path.join(savepath, name + ".svg")

            if os.path.exists(filename) and not nodialog:
                overwrite = choice()

                if overwrite:
                    triptych.save(filename)
            else:
                triptych.save(filename)
        elif show:
            triptych.show()

        return triptych
