"""
Much of the contents of this file has been written by Gabriel Ouellet, 
fellow master's student. All his code is available on the JosePh Gitlab.
"""

import os
import lmfit
import numpy as np

from pathlib import Path
from resonator import background, shunt, reflection, see

from .file_handler import writer
from .util import choice, convert_magphase_to_complex, convert_complex_to_magphase


def resonator_fitter(
    data,
    freq,
    power=None,
    params=None,
    fit_method="shunt",
    bg=background.MagnitudePhaseDelay(),
    cut=None,
):
    """
    Wrapper around resonator package.

    Parameters
    ----------
    data : np.array
        Data array of complex values to be fitted.
    freq : np.array
        DESCRIPTION.
    power : int or float
        Power input in the resonator used to calculate number of photons inside fit package. The default is False.
    params : lmfit.Parameters
        DESCRIPTION. The default is None.
    fit_method : str, optional
        Fit method to be used, can be "reflection", "shunt" or . The default is "shunt".
    bg : class-like object, optional
        Background model from resonator library. The default is background.MagnitudePhaseDelay().
    coupe : list, optional
        List describing where data wants to be cut for analysis [start, end]. All data is fitted
        but the corresponding area will be fitted separately.The default is [].

    Returns
    -------
    Resonator fitter result object and two dictionaries : one with the wanted data fit results and the other
    with all the data fit results (they are identical if coupe is False).

    """

    if fit_method == "shunt":

        if cut:
            r = shunt.LinearShuntFitter(
                frequency=freq[cut[0] : -cut[1]],
                data=data[cut[0] : -cut[1]],
                params=params,
                background_model=bg,
            )
        else:
            r = shunt.LinearShuntFitter(
                frequency=freq, data=data, params=params, background_model=bg
            )

    elif fit_method == "reflection":

        if cut:
            r = reflection.LinearReflectionFitter(
                frequency=freq[cut[0] : -cut[1]],
                data=data[cut[0] : -cut[1]],
                params=params,
                background_model=bg,
            )
        else:
            r = reflection.LinearReflectionFitter(
                frequency=freq, data=data, params=params, background_model=bg
            )

    else:
        print("Fit method not recognized")
        return

    if power is not None:
        photon = r.photon_number_from_power(r.f_r, power)
    else:
        photon = 0
    result = {
        "coupling loss": r.coupling_loss,
        "internal loss": r.internal_loss,
        "total loss": r.total_loss,
        "coupling loss error": r.coupling_loss_error,
        "internal loss error": r.internal_loss_error,
        "total loss error": r.total_loss_error,
        "coupling quality factor": r.Q_c,
        "internal quality factor": r.Q_i,
        "total quality factor": r.Q_t,
        "coupling quality factor error": r.Q_c_error,
        "internal quality factor error": r.Q_i_error,
        "total quality factor error": r.Q_t_error,
        "background model": r.background_model.name,
        "resonance frequency": r.f_r,
        "resonance frequency error": r.f_r_error,
        "photon number": photon,
    }

    return result, r


def check_fit(result, threshold=1, verbose=False):
    """
    Function taken from Nicolas Bourlet (thanks), available on Gitlab of JosePh group. Check whether the lmfit result is
    correct or not.

    Parameters
    ----------
    result : MinimizerResult class object
        Output from lmfit, is the result of a minimization made by lmfit fit functions.
    threshold : float, optional
        A value greater than 0 which determine the error tolerance on fit values. The default is 1 (100%).
    verbose : bool, optional
        If True, displays if the threshold has been reached or if there were no errorbars found. The default is False

    Returns
    -------

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


def fit_resonator_test(
    datalist: list,
    filelist: list,
    powers,
    f_r=None,
    couploss=1e-6,
    intloss=1e-6,
    bg=background.MagnitudePhaseDelay(),
    savepic=False,
    savepath="",
    write=False,
    basepath=os.getcwd(),
    threshold=0.5,
    start=0,
    jump=10,
    nodialog=False,
):
    """


    Parameters
    ----------
    datalist : list
        List of data arrays to analyse.
    filelist : list
        List of all filenames of datafiles.
    basepath : str
        Path to the folder of all data and Images, Fit results and Codes folders.
    powers : np.array
        Powers at which resonances are observed.
    f_r : int, optional
        Value of resonance frequency, adds to fit parameters. The default is None.
    couploss : int, optional
        Coupling loss of resonator, adds to fit parameters. The default is 1e6.
    intloss : int, optional
        Internal loss of resonator, adds to fit parameters. The default is 1e6.
    bg : class-like object, optional
        Background model from resonator library. The default is background.MagnitudePhaseDelay().
    savepic : bool, optional
        If True, saves the fit plots generated by the function in the Images folder in the same
        directory. The default is False.
    savepath : str, optional
        Where to save the graph, savepic must be True. The default is "".
    write : bool, optional
        Choose whether to write the fit results in a .txt file in the Fit results folder in the
        same directory or not. The default is False.
    threshold : float, optional
        A value greater than 0 which determine the error tolerance on fit values.
        The default is 1 (100%).
    start : int, optional
        Where to start shrunking dataset. The default is 0.
    jump : int, optional
        Step between the shrunking of dataset to fit better. The default is 10.
    nodialog : bool, optional
        If True, does not display any dialog box (watch out for overwriting). The default is False.

    Returns
    -------
    dict(result : value, ...)

    """
    if savepath == "":
        savepath = os.path.join(basepath, "Images")
    dictoflist = {}
    dictio, r = {}, {}
    data_store = {
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
        "photnum": [],
    }
    params = lmfit.Parameters()
    params.add(name="internal_loss", value=intloss)
    params.add(name="coupling_loss", value=couploss)
    params.add(name="resonance_frequency", value=f_r, min=1e9, max=1e10)

    if isinstance(filelist, str):
        filelist = [filelist]

    number_of_power_points = powers.shape[1] if len(powers.shape) > 1 else len(powers)

    for i in range(number_of_power_points):
        if Path(filelist[0]).suffix == ".hdf5":
            freq = datalist[i][0]
            s21_real = datalist[i][1]
            s21_imag = datalist[i][2]
            s21_complex = s21_real + 1j * s21_imag
            mag, phase = convert_complex_to_magphase(s21_real, s21_imag, deg=True)
        elif len(filelist) == 1:
            freq = datalist[0]
            s21_complex = convert_magphase_to_complex(
                datalist[1], datalist[2], deg=True, dBm=True
            )
            mag = datalist[1]
            phase = datalist[2]
        elif len(filelist) > 1:
            freq = datalist[i][0]
            mag = datalist[i][1]
            phase = datalist[i][2]
            s21_complex = convert_magphase_to_complex(
                datalist[i][1], datalist[i][2], deg=True, dBm=True
            )
        elif len(filelist) == 0:
            raise ValueError("File list given is empty")
        else:
            raise TypeError("Unrecognized data format")

        for p in np.arange(start, len(freq) // 2, jump):
            # Trim data, unwrap and S21 complex creation
            if p == 0:
                freq_cut = freq
                s21_complex_cut = s21_complex
            else:
                freq_cut = freq[p:-p]
                s21_complex_cut = s21_complex[p:-p]

            dictio, r = resonator_fitter(
                s21_complex_cut,
                freq_cut,
                power=powers[0][i] if len(powers.shape) > 1 else powers[i],
                bg=bg,
            )

            # Filter out bad fits

            if check_fit(r.result, verbose=False, threshold=threshold):
                dictoflist = dict_filler(dictio, data_store)
                break

        if savepic:
            a = str(np.mean(freq / 1e9))[:5].replace(".", "_")
            plot_fit(
                r,
                save=savepic,
                savepath=savepath,
                name=f"{a}GHz_{powers[0][i] if len(powers.shape) > 1 else powers[i]}_dBm",
                nodialog=nodialog,
                cut=True,
                cutted_data={
                    "real": np.real(s21_complex),
                    "imag": np.imag(s21_complex),
                    "phase": phase,
                    "mag": mag,
                    "freq": freq,
                },
            )
        else:
            plot_fit(
                r,
                cut=True,
                cutted_data={
                    "real": np.real(s21_complex),
                    "imag": np.imag(s21_complex),
                    "phase": phase,
                    "mag": mag,
                    "freq": freq,
                },
            )
        if write:
            a = str(np.mean(freq / 1e9))[:5].replace(".", "_")
            power_tag = powers[0][i] if len(powers.shape) > 1 else powers[i]
            writer(
                dictio,
                os.path.join(basepath, "Fit results"),
                name=f"{a}GHz_{power_tag}_dBm",
                nodialog=nodialog,
            )

    dictofarr = lst_to_arrays(dictoflist)
    return dictofarr


def dict_filler(dictio, another_dict):
    """
    Fills a dictionary (in the way I want) with results info provided in dictio

    Parameters
    ----------
    dictio : dict
        Dictionary containing fit results info (output of resonator_fitter function).
    another_dict : dict
        The other dictionary that is going to be filled the right way.

    """
    another_dict["Q_c"].append(dictio["coupling quality factor"])
    another_dict["Q_c_err"].append(dictio["coupling quality factor error"])
    another_dict["Q_i"].append(dictio["internal quality factor"])
    another_dict["Q_i_err"].append(dictio["internal quality factor error"])
    another_dict["Q_t"].append(dictio["total quality factor"])
    another_dict["Q_t_err"].append(dictio["total quality factor error"])
    another_dict["L_c"].append(dictio["coupling loss"])
    another_dict["L_c_err"].append(dictio["coupling loss error"])
    another_dict["L_i"].append(dictio["internal loss"])
    another_dict["L_i_err"].append(dictio["internal loss error"])
    another_dict["L_t"].append(dictio["total loss"])
    another_dict["L_t_err"].append(dictio["total loss error"])
    another_dict["f_r"].append(dictio["resonance frequency"])
    another_dict["f_r_err"].append(dictio["resonance frequency error"])
    another_dict["photnum"].append(dictio["photon number"])

    return another_dict


def lst_to_arrays(dictoflist):
    """
    Transforms the lists in the data dictionary into numpy arrays.

    Parameters
    ----------
    dictoflist : dict
        Dictionary with lists to transform into arrays.

    Returns
    -------
    dictoflist but with arrays

    """

    for key in dictoflist:
        if type(dictoflist[key]) is list:
            dictoflist[key] = np.array(dictoflist[key])
    return dictoflist


def plot_fit(
    r,
    cut=False,
    cutted_data=None,
    save=False,
    plot_cut=True,
    savepath="",
    name="",
    nodialog=False,
):
    """
    Is a copy of plot_fit but improved

    Parameters
    ----------
    r : resonator fit object or list of resonator fit object
        Fitted data, result from fitter function.
    cut : bool, optional
        State if data is cut, will rearrange plot accordingly (r must be a list with
        small fit first, all data fit second). The default is False.
    cutted_data : resonator fit object or list of resonator fit object
        Fitted data, result from fitter function.
    save : bool, optional
        To choose whether to save the plot or not. The default is False.
    plot_cut : bool, optional
        Whether to plot cutted data or not. The default is True.
    savepath : str, optional
        Path to save the plot, save must be True. The default is "".
    name : str, optional
        Name of saved plot. The default is "".
    nodialog : bool, optional
        If True, does not display any dialog box (watch out for overwriting). The default is False.

    Returns
    -------
    figure, (axes)

    """
    if cutted_data is None:
        cutted_data = {"real": [], "imag": [], "phase": [], "mag": [], "freq": []}

    if cut:

        fig, (ax_magnitude, ax_phase, ax_complex) = see.triptych(
            resonator=r,
            frequency_scale=1e-9,
            three_ticks=False,
            figure_settings={"figsize": (10, 6), "dpi": 120},
            data_settings={"markersize": 1, "label": ""},
            fit_settings={"label": ""},
            resonance_settings={"label": ""},
        )
        if plot_cut:
            cutted_data["freq"] = np.asarray(cutted_data["freq"], dtype=np.float64)
            ax_complex.plot(cutted_data["real"], cutted_data["imag"])
            ax_phase.plot(cutted_data["freq"] * 1e-9, cutted_data["phase"])
            ax_magnitude.plot(cutted_data["freq"] * 1e-9, cutted_data["mag"])

        see.triptych(
            resonator=r,
            frequency_scale=1e-9,
            three_axes=(ax_magnitude, ax_phase, ax_complex),
            three_ticks=False,
            data_settings={"markersize": 1},
            fit_settings={"linewidth": 1, "color": "black"},
            resonance_settings={"markersize": 7, "color": "black"},
        )

    else:
        fig, (ax_magnitude, ax_phase, ax_complex) = see.triptych(
            resonator=r,
            frequency_scale=1e-9,
            figure_settings={"figsize": (10, 6), "dpi": 120},
            data_settings={"markersize": 2},
            fit_settings={"linewidth": 0.5, "color": "black"},
            resonance_settings={"markersize": 5, "color": "black"},
        )

    ax_complex.legend()

    if save:
        filename = os.path.join(savepath, name + ".svg")

        if os.path.exists(filename) and not nodialog:
            overwrite = choice()

            if overwrite:
                fig.savefig(filename, dpi=300, transparent=True)
        else:
            fig.savefig(filename, dpi=300, transparent=True)

    # plt.show()

    return fig, (ax_magnitude, ax_phase, ax_complex)
