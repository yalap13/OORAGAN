import numpy as np

from numpy.typing import ArrayLike
from dataclasses import dataclass
from typing import Optional
from scipy.special import ellipk
from scipy.constants import epsilon_0, mu_0
from tabulate import tabulate


@dataclass
class Resonator:
    """
    Container of a CPW resonator's parameter.

    Parameters
    ----------
    length : float
        Resonator total lenght in microns.
    conductor_width : float
        Width of the central conductor of the CPW in microns.
    gap_to_ground : float
        Gap between the central conductor and the ground plane in microns.
    thickness : float
        Thickness of the metal in nanometers.
    coupling_gap : float
        Coupling distance measured between the gap of the feedline and the gap
        of the resonator. Distance in microns.
    substrate_dielectric_const : float, optional
        Dielectric constant of the substrate. Defaults to 11.68 (silicon).
    Qc_estimated : float, optional
        Estimation of the coupling quality factor of the resonator.
    type : str, optional
        Type of resonator, between "lambda2" for half-wave or "lambda4" for quarter wave.
        Defaults to "lambda4".

    Attributes
    ----------
    conductor_width : float
        Width of the central conductor of the CPW in microns.
    coupling_gap : float
        Coupling distance measured between the gap of the feedline and the gap
        of the resonator. Distance in microns.
    freq_geo : float
        Geometrical frequency of the resonator in Hz.
    gap_to_ground : float
        Gap between the central conductor and the ground plane in microns.
    length : float
        Resonator total lenght in microns.
    Qc_estimated : float
        Estimation of the coupling quality factor of the resonator.
    substrate_dielectric_const : float
        Dielectric constant of the substrate.
    thickness : float
        Thickness of the metal in nanometers.
    type : str
        Type of resonator, between "lambda2" for half-wave or "lambda4" for quarter wave.
    """

    length: float
    conductor_width: float
    gap_to_ground: float
    thickness: float
    coupling_gap: float
    london_eff: float
    substrate_dielectric_const: float = 11.68
    Qc_estimated: Optional[float] = None
    type: str = "lambda4"

    def __post_init__(self) -> None:
        s = self.gap_to_ground * 1e-6
        w = self.conductor_width * 1e-6
        t = self.thickness * 1e-9
        lamb = self.london_eff * 1e-9
        lres = self.length * 1e-6
        k_zero = w / (w + 2 * s)
        k1_zero = np.sqrt(1 - k_zero ** 2)
        k = self._u1(t, w, s) / self._u2(t, w, s)
        k_half = self._u1(t / 2, w, s) / self._u2(t / 2, w, s)
        k1 = np.sqrt(1 - k ** 2)
        k1_half = np.sqrt(1 - k_half ** 2)
        g = (-k * np.log(t / (4 * (w + 2 * s))) - np.log(t / (4 * w)) + (2 * (w + s) / (w + 2 * s)) * np.log(
            s / (w + s))) / (2 * k ** 2 * ellipk(k ** 2) ** 2)

        self.c_geo = 2 * epsilon_0 * ellipk(k ** 2) / ellipk(
            k1 ** 2
        ) + 2 * self.substrate_dielectric_const * epsilon_0 * ellipk(
            k_zero ** 2
        ) / ellipk(
            k1_zero ** 2
        )
        self.l_geo = mu_0 * ellipk(k1_half ** 2) / (4 * ellipk(k_half ** 2))
        self.l_kin = (mu_0 * g * lamb ** 2) / (w * t)

        if self.type == "lambda4":
            res_type = 4
        elif self.type == "lambda2":
            res_type = 2
        else:
            raise ValueError("Only types 'lambda2' and 'lambda4' are accepted")

        self.freq_geo = (1 / (np.sqrt(self.l_geo * self.c_geo) * res_type * lres)) / 1e9
        self.freq_kin = (1 / (np.sqrt((self.l_geo + self.l_kin) * self.c_geo) * res_type * lres)) / 1e9

    def _u1(self, t: float, w: float, s: float) -> float:
        """
        Definition of u-parameter for Gao's CPW inductance and capacitance calculation derivation.

        Parameters
        ----------
        t : float
            Thickness of metal film in nanometers.
        w : float
            Width of resonator central conductor in microns.
        s : float
            Separation between resonator central conductor and ground plane in microns.
        """
        tm = t * 1e-3
        d = 2 * tm / np.pi
        a = w
        b = w + 2 * s
        return (
                a
                + (d / 2)
                + (3 * np.log(2) * d / 2)
                - (d * np.log(d / a) / 2)
                + (d * np.log((b - a) / (b + a)) / 2)
        )

    def _u2(self, t: float, w: float, s: float) -> float:
        """
        Definition of u-parameter for Gao's CPW inductance and capacitance calculation derivation.

        Parameters
        ----------
        t : float
            Thickness of metal film in nanometers.
        w : float
            Width of resonator central conductor in microns.
        s : float
            Separation between resonator central conductor and ground plane in microns.
        """
        tm = t * 1e-3
        d = 2 * tm / np.pi
        a = w
        b = w + 2 * s
        return (
                b
                - (d / 2)
                - (3 * np.log(2) * d / 2)
                + (d * np.log(d / a) / 2)
                + (d * np.log((b - a) / (b + a)) / 2)
        )


class ResonatorAttribution:
    """
    Resonator attribution takes the fit results and tries to attribute the resonance peaks
    to physical resonators present on the feedline.

    Parameters
    ----------
    fit_results : ArrayLike
        Results from the fit of the resonators given by the ResonatorFitter class.
    dc_kinetic_induct : float
        DC measured sheet kinetic inductance in nH/square.
    resonators : Resonator
        The physical resonators present on the feedline.

    Attributes
    ----------
    _fit_results : ArrayLike
        Results from the fit of the resonators given by the ResonatorFitter class.
    _dc_Lkin : float
        DC measured sheet kinetic inductance in nH/square.
    _res_on_line : Resonator
        The physical resonators present on the feedline.
    """

    def __init__(
            self,
            fit_results: ArrayLike,
            dc_kinetic_induct: float,
            resonators: list[Resonator],
    ):
        """
        Resonator attribution takes the fit results and tries to attribute the resonance peaks
        to physical resonators present on the feedline.

        Parameters
        ----------
        fit_results : ArrayLike
            Results from the fit of the resonators given by the ResonatorFitter class.
        dc_kinetic_induct : float
            DC measured sheet kinetic inductance in nH/square.
        resonators : Resonator
            The physical resonators present on the feedline.
        """

        self.result = None
        self._fit_results = fit_results
        self._res_on_line = resonators
        self._dc_Lkin = dc_kinetic_induct * 1e-9

    def minimize_lambda_eff(self):
        raise NotImplementedError

    def minimize_Lk(self, printer=True):
        """
        Tries to attribute each resonance measured with a resonator object provided to the class by calculating a
        kinetic inductance from the resonance shift with the geometrical resonance. Compares this kinetic inductance
        with the one provided to the class (ideally from DC measurements).

        Parameters
        ----------
        printer : bool, optional
            Choose whether to display the results dictionary (with fashion!) or not.
        """

        # Initializing the results dictionary
        result = {}

        # Kinetic inductance calculation and minimization
        for fitnb, fitdict in enumerate(self._fit_results):
            lkarr = np.zeros(len(self._res_on_line))
            fmes = fitdict["f_r"][0]
            for resnb, res in enumerate(self._res_on_line):
                lkarr[resnb] = ((1 / (16 * res.c_geo * (res.length * 1e-6) ** 2 * fmes ** 2))
                                - res.l_geo)
            # Minimization
            id_closest_res = (np.abs(lkarr - self._dc_Lkin)).argmin()

            # Resonator allocation and variable storage
            vals = [self._res_on_line[id_closest_res].length, self._res_on_line[id_closest_res].freq_geo,
                    self._res_on_line[id_closest_res].freq_kin, fmes, lkarr[id_closest_res], self._dc_Lkin * 1e9]
            result[f"res {fitnb}"] = np.array(vals)

        if printer:
            keys = sorted(result.keys())
            heads = ["Resonator length (μm)",
                     "Geometrical resonance frequency (GHz)",
                     "Resonance frequency with kinetic inductance (GHz)",
                     "Measured resonance frequency (GHz)",
                     "Measured kinetic inductance (pH/sq)",
                     "Kinetic inductance from DC measurements (pH/sq)"]

            to_print = []
            for row in range(len(result[keys[0]])):
                to_print.append([heads[row]] + [result[k][row] for k in keys])
            print(tabulate(to_print, headers=keys, tablefmt="rounded_grid"))

        return result

    def minimize_Qc(self):
        raise NotImplementedError
