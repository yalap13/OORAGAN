import numpy as np

from numpy.typing import ArrayLike
from dataclasses import dataclass
from typing import Optional
from scipy.special import ellipk
from scipy.constants import epsilon_0, mu_0


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
        Type of resonator, between "lambda2" for half wave or "lambda4" for quarter wave.
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
        Type of resonator, between "lambda2" for half wave or "lambda4" for quarter wave.
    """

    length: float
    conductor_width: float
    gap_to_ground: float
    thickness: float
    coupling_disctance: float
    substrate_dielectric_const: float = 11.68
    Qc_estimated: Optional[float] = None
    type: str = "lambda4"

    def __post_init__(self) -> None:
        thick_micron = self.thickness * 1e-3
        k_zero = self.conductor_width / (self.conductor_width + 2 * self.gap_to_ground)
        k1_zero = np.sqrt(1 - k_zero**2)
        k = self._u1(thick_micron, self.conductor_width, self.gap_to_ground) / self._u2(
            thick_micron, self.conductor_width, self.gap_to_ground
        )
        k_half = self._u1(
            thick_micron / 2, self.conductor_width, self.gap_to_ground
        ) / self._u2(thick_micron / 2, self.conductor_width, self.gap_to_ground)
        k1 = np.sqrt(1 - k**2)
        k1_half = np.sqrt(1 - k_half**2)

        c_geo = 2 * epsilon_0 * ellipk(k**2) / ellipk(
            k1**2
        ) + 2 * self.substrate_dielectric_const * epsilon_0 * ellipk(
            k_zero**2
        ) / ellipk(
            k1_zero**2
        )
        l_geo = mu_0 * ellipk(k1_half**2) / (4 * ellipk(k_half**2))

        if self.type == "lambda4":
            res_type = 4
        elif self.type == "lambda2":
            res_type = 2
        else:
            raise ValueError("Only types 'lambda2' and 'lambda4' are accepted")

        self.freq_geo = 1 / (np.sqrt(l_geo * c_geo) * res_type * self.length)

    def _u1(t: float, w: float, s: float) -> float:
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
        d = 2 * t / np.pi
        a = w
        b = w + 2 * s
        return (
            a
            + (d / 2)
            + (3 * np.log(2) * d / 2)
            - (d * np.log(d / a) / 2)
            + (d * np.log((b - a) / (b + a)) / 2)
        )

    def _u2(t: float, w: float, s: float) -> float:
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
        d = 2 * t / np.pi
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
    dc_kinetic_inductance : float
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
        *resonators: Resonator,
    ):
        """
        Resonator attribution takes the fit results and tries to attribute the resonance peaks
        to physical resonators present on the feedline.

        Parameters
        ----------
        fit_results : ArrayLike
            Results from the fit of the resonators given by the ResonatorFitter class.
        dc_kinetic_inductance : float
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

        self._fit_results = fit_results
        self._res_on_line = [res for res in resonators]
        self._dc_Lkin = dc_kinetic_induct * 1e-9
