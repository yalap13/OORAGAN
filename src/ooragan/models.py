import numpy as np
from resonator import base, background, guess


class StandingWaveInterferenceReflection(base.ResonatorModel):
    r"""
    This class represent the reflection seen from the input port of a measurement chain
    containing a parasitic impedance mismatch causing a standing wave interfering with
    the resonance.

    Parameters
    ----------
    center_frequency : float
        Center frequency of the measured span used as phase reference.
    standing_wave_delay : float, optional
        Delay calculated from the period of the standing wave, measured on
        a wide scan of the reflection.
    vary_standing_wave_delay : bool, optional
        Controls wheter or not to leave the standing wave delay as a free
        fitting parameter. By default, ``False``.

    Notes
    -----
    The fitted model is given by

    .. math::

        S_{11} (f) = e^{i\chi}\cos\theta-\frac{e^{2i\chi}\sin^2\theta e^{-2\pi i(f-f_c)\tau_{sw}}\Gamma_r (f)}{1-e^{i\chi}\cos\theta e^{-2\pi i(f-f_c)\tau_{sw}}\Gamma_r(f)}

    with

    .. math:: \Gamma_r (f) = 1-\frac{2Q_l/Q_c}{1+2iQ_l\frac{f-f_r}{f_r}}

    and

    .. math:: Q_l = \frac{Q_c Q_i}{Q_c+Q_i}.
    """

    # This is only needed by ResonatorFitter.guess().
    # Since this model contains its own absolute background, use unity.
    reference_point = 1.0 + 0.0j

    def __init__(
        self,
        center_frequency,
        standing_wave_delay=20e-9,
        vary_standing_wave_delay=False,
        *args,
        **kwargs,
    ):
        self.center_frequency = center_frequency
        self._standing_wave_delay = standing_wave_delay
        self._vary_standing_wave_delay = vary_standing_wave_delay

        def standing_wave_interference_reflection(
            frequency,
            resonance_frequency,
            internal_quality_factor,
            coupling_quality_factor,
            standing_wave_delay,
            theta,
            beta,
            chi,
        ):
            df = frequency - self.center_frequency
            s11 = np.exp(1j * (chi + beta)) * np.cos(theta)
            s22 = np.exp(1j * (chi - beta)) * np.cos(theta)
            s12 = 1j * np.exp(1j * chi) * np.sin(theta)
            loaded_quality_factor = (
                internal_quality_factor
                * coupling_quality_factor
                / (internal_quality_factor + coupling_quality_factor)
            )

            standing_wave = np.exp(-2j * np.pi * df * standing_wave_delay)
            detuning = (frequency - resonance_frequency) / resonance_frequency
            resonance = 1 - (2 * loaded_quality_factor / coupling_quality_factor) / (
                1 + 2j * loaded_quality_factor * detuning
            )
            return s11 - (s12**2 * standing_wave * resonance) / (
                1 - s22 * standing_wave * resonance
            )

        super().__init__(
            func=standing_wave_interference_reflection,
            *args,
            **kwargs,
        )

    def guess(self, data=None, frequency=None, **kwargs):
        if data is None or frequency is None:
            return self.make_params()

        params = self.make_params()
        frequency = np.asarray(frequency)

        # ---------------------------------------------------------
        # Fill lmfit parameters.
        # ---------------------------------------------------------

        params["resonance_frequency"].set(
            value=self.center_frequency,
            min=frequency.min(),
            max=frequency.max(),
        )
        params["internal_quality_factor"].set(
            value=1e3,
            min=1,
            max=1e7,
        )
        params["coupling_quality_factor"].set(
            value=1e3,
            min=1,
            max=1e7,
        )
        params["standing_wave_delay"].set(
            value=self._standing_wave_delay,
            min=0,
            vary=self._vary_standing_wave_delay,
        )
        params["theta"].set(
            value=1.5,
            min=0,
            max=np.pi / 2,
        )
        params["chi"].set(
            value=0,
            min=-np.pi,
            max=np.pi,
        )

        return params


class StandingWaveInterferenceReflectionFitter(base.ResonatorFitter):
    r"""
    This class implements the fitter for the StandingWaveInterferenceReflection.

    .. attention::

        Because this class inherits from :class:`resonator.base`ResonatorFitter`, it looks
        like all attributes exist, but most are yet unimplemented for this class. The ones that
        are implemented are ``f_r``, ``f_r_error``, ``Q_i``, ``Q_c`` and ``Q_l``.

    Parameters
    ----------
    frequency : ArrayLike
        Array of the frequency of the sweep.
    data : ArrayLike
        Array of the complex data.
    errors : ArrayLike
        An array of complex numbers containing the standard errors of the mean of the data points.
    standing_wave_delay : float, optional
        Delay calculated from the period of the standing wave, measured on
        a wide scan of the reflection.
    vary_standing_wave_delay : bool, optional
        Controls wheter or not to leave the standing wave delay as a free
        fitting parameter. By default, ``False``.
    params : lmfit.parameter.Parameters, optional
        Starting points and bounds for the fit paramters.
    background_model : lmfit.model.Model, optional
        Model to represent the background of the signal. By default uses the MagnitudePhaseDelay model from
        resonator.background.

    Notes
    -----
    The fitted model is given by

    .. math::

        S_{11} (f) = e^{i\chi}\cos\theta-\frac{e^{2i\chi}\sin^2\theta e^{-2\pi i(f-f_c)\tau_{sw}}\Gamma_r (f)}{1-e^{i\chi}\cos\theta e^{-2\pi i(f-f_c)\tau_{sw}}\Gamma_r(f)}

    with

    .. math:: \Gamma_r (f) = 1-\frac{2Q_l/Q_c}{1+2iQ_l\frac{f-f_r}{f_r}}

    and

    .. math:: Q_l = \frac{Q_c Q_i}{Q_c+Q_i}.

    """

    def __init__(
        self,
        frequency,
        data,
        errors=None,
        standing_wave_delay=20e-9,
        vary_standing_wave_delay=False,
        params=None,
        background_model=None,
        **fit_kwargs,
    ):
        frequency = np.asarray(frequency)
        data = np.asarray(data)

        center_frequency = np.mean(frequency)

        foreground_model = StandingWaveInterferenceReflection(
            center_frequency=center_frequency,
            standing_wave_delay=standing_wave_delay,
            vary_standing_wave_delay=vary_standing_wave_delay,
        )

        if background_model is None:
            background_model = background.MagnitudePhaseDelay()

        super().__init__(
            frequency=frequency,
            data=data,
            foreground_model=foreground_model,
            background_model=background_model,
            errors=errors,
            params=params,
            **fit_kwargs,
        )

    # --------------------------------------------------------------
    # Resonator-style aliases
    # --------------------------------------------------------------

    @property
    def f_r(self):
        """The resonance frequency"""
        return self.resonance_frequency

    @property
    def f_r_error(self):
        """The resonance frequency"""
        return self.resonance_frequency_error

    @property
    def Q_i(self):
        """The internal quality factor"""
        return self.internal_quality_factor

    @property
    def Q_c(self):
        """The coupling quality factor"""
        return self.coupling_quality_factor

    @property
    def Q_l(self):
        """The loaded (or total) quality factor"""
        return (
            self.internal_quality_factor
            * self.coupling_quality_factor
            / (self.internal_quality_factor + self.coupling_quality_factor)
        )


class SlowlyVaryingMagnitudePhaseDelay(base.BackgroundModel):
    r"""
    This class implements a background model which represents a slowly varying global magnitude,
    phase and delay.

    Notes
    -----
    The model is given by

    .. math::

        A [1+b (f-f_c)] e^{i\alpha}e^{-2\pi i (f-f_c)\tau_0}

    where :math:`b \in \mathbb{C}`.
    """

    def __init__(self, *args, **kwds):
        def slowly_varying_magnitude_phase_delay(
            frequency, frequency_reference, b1_real, b1_imag, magnitude, phase, delay
        ):
            b1 = b1_real + 1j * b1_imag
            slow_varying_magnitude = 1 + b1 * (frequency - frequency_reference)
            return (
                magnitude
                * slow_varying_magnitude
                * np.exp(
                    1j * (2 * np.pi * (frequency - frequency_reference) * delay + phase)
                )
            )

        super(SlowlyVaryingMagnitudePhaseDelay, self).__init__(
            func=slowly_varying_magnitude_phase_delay, *args, **kwds
        )

    def guess(self, data, frequency, fraction=0.1, **kwds):
        """
        :param data: complex scattering parameter data.
        :param frequency: the frequencies corresponding to the data points.
        :param fraction: the fraction of points with lowest nearest-neighbor distances to use to estimate the magnitude.
        :param kwds: ignored, for now.
        :return: lmfit.Parameters
        """
        params = self.make_params()
        frequency_reference = frequency.mean()
        params["frequency_reference"].set(value=frequency_reference, vary=False)
        # Use the points with smallest nearest-neighbor distances per frequency difference
        indices = guess.smallest(
            guess.distances_per_frequency(frequency=frequency, data=data),
            fraction=fraction,
        )
        phase, delay = guess.polyfit_phase_delay(
            frequency=frequency[indices] - frequency_reference, data=data[indices]
        )
        params["phase"].set(value=phase)
        params["delay"].set(value=delay)
        _, offset = guess.polyfit_magnitude_slope_offset(
            frequency[indices] - frequency_reference, data=data[indices]
        )
        params["magnitude"].set(value=offset, min=0)
        params["b1_real"].set(value=0, min=-0.5, max=0.5)
        params["b1_imag"].set(value=0, min=-0.5, max=0.5)
        return params
