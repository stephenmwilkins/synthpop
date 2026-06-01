"""A submodule for creating and manipulating star formation histories.

NOTE: This module is imported as SFH in parametric.__init__ enabling the syntax
      shown below.

Example usage:

    from synthesizer.parametric import SFH

    print(SFH.parametrisations)

    sfh = SFH.Constant(...)
    sfh = SFH.Exponential(...)
    sfh = SFH.LogNormal(...)

    sfh.calculate_sfh()

"""

import inspect
import io
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from unyt import Gyr, unyt_array, yr, Msun, Myr



# Define a list of the available parametrisations
parametrisations = (
    "Constant",
)


class Common:
    """The parent class for all accretion history parametrisations.

    Attributes:
        name (str):
            The name of this accretion history. This is set by the child and encodes
            the type of the accretion history. Possible values are defined in
            parametrisations above.
        parameters (dict):
            A dictionary containing the parameters of the model.
    """

    def __init__(self, name, **kwargs):
        """Initialise the parent.

        Args:
            name (str):
                The name of this accretion history. This is set by the child and encodes
                the type of the accretion history. Possible values are defined in
                parametrisations above.
            **kwargs (dict):
                A dictionary containing the parameters of the model.
        """
        # Set the name string
        self.name = name

        # Store the model parameters (defined as kwargs)
        self.parameters = kwargs


    def get_accretion_rate(self, age):
        """Calculate the accretion rate in each bin.

        Args:
            age (float):
                The age at which to calculate the accretion rate.

        Returns:
            accretion_rate (float/np.ndarray of float)
                The accretion rate at the passed age. Either as a single value
                or an array for each age in age.
        """
        # If we have been handed an array we need to loop
        if isinstance(age, (np.ndarray, list, tuple)) or hasattr(age, "ndim"):
            if hasattr(self, "_accretion_rates"):
                # If the child has defined a vectorised version of _accretion_rate
                return self._accretion_rates(age)
            else:
                return np.array([self._accretion_rate(a) for a in age])

        return self._accretion_rate(age)


class Constant(Common):
    """A constant accretion history.

    The accretion rate is defined such that:
        accretion_rate = 1; min_age<t<=max_age
        accretion_rate = 0; t>max_age, t<min_age

    Attributes:
        max_age (unyt_quantity):
            The age above which the accretion history is truncated.
        min_age (unyt_quantity):
            The age below which the accretion history is truncated.
    """

    def __init__(self, max_age=100 * yr, min_age=0 * yr, accretion_rate=None, seed_mass=1E5*Msun, final_mass=None):
        """Initialise the parent and this parametrisation of the SFH.

        Args:
            max_age (unyt_quantity):
                The age above which the accretion history is truncated.
                If min_age = 0 then this is the duration of accretion.
            min_age (unyt_quantity):
                The age below which the accretion history is truncated.
            accretion_rate (unyt_quantity):
                The rate at which mass is accreted.
            seed_mass (unyt_quantity):
                The initial mass of the black hole.
            final_mass (unyt_quantity):
                The final mass of the black hole.
                The duration of the accretion history. This is deprecated
                in favour of max_age.

        """

        # Initialise the parent
        Common.__init__(
            self, name="Constant", min_age=min_age, max_age=max_age
        )

        duration = max_age - min_age

        if (accretion_rate is None) & (final_mass is None):
            raise Exception("Must provide either accretion_rate or final_mass.")

        if accretion_rate is not None:
            if final_mass is not None:
                raise Exception("Both final_mass and accretion_rate were provided; "
                    "ignoring final_mass in favour of accretion_rate.")

            final_mass = seed_mass + accretion_rate * duration

        elif final_mass is not None:
            accretion_rate = (final_mass - seed_mass) / duration

        # Set the model parameters
        self.max_age = max_age
        self.min_age = min_age
        self._max_age = max_age.to('yr').value
        self._min_age = min_age.to('yr').value
        self.seed_mass = seed_mass
        self.accretion_rate = accretion_rate
        self._accretion_rate = accretion_rate.to('Msun/yr').value
        self.final_mass = final_mass
        
    def _accretion_rate(self, age):
        """Get the amount accretion rate in a single age bin.

        Args:
            age (float):
                The age (in years) at which to evaluate the accretion rate.
        """

        # Set the accretion rate based on the duration.
        if (age <= self._max_age) & (age >= self._min_age):
            return self.accretion_rate
        return 0.0

    def _accretion_rates(self, ages):
        """Vectorised version of _accretion_rate for multiple ages.

        Args:
            ages (np.ndarray of float, unyt_array):
                The ages (in years) at which to evaluate the accretion rate.
        """

        if isinstance(ages, unyt_array):
            ages = ages.to('yr').value
        else:
            ages = np.asarray(ages)

        accretion_rates = np.zeros_like(ages)
        mask = (ages <= self._max_age) & (ages >= self._min_age)
        accretion_rates[mask] = self._accretion_rate
        return accretion_rates * Msun/yr
    


class Stochastic(Common):
    """A stochastic black hole accretion history.

    The accretion history is generated from a random process and then
    normalised such that the integrated accreted mass matches the
    requested final mass.

    By default this uses a lognormal distribution of accretion-rate
    fluctuations around a mean value.

    Attributes:
        max_age (unyt_quantity):
            The age above which the accretion history is truncated.
        min_age (unyt_quantity):
            The age below which the accretion history is truncated.
        n_steps (int):
            Number of time bins used to sample the accretion history.
        seed_mass (unyt_quantity):
            Initial black hole mass.
        final_mass (unyt_quantity):
            Final black hole mass.
        sigma (float):
            Width of the lognormal fluctuations.
        random_seed (int):
            Seed for reproducible stochastic histories.
    """

    def __init__(
        self,
        max_age=100 * yr,
        min_age=0 * yr,
        seed_mass=1e5 * Msun,
        final_mass=1e8 * Msun,
        n_steps=1000,
        sigma=1.0,
        random_seed=None,
    ):
        """Initialise the stochastic accretion history.

        Args:
            max_age (unyt_quantity):
                The age above which the accretion history is truncated.
            min_age (unyt_quantity):
                The age below which the accretion history is truncated.
            seed_mass (unyt_quantity):
                Initial black hole mass.
            final_mass (unyt_quantity):
                Final black hole mass.
            n_steps (int):
                Number of timesteps used to sample the accretion history.
            sigma (float):
                Width of the lognormal fluctuations.
            random_seed (int):
                Seed for reproducibility.
        """

        # Initialise parent
        Common.__init__(
            self,
            name="Stochastic",
            min_age=min_age,
            max_age=max_age,
        )

        if final_mass <= seed_mass:
            raise ValueError(
                "final_mass must be greater than seed_mass."
            )

        self.max_age = max_age
        self.min_age = min_age
        self._max_age = max_age.to("yr").value
        self._min_age = min_age.to("yr").value

        self.seed_mass = seed_mass
        self.final_mass = final_mass

        self.n_steps = n_steps
        self.sigma = sigma
        self.random_seed = random_seed

        # Duration and timestep
        duration = max_age - min_age
        self.dt = duration / n_steps
        self._dt = self.dt.to("yr").value

        # Construct time array
        self.ages = np.linspace(
            self._min_age,
            self._max_age,
            n_steps,
        )

        # Generate stochastic fluctuations
        rng = np.random.default_rng(random_seed)

        fluctuations = rng.lognormal(
            mean=0.0,
            sigma=sigma,
            size=n_steps,
        )

        # Compute the required total accreted mass
        total_accreted_mass = final_mass - seed_mass

        # Normalise fluctuations to match final mass
        weights = fluctuations / np.sum(fluctuations)

        accreted_mass_per_bin = (
            weights * total_accreted_mass.to("Msun").value
        ) * Msun

        self.accretion_rates = (
            accreted_mass_per_bin / self.dt
        ).to("Msun/yr")

        self._accretion_rates_array = (
            self.accretion_rates.to("Msun/yr").value
        )

        # Compute cumulative mass history
        cumulative_mass = (
            np.cumsum(accreted_mass_per_bin.to("Msun").value)
            * Msun
        )

        self.mass_history = seed_mass + cumulative_mass

    def _accretion_rate(self, age):
        """Get the accretion rate at a given age.

        Args:
            age (float):
                Age in years.
        """

        if (age < self._min_age) or (age > self._max_age):
            return 0.0 * Msun / yr

        idx = np.searchsorted(self.ages, age) - 1
        idx = np.clip(idx, 0, self.n_steps - 1)

        return self.accretion_rates[idx]

    def _accretion_rates(self, ages):
        """Vectorised accretion rates.

        Args:
            ages (np.ndarray or unyt_array):
                Ages at which to evaluate the accretion rate.
        """

        if isinstance(ages, unyt_array):
            ages = ages.to("yr").value
        else:
            ages = np.asarray(ages)

        rates = np.zeros_like(ages, dtype=float)

        mask = (
            (ages >= self._min_age)
            & (ages <= self._max_age)
        )

        idx = np.searchsorted(self.ages, ages[mask]) - 1
        idx = np.clip(idx, 0, self.n_steps - 1)

        rates[mask] = self._accretion_rates_array[idx]

        return rates * Msun / yr
    


class ConstantEddington(Common):
    """A black hole accretion history with a constant Eddington ratio.

    The black hole grows exponentially according to:

        dM/dt = lambda_edd * M / t_sal

    where:
        lambda_edd : Eddington ratio
        t_sal      : Salpeter timescale

    This produces a mass-dependent accretion rate.

    Attributes:
        duration (unyt_quantity):
            Total duration of the accretion episode.
        seed_mass (unyt_quantity):
            Initial black hole mass.
        eddington_ratio (float):
            Constant Eddington ratio.
        epsilon (float):
            Radiative efficiency.
        salpeter_time (unyt_quantity):
            Salpeter timescale.
    """

    def __init__(
        self,
        duration=100 * Myr,
        seed_mass=1e5 * Msun,
        eddington_ratio=1.0,
        epsilon=0.1,
    ):
        """Initialise the accretion history.

        Args:
            duration (unyt_quantity):
                Duration of the accretion episode.
            seed_mass (unyt_quantity):
                Initial black hole mass.
            eddington_ratio (float):
                Constant Eddington ratio.
            epsilon (float):
                Radiative efficiency.
        """

        if seed_mass is None:
            raise ValueError(
                "seed_mass must be provided."
            )

        if eddington_ratio <= 0:
            raise ValueError(
                "eddington_ratio must be positive."
            )

        # Initialise parent
        Common.__init__(
            self,
            name="ConstantEddington",
            min_age=0 * yr,
            max_age=duration,
        )

        # Salpeter time:
        #
        # t_sal ~= 4.5e7 yr * (epsilon / 0.1)
        #
        self.epsilon = epsilon

        self.salpeter_time = (
            4.5e7 * yr * (epsilon / 0.1)
        )

        self.duration = duration
        self.seed_mass = seed_mass
        self.eddington_ratio = eddington_ratio

        self._duration = duration.to("yr").value
        self._salpeter_time = (
            self.salpeter_time.to("yr").value
        )

        # Final mass from exponential growth
        growth_factor = np.exp(
            eddington_ratio
            * duration.to("yr").value
            / self._salpeter_time
        )

        self.final_mass = seed_mass * growth_factor

    def _mass(self, age):
        """Black hole mass at a given age.

        Args:
            age (float):
                Age in years.
        """

        if age < 0:
            return self.seed_mass

        if age > self._duration:
            age = self._duration

        growth_factor = np.exp(
            self.eddington_ratio
            * age
            / self._salpeter_time
        )

        return self.seed_mass * growth_factor

    def _accretion_rate(self, age):
        """Accretion rate at a given age.

        Args:
            age (float):
                Age in years.
        """

        if (age < 0) or (age > self._duration):
            return 0.0 * Msun / yr

        mass = self._mass(age)

        return (
            self.eddington_ratio
            * mass
            / self.salpeter_time
        ).to("Msun/yr")

    def _accretion_rates(self, ages):
        """Vectorised accretion rates.

        Args:
            ages (np.ndarray or unyt_array):
                Ages at which to evaluate the accretion rate.
        """

        if isinstance(ages, unyt_array):
            ages = ages.to("yr").value
        else:
            ages = np.asarray(ages)

        rates = np.zeros_like(ages, dtype=float)

        mask = (
            (ages >= 0.0)
            & (ages <= self._duration)
        )

        growth_factor = np.exp(
            self.eddington_ratio
            * ages[mask]
            / self._salpeter_time
        )

        masses = (
            self.seed_mass.to("Msun").value
            * growth_factor
        )

        rates[mask] = (
            self.eddington_ratio
            * masses
            / self._salpeter_time
        )

        return rates * Msun / yr

    def _masses(self, ages):
        """Vectorised black hole masses.

        Args:
            ages (np.ndarray or unyt_array):
                Ages at which to evaluate the BH mass.
        """

        if isinstance(ages, unyt_array):
            ages = ages.to("yr").value
        else:
            ages = np.asarray(ages)

        masses = np.full_like(
            ages,
            self.seed_mass.to("Msun").value,
            dtype=float,
        )

        mask = (
            (ages >= 0.0)
            & (ages <= self._duration)
        )

        growth_factor = np.exp(
            self.eddington_ratio
            * ages[mask]
            / self._salpeter_time
        )

        masses[mask] *= growth_factor

        masses[ages > self._duration] = (
            self.final_mass.to("Msun").value
        )

        return masses * Msun