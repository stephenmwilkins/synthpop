import inspect
import io
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from unyt import Gyr, unyt_array, yr, Msun

from synthpop.blackholes.accretion_history_models import Common, parametrisations



class AccretionHistory:
  
    def __init__(self, accretion_history_model, age_range=(0, 10**10), dt=10**6, **kwargs):
        """Initialise the accretion history.

        Args:
            accretion_history_model (str or Common):
                The name of the accretion history model to use, or an instance of
                a child of Common.
        """
        if isinstance(accretion_history_model, str):
            if accretion_history_model not in parametrisations:
                raise ValueError(f"Model {accretion_history_model} not recognised. Must be one of {parametrisations}.")
            self.model = eval(accretion_history_model)(**kwargs)
        elif isinstance(accretion_history_model, Common):
            self.model = accretion_history_model
        else:
            raise ValueError("Model must be either a string or an instance of Common.")

        self.accretion_history_model = accretion_history_model
        self.final_mass = self.accretion_history_model.final_mass 
        self.seed_mass = self.accretion_history_model.seed_mass 

        # Define the age array
        self.ages = np.arange(*age_range, dt) * yr

        self._ages = self.ages.to(yr).value

        self.accretion_history = self._calculate_accretion_history()

        self.mass_history = self._calculate_mass_history()

        self.eddington_ratio_history = self._calaculate_eddington_ratio_history()


    def _calculate_accretion_history(self):
        """Calculate the accretion history. This is a private method that is called by the public method calculate_accretion_history. The public method is responsible for setting the parameters of the model and then calling this method to do the actual calculation."""

        # Evaluate the array
        return self.model.get_accretion_rate(self._ages)

    def _calculate_mass_history(self):
        """Calculate the mass history by integrating the accretion history."""
        return cumtrapz(-self.accretion_history[::-1].to('Msun/yr').value, self.ages[::-1].to('yr').value, initial=0)[::-1] * Msun + self.seed_mass
    
    def _calaculate_eddington_ratio_history(self):
        """Calculate the Eddington ratio history."""
        eddington_luminosity = 1.26e38 * self.mass_history.to(Msun).value  # in erg/s
        bolometric_luminosity = self.accretion_history.to(Msun/yr).value * 1.5e38  # in erg/s, assuming 10% efficiency
        eddington_ratio = bolometric_luminosity / eddington_luminosity
        return eddington_ratio
    

    def plot_accretion_history(self):
        """Plot the accretion history."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.ages.to(Gyr), self.accretion_history.to(Msun/yr))
        plt.yscale("log")
        plt.xlabel("Age (Gyr)")
        plt.ylabel("$log_{10}(Accretion Rate (Msun/yr))$")
        plt.title(f"Accretion History: {self.accretion_history_model}")
        plt.grid()
        plt.show()

    def plot_mass_history(self):
        """Plot the mass history."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.ages.to(Gyr), self.mass_history.to(Msun))
        plt.yscale("log")
        plt.xlabel("Age (Gyr)")
        plt.ylabel("$log_{10}(Mass (Msun))$")
        plt.title(f"Mass History: {self.accretion_history_model}")
        plt.grid()
        plt.show()

    def plot_eddington_ratio_history(self):
        """Plot the Eddington ratio history."""

        plt.figure(figsize=(8, 5))
        plt.plot(self.ages.to(Gyr), self.eddington_ratio_history)
        plt.yscale("log")
        plt.xlabel("Age (Gyr)")
        plt.ylabel("$log_{10}(Eddington Ratio)$")
        plt.title(f"Eddington Ratio History: {self.accretion_history_model}")
        plt.grid()
        plt.show()