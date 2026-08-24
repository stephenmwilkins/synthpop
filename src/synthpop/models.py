
import numpy as np
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric.sf_hist import Common
from synthesizer import exceptions
from synthpop.distribution_functions import Schechter, Driver2022_DoubleSchechter, Driver2022_SingleSchechter
from unyt import yr, Myr, Msun, Gyr, unyt_quantity, Mpc, dimensionless

class DriverTwoPhase(Common):
    """
    Two-phase star formation history (Driver et al. 2013).
    """

    def __init__(self, tau, n, max_age, min_age=0.0 * yr):
        """
        __init__ method for the DriverTwoPhase class.
        
        Arguments
        ---------
        tau (unyt_quantity)
            The characteristic timescale of the star formation history.
        n (float)
            The power-law index of the delaying term.
        max_age (unyt_quantity)
            The maximum age of the stellar population.
        min_age (unyt_quantity, optional)
            The minimum age of the stellar population.
        """

        if tau <= 0.0 * Gyr or n <= 0:
            raise exceptions.InconsistentArguments(
                "tau and n must be positive."
            )

        # Initialise a new Synthesizer SFH parameterisation.
        Common.__init__(
            self, name="DriverTwoPhase", tau=tau, n=n,
            max_age=max_age, min_age=min_age,
        )

        self.tau = tau.to("yr").value
        self.n = n
        self.max_age = max_age.to("yr").value
        self.min_age = min_age.to("yr").value

    def _sfr(self, age):
        """
        Compute the star formation rate (SFR) at a given age.
        
        Arguments
        ---------
        age (unyt_quantity)
            The age of the stellar population.
        
        Returns
        -------
        float
            The star formation rate at the given age.
        """

        t = self.max_age - age
        if (age < self.max_age) and (age >= self.min_age) and t > 0:
            return ((self.tau / t) ** self.n) * np.exp(-self.tau / t)
        return 0.0

    def _sfrs(self, ages):
        """
        Compute the star formation rates (SFRs) for an array of ages.
        
        Arguments
        ---------
        ages (array-like)
            An array of ages of the stellar population.
        
        Returns
        -------
        np.ndarray
            An array of star formation rates corresponding to the input ages.
        """

        t = self.max_age - ages
        sfrs = np.zeros_like(ages, dtype=np.float64)
        mask = (ages < self.max_age) & (ages >= self.min_age) & (t > 0)
        sfrs[mask] = ((self.tau / t[mask]) ** self.n) * np.exp(-self.tau / t[mask])
        return sfrs

# Convenience presets using the paper's recommended (eq. 4/5) shape parameters
class DriverSpheroid(DriverTwoPhase):
    """
    Convenience function for Spheroid SFH (Driver et al. 2013).
    """
    def __init__(self, max_age=1.37e10 * yr, min_age=0.0 * yr):
        super().__init__(tau=21.86 * Gyr, n=8.57, max_age=max_age, min_age=min_age)

class DriverSpheroid2(DriverTwoPhase):
    """
    Convenience function for Spheroid SFH including AGN (Driver et al. 2013).
    """
    def __init__(self, max_age=1.37e10 * yr, min_age=0.0 * yr):
        super().__init__(tau=16.82 * Gyr, n=6.97, max_age=max_age, min_age=min_age)


class DriverDisc(DriverTwoPhase):
    """
    Convenience function for Disc SFH (Driver et al. 2013).
    """
    def __init__(self, max_age=1.37e10 * yr, min_age=0.0 * yr):
        super().__init__(tau=29.39 * Gyr, n=5.50, max_age=max_age, min_age=min_age)

class Model:
    def __init__(self,            
        galaxy_stellar_mass_function=None,
        sfh_function=None,
        sfh_parameters=None,
        metal_dist_function=None,
        metal_dist_parameters=None,
        dust_attenuation_function=None,
    ):
                          
        self.galaxy_stellar_mass_function = galaxy_stellar_mass_function
        self.sfh_function = sfh_function
        self.sfh_parameters = sfh_parameters
        self.metal_dist_function = metal_dist_function
        self.metal_dist_parameters = metal_dist_parameters
        self.dust_attenuation_function = dust_attenuation_function

    def get_sfh(self, mass, time_array):

        sfh_params = {key: (value(mass) if callable(value) else value) for key, value in self.sfh_parameters.items()}
        print(f"SFH parameters for mass {mass.to('Msun'):.2e}: {sfh_params}")

        return self.sfh_function(**sfh_params).get_sfr(time_array.to('yr').value) 

    def plot_sfh(self, mass, time_array):

        sfh = self.get_sfh(mass, time_array)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(time_array.to('Gyr'), sfh)
        plt.xlabel("Time (Gyr)")
        plt.ylabel("SFR (Msun/yr)")
        plt.title(f"SFH for Mass = {mass.to('Msun'):.2e}")
        plt.grid()
        plt.show()


class Default(Model):
    def __init__(self):

        galaxy_stellar_mass_function = Driver2022_DoubleSchechter

        # Define a delta function for metallicity
        metal_dist_function = ZDist.DeltaConstant

        metal_dist_parameters = {
            "log10metallicity": -2.5
        }

        sfh_function = SFH.LogNormal

        sfh_parameters = {
            "tau": 0.6 * dimensionless,
            "peak_age": 1E10 * yr,
            "max_age": 1.37E10 * yr
        }

        super().__init__(
            galaxy_stellar_mass_function=galaxy_stellar_mass_function,
            sfh_function=sfh_function,
            sfh_parameters=sfh_parameters,
            metal_dist_function=metal_dist_function,
            metal_dist_parameters=metal_dist_parameters
        )


class Spheroid(Model):
    def __init__(self):

        galaxy_stellar_mass_function = Driver2022_DoubleSchechter

        # Define a delta function for metallicity
        metal_dist_function = ZDist.DeltaConstant

        metal_dist_parameters = {
            "log10metallicity": -2.5
        }

        sfh_function = SFH.DecliningExponential

        sfh_parameters = {
            "tau": 0.6 * Gyr,
            "max_age": 1.2E10 * yr,
        }

        super().__init__(
            galaxy_stellar_mass_function=galaxy_stellar_mass_function,
            sfh_function=sfh_function,
            sfh_parameters=sfh_parameters,
            metal_dist_function=metal_dist_function,
            metal_dist_parameters=metal_dist_parameters
        )



class Default2(Model):
    def __init__(self):

        galaxy_stellar_mass_function = Driver2022_DoubleSchechter

        # Define a delta function for metallicity
        metal_dist_function = ZDist.DeltaConstant

        metal_dist_parameters = {
            "log10metallicity": -2.5
        }

        sfh_function = SFH.LogNormal

        def peak_age_function(mass, age_of_Universe=1.37E10 * yr):

            value = (mass/(1E9*Msun))**2.5 * 1E9 + np.random.normal(0, 1e9)

            return np.min((value, age_of_Universe.to('yr').value)) * yr

        def tau_function(mass):
            return np.clip(np.random.normal(0.6, 0.1) * dimensionless, 0.1, 1.0)

        sfh_parameters = {
            "tau": tau_function,
            "peak_age": peak_age_function,
            "max_age": 1.37E10 * yr
        }

        super().__init__(
            galaxy_stellar_mass_function=galaxy_stellar_mass_function,
            sfh_function=sfh_function,
            sfh_parameters=sfh_parameters,
            metal_dist_function=metal_dist_function,
            metal_dist_parameters=metal_dist_parameters
        )



class Constant(Model):
    def __init__(self):

        galaxy_stellar_mass_function = Driver2022_DoubleSchechter

        # Define a delta function for metallicity
        metal_dist_function = ZDist.DeltaConstant

        metal_dist_parameters = {
            "log10metallicity": -2.5
        }

        # Define a constant star formation history (SFH)
        sfh_function = SFH.Constant

        sfh_parameters = {
            "max_age": 1.37E10 * yr
        }       

        # Define dust attenuation as a function of stellar mass
        def dust_attenuation_function(mass):

            tau_v = 0.1 * ((mass-1E9*Msun)/(1E9*Msun))  # Example: tau_v decreases with mass
            r = np.random.normal(0, 0.5, size=mass.shape)  # Add some scatter
            return np.maximum(0.0, tau_v * (1 + r)) * dimensionless

        super().__init__(
            galaxy_stellar_mass_function=galaxy_stellar_mass_function,
            sfh_function=sfh_function,
            sfh_parameters=sfh_parameters,
            metal_dist_function=metal_dist_function,
            metal_dist_parameters=metal_dist_parameters,
            dust_attenuation_function=dust_attenuation_function,
        )

class Constant100(Model):
    def __init__(self):

        galaxy_stellar_mass_function = Driver2022_DoubleSchechter

        # Define a delta function for metallicity
        metal_dist_function = ZDist.DeltaConstant

        metal_dist_parameters = {
            "log10metallicity": -2.5
        }

        # Define a constant star formation history (SFH)
        sfh_function = SFH.Constant

        sfh_parameters = {
            "max_age": 100 * Myr
        }       

        # Define dust attenuation as a function of stellar mass
        def dust_attenuation_function(mass):

            tau_v = 0.1 * ((mass-1E9*Msun)/(1E9*Msun))  # Example: tau_v decreases with mass
            r = np.random.normal(0, 0.5, size=mass.shape)  # Add some scatter
            return np.maximum(0.0, tau_v * (1 + r)) * dimensionless

        super().__init__(
            galaxy_stellar_mass_function=galaxy_stellar_mass_function,
            sfh_function=sfh_function,
            sfh_parameters=sfh_parameters,
            metal_dist_function=metal_dist_function,
            metal_dist_parameters=metal_dist_parameters,
            dust_attenuation_function=dust_attenuation_function,
        )


class DriverMoffettSpheroid(Model):
    def __init__(self):

        galaxy_stellar_mass_function = Schechter(
            phi_star=3.70e-3 / Mpc**3, alpha=-0.623 * dimensionless, x_star=10**10.60 * Msun)

        # Define a delta function for metallicity
        metal_dist_function = ZDist.DeltaConstant

        metal_dist_parameters = {
            "log10metallicity": -2.5
        }

        sfh_function = DriverTwoPhase

        sfh_parameters = {
            "tau": 21.86 * Gyr,
            "n": 8.57 * dimensionless,
            "max_age": 1.37E10 * yr,
            "min_age": 0.0 * yr
        }

        # Define dust attenuation as a function of stellar mass
        def dust_attenuation_function(mass):

            tau_v = 0.1 * ((mass-1E9*Msun)/(1E9*Msun))  # Example: tau_v decreases with mass
            r = np.random.normal(0, 0.5, size=mass.shape)  # Add some scatter
            return np.maximum(0.0, tau_v * (1 + r)) * dimensionless

        super().__init__(
            galaxy_stellar_mass_function=galaxy_stellar_mass_function,
            sfh_function=sfh_function,
            sfh_parameters=sfh_parameters,
            metal_dist_function=metal_dist_function,
            metal_dist_parameters=metal_dist_parameters,
            dust_attenuation_function=dust_attenuation_function
        )

class DriverMoffettSpheroid2(Model):
    def __init__(self):

        galaxy_stellar_mass_function = Schechter(
            phi_star=3.70e-3 / Mpc**3, alpha=-0.623 * dimensionless, x_star=10**10.60 * Msun)

        # Define a delta function for metallicity
        metal_dist_function = ZDist.DeltaConstant

        metal_dist_parameters = {
            "log10metallicity": -2.5
        }

        sfh_function = DriverTwoPhase

        sfh_parameters = {
            "tau": 16.82 * Gyr,
            "n": 6.97 * dimensionless,
            "max_age": 1.37E10 * yr,
            "min_age": 0.0 * yr
        }

        # Define dust attenuation as a function of stellar mass
        def dust_attenuation_function(mass):

            tau_v = 0.1 * ((mass-1E9*Msun)/(1E9*Msun))  # Example: tau_v decreases with mass
            r = np.random.normal(0, 0.5, size=mass.shape)  # Add some scatter
            return np.maximum(0.0, tau_v * (1 + r)) * dimensionless

        super().__init__(
            galaxy_stellar_mass_function=galaxy_stellar_mass_function,
            sfh_function=sfh_function,
            sfh_parameters=sfh_parameters,
            metal_dist_function=metal_dist_function,
            metal_dist_parameters=metal_dist_parameters,
            dust_attenuation_function=dust_attenuation_function
        )

class DriverMoffettDisc(Model):
    def __init__(self):

        galaxy_stellar_mass_function = Schechter(
            phi_star=1.72e-3 / Mpc**3, alpha=-1.20 * dimensionless, x_star=10**10.73 * Msun)

        # Define a delta function for metallicity
        metal_dist_function = ZDist.DeltaConstant

        metal_dist_parameters = {
            "log10metallicity": -2.5
        }

        sfh_function = DriverTwoPhase

        sfh_parameters = {
            "tau": 29.39 * Gyr,
            "n": 5.50 * dimensionless,
            "max_age": 1.37E10 * yr,
            "min_age": 0.0 * yr
        }

        # Define dust attenuation as a function of stellar mass
        def dust_attenuation_function(mass):

            tau_v = 0.1 * ((mass-1E9*Msun)/(1E9*Msun))  # Example: tau_v decreases with mass
            r = np.random.normal(0, 0.5, size=mass.shape)  # Add some scatter
            return np.maximum(0.0, tau_v * (1 + r)) * dimensionless

        super().__init__(
            galaxy_stellar_mass_function=galaxy_stellar_mass_function,
            sfh_function=sfh_function,
            sfh_parameters=sfh_parameters,
            metal_dist_function=metal_dist_function,
            metal_dist_parameters=metal_dist_parameters,
            dust_attenuation_function=dust_attenuation_function
        )