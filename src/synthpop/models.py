
import numpy as np
from synthesizer.parametric import SFH, ZDist
from synthpop.distribution_functions import Schechter, Driver2022_DoubleSchechter, Driver2022_SingleSchechter
from unyt import yr, Myr, Msun, Gyr, unyt_quantity, Mpc, dimensionless

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





