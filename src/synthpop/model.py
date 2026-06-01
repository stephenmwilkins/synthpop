
import numpy as np
from synthesizer.parametric import SFH, ZDist
from synthpop.distribution_functions import Schechter
from unyt import yr, Myr, Msun, Gyr, unyt_quantity, Mpc, dimensionless

class Model:
    def __init__(self,            
        galaxy_stellar_mass_function=None,
        sfh_function=None,
        sfh_parameters=None,
        metal_dist_function=None,
        metal_dist_parameters=None,
    ):
                          
        self.galaxy_stellar_mass_function = galaxy_stellar_mass_function
        self.sfh_function = sfh_function
        self.sfh_parameters = sfh_parameters
        self.metal_dist_function = metal_dist_function
        self.metal_dist_parameters = metal_dist_parameters



class Default(Model):
    def __init__(self):

        x_star = 10**10.745 * Msun     
        phi_star = 10**(-2.437) * Mpc**-3  
        alpha = -1.465
        galaxy_stellar_mass_function = Schechter(x_star=x_star, alpha=alpha, phi_star=phi_star)

        # Define a delta function for metallicity
        metal_dist_function = ZDist.DeltaConstant

        metal_dist_parameters = {
            "log10metallicity": -2.5
        }

        sfh_function = SFH.LogNormal

        sfh_parameters = {
            "tau": 0.6 * dimensionless,
            "peak_age": 1E10 * yr,
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

        x_star = 10**10.745 * Msun     
        phi_star = 10**(-2.437) * Mpc**-3  
        alpha = -1.465
        galaxy_stellar_mass_function = Schechter(x_star=x_star, alpha=alpha, phi_star=phi_star)

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

        x_star = 10**10.745 * Msun     
        phi_star = 10**(-2.437) * Mpc**-3  
        alpha = -1.465
        galaxy_stellar_mass_function = Schechter(x_star=x_star, alpha=alpha, phi_star=phi_star)

        # Define a delta function for metallicity
        metal_dist_function = ZDist.DeltaConstant

        metal_dist_parameters = {
            "log10metallicity": -2.5
        }

        # Define a constant star formation history (SFH)
        sfh_function = SFH.Constant

        sfh_parameters = {
        }       

        super().__init__(
            galaxy_stellar_mass_function=galaxy_stellar_mass_function,
            sfh_function=sfh_function,
            sfh_parameters=sfh_parameters,
            metal_dist_function=metal_dist_function,
            metal_dist_parameters=metal_dist_parameters
        )





