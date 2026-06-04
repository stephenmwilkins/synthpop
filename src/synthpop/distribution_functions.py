from unittest import result

import numpy as np
from unyt import unyt_array, unyt_quantity, Mpc, Msun
from unyt.dimensions import mass
import matplotlib.pyplot as plt
from scipy.special import gammaincc, gamma
from scipy.integrate import quad

class Common:
    def __init__(self, name=None, parameters=None, x_units=None):
        self.name = name
        self.parameters = parameters
        self.x_units = x_units

    def __str__(self):
        """Print basic summary of the distribution function."""
        pstr = ""
        pstr += "-" * 10 + "\n"
        pstr += "SUMMARY OF DISTRIBUTION FUNCTION" + "\n"
        pstr += f"Name: {self.name}" + "\n"
        for key, value in self.parameters.items():
            pstr += f"{key}: {value}" + "\n"
        pstr += "-" * 10 + "\n"
        return pstr

    def first_moment_quad(self, xmin=0.0, xmax=np.inf):
        """
        Compute the first moment:

            ∫ x φ(x) dx
        """
        result, _ = quad(
            lambda x: x * self.phi(x),
            xmin,
            xmax,
        )

        return result

    def sample(self, xmin=0.01, xmax=100, volume=1e6):
        """
        Sample values from the Schechter function using inverse transform sampling.

        Parameters
        ----------
        xmin, xmax : float
            Quantity limits
        volume : float
            Survey volume (same units as phi_star^-1)
        
        Returns
        -------
        x_samples : ndarray
            Array of sampled quantities
        """

        volume = volume.to("Mpc**3").value 
        
        xmin = xmin.to(self.x_units).value
        xmax = xmax.to(self.x_units).value

        # Compute expected number of galaxies in the volume
        # Integrate Schechter function numerically using bins in log space
        x_bins = np.logspace(np.log10(xmin), np.log10(xmax), 1000)
        phi_vals = self.phi(x_bins)

        dx = np.diff(x_bins)
        # Mid-bin values
        phi_mid = 0.5 * (phi_vals[:-1] + phi_vals[1:])
        # Expected number in each bin
        N_expected = phi_mid * dx * volume

        # Poisson sample the number of galaxies per bin
        N_gal = np.random.poisson(N_expected)
        
        # For each bin, sample galaxy luminosities uniformly within the bin
        x_samples = []
        for i, N in enumerate(N_gal):
            if N > 0:
                x_samples.append(np.random.uniform(x_bins[i], x_bins[i+1], N))
        if x_samples:
            return np.concatenate(x_samples)
        else:
            return np.array([])


    def calculate_number_density(self, xmin=0.0, xmax=np.inf):
        """
        Compute the number density, i.e. the integral of phi(x) dx between xmin and xmax.

        Parameters
        ----------
        xmin, xmax : float
            Quantity limits
        
        Returns
        -------
        number_density : unyt_quantity
            Number density of galaxies between xmin and xmax.
        """

        # Strip units from xmin and xmax if they are unyt objects, since the first_moment method expects unitless inputs in the same units as x_star
        if isinstance(xmin, (unyt_array, unyt_quantity)):
            xmin = xmin.to(self.x_units).value

        if isinstance(xmax, (unyt_array, unyt_quantity)):
            xmax = xmax.to(self.x_units).value

        return self.zeroth_moment(xmin=xmin, xmax=xmax) / Mpc**3


    def calculate_mass_density(self, xmin=0.0, xmax=np.inf):
        """
        Compute the mass integral, i.e. the integral of x * phi(x) dx between xmin and xmax.

        Parameters
        ----------
        xmin, xmax : float
            Quantity limits
        
        Returns
        -------
        mass_density : unyt_quantity
            Mass density of galaxies between xmin and xmax.
        """

        # Strip units from xmin and xmax if they are unyt objects, since the first_moment method expects unitless inputs in the same units as x_star
        if isinstance(xmin, (unyt_array, unyt_quantity)):
            xmin = xmin.to(self.x_units).value

        if isinstance(xmax, (unyt_array, unyt_quantity)):
            xmax = xmax.to(self.x_units).value

        return self.first_moment(xmin=xmin, xmax=xmax) * Msun / Mpc**3
    
    def calculate_luminosity_density(self, xmin=0.0, xmax=np.inf):

        """
        Compute the luminosity integral, i.e. the integral of x * phi(x) dx between xmin and xmax.
        """
        return self.first_moment(xmin=xmin, xmax=xmax) * self.x_units / Mpc**3


    def calculate_omega_star(self, cosmology):
        """
        Compute the stellar mass density parameter Omega_star.

        Parameters
        ----------
        cosmology : astropy.cosmology instance
            Cosmological model to use for critical density calculation.

        Returns
        -------
        omega_star : float
            Stellar mass density parameter.
        """
        rho_crit = cosmology.critical_density0.to("Msun/Mpc**3").value * Msun / Mpc**3
        rho_star = self.calculate_mass_density()
        return rho_star / rho_crit



    def plot(self, xmin=0.001, xmax=100, ylimits=None, grid=True, observations=None, samples=None, bins=30, volume=None):
        """
        Plot the distribution function.

        Parameters
        ----------
        xmin, xmax : float
            Quantity limits in units of x_star
        """


        plt.figure(figsize=(8, 5))

        # Plot the analytical distribution function in log space
        x = np.logspace(np.log10(xmin*self.x_star), np.log10(xmax*self.x_star), 1000)
        log10phi_vals = np.log10(self.phi_logx(np.log10(x)))
        plt.plot(np.log10(x), log10phi_vals, label=self.name)

        # Plot any samples if provided
        if samples is not None:

            # Create histogram of samples in log space
            hist, edges = np.histogram(np.log10(samples), range=[np.log10(xmin*self.x_star), np.log10(xmax*self.x_star)], bins=bins)

            # Calculate bin width in log space
            bin_width = edges[1] - edges[0]

            # Convert histogram to φ(L)
            phi_sampled = hist / (bin_width * volume.to("Mpc**3").value) 

            # Plot histogram
            plt.step(edges[:-1], np.log10(phi_sampled), where='post')
            
        if ylimits is not None:
            plt.ylim(ylimits)

        plt.xlabel("Quantity (e.g. luminosity or mass)")
        plt.ylabel(r"$\phi$ (number density per unit quantity)")
        plt.title(f"{self.name} Distribution Function")
        plt.legend()
        if grid:
            plt.grid()
        plt.show()


class Schechter(Common):

    def __init__(self, phi_star=None, alpha=None, x_star=None):
        """
        Initialize the Schechter function parameters.

        Parameters
        ----------
        phi_star : float
            Normalization.
        alpha : float
            Faint-end slope.
        x_star : float
            Characteristic scale.
        """

        if isinstance(x_star, (unyt_quantity, unyt_array)):
            if x_star.units.dimensions == mass:
                self.x_units = "Msun"
                x_star = x_star.to("Msun").value
            elif x_star.units == "erg/(Hz*s)":
                self.x_units = "erg/(Hz*s)"
                x_star = x_star.to("erg/(Hz*s)").value
        else:
            self.x_units = None        

        if x_star is None:
            raise ValueError("Must specify x_star.")

        self.x_star = x_star
        self.phi_star = phi_star.to("Mpc**-3").value if phi_star is not None else None
        self.alpha = alpha

        self.parameters = {
            "x_star": self.x_star,
            "phi_star": self.phi_star,
            "alpha": self.alpha,
        }

        # Instantiate the parent
        Common.__init__(
            self,
            name="Schechter",
            parameters=self.parameters,
            x_units=self.x_units
        )

    def phi(self, x):
        """
        Compute the Schechter function.

        Parameters
        ----------
        x : array-like
            Quantity array.
        x_star : float
            Characteristic scale.
        phi_star : float
            Normalization.
        alpha : float
            Faint-end slope.

        Returns
        -------
        phi : array-like
            Schechter function evaluated at x.
        """
        return (
            (self.phi_star / self.x_star)
            * (x / self.x_star) ** self.alpha
            * np.exp(-x / self.x_star)
        )

    def phi_logx(self, logx):
        """
        Schechter function in log space.

        Parameters
        ----------
        logx : array-like
            log10(x), where x has the same units as x_star.
        phi_star : float
            Normalization.
        alpha : float
            Faint-end slope.

        Returns
        -------
        phi_logx : array
            Schechter function values per dex.
        """
        x = 10**logx

        phi = (
            np.log(10)
            * self.phi_star
            * (x / self.x_star) ** (self.alpha + 1)
            * np.exp(-x / self.x_star)
        )

        return phi

    def first_moment(self, xmin=0.0, xmax=np.inf):

        """

        Integrate x * phi(x) dx between xmin and xmax.

        """

        s = self.alpha + 2
        x1 = xmin / self.x_star
        x2 = xmax / self.x_star
        if np.isinf(xmax):
            upper = 0.0
        else:
            upper = gamma(s) * gammaincc(s, x2)
        lower = gamma(s) * gammaincc(s, x1)

        return self.phi_star * self.x_star * (lower - upper)

    def zeroth_moment(self, xmin=0.0, xmax=np.inf):
        """
        Integrate phi(x) dx between xmin and xmax.

        Returns
        -------
        float
            Number density.
        """

        s = self.alpha + 1

        x1 = xmin / self.x_star
        x2 = xmax / self.x_star

        if np.isinf(xmax):
            upper = 0.0
        else:
            upper = gamma(s) * gammaincc(s, x2)

        lower = gamma(s) * gammaincc(s, x1)

        return self.phi_star * (lower - upper)



class DoubleSchechter(Common):

    def __init__(
        self,
        phi1_star=None,
        alpha1=None,
        phi2_star=None,
        alpha2=None,
        x_star=None,
    ):
        """
        Initialize the Double Schechter function parameters.

        Parameters
        ----------
        phi1_star : float
            Normalization of first component.
        alpha1 : float
            Faint-end slope of first component.
        phi2_star : float
            Normalization of second component.
        alpha2 : float
            Faint-end slope of second component.
        x_star : float
            Characteristic scale (e.g. L* or M*).
        """

        if isinstance(x_star, (unyt_quantity, unyt_array)):
            if x_star.units.dimensions == mass:
                self.x_units = "Msun"
                x_star = x_star.to("Msun").value
            elif x_star.units == "erg/(Hz*s)":
                self.x_units = "erg/(Hz*s)"
                x_star = x_star.to("erg/(Hz*s)").value
        else:
            self.x_units = None        

        if x_star is None:
            raise ValueError("Must specify x_star.")

        self.x_star = x_star
        self.phi1_star = phi1_star.to("Mpc**-3").value if phi1_star is not None else None
        self.alpha1 = alpha1
        self.phi2_star = phi2_star.to("Mpc**-3").value if phi2_star is not None else None
        self.alpha2 = alpha2

        self.parameters = {
            "x_star": self.x_star,
            "phi1_star": self.phi1_star,
            "alpha1": self.alpha1,
            "phi2_star": self.phi2_star,
            "alpha2": self.alpha2,
        }

        # Instantiate the parent
        Common.__init__(
            self,
            name="DoubleSchechter",
            parameters=self.parameters,
            x_units=self.x_units
        )

    def phi(self, x):
        """
        Double Schechter function.

        Parameters
        ----------
        x : array-like
            Quantity being modeled (e.g. luminosity or mass).

        Returns
        -------
        phi : array-like
            Number density per unit x.
        """
        xx = x / self.x_star

        return (
            np.exp(-xx) / self.x_star
            * (
                self.phi1_star * xx**self.alpha1
                + self.phi2_star * xx**self.alpha2
            )
        )

    def phi_logx(self, logx):
        """
        Double Schechter function in log10 space.

        Parameters
        ----------
        logx : array-like
            log10(x).

        Returns
        -------
        phi_logx : array-like
            Number density per dex.
        """
        x = 10**logx
        xx = x / self.x_star

        return (
            np.log(10)
            * np.exp(-xx)
            * (
                self.phi1_star * xx**(self.alpha1 + 1)
                + self.phi2_star * xx**(self.alpha2 + 1)
            )
        )
    
    def first_moment(self, xmin=0.0, xmax=np.inf):
        """
        Integral of x * phi(x) dx.
        """

        x1 = xmin / self.x_star
        x2 = xmax / self.x_star

        def component(phi_star, alpha):
            s = alpha + 2

            lower = gamma(s) * gammaincc(s, x1)

            if np.isinf(xmax):
                upper = 0.0
            else:
                upper = gamma(s) * gammaincc(s, x2)

            return phi_star * (lower - upper)

        return self.x_star * (
            component(self.phi1_star, self.alpha1)
            + component(self.phi2_star, self.alpha2)
        )    

    def zeroth_moment(self, xmin=0.0, xmax=np.inf):

        x1 = xmin / self.x_star
        x2 = xmax / self.x_star

        def integral(alpha):

            s = alpha + 1

            if np.isinf(xmax):
                upper = 0.0
            else:
                upper = gamma(s) * gammaincc(s, x2)

            lower = gamma(s) * gammaincc(s, x1)

            return lower - upper

        return (
            self.phi1_star * integral(self.alpha1)
            + self.phi2_star * integral(self.alpha2)
        )



class DoublePowerLaw(Common):
    """
    Double power-law distribution function.

    \[
    \phi(x) = \frac{\phi_*}{x_*}
    \left[
    \left(\frac{x}{x_*}\right)^{-\alpha}
    +
    \left(\frac{x}{x_*}\right)^{-\beta}
    \right]^{-1}, 
    \]
    """


    def __init__(
        self,
        phi_star=None,
        alpha=None,
        beta=None,
        x_star=None,
    ):
        """
        Initialize the double power-law parameters.

        Parameters
        ----------
        phi_star : float
            Normalization.
        alpha : float
            Low-x slope.
        beta : float
            High-x slope.
        x_star : float
            Break scale.
        """

        if isinstance(x_star, (unyt_quantity, unyt_array)):
            if x_star.units.dimensions == mass:
                self.x_units = "Msun"
                x_star = x_star.to("Msun").value
            elif x_star.units == "erg/(Hz*s)":
                self.x_units = "erg/(Hz*s)"
                x_star = x_star.to("erg/(Hz*s)").value
        else:
            self.x_units = None      

        if x_star is None:
            raise ValueError("Must specify x_star.")

        self.x_star = x_star
        self.phi_star = (
            phi_star.to("Mpc**-3").value
            if phi_star is not None
            else None
        )
        self.alpha = alpha
        self.beta = beta

        self.parameters = {
            "x_star": self.x_star,
            "phi_star": self.phi_star,
            "alpha": self.alpha,
            "beta": self.beta,
        }

        Common.__init__(
            self,
            name="DoublePowerLaw",
            parameters=self.parameters,
            x_units=self.x_units,
        )

    def phi(self, x):
        """
        Double power-law per unit x.
        """

        xx = x / self.x_star

        return (
            (self.phi_star / self.x_star)
            / (
                xx**(-self.alpha)
                + xx**(-self.beta)
            )
        )

    def phi_logx(self, logx):
        """
        Double power-law per dex.
        """

        x = 10**logx
        xx = x / self.x_star

        return (
            np.log(10)
            * self.phi_star
            * xx
            / (
                xx**(-self.alpha)
                + xx**(-self.beta)
            )
        )

    def first_moment(self, xmin=0.0, xmax=np.inf):
        """
        Integrate x * phi(x) dx.

        Numerical implementation because the
        general double power-law does not have
        a simple closed-form primitive.
        """

        from scipy.integrate import quad

        result, _ = quad(
            lambda x: x * self.phi(x),
            xmin,
            xmax,
        )

        return result


    def zeroth_moment(self, xmin=0.0, xmax=np.inf):

        result, _ = quad(
            self.phi,
            xmin,
            xmax,
        )

        return result






predefined_distribution_functions = ['Driver2022_SingleSchechter', 'Driver2022_DoubleSchechter']


Driver2022_SingleSchechter = Schechter(
    x_star=10**10.745 * Msun,
    phi_star=10**(-2.437) * Mpc**-3,
    alpha=-0.465,
)

Driver2022_DoubleSchechter = DoubleSchechter(
    x_star=10**10.745 * Msun,
    phi1_star=10**(-2.437) * Mpc**-3,
    alpha1=-0.466,
    phi2_star=10**(-3.201) * Mpc**-3,
    alpha2=-1.530,
)