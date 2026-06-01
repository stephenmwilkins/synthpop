import numpy as np
from unyt import unyt_array, unyt_quantity
from unyt.dimensions import mass
import matplotlib.pyplot as plt

class Common:
    def __init__(self, name=None, parameters=None):
        self.name = name
        self.parameters = parameters

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

    def sample(self, x_min=0.01, x_max=100, phi_star=1.0, volume=1e6):
        """
        Sample values from the Schechter function using inverse transform sampling.

        Parameters
        ----------
        x_min, x_max : float
            Quantity limits
        volume : float
            Survey volume (same units as phi_star^-1)
        
        Returns
        -------
        x_samples : ndarray
            Array of sampled quantities
        """

        # Compute expected number of galaxies in the volume
        # Integrate Schechter function numerically using bins in log space
        x_bins = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
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
        
    def plot(self, x_min=0.01, x_max=100):
        """
        Plot the distribution function.

        Parameters
        ----------
        x_min, x_max : float
            Quantity limits in units of x_star
        """
        x = np.logspace(np.log10(x_min*self.x_star), np.log10(x_max*self.x_star), 1000)
        phi_vals = self.phi(x)

        plt.figure(figsize=(8, 5))
        plt.plot(x, phi_vals, label=self.name)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Quantity (e.g. luminosity or mass)")
        plt.ylabel(r"$\phi$ (number density per unit quantity)")
        plt.title(f"{self.name} Distribution Function")
        plt.legend()
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
                x_star = x_star.to("Msun").value
            else:
                x_star = x_star.value

        if x_star is None:
            raise ValueError("Must specify x_star.")

        self.x_star = x_star
        self.phi_star = phi_star
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
                x_star = x_star.to("Msun").value
            else:
                x_star = x_star.value

        if x_star is None:
            raise ValueError("Must specify x_star.")

        self.x_star = x_star
        self.phi1_star = phi1_star
        self.alpha1 = alpha1
        self.phi2_star = phi2_star
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