import types
import numpy as np
from unyt import yr, Myr, Msun, Gyr, unyt_quantity, Mpc, sr, unyt_array
import matplotlib.pyplot as plt
from synthesizer.parametric import Stars, Galaxy

class Lightcone:

    def __init__(self, 
        model=None,
        minimum_stellar_mass=1e6*Msun,
        maximum_stellar_mass=1e12*Msun,                 
        grid=None,
        cosmology=None,
        redshift_range=(0, 10.),
        solid_angle=4*np.pi * sr,
        random_seed=42
        ):
        

        self.model = model
        self.grid = grid
        self.cosmology = cosmology
        self.redshift_range = redshift_range
        self.solid_angle = solid_angle.to("deg**2")
        self.star_formation_rates = None
        self.random_seed = random_seed

        self.age_of_the_universe = self.cosmology.age(redshift_range[0]).to("yr").value * yr
        self.model.sfh_parameters['max_age'] = self.age_of_the_universe

        # Grids
        z_grid = np.linspace(*self.redshift_range, 500)
        logM_grid = np.linspace(
            np.log10(minimum_stellar_mass.to("Msun").value), 
            np.log10(maximum_stellar_mass.to("Msun").value), 
            500)

        # Volume element (Mpc^3 sr^-1 dz^-1)
        dV_dz = cosmology.differential_comoving_volume(z_grid).to("Mpc**3/sr").value

        # Evaluate Φ(logM)
        phiM = self.model.galaxy_stellar_mass_function.phi_logx(logM_grid)

        # Joint PDF: Φ(M) × dV/dz
        dz = np.diff(z_grid).mean()
        dlogM = np.diff(logM_grid).mean()
        pdf = np.outer(dV_dz, phiM)

        Nexp = (
            np.sum(pdf)
            * dz
            * dlogM
            * self.solid_angle.to("sr").value
        )

        norm = np.sum(pdf) * dz * dlogM
        pdf /= norm

        N = np.random.poisson(Nexp)

        # Flatten and normalize
        pdf_flat = pdf.ravel()
        pdf_flat /= pdf_flat.sum()

        # Draw samples
        idx = np.random.choice(
            pdf_flat.size,
            size=N,
            p=pdf_flat
        )

        iz, iM = np.unravel_index(idx, pdf.shape)

        self.redshifts = z_grid[iz]
        self.final_surviving_masses = 10**logM_grid[iM] * Msun

        self.lookback_times = cosmology.lookback_time(self.redshifts).to("Myr").value * Myr

        self.N = N
        
        print(self.N)

        self._create_galaxies()

        self.surviving_masses = np.array([galaxy.stars.surviving_mass.to("Msun").value for galaxy in self.galaxies]) * Msun

    
    def __str__(self):
        """Print basic summary of the galaxy population."""
        pstr = ""
        pstr += "-" * 10 + "\n"
        pstr += "SUMMARY OF LIGHTCONE" + "\n"
        pstr += f"Number of galaxies: {self.N}" + "\n"
        pstr += f"Redshift range: {self.redshift_range}" + "\n"
        # pstr += f"Total surviving stellar mass density: {(self.total_surviving_stellar_mass/self.volume).to('Msun/Mpc**3'):.2e}" + "\n"
        # pstr += f"Range of surviving stellar masses: {self.surviving_mass_range[0]:.2e} - {self.surviving_mass_range[1]:.2e}" + "\n"
        pstr += "-" * 10 + "\n"
        return pstr


    def _create_galaxies(self):
        """Create galaxy objects with properties sampled from the population."""


        self.galaxies = []

        for redshift, lookback_time, final_surviving_mass in zip(self.redshifts, self.lookback_times, self.final_surviving_masses):

            sfh_parameters = {}
            for key, value in self.model.sfh_parameters.items():  
                if isinstance(value, unyt_quantity):
                    sfh_parameters[key] = value
                elif isinstance(value, types.FunctionType):
                    sfh_parameters[key] = value(final_surviving_mass)
                else:                    
                    raise ValueError(f"Unsupported type for sfh parameter '{key}': {type(value)}")

            sfh_model = self.model.sfh_function(**sfh_parameters) if self.model.sfh_function else None
            
            metal_dist_model = self.model.metal_dist_function(**self.model.metal_dist_parameters) if self.model.metal_dist_function else None

            # Create the Stars object
            final_stars = Stars(
                self.grid.log10ages,
                self.grid.metallicities,
                sf_hist=sfh_model,
                metal_dist=metal_dist_model,
                surviving_mass=final_surviving_mass,
                grid=self.grid,
            )

            # Create the new Stars object
            stars = final_stars.get_at_earlier_time(lookback_time)

            self.galaxies.append(Galaxy(stars=stars, redshift=redshift))


    def calculate_star_formation_rates(self, age=10*Myr):
        """
        Calculate the average star formation rate for each galaxy over a specified age interval.
        
        Parameters
        ----------
        age : unyt_quantity
            The age interval over which to calculate the average star formation rate (e.g., 10 Myr). 
        Returns
        -------
        sfr : np.ndarray
            Array of average star formation rates for each galaxy.
        """

        self.star_formation_rates = np.array([
            galaxy.stars.calculate_average_sfr(t_range=(0, age))
            for galaxy in self.galaxies
        ]) * (Msun / yr)

        return self.star_formation_rates


    def calculate_volume(self, redshift_range=None):
        """Calculate the comoving volume of the universe within a given redshift range.
        
        Parameters
        ----------
        redshift_range : list
            A list of two values specifying the minimum and maximum redshift to consider.   
        
        Returns
        -------
        unyt_quantity
            The comoving volume in units of Mpc^3.

        """

        # calculate total volume of the universe in the redshift range
        volume = (self.cosmology.comoving_volume(redshift_range[1]).to("Mpc**3").value - self.cosmology.comoving_volume(redshift_range[0]).to("Mpc**3").value) * Mpc**3

        return volume * self.solid_angle.to("sr").value / (4 * np.pi)


    def calculate_total_stellar_mass_density(self, redshift_range=[0, 0.5]):
        """Calculate the total stellar mass density at a given redshift range.
        
        Parameters
        ----------
        redshift_range : list
            A list of two values specifying the minimum and maximum redshift to consider.   
        
        Returns
        -------
        unyt_quantity
            The total stellar mass density in units of Msun/Mpc^3.

        """

        selection = (self.redshifts >= redshift_range[0]) & (self.redshifts <= redshift_range[1])
        total_stellar_mass = np.sum(self.surviving_masses[selection])
        volume = self.calculate_volume(redshift_range=redshift_range)

        return total_stellar_mass / volume


    def calculate_cosmic_stellar_mass_density(self, age_bins=None, redshift_bins=None):

        redshift_bin_centres = 0.5 * (redshift_bins[:-1] + redshift_bins[1:])
        stellar_mass_density = np.zeros_like(redshift_bin_centres) * (Msun / Mpc**3)        

        for i in range(len(redshift_bins) - 1):
            stellar_mass_density[i] = self.calculate_total_stellar_mass_density(redshift_range=[redshift_bins[i], redshift_bins[i+1]])
        
        return redshift_bin_centres, stellar_mass_density


    def plot_cosmic_stellar_mass_density(self, age_bins=None, redshift_bins=None):

        redshift_bin_centres, stellar_mass_density = self.calculate_cosmic_stellar_mass_density(age_bins=age_bins, redshift_bins=redshift_bins)

        plt.plot(redshift_bin_centres, stellar_mass_density.to("Msun/Mpc**3").value, marker='o')
        plt.xlabel("Redshift")
        plt.ylabel(r"Cosmic Stellar Mass Density ($M_{\odot}/\mathrm{Mpc}^3$)")
        plt.yscale("log")
        plt.xlim(self.redshift_range)
        plt.show()



    def calculate_cosmic_sfrd(self, age_bins=None, redshift_bins=None):

        if self.star_formation_rates is None:
            self.calculate_star_formation_rates()

        redshift_bin_centres = 0.5 * (redshift_bins[:-1] + redshift_bins[1:])
        sfrd = np.zeros_like(redshift_bin_centres) * (Msun / yr / Mpc**3)
        for i in range(len(redshift_bins) - 1):
            selection = (self.redshifts >= redshift_bins[i]) & (self.redshifts < redshift_bins[i+1])
            total_sfr = np.sum(self.star_formation_rates[selection])
            volume = self.calculate_volume(redshift_range=[redshift_bins[i], redshift_bins[i+1]])
            sfrd[i] = total_sfr / volume    

        return redshift_bin_centres, sfrd
       

    def plot_cosmic_sfrd(self, age_bins=None, redshift_bins=None):

        redshift_bin_centres, sfrd = self.calculate_cosmic_sfrd(age_bins=age_bins, redshift_bins=redshift_bins)

        plt.plot(redshift_bin_centres, sfrd.to("Msun/yr/Mpc**3").value, marker='o')
        plt.xlabel("Redshift")
        plt.ylabel(r"Cosmic Star Formation Rate Density ($M_{\odot}/\mathrm{yr}/\mathrm{Mpc}^3$)")
        plt.yscale("log")
        plt.xlim(self.redshift_range)
        plt.show()




    def generate_spectra(self, emission_model):
        for galaxy in self.galaxies:
            # Get the rest-frame spectra
            galaxy.stars.get_spectra(emission_model)
            # Get the observed-frame spectra
            galaxy.get_observed_spectra(self.cosmology)

        return

    def generate_photometry(
            self, 
            spec_id,
            filters):

        for galaxy in self.galaxies:
            galaxy.stars.spectra[spec_id].get_photo_lnu(filters)
            galaxy.stars.spectra[spec_id].get_photo_fnu(filters)

        return
    
    def get_rest_photometry(self, spec_id, filter_code):
        photometry = []
        for galaxy in self.galaxies:
            photometry.append(galaxy.stars.spectra[spec_id].photo_lnu[filter_code])

        return unyt_array(photometry)
    
    def get_observed_photometry(self, spec_id, filter_code):
        photometry = []
        for galaxy in self.galaxies:
            photometry.append(galaxy.stars.spectra[spec_id].photo_fnu[filter_code])

        return unyt_array(photometry)

    def plot_redshift_final_surviving_mass(self):

        plt.scatter(self.redshifts, self.final_surviving_masses, alpha=0.5, c='k', s=10)

        plt.xlabel("Redshift")
        plt.ylabel("Final surviving stellar mass (Msun)")
        plt.xlim(self.redshift_range)
        plt.yscale("log")
        plt.show()

    def plot_redshift_surviving_mass(self):

        plt.scatter(self.redshifts, self.surviving_masses.to("Msun"), alpha=0.5, c='k', s=10)

        plt.xlabel("Redshift")
        plt.ylabel("Final surviving stellar mass (Msun)")
        plt.xlim(self.redshift_range)
        plt.yscale("log")
        plt.show()


    def plot_number_counts(self, filter_code, bin_edges=None, bin_width=0.2, magnitude=False, observations=None):

        log10f = np.log10(self.get_observed_photometry("incident", filter_code).to("nJy").value)

        if magnitude:
            x = -2.5 * log10f + 31.4  # Convert to AB magnitudes
            xlabel = r"$m$"
        else:
            x = log10f
            xlabel = r"$\log_{10}(F_{\nu}/erg/s/cm^2/Hz)$"

        if bin_edges is None:
            bin_edges = np.arange(np.min(x), np.max(x) + bin_width, bin_width)

        hist, edges = np.histogram(x, bins=bin_edges)

        # Bin centres in linear space
        bin_centres = (edges[:-1] + edges[1:]) / 2

        # Convert histogram to φ(L)
        surface_density = hist / (bin_width * self.solid_angle.to("deg**2").value) 

        # Plot histogram
        plt.plot(bin_centres, np.log10(surface_density), alpha=1, lw=1)

        if observations is not None:
            plt.plot(observations['x'], np.log10(observations['phi']), label='Observations')

        plt.xlabel(xlabel)
        plt.ylabel(r"$\log_{10}(\Sigma/deg^{-2} dex^{-1})$")
        plt.show()