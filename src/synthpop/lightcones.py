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

    def __add__(self, lightcone2):
        """
        Add two Lightcone instances by concatenating their galaxy lists.
        
        Parameters
        ----------
        lightcone2 : Lightcone
            Another Lightcone instance to add.

        Returns
        -------
        Lightcone
            A new Lightcone instance containing the combined galaxy 
            populations of both lightcones.
        """


        if not isinstance(lightcone2, Lightcone):
            return NotImplemented

        # Ensure the lightcones have the same cosmology, extent and grid.
        if self.cosmology != lightcone2.cosmology:
            raise ValueError("Cannot add Lightcone instances with different cosmologies.")
        
        if self.redshift_range != lightcone2.redshift_range:
            raise ValueError("Cannot add Lightcone instances with different redshift ranges.")
        
        if self.grid != lightcone2.grid:
            raise ValueError("Cannot add Lightcone instances with different SPS grids.")

        if not np.isclose(
            self.solid_angle.to("sr").value,
            lightcone2.solid_angle.to("sr").value,
        ):
            raise ValueError("Cannot add Lightcone instances with different solid angles.")

        lightcone3 = Lightcone.__new__(Lightcone)

        # Copy over the attributes associated with these elements.
        lightcone3.model = self.model
        lightcone3.grid = self.grid
        lightcone3.cosmology = self.cosmology
        lightcone3.redshift_range = self.redshift_range
        lightcone3.solid_angle = self.solid_angle
        lightcone3.random_seed = self.random_seed
        lightcone3.age_of_the_universe = self.age_of_the_universe

        # Concatenate the galaxy lists and update the number of galaxies.
        lightcone3.galaxies = self.galaxies + lightcone2.galaxies
        lightcone3.N = len(lightcone3.galaxies)

        lightcone3.redshifts = np.concatenate([self.redshifts, lightcone2.redshifts])

        lightcone3.lookback_times = np.concatenate([
            self.lookback_times.to("Myr").value,
            lightcone2.lookback_times.to("Myr").value,
        ]) * Myr

        lightcone3.final_surviving_masses = np.concatenate([
            self.final_surviving_masses.to("Msun").value,
            lightcone2.final_surviving_masses.to("Msun").value,
        ]) * Msun

        lightcone3.surviving_masses = np.concatenate([
            self.surviving_masses.to("Msun").value,
            lightcone2.surviving_masses.to("Msun").value,
        ]) * Msun

        lightcone3.star_formation_rates = None
        if (
            self.star_formation_rates is not None
            and lightcone2.star_formation_rates is not None
        ):
            lightcone3.star_formation_rates = np.concatenate([
                self.star_formation_rates.to("Msun/yr").value,
                lightcone2.star_formation_rates.to("Msun/yr").value,
            ]) * (Msun / yr)

        return lightcone3

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

            # Create the new Stars object at the earlier lookback time.
            stars = final_stars.get_at_earlier_time(lookback_time)
            stars.surviving_mass = stars.calculate_surviving_mass(self.grid)

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


    def plot_number_counts(self, filter_code, bin_edges=None, bin_width=0.5, magnitude=False,
                           observations=None):
        """
        Plot the galaxy number counts in a given filter, optionally 
        comparing to observational data.

        Parameters
        ----------
        filter_code : str
            The filter code for which to plot the number counts.
        bin_edges : array-like, optional
            The edges of the bins for the histogram. 
            If None, use full range.
        bin_width : float, optional
            The bin width when using the full range.
        magnitude : bool, optional
            If True, plot in and assume bin_edges are in AB magnitudes.
        observations : dict, optional
            Observational data to compare against. Should have keys 'x' 
            and 'phi' for the x-values and number counts, respectively.
        """

        flux = self.get_observed_photometry("incident", filter_code).to("Jy").value

        if magnitude:
            x = -2.5 * np.log10(flux) + 8.90  # convert to AB magnitudes.
            xlabel = r"$m_{\mathrm{AB}}$"
        else:
            x = np.log10(flux)
            xlabel = r"$\log_{10}(F_{\nu} \, / \, \mathrm{Jy})$"

        # Bin and normalise by bin width and solid angle.
        if bin_edges is None:
            bin_edges = np.arange(np.min(x), np.max(x) + bin_width, bin_width)

        hist, edges = np.histogram(x, bins=bin_edges)
        surface_density = hist / (bin_width * self.solid_angle.to("deg**2").value)

        # Plot and add any observations.
        bin_centres = (edges[:-1] + edges[1:]) / 2
        plt.plot(bin_centres, np.log10(surface_density))

        if observations is not None:
            plt.plot(observations['x'], np.log10(observations['phi']), label='Observations')

        plt.xlabel(xlabel)
        plt.ylabel(r"$\log_{10}(\mathrm{N} \, / \, \mathrm{deg}^{-2} \, \mathrm{dex}^{-1})$")
        plt.show()