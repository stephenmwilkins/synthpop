import types
import numpy as np
from unyt import unyt_array, yr, Myr, Msun, Gyr, unyt_quantity, Mpc
import matplotlib.pyplot as plt
from synthesizer.parametric import Stars, Galaxy
import pickle
import dill
import warnings
import copy

class GalaxyPopulation:
    """A single epoch galaxy population based on a Synthpop model."""

    def __init__(self, model=None, minimum_stellar_mass=1e6*Msun, maximum_stellar_mass=1e12*Msun,
                 volume=1E6*Mpc**3, grid=None, cosmology=None, redshift=0.0, random_seed=42,
                 galaxies=None, final_surviving_masses=None):
        """__init__ method for the GalaxyPopulation.
        
        Parameters
        ----------
        model : Model
            The Synthpop model to use.
        minimum_stellar_mass : unyt_quantity
            The minimum stellar mass at z=0 to consider.
        maximum_stellar_mass : unyt_quantity
            The maximum stellar mass at z=0 to consider.
        volume : unyt_quantity
            The volume from which to sample the galaxy population.
        grid : object
            The Synthesizer SPS grid. Spectra are not required to build 
            the population, so this can be loaded with 
            ignore_spectra=True to reduce memory usage.
        cosmology : astropy.cosmology object
            The cosmology to assume.
        redshift : float
            The redshift of the population.
        random_seed : int
            The random seed to use.
        galaxies : list[Galaxy]
            A list of existing Galaxies to use. Used when generating
            a population from a later population.
        final_surviving_masses : list[unyt_quantity]
            A list of z=0 surviving masses for Galaxies in galaxies.
            Used when generating a population from a later population.
        """

        # Assign the input parameters to attributes.
        self.model = copy.deepcopy(model) if model is not None else None
        self.minimum_stellar_mass = minimum_stellar_mass
        self.maximum_stellar_mass = maximum_stellar_mass
        self.volume = volume
        self.grid = grid
        self.cosmology = cosmology
        self.redshift = redshift
        self.random_seed = random_seed

        self.lookback_time = self.cosmology.lookback_time(redshift).to("Myr").value * Myr
        self.age_of_the_universe = self.cosmology.age(self.redshift).to("Myr").value * Myr
        self.final_age_of_the_universe = self.cosmology.age(0).to("Myr").value * Myr

        # If a list of galaxies is provided, it is the population.
        if galaxies is not None:
            if final_surviving_masses is None:
                raise ValueError("If galaxies is provided, so must final_surviving_masses.")
            self.galaxies = galaxies
            self.final_surviving_masses = final_surviving_masses

        # Otherwise, we need to construct it.
        else:

            # Sample the surviving masses at z=0 from the GSMF.
            self.final_surviving_masses = self.model.galaxy_stellar_mass_function.sample(
                xmin=minimum_stellar_mass, 
                xmax=maximum_stellar_mass, 
                volume=volume,
            ) 
            self.final_surviving_masses = self.final_surviving_masses.to("Msun")

            # Ensure SFH max_age is consistent with the age of the 
            # Universe at z=0.
            if self.model.sfh_parameters.get('max_age', None) is None:
                self.model.sfh_parameters['max_age'] = self.final_age_of_the_universe
            else:
                if self.model.sfh_parameters['max_age'] > self.final_age_of_the_universe:
                    self.model.sfh_parameters['max_age'] = self.final_age_of_the_universe

            # Create the galaxies.
            self._create_galaxies()

        # Get the surviving mass at the target redshift.
        self.surviving_masses = np.array(
            [galaxy.stars.surviving_mass.to("Msun").value for galaxy in self.galaxies]) * Msun
        self.N = len(self.surviving_masses)

        if self.N == 0:
            raise ValueError("No galaxies were created. Consider adjusting the mass range "
            "or volume.")

        # Compute some basic statistics.
        self.total_surviving_stellar_mass = np.sum(self.surviving_masses)

        self.total_surviving_stellar_mass_density = self.total_surviving_stellar_mass / self.volume

        self.final_surviving_mass_range = (self.final_surviving_masses.min(),
                                            self.final_surviving_masses.max()) 

        self.surviving_mass_range = (self.surviving_masses.min(),
                                        self.surviving_masses.max())

    def __add__(self, galpop2):
        """Add two GalaxyPopulation instances.
        
        Parameters
        ----------
        galpop2 : GalaxyPopulation
            Another GalaxyPopulation instance to add.

        Returns
        -------
        GalaxyPopulation
            A new GalaxyPopulation instance containing the combined galaxy 
            populations of both galaxy populations.
        """

        # Ensure the galaxy populations have the same cosmology, 
        # extent and grid.
        if self.volume != galpop2.volume:
            raise ValueError("Cannot add GalaxyPopulation instances with different volumes.")
        
        if self.cosmology != galpop2.cosmology:
            raise ValueError("Cannot add GalaxyPopulation instances with different cosmologies.")
        
        if self.redshift != galpop2.redshift:
            raise ValueError("Cannot add GalaxyPopulation instances with different redshifts.")
        
        if self.grid != galpop2.grid:
            raise ValueError("Cannot add GalaxyPopulation instances with different SPS grids.")

        # Instantiate a new GalaxyPopulation with the combined galaxies and final surviving masses.
        galpop3_galaxies = self.galaxies + galpop2.galaxies
        galpop3_final_surviving_masses = np.concatenate([
            self.final_surviving_masses.to("Msun").value,
            galpop2.final_surviving_masses.to("Msun").value,
        ]) * Msun

        minimum_stellar_mass = min(self.minimum_stellar_mass, galpop2.minimum_stellar_mass)
        maximum_stellar_mass = max(self.maximum_stellar_mass, galpop2.maximum_stellar_mass)

        return GalaxyPopulation(model=None,
                                minimum_stellar_mass=minimum_stellar_mass,
                                maximum_stellar_mass=maximum_stellar_mass,
                                volume=self.volume,
                                grid=self.grid,
                                cosmology=self.cosmology,
                                redshift=self.redshift,
                                random_seed=self.random_seed,
                                galaxies=galpop3_galaxies,
                                final_surviving_masses=galpop3_final_surviving_masses)

    def __str__(self):
        """Print basic summary of the galaxy population."""
        pstr = ""
        pstr += "-" * 10 + "\n"
        pstr += "SUMMARY OF GALAXY POPULATION" + "\n"
        pstr += f"Number of galaxies: {self.N}" + "\n"
        pstr += f"Volume: {self.volume.to('Mpc**3'):.2e}" + "\n"
        pstr += f"Redshift: {self.redshift}" + "\n"
        pstr += f"Total stellar mass density: {self.total_surviving_stellar_mass_density.to('Msun/Mpc**3'):.2e}" + "\n"
        pstr += f"Range of stellar masses: {self.surviving_mass_range[0].to('Msun').value:.2e} - {self.surviving_mass_range[1].to('Msun').value:.2e}" + "\n"
        pstr += f"Age of the universe at z={self.redshift}: {self.age_of_the_universe:.2e}" + "\n"
        pstr += "-" * 10 + "\n"
        return pstr

    def save(self, filename):
        """Save the galaxy population object to a file using dill.
        
        Dill is required instead of pickle to handle the complex objects 
        (e.g., functions) that may be present in the model and galaxy 
        properties.
        
        Args:
            filename (str): The name of the file to save the galaxy 
            population to.
        """
        with open(filename, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Load a galaxy population object from a pickle/dill file.
        
        Args:
            filename (str): The name of the file to load the galaxy 
            population from.
        """
        with open(filename, "rb") as f:
            return dill.load(f)


    def _create_galaxies(self):
        """Create galaxy objects with properties sampled from the population."""

        self.galaxies = []

        for i, final_surviving_mass in enumerate(self.final_surviving_masses):

            sfh_parameters = {}
            for key, value in self.model.sfh_parameters.items():  
                if isinstance(value, unyt_quantity):
                    sfh_parameters[key] = value
                elif isinstance(value, (list, np.ndarray)):
                    sfh_parameters[key] = value[i]
                elif isinstance(value, types.FunctionType):
                    sfh_parameters[key] = value(final_surviving_mass)
                else:                    
                    raise ValueError(f"Unsupported type for sfh parameter '{key}': {type(value)}")

            sfh_model = self.model.sfh_function(**sfh_parameters) if self.model.sfh_function else None
            
            metal_dist_model = self.model.metal_dist_function(**self.model.metal_dist_parameters) if self.model.metal_dist_function else None

            # Create the Stars object as it appears at z=0.
            final_stars = Stars(
                self.grid.log10ages,
                self.grid.metallicities,
                sf_hist=sfh_model,
                metal_dist=metal_dist_model,
                surviving_mass=final_surviving_mass,
                grid=self.grid,
            )

            # Create the new Stars object at the earlier lookback time.
            stars = final_stars.get_at_earlier_time(self.lookback_time)
            stars.surviving_mass = stars.calculate_surviving_mass(self.grid)

            self.galaxies.append(Galaxy(stars=stars, redshift=self.redshift))

    def project_to_earlier_epoch(self, redshift):
        """Project a galaxy population to an earlier epoch.

        Due to Synthesizer quirks, get_earlier_earlier_time is less 
        accurate when applied to the same Stars object multiple times.
        This should be kept in mind when using this functionality.
         
        Parameters
        ----------
        redshift : float
            The redshift to project to.

        Returns
        -------
        GalaxyPopulation
            A new galaxy population at the specified epoch.
        """

        # Calculate the age offset required to reach the target redshift.
        if redshift < self.redshift:
            raise ValueError(f"redshift {redshift} is less than the current redshift {self.redshift}.")
        age_offset = (self.cosmology.lookback_time(redshift).to("Myr").value * Myr) - self.lookback_time

        # Apply it to each Galaxy.
        new_galaxies = []
        for galaxy in self.galaxies:

            new_stars = galaxy.stars.get_at_earlier_time(age_offset)
            new_stars.surviving_mass = new_stars.calculate_surviving_mass(self.grid)
            new_galaxies.append(Galaxy(stars=new_stars, redshift=redshift))

        # Construct a new GalaxyPopulation from the new Galaxies.
        return GalaxyPopulation(
            model=self.model,
            minimum_stellar_mass=self.minimum_stellar_mass,
            maximum_stellar_mass=self.maximum_stellar_mass,
            volume=self.volume,
            grid=self.grid,
            cosmology=self.cosmology,
            redshift=redshift,
            random_seed=self.random_seed,
            galaxies=new_galaxies,
            final_surviving_masses=self.final_surviving_masses
        )

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

        sfr = []
        for galaxy in self.galaxies:
            sfr.append(galaxy.stars.calculate_average_sfr(t_range = (0, age)))
        return unyt_array(sfr).to('Msun/yr')

    def calculate_combined_sfh(self):
        """Calculate the combined star formation history of all galaxies in the population."""
        combined_sfh = np.zeros_like(self.grid.log10ages)

        for galaxy in self.galaxies:
            combined_sfh += galaxy.stars.sf_hist

        return combined_sfh



    def generate_spectra(self, emission_model):
        for galaxy in self.galaxies:
            galaxy.stars.get_spectra(emission_model)

        return

    def generate_photometry(
            self, 
            spec_id,
            filters):

        for galaxy in self.galaxies:
            galaxy.stars.spectra[spec_id].get_photo_lnu(filters)

        return

    def get_photometry(self, spec_id, filter_code):
        photometry = []
        for galaxy in self.galaxies:
            photometry.append(galaxy.stars.spectra[spec_id].photo_lnu[filter_code])

        return np.array(photometry)


    def plot_sfhs(self, N=10):
        """Plot the star formation histories of all galaxies in the population."""

        if N == 'all':
            N = self.N
        if N > self.N:
            N = self.N
        if self.N > N:
            print(f"Plotting SFHs for {N} out of {self.N} galaxies in the population.")
        if N > 100:
            print("Warning: Plotting SFHs for a large number of galaxies may result in a crowded plot.")

        # Trust that Synthesizer returns the SFH in Msun.
        for galaxy in self.galaxies[:N]:
            plt.plot(galaxy.stars.ages.to('Gyr').value, np.log10(galaxy.stars.sf_hist), 
                     alpha=0.1, c='k')

        plt.xlim((0, self.age_of_the_universe.to('Gyr').value))
        plt.xlabel('Stellar Age / Gyr')
        plt.ylabel('Stellar Mass Formed / M$_\odot$')
        plt.show()

    # def plot_sfh(self, rate=False):
    #     """Plot the combined star formation history of all galaxies in the population."""

    #     combined_sfh = self.calculate_combined_sfh()
    #     if rate:
    #         combined_sfr = combined_sfh / age_bin_widths
    #         plt.plot(self.grid.log10ages, combined_sfr, alpha=0.1, c='k')
    #     else:
    #         plt.plot(self.grid.log10ages, combined_sfh, alpha=0.1, c='k')

    #     plt.xlim((0, self.age_of_the_universe.to('yr').value))
    #     # plt.ylim(log_flux_range)
    #     # plt.xlabel(r'$\rm \lambda\ (Angstrom)$')
    #     # plt.ylabel(r'$\rm \log_{10}(F_{\lambda}/erg\ s^{-1}\ cm^{-2}\ \AA^{-1})$')
    #     plt.show()



    def plot_spectra(self, spec_id, wavelength_range=(1000, 10000), N=10):
        """Plot the spectra of all galaxies in the population for a given spectrum ID."""

        if N > self.N:
            N = self.N
        if self.N > N:
            print(f"Plotting spectra for {N} out of {self.N} galaxies in the population.")
        if N > 100:
            print("Warning: Plotting spectra for a large number of galaxies may result in a crowded plot.")


        for galaxy in self.galaxies[:N]:
            plt.plot(galaxy.stars.spectra[spec_id].lam, galaxy.stars.spectra[spec_id].lnu, alpha=0.1, c='k')

        plt.xlim(wavelength_range)
        # plt.ylim(log_flux_range)
        # plt.xlabel(r'$\rm \lambda\ (Angstrom)$')
        # plt.ylabel(r'$\rm \log_{10}(F_{\lambda}/erg\ s^{-1}\ cm^{-2}\ \AA^{-1})$')
        plt.show()



    def plot_stellar_mass_function(self, at_z0=False, bin_width=0.1, step=True):
        """Plot the stellar mass function of the galaxy population.
        
        Parameters
        ----------
        at_z0 : bool
            If True, plot the z=0 stellar masses, otherwise plot at the 
            current redshift.
        bin_width : float
            Dex bin width of the histogram bins.
        step : bool
            If True, plot the histogram as a step function, otherwise plot as a line.
        """

        # Get the appropriate masses.
        if at_z0 == False:
            mass_range = [i.to("Msun").value for i in self.surviving_mass_range]
            surving_masses = self.surviving_masses.to("Msun").value
            log_surviving_mass_bins = np.arange(*np.log10(mass_range), bin_width)
        else:
            mass_range = [i.to("Msun").value for i in self.final_surviving_mass_range]
            surving_masses = self.final_surviving_masses.to("Msun").value
            log_surviving_mass_bins = np.arange(*np.log10(mass_range), bin_width)

            # If using z=0 masses, also plot the input GSMF if available.
            if hasattr(self.model, "galaxy_stellar_mass_function"):
                
                # Evaluate LF
                phi = self.model.galaxy_stellar_mass_function.phi_logx(log_surviving_mass_bins)

                # Plot the input GSMF
                plt.plot(log_surviving_mass_bins, np.log10(phi), ls='-', c='k', alpha=0.2, lw=2)
        
        # Construct a histogram of sampled masses.
        hist, edges = np.histogram(np.log10(surving_masses), 
                                   bins=log_surviving_mass_bins)
        bin_centres = 10**((edges[:-1] + edges[1:]) / 2)

        # Convert to a GSMF and plot.
        phi_sampled = hist / (bin_width * self.volume.to("Mpc**3").value) 

        if step:
            plt.step(np.log10(bin_centres), np.log10(phi_sampled), c='k', alpha=1, lw=1, where='mid')
        else:
            plt.plot(np.log10(bin_centres), np.log10(phi_sampled), c='k', alpha=1, lw=1)

        plt.xlabel(r'$\rm \log_{10}(M_{\star}/M_{\odot})$')
        plt.ylabel(r'$\rm \log_{10}(\phi(M_{\star})/Mpc^{-3}\ dex^{-1})$')
        plt.show()

    def plot_star_formation_rate_distribution_function(self, bin_width=0.1):
       
        # Calculate SFRs
        sfrs = self.calculate_star_formation_rates()    

        sfr_range = (np.max((sfrs.min(), 0.001)), sfrs.max())    

        # Plot histogram of sampled GSMF
        
        log_sfr_bins = np.arange(*np.log10(sfr_range), bin_width)

        hist, edges = np.histogram(np.log10(sfrs), bins=log_sfr_bins)

        # Bin centres in linear space
        bin_centres = 10**((edges[:-1] + edges[1:]) / 2)

        # Convert histogram to φ(L)
        phi_sampled = hist / (bin_width * self.volume.to("Mpc**3").value) 

        # Plot histogram
        plt.plot(np.log10(bin_centres), np.log10(phi_sampled), c='k', alpha=1, lw=1)

        plt.xlabel(r'$\rm \log_{10}(SFR/M_{\odot} yr^{-1})$')
        plt.ylabel(r'$\rm \log_{10}(\phi(SFR)/Mpc^{-3}\ dex^{-1})$')
        plt.legend()
        plt.show()

    def plot_luminosity_function(self, spec_id,filter_code, bin_width=0.1):

        # Extract luminosities for the specified spectrum ID and filter code
        luminosities = self.get_photometry(spec_id, filter_code)

        # Range of luminosities
        luminosity_range = (luminosities.min(), luminosities.max())    

        # Define histogram bins in log space
        log_luminosity_bins = np.arange(*np.log10(luminosity_range), bin_width)

        # 
        hist, edges = np.histogram(np.log10(luminosities), bins=log_luminosity_bins)

        # Bin centres in linear space
        bin_centres = 10**((edges[:-1] + edges[1:]) / 2)

        # Convert histogram to φ(L)
        phi_sampled = hist / (bin_width * self.volume.to("Mpc**3").value) 

        # Plot histogram
        plt.plot(np.log10(bin_centres), np.log10(phi_sampled), c='k', alpha=1, lw=1)

        plt.xlabel(r'$\rm \log_{10}(L_{\nu}/erg\ s^{-1}\ cm^{-2}\ Hz^{-1})$')
        plt.ylabel(r'$\rm \log_{10}(\phi(L_{\nu})/Mpc^{-3}\ dex^{-1})$')
        plt.legend()
        plt.show()



    def plot_sfr_vs_stellar_mass(self):
        
        # Calculate SFRs
        sfrs = self.calculate_star_formation_rates()    

        # Extract stellar masses
        stellar_masses = self.surviving_masses.to("Msun").value

        plt.scatter(np.log10(stellar_masses), np.log10(sfrs.value), alpha=0.5, c='k', s=10)

        plt.xlabel(r'$\rm \log_{10}(M_{\star}/M_{\odot})$')
        plt.ylabel(r'$\rm \log_{10}(SFR/M_{\odot} yr^{-1})$')
        plt.legend()
        plt.show()


    def plot_ssfr_vs_stellar_mass(self):
        
        # Calculate sSFRs
        sfrs = self.calculate_star_formation_rates()    
        ssfrs = sfrs / self.surviving_masses
        ssfrs = ssfrs.to('Gyr**-1').value

        plt.scatter(np.log10(self.surviving_masses.to('Msun').value), np.log10(ssfrs), alpha=0.5, c='k', s=10)

        plt.xlabel(r'$\rm \log_{10}(M_{\star}/M_{\odot})$')
        plt.ylabel(r'$\rm \log_{10}(SSFR/Gyr^{-1})$')
        plt.legend()
        plt.show()

    
    def get_color(self, spec_id, filter_code1, filter_code2):
        """
        Get the color (magnitude difference) between two filters for a given spectrum ID.

        Parameters        ----------
        spec_id : str
            The ID of the spectrum to extract photometry from.
        filter_code1 : str
            The code of the first filter (e.g., 'FUV').
        filter_code2 : str
            The code of the second filter (e.g., 'NUV').

        Returns
        
        """


        photometry1 = self.get_photometry(spec_id, filter_code1)
        photometry2 = self.get_photometry(spec_id, filter_code2)

        color = -2.5*np.log10(photometry1/photometry2)

        return color

    def plot_color_color_diagram(self, spec_id, filter_codes):

        # Extract photometry for the specified spectrum ID and filter codes
        
        color1 = self.get_color(spec_id, filter_codes[0], filter_codes[1])
        color2 = self.get_color(spec_id, filter_codes[1], filter_codes[2])

        plt.scatter(color1, color2, alpha=0.5, c='k', s=10)

        plt.xlabel(rf'$\rm {filter_codes[0]}-{filter_codes[1]}$')
        plt.ylabel(rf'$\rm {filter_codes[1]}-{filter_codes[2]}$')
        plt.legend()
        plt.show()



    


class MultiEpochGalaxyPopulation:
    """A multi-epoch galaxy population based on a Synthpop model."""

    def __init__(self, model=None, minimum_stellar_mass=1e6*Msun, maximum_stellar_mass=1e12*Msun, 
                 volume=1E6*Mpc**3, grid=None, cosmology=None, redshifts=None, random_seed=42,
                 same_galaxies_across_epochs=True):
        """__init__ method for the MultiEpochGalaxyPopulation.
        
        Parameters
        ----------
        model : Model
            The Synthpop model to use.
        minimum_stellar_mass : unyt_quantity
            The minimum stellar mass at z=0 to consider.
        maximum_stellar_mass : unyt_quantity
            The maximum stellar mass at z=0 to consider.
        volume : unyt_quantity
            The volume from which to sample the galaxy population.
        grid : object
            The Synthesizer SPS grid. Spectra are not required to build 
            the population, so this can be loaded with 
            ignore_spectra=True to reduce memory usage.
        cosmology : astropy.cosmology object
            The cosmology to assume.
        redshifts : list[float]
            The redshifts at which to construct populations.
        random_seed : int
            The random seed to use.
        same_galaxies_across_epochs : bool
            If True, the same galaxies are projected to earlier epochs.
        """

        # Assign the input parameters to attributes.
        self.model = model
        self.volume = volume
        self.grid = grid
        self.cosmology = cosmology
        self.redshifts = np.sort(redshifts)
        self.random_seed = random_seed
        self.same_galaxies_across_epochs = same_galaxies_across_epochs

        if same_galaxies_across_epochs:

            warnings.warn("Due to Synthesizer quirks, projecting to an earlier time becomes less " \
            "accurate after the first iteration.")
            
            # Instantiate the galaxy population at the lowest redshift.
            galpop = GalaxyPopulation(
                model=self.model,
                minimum_stellar_mass=minimum_stellar_mass, 
                maximum_stellar_mass=maximum_stellar_mass, 
                volume=volume,
                grid=self.grid,
                cosmology=self.cosmology,
                redshift=self.redshifts[0],
                random_seed=self.random_seed)

            self.epochs = [galpop]

            # Project the population back to earlier epochs.
            for redshift in self.redshifts[1:]:
                epoch_population = galpop.project_to_earlier_epoch(redshift=redshift)
                self.epochs.append(epoch_population)

        # Otherwise, create a new population at each redshift.
        else:
            self.epochs = []

            for redshift in self.redshifts:

                # Instantiate the galaxy population.
                galpop = GalaxyPopulation(
                    model=self.model,
                    minimum_stellar_mass=minimum_stellar_mass, 
                    maximum_stellar_mass=maximum_stellar_mass, 
                    volume=volume,
                    grid=self.grid,
                    cosmology=self.cosmology,
                    redshift=redshift,
                    random_seed=self.random_seed)
                self.epochs.append(galpop)


    def __str__(self):
        """Print basic summary of the galaxy population."""
        pstr = ""
        pstr += "-" * 10 + "\n"
        pstr += "SUMMARY OF GALAXY POPULATION" + "\n"
        pstr += f"Volume: {self.volume.to('Mpc**3'):.2e}" + "\n"
        pstr += f"Redshifts: {self.redshifts}" + "\n"
        pstr += "-" * 10 + "\n"
        return pstr
    

    def plot_stellar_mass_function(self,  at_z0=False, bin_width=0.1, step=True):
        """Plot the stellar mass function at each epoch.
        
        Parameters
        ----------
        at_z0 : bool
            If True, plot the z=0 stellar masses, otherwise plot at each 
            epoch redshift.
        bin_width : float
            Dex width of the histogram bins.
        step : bool
            If True, plot the histogram as a step function, otherwise plot as a line.
        """

        for epoch, redshift in zip(self.epochs, self.redshifts):

            # Get the appropriate masses.
            if at_z0 == False:
                mass_range = [i.to("Msun").value for i in epoch.surviving_mass_range]
                surving_masses = epoch.surviving_masses.to("Msun").value
                log_surviving_mass_bins = np.arange(*np.log10(mass_range), bin_width)
            else:
                mass_range = [i.to("Msun").value for i in epoch.final_surviving_mass_range]
                surving_masses = epoch.final_surviving_masses.to("Msun").value
                log_surviving_mass_bins = np.arange(*np.log10(mass_range), bin_width)

            # Construct a histogram of sampled masses.
            hist, edges = np.histogram(np.log10(surving_masses), bins=log_surviving_mass_bins)
            bin_centres = 10**((edges[:-1] + edges[1:]) / 2)

            # Convert to a GSMF and plot.
            phi_sampled = hist / (bin_width * epoch.volume.to("Mpc**3").value) 
            if step:
                plt.step(np.log10(bin_centres), np.log10(phi_sampled), alpha=1, lw=1, where='mid', label=f'z={redshift}')
            else:
                plt.plot(np.log10(bin_centres), np.log10(phi_sampled), alpha=1, lw=1, label=f'z={redshift}')

        # Try to plot the input GSMF if plotting at z=0.
        if at_z0:
            if hasattr(self.model, "galaxy_stellar_mass_function"):
                phi = self.model.galaxy_stellar_mass_function.phi_logx(log_surviving_mass_bins)
                plt.plot(log_surviving_mass_bins, np.log10(phi), ls='-', c='k', alpha=0.2, lw=2)

        plt.xlabel(r'$\rm \log_{10}(M_{\star}/M_{\odot})$')
        plt.ylabel(r'$\rm \log_{10}(\phi(M_{\star})/Mpc^{-3}\ dex^{-1})$')
        plt.legend()
        plt.show()