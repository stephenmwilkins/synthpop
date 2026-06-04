import types
import numpy as np
from unyt import yr, Myr, Msun, Gyr, unyt_quantity, Mpc
import matplotlib.pyplot as plt
from synthesizer.parametric import Stars, Galaxy


class GalaxyPopulation:

    def __init__(self, 
        model=None,
        minimum_stellar_mass=1e6*Msun,
        maximum_stellar_mass=1e12*Msun,                 
        volume = 1E6*Mpc**3,
        grid=None,
        cosmology=None,
        redshift=0.0,
        random_seed=42,
        galaxies=None,
        ):
        
        self.model = model
        self.volume = volume
        self.grid = grid
        self.cosmology = cosmology
        self.redshift = redshift
        self.random_seed = random_seed

        self.age_of_the_universe = self.cosmology.age(self.redshift).to("Myr").value * Myr

        if galaxies is not None:
            self.galaxies = galaxies
            self.surviving_masses = np.array([galaxy.stars.surviving_mass.to("Msun").value for galaxy in self.galaxies]) * Msun
            self._surviving_masses = self.surviving_masses.to('Msun').value

        else:
            # Sample the surviving masses from the galaxy stellar mass function
            self._surviving_masses = self.model.galaxy_stellar_mass_function.sample(
                xmin=minimum_stellar_mass, 
                xmax=maximum_stellar_mass, 
                volume=volume,
            ) 

            # Store the surviving mass with units for later use
            self.surviving_masses = self._surviving_masses * Msun

            # Set the max_age parameter for the SFH function to the age of the universe at this redshift, if it is not already set in the model
            self.model.sfh_parameters['max_age'] = self.age_of_the_universe

            # Calculate dust attenuation if a function is provided in the model
            self.tau_v = self.model.dust_attenuation_function(self.surviving_masses) if self.model.dust_attenuation_function else None
            self._create_galaxies()

        # Calculate the number of galaxies in the population
        self.N = len(self.surviving_masses)

        # Calculate the total surviving stellar mass in the population
        self.total_surviving_stellar_mass = np.sum(self.surviving_masses)

        # Calculate the total surviving stellar mass density in the population
        self.total_surviving_stellar_mass_density = self.total_surviving_stellar_mass / self.volume

        # Calculate the range of surviving stellar masses in the population
        self.surviving_mass_range = (self._surviving_masses.min(), self._surviving_masses.max())

        


    def __str__(self):
        """Print basic summary of the galaxy population."""
        pstr = ""
        pstr += "-" * 10 + "\n"
        pstr += "SUMMARY OF GALAXY POPULATION" + "\n"
        pstr += f"Number of galaxies: {self.N}" + "\n"
        pstr += f"Volume: {self.volume.to('Mpc**3'):.2e}" + "\n"
        pstr += f"Redshift: {self.redshift}" + "\n"
        pstr += f"Total stellar mass density: {self.total_surviving_stellar_mass_density.to('Msun/Mpc**3'):.2e}" + "\n"
        pstr += f"Range of stellar masses: {self.surviving_mass_range[0]:.2e} - {self.surviving_mass_range[1]:.2e}" + "\n"
        pstr += f"Age of the universe at z={self.redshift}: {self.age_of_the_universe:.2e}" + "\n"
        pstr += "-" * 10 + "\n"
        return pstr


    def _create_galaxies(self):
        """Create galaxy objects with properties sampled from the population."""


        self.galaxies = []

        for i, surviving_mass in enumerate(self.surviving_masses):

            sfh_parameters = {}
            for key, value in self.model.sfh_parameters.items():  
                if isinstance(value, unyt_quantity):
                    sfh_parameters[key] = value
                elif isinstance(value, (list, np.ndarray)):
                    sfh_parameters[key] = value[i]
                elif isinstance(value, types.FunctionType):
                    sfh_parameters[key] = value(surviving_mass)
                else:                    
                    raise ValueError(f"Unsupported type for sfh parameter '{key}': {type(value)}")

            sfh_model = self.model.sfh_function(**sfh_parameters) if self.model.sfh_function else None
            
            metal_dist_model = self.model.metal_dist_function(**self.model.metal_dist_parameters) if self.model.metal_dist_function else None

            if self.tau_v is not None:
                tau_v = float(self.tau_v[i])
            else:
                tau_v = None

            # Create the Stars object
            stars = Stars(
                self.grid.log10ages,
                self.grid.metallicities,
                sf_hist=sfh_model,
                metal_dist=metal_dist_model,
                surviving_mass=surviving_mass,
                grid=self.grid,
                tau_v=tau_v,
            )

            self.galaxies.append(Galaxy(stars=stars, redshift=self.redshift))

    def project_to_earlier_epoch(self, redshift=None, age=None):

        if redshift is not None:
            target_age = self.cosmology.lookback_time(redshift).to("Myr").value * Myr
        elif age is not None:
            target_age = age
        else:
            raise ValueError("Must specify either redshift or age to project to.")

        print(target_age)

        new_galaxies = []

        for galaxy in self.galaxies:

            # determine the surviving mass at the target age
            target_surviving_mass = galaxy.stars.calculate_surviving_mass_at_age(target_age, self.grid) 

            # Create the new Stars object
            stars = Stars(
                self.grid.log10ages,
                self.grid.metallicities,
                sf_hist=galaxy.stars.sf_hist,
                metal_dist=galaxy.stars.metal_dist,
                surviving_mass=target_surviving_mass,
                grid=self.grid,
                age_offset=target_age,
            )

            new_galaxies.append(Galaxy(stars=stars,redshift=redshift))

        return GalaxyPopulation(
            cosmology=self.cosmology,
            grid=self.grid,
            volume=self.volume,
            redshift=redshift,
            galaxies=new_galaxies
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
        return np.array(sfr)

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


        for galaxy in self.galaxies[:N]:
            plt.plot(galaxy.stars.ages, galaxy.stars.sf_hist, alpha=0.1, c='k')

        plt.xlim((0, self.age_of_the_universe.to('yr').value))
        # plt.ylim(log_flux_range)
        # plt.xlabel(r'$\rm \lambda\ (Angstrom)$')
        # plt.ylabel(r'$\rm \log_{10}(F_{\lambda}/erg\ s^{-1}\ cm^{-2}\ \AA^{-1})$')
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



    def plot_stellar_mass_function(self, bin_width=0.1, step=True):

        if hasattr(self.model, "galaxy_stellar_mass_function"):

            # Range of masses
            log_surviving_mass_bins = np.linspace(*np.log10(self.surviving_mass_range), 200)

            # Evaluate LF
            phi = self.model.galaxy_stellar_mass_function.phi_logx(log_surviving_mass_bins)

            # Plot the input GSMF
            plt.plot(
                log_surviving_mass_bins, 
                np.log10(phi), 
                ls='-',
                c='k',
                alpha=0.2,
                lw=2,
                label=None,
                )
        
        # Plot histogram of sampled GSMF
        
        log_surviving_mass_bins = np.arange(*np.log10(self.surviving_mass_range), bin_width)

        hist, edges = np.histogram(np.log10(self._surviving_masses), bins=log_surviving_mass_bins)

        # Bin centres in linear space
        bin_centres = 10**((edges[:-1] + edges[1:]) / 2)

        # Convert histogram to φ(L)
        phi_sampled = hist / (bin_width * self.volume.to("Mpc**3").value) 

        # Plot histogram
        if step:
            plt.step(np.log10(bin_centres), np.log10(phi_sampled), c='k', alpha=1, lw=1, where='mid')
        else:
            plt.plot(np.log10(bin_centres), np.log10(phi_sampled), c='k', alpha=1, lw=1)

        plt.xlabel(r'$\rm \log_{10}(M_{\star}/M_{\odot})$')
        plt.ylabel(r'$\rm \log_{10}(\phi(M_{\star})/Mpc^{-3}\ dex^{-1})$')
        plt.legend()
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
        stellar_masses = self._surviving_masses

        plt.scatter(np.log10(stellar_masses), np.log10(sfrs), alpha=0.5, c='k', s=10)

        plt.xlabel(r'$\rm \log_{10}(M_{\star}/M_{\odot})$')
        plt.ylabel(r'$\rm \log_{10}(SFR/M_{\odot} yr^{-1})$')
        plt.legend()
        plt.show()


    def plot_ssfr_vs_stellar_mass(self):
        
        # Calculate SFRs
        sfrs = self.calculate_star_formation_rates()    
        ssfrs = sfrs / self._surviving_masses
        # Extract stellar masses
        stellar_masses = self._surviving_masses

        plt.scatter(np.log10(stellar_masses), np.log10(ssfrs), alpha=0.5, c='k', s=10)

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

    def __init__(self, 
        model=None,
        minimum_stellar_mass=1e6*Msun,
        maximum_stellar_mass=1e12*Msun,                 
        volume = 1E6*Mpc**3,
        grid=None,
        cosmology=None,
        redshifts=None,
        random_seed=42,
        same_galaxies_across_epochs=True,
        ):
        
        self.model = model
        self.volume = volume
        self.grid = grid
        self.cosmology = cosmology
        self.redshifts = redshifts
        self.random_seed = random_seed

        self.final_redshift = self.redshifts[0]

        if same_galaxies_across_epochs:
            self.age_of_the_universe = self.cosmology.age(self.redshifts[0]).to("Myr").value * Myr
            
            # Instantiate the galaxy population
            galpop = GalaxyPopulation(
                model=self.model,
                minimum_stellar_mass=minimum_stellar_mass, 
                maximum_stellar_mass=maximum_stellar_mass, 
                volume=volume,
                grid=self.grid,
                cosmology=self.cosmology,
                redshift=self.final_redshift,
                random_seed=self.random_seed)

            self.epochs = [galpop]

            for redshift in self.redshifts:
                epoch_population = galpop.project_to_earlier_epoch(redshift=redshift)
                self.epochs.append(epoch_population)

        else:
            self.epochs = []

            for redshift in self.redshifts:
                # Instantiate the galaxy population
                galpop = GalaxyPopulation(
                    model=self.model,
                    minimum_stellar_mass=minimum_stellar_mass, 
                    maximum_stellar_mass=maximum_stellar_mass, 
                    volume=volume,
                    grid=self.grid,
                    cosmology=self.cosmology,
                    redshift=self.final_redshift,
                    random_seed=self.random_seed)
                epoch_population = galpop.project_to_earlier_epoch(redshift=redshift)
                self.epochs.append(epoch_population)


    def __str__(self):
        """Print basic summary of the galaxy population."""
        pstr = ""
        pstr += "-" * 10 + "\n"
        pstr += "SUMMARY OF GALAXY POPULATION" + "\n"
        pstr += f"Volume: {self.volume.to('Mpc**3'):.2e}" + "\n"
        pstr += f"Redshifts: {self.redshifts}" + "\n"
        pstr += "-" * 10 + "\n"
        return pstr
    

    def plot_stellar_mass_function(self, bin_width=0.1):

        for epoch, redshift in zip(self.epochs, self.redshifts):

            print(epoch.N, epoch.surviving_mass_range)

            # Plot histogram of sampled GSMF
            
            log_surviving_mass_bins = np.arange(*np.log10(epoch.surviving_mass_range), bin_width)

            hist, edges = np.histogram(np.log10(epoch._surviving_masses), bins=log_surviving_mass_bins)

            # Bin centres in linear space
            bin_centres = 10**((edges[:-1] + edges[1:]) / 2)

            # Convert histogram to φ(L)
            phi_sampled = hist / (bin_width * epoch.volume.to("Mpc**3").value) 

            # Plot histogram
            plt.plot(np.log10(bin_centres), np.log10(phi_sampled), alpha=1, lw=1, label=f'z={redshift}')

        plt.xlabel(r'$\rm \log_{10}(M_{\star}/M_{\odot})$')
        plt.ylabel(r'$\rm \log_{10}(\phi(M_{\star})/Mpc^{-3}\ dex^{-1})$')
        plt.legend()
        plt.show()