import types
import dill
import warnings

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
from unyt import yr, Myr, Msun, Gyr, unyt_quantity, Mpc, sr, unyt_array
import matplotlib.pyplot as plt

from synthesizer.parametric import Stars, Galaxy

# Set up multiprocessing capability when constructing Lightcone galaxies.
def _build_galaxy(model, grid, redshift, lookback_time, final_surviving_mass):
    """Build a galaxy object with a given model.
    
    Parameters
    ----------
    model : object
        The Synthpop galaxy model to use for building the galaxy.
    grid : object
        The SPS grid to use for building the galaxy.
    redshift : float
        The redshift of the galaxy.
    lookback_time : unyt_quantity
        The lookback time corresponding to the redshift and cosmology.
    final_surviving_mass : unyt_quantity
        The surviving stellar mass of the galaxy at redshift zero.
        
    Returns
    -------
    Galaxy
        A Galaxy object with the specified properties.
    """

    # Unpack the SFH parameters.
    sfh_parameters = {}
    for key, value in model.sfh_parameters.items():
        if isinstance(value, unyt_quantity):
            sfh_parameters[key] = value
        elif isinstance(value, types.FunctionType):
            sfh_parameters[key] = value(final_surviving_mass)
        else:
            raise ValueError(f"Unsupported type for sfh parameter '{key}': {type(value)}")

    # Get functional form of the SF and metallicity distributions.
    sfh_model = model.sfh_function(**sfh_parameters) if model.sfh_function else None
    metal_dist_model = (
        model.metal_dist_function(**model.metal_dist_parameters)
        if model.metal_dist_function
        else None
    )

    # Build the Synthesizer Stars object at z=0.
    final_stars = Stars(
        grid.log10ages,
        grid.metallicities,
        sf_hist=sfh_model,
        metal_dist=metal_dist_model,
        surviving_mass=final_surviving_mass,
        grid=grid,
    )

    # Trace the Stars back to the desired lookback time.
    stars = final_stars.get_at_earlier_time(lookback_time)
    stars.surviving_mass = stars.calculate_surviving_mass(grid)

    return Galaxy(stars=stars, redshift=redshift)

# The SFH model and Grid are the same for every galaxy, so we can store
# them globally for each worker process.
_GALAXY_WORKER_MODEL = None
_GALAXY_WORKER_GRID = None

def _init_galaxy_worker(model_and_grid=None):
    """Initialise the SFH model and Grid for a worker process."""
    global _GALAXY_WORKER_MODEL, _GALAXY_WORKER_GRID
    if model_and_grid is not None:
        _GALAXY_WORKER_MODEL, _GALAXY_WORKER_GRID = dill.loads(model_and_grid)

def _build_galaxy_worker(arguments):
    """Wrapper function to build a galaxy in a worker process."""
    return _build_galaxy(_GALAXY_WORKER_MODEL, _GALAXY_WORKER_GRID, *arguments)

class Lightcone:
    """A lightcone of galaxies generated using a Synthpop model."""

    def __init__(self, model=None, minimum_stellar_mass=1e6*Msun, maximum_stellar_mass=1e12*Msun, 
                 grid=None, merger_grid=None, cosmology=None, redshift_range=(0, 10.),
                 solid_angle=4*np.pi * sr, random_seed=42, n_jobs=1):
        """__init__ method for the Lightcone class.
        
        Parameters
        ----------
        model : object
            The Synthpop galaxy model to use.
        minimum_stellar_mass : unyt_quantity
            The minimum stellar mass of galaxies at z=0.
        maximum_stellar_mass : unyt_quantity
            The maximum stellar mass of galaxies at z=0.
        grid : object
            The Synthesizer SPS grid. Spectra are not required to build 
            the population, so this can be loaded with 
            ignore_spectra=True to reduce memory usage.
        merger_grid : object
            The Synthpop merger grid describing progenitor counts as a 
            function of z=0 mass and redshift.
        cosmology : astropy.cosmology object
            The cosmology to assume.
        redshift_range : tuple
            The lightcone will be populated over this redshift range.
        solid_angle : unyt_quantity
            The solid angle of the lightcone.
        random_seed : int
            The random seed to use.
        n_jobs : int
            The number processes to use when constructing galaxies.
            Use -1 to use all cores, or 1 for no multiprocessing."""
        
        # Assign the input parameters to attributes.
        self.model = model
        self.grid = grid
        self.merger_grid = merger_grid
        self.cosmology = cosmology
        self.redshift_range = redshift_range
        self.solid_angle = solid_angle.to("deg**2")
        self.star_formation_rates = None
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.age_of_the_universe = self.cosmology.age(redshift_range[0]).to("yr").value * yr
        self.model.sfh_parameters['max_age'] = self.age_of_the_universe

        # Configure multiprocessing jobs and method.
        if not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
            raise ValueError("n_jobs must be a positive integer or -1")
        if n_jobs > 1:
            warnings.warn(
                "Multiprocessing is enabled. You may consider providing a grid loaded with " \
                "ignore_spectra=True to reduce memory usage."
            )

        available_start_methods = mp.get_all_start_methods()
        start_method = "spawn" if "spawn" in available_start_methods else "fork"

        # Construct redshift and compute volume within each shell.
        z_edges = np.linspace(*self.redshift_range, 501)
        shell_volume = (
            cosmology.comoving_volume(z_edges[1:])
            - cosmology.comoving_volume(z_edges[:-1])
        ).to("Mpc**3").value / (4 * np.pi)

        # Construct mass grid and sample the z=0 stellar mass function.
        logM_edges = np.linspace(
            np.log10(minimum_stellar_mass.to("Msun").value), 
            np.log10(maximum_stellar_mass.to("Msun").value), 
            501)
        logM_grid = 0.5 * (logM_edges[:-1] + logM_edges[1:])
        phiM = self.model.galaxy_stellar_mass_function.phi_logx(logM_grid)

        # Combine the grids to get a PDF and the expected galaxy count.
        dlogM = np.diff(logM_grid).mean()
        pdf = np.outer(shell_volume, phiM) * dlogM

        Nexp = (
            np.sum(pdf)
            * self.solid_angle.to("sr").value
        )
        N = self.rng.poisson(Nexp)

        # Flatten and normalise the PDF.
        pdf_flat = pdf.ravel()
        pdf_flat /= pdf_flat.sum()

        # Draw samples to get the redshifts and z=0 masses of galaxies.
        idx = self.rng.choice(pdf_flat.size, size=N, p=pdf_flat)

        iz, iM = np.unravel_index(idx, pdf.shape)

        self.final_surviving_masses = (
            10**self.rng.uniform(logM_edges[iM], logM_edges[iM + 1]) * Msun
        )

        self.redshifts = self.rng.uniform(z_edges[iz], z_edges[iz + 1])
        self.lookback_times = cosmology.lookback_time(self.redshifts).to("Myr").value * Myr
        self.N = N

        # Split z>0 galaxies into progenitors.
        if self.merger_grid is not None:
            self._split_by_progenitors()

        # Create the final galaxy objects and extract the masses.
        self._create_galaxies(n_jobs=n_jobs, start_method=start_method)

        self.surviving_masses = np.array(
            [galaxy.stars.surviving_mass.to("Msun").value for galaxy in self.galaxies]) * Msun

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

        if self.merger_grid != lightcone2.merger_grid:
            raise ValueError("Cannot add Lightcone instances with different merger grids.")

        if not np.isclose(
            self.solid_angle.to("sr").value,
            lightcone2.solid_angle.to("sr").value,
        ):
            raise ValueError("Cannot add Lightcone instances with different solid angles.")

        lightcone3 = Lightcone.__new__(Lightcone)

        # Copy over the attributes associated with these elements.
        lightcone3.model = self.model
        lightcone3.grid = self.grid
        lightcone3.merger_grid = self.merger_grid
        lightcone3.cosmology = self.cosmology
        lightcone3.redshift_range = self.redshift_range
        lightcone3.solid_angle = self.solid_angle
        lightcone3.random_seed = self.random_seed
        lightcone3.age_of_the_universe = self.age_of_the_universe

        # Add the galaxy lists and update the number of galaxies.
        lightcone3.galaxies = self.galaxies + lightcone2.galaxies
        lightcone3.N = len(lightcone3.galaxies)

        # Concatenate each of the property arrays.
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

    def save(self, filename):
        """Save the Lightcone object to a file using dill.
        
        Args:
            filename (str): The name of the file to save the Lightcone 
            object to.
        """
        with open(filename, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Load a Lightcone object from a dill file.
        
        Args:
            filename (str): The name of the file to load the galaxy 
            population from.
        """
        with open(filename, "rb") as f:
            return dill.load(f)

    def _split_by_progenitors(self):
        """Split galaxies at z>0 into progenitors based on z=0 mass."""

        # Get the number of progenitors by sampling from a Gaussian.
        N_prog = self.merger_grid.get(
            'N_mean',interpolate=True, mass=self.final_surviving_masses, redshift=self.redshifts)
        N_std = self.merger_grid.get(
            'N_std', interpolate=True, mass=self.final_surviving_masses, redshift=self.redshifts)
        
        N_prog = np.round(np.random.normal(N_prog.to("dimensionless").value, N_std.to("dimensionless").value))
        N_prog = np.where(np.isnan(N_prog), 1, N_prog)
        N_prog = np.maximum(N_prog, 1).astype(int)

        # Now get the mass fraction in the two most massive progenitors.
        frac1 = self.merger_grid.get(
            'frac1_median', interpolate=True, mass=self.final_surviving_masses, redshift=self.redshifts)
        frac2 = self.merger_grid.get(
            'frac2_median', interpolate=True, mass=self.final_surviving_masses, redshift=self.redshifts)

        # Replace missing coverage with equal distributions.
        frac1 = np.where(np.isnan(frac1), 1.0 / N_prog, frac1)
        frac2 = np.where(np.isnan(frac1) | np.isnan(frac2), 1.0 / N_prog, frac2)

        # Enforce f1 + f2 <= 1.
        frac1 = np.clip(frac1, 0.0, 1.0)
        frac2 = np.clip(frac2, 0.0, 1.0 - frac1)

        # The mass and lookback times of the original galaxies.
        z0_total_masses = self.final_surviving_masses.to("Msun").value
        lookback_times = self.lookback_times.to("Myr").value

        new_redshifts = []
        new_lookback = []
        new_masses = []

        # For each original galaxy.
        for i in range(self.N):

            # Get its final mass and the number of progenitors.
            z0_total_mass = z0_total_masses[i]
            n = int(N_prog[i])

            # No splitting required.
            if n == 1:
                components = [z0_total_mass]

            # If only two progenitors f1 + f2 = 1.
            elif n == 2:

                f1, f2 = frac1[i], frac2[i]
                norm = f1 + f2

                f1, f2 = f1 / norm, f2 / norm
                components = [f1 * z0_total_mass, f2 * z0_total_mass]

            # Assign mass to additional progenitors equally.
            else:
                f1, f2 = frac1[i], frac2[i]
                remainder_frac = max(1.0 - f1 - f2, 0.0)
                remainder_each = (remainder_frac * z0_total_mass) / (n - 2)
                components = [f1 * z0_total_mass, f2 * z0_total_mass] + [remainder_each] * (n - 2)

            new_redshifts.extend([self.redshifts[i]] * n)
            new_lookback.extend([lookback_times[i]] * n)
            new_masses.extend(components)

        # Overwrite the original arrays.
        self.redshifts = np.array(new_redshifts)
        self.lookback_times = np.array(new_lookback) * Myr
        self.final_surviving_masses = np.array(new_masses) * Msun
        self.N = len(self.redshifts)

    def _create_galaxies(self, n_jobs, start_method):
        """Create galaxy objects using the model and sampled parameters.
        
        Parameters
        ----------
        n_jobs : int
            The number of parallel jobs to use.
        start_method : str
            The method to use for starting new processes.
        """

        tasks = list(zip(self.redshifts, self.lookback_times, self.final_surviving_masses))

        # Just loop over tasks if no multiprocessing is requested.
        if n_jobs == 1 or len(tasks) < 2:
            self.galaxies = [
                _build_galaxy(self.model, self.grid, *task)
                for task in tasks
            ]
            return

        # Otherwise, configure multiprocessing.
        workers = n_jobs if n_jobs > 0 else mp.cpu_count()
        context = mp.get_context(start_method)
        chunksize = max(1, len(tasks) // (workers * 4))

        if start_method == "fork":
            global _GALAXY_WORKER_MODEL, _GALAXY_WORKER_GRID
            _GALAXY_WORKER_MODEL = self.model
            _GALAXY_WORKER_GRID = self.grid
            initializer_args = ()
        else:
            initializer_args = (dill.dumps((self.model, self.grid)),)

        # Initialize and execue the worker processes.
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_galaxy_worker,
            initargs=initializer_args,
        ) as executor:
            self.galaxies = list(
                executor.map(_build_galaxy_worker, tasks, chunksize=chunksize)
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

        plt.scatter(self.redshifts, self.final_surviving_masses.to("Msun"), alpha=0.5, c='k', s=10)

        plt.xlabel("Redshift")
        plt.ylabel("Final surviving stellar mass (Msun)")
        plt.xlim(self.redshift_range)
        plt.yscale("log")
        plt.show()

    def plot_redshift_surviving_mass(self):

        c = np.log10(self.final_surviving_masses.to("Msun").value)
        plt.scatter(self.redshifts, self.surviving_masses.to("Msun"), alpha=0.5, c=c, s=10, cmap='viridis')

        plt.colorbar(label='Final Mass (Msun)')
        plt.xlabel("Redshift")
        plt.ylabel("Current surviving mass (Msun)")
        plt.yscale("log")
        plt.show()

    def plot_mass_ratio(self):
        """ Plot the ratio between final and current stellar mass."""
        
        mass_ratio = self.final_surviving_masses.to("Msun") / self.surviving_masses.to("Msun")
        plt.scatter(self.redshifts, mass_ratio, alpha=0.5, s=10)

        plt.xlabel("Redshift")
        plt.ylabel("Mass ratio (Final / Current)")
        plt.axhline(1, color='r', linestyle='--')
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
        bin_width = np.diff(edges)

        surface_density = hist / (bin_width * self.solid_angle.to("deg**2").value)

        # Plot and add any observations.
        bin_centres = (edges[:-1] + edges[1:]) / 2
        plt.plot(bin_centres, np.log10(surface_density))

        if observations is not None:
            plt.plot(observations['x'], np.log10(observations['phi']), label='Observations')

        plt.xlabel(xlabel)
        plt.ylabel(r"$\log_{10}(\mathrm{N} \, / \, \mathrm{deg}^{-2} \, \mathrm{dex}^{-1})$")
        plt.show()