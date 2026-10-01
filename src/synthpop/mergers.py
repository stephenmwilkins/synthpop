import os
import glob
import requests

import h5py
import numpy as np
from unyt import unyt_array, Msun
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
 
class MergerGrid:
    """Grid of redshift and z=0 mass dependent progenitor counts."""
 
    def __init__(self, path):
        """__init__ method for MergerGrid.
        
        Parameters
        ----------
        path : str
            Path to the HDF5 file containing the merger grid data.
        """
 
        self.grid_name = os.path.basename(path)
 
        with h5py.File(path, "r") as hf:
            self.axis_names = list(hf.attrs["axes"])
 
            # Get the axes and reassign units.
            self.axes = {}
            for name in self.axis_names:
                dset = hf["axes"][name]
                self.axes[name] = unyt_array(dset[...], dset.attrs["Units"])
 
            # Extract the data for progenitors.
            self.data = {}
            for group_name in ("progenitors", "counts", "mass_fractions"):
                if group_name not in hf:
                    continue
                for key in hf[group_name].keys():
                    dset = hf[group_name][key]
                    self.data[key] = unyt_array(dset[...], dset.attrs["Units"])
 
            # Extract the model information.
            self.model = {}
            for key, value in hf["model"].attrs.items():
                self.model[key] = value
 
        # Sort each axis ready for interpolation.
        for dim, name in enumerate(self.axis_names):
            order = np.argsort(self.axes[name].value)
            self.axes[name] = self.axes[name][order]
            for key in self.data:
                self.data[key] = np.take(self.data[key], order, axis=dim)
 
        # Cache of constructed interpolators.
        self._interpolators = {}
 
    def __str__(self):
        """Return a string representation of the MergerGrid object."""
        lines = [f"MergerGrid: {self.grid_name}", ""]
 
        lines.append("Axes:")
        for name in self.axis_names:
            axis = self.axes[name]
            lines.append(
                f"  {name}: {len(axis)} points, "
                f"[{axis.min().value:.4g}, {axis.max().value:.4g}] {axis.units}"
            )
 
        lines.append("")
        lines.append("Data:")
        for key, arr in self.data.items():
            lines.append(f"  {key}: shape {arr.shape}, units {arr.units}")
 
        lines.append("")
        lines.append("Model:")
        for key, value in self.model.items():
            if key not in ["mass_bin_edges_Msun", "n_galaxies_total"]:
                lines.append(f"  {key}: {value}")
 
        return "\n".join(lines)
 
    def _to_axis_units(self, name, value):
        """Convert a value to the units of the specified axis.
        
        Parameters
        ----------
        name : str
            Name of the axis.
        value : unyt_quantity or float
            Value to convert to the axis units.
 
        Returns
        -------
        unyt_quantity or float
            Value in the units of the specified axis.
        """
 
        axis = self.axes[name]
        if not hasattr(value, "units"):
            if not axis.units.is_dimensionless:
                raise ValueError(
                    f"'{name}' axis has units '{axis.units}', but the value "
                    f"given has none. Pass a unyt_quantity."
                )
            return value
        return value.to(axis.units)
 
    def nearest_index(self, **axis_values):
        """Get the index of the nearest point on the specified axes.
 
        Accepts either scalars or equal-length arrays for every axis --
        arrays are looked up vectorised (no Python loop), scalars return
        plain int indices as before.
        
        Parameters
        ----------
        **axis_values : dict
            Axis values to find the nearest index for. The keys should 
            match the axis names of the grid.
            
        Returns
        -------
        tuple of int, or tuple of ndarray
            Indices of the nearest point on the grid for each axis.
        """
        scalar = self._check_axis_value_shapes(axis_values)
 
        idx = []
        for name in self.axis_names:
 
            # Convert to the axis units.
            axis = self.axes[name]
            target = self._to_axis_units(name, axis_values[name])
            target_val = np.atleast_1d(target.value if hasattr(target, "value") else target)
 
            # Get the indices into the nearest grid points.
            diffs = np.abs(axis.value[:, None] - target_val[None, :])
 
            nearest = np.argmin(diffs, axis=0)
            idx.append(int(nearest[0]) if scalar else nearest)
 
        return tuple(idx)
 
    def _check_axis_value_shapes(self, axis_values):
        """Check axis_values are all scalar, or all equal-length arrays.
 
        Parameters
        ----------
        axis_values : dict
            The axis_values passed to nearest_index/get.
 
        Returns
        -------
        scalar : bool
            True if every axis value is a scalar.
        """
        ndims = [np.ndim(axis_values[name]) for name in self.axis_names]
        if len(set(ndims)) > 1:
            raise ValueError(f"Mix of scalar and array axis values is not supported.")
        scalar = all(n == 0 for n in ndims)
 
        if not scalar:
            sizes = [len(np.atleast_1d(axis_values[name])) for name in self.axis_names]
            if len(set(sizes)) > 1:
                raise ValueError(f"Axis value arrays must all have the same length.")
            
        return scalar
 
    def get(self, key, interpolate=False, **axis_values):
        """Get the value of a quantity at specified axis values.
 
        Accepts either scalars or equal-length arrays for every axis --
        arrays are looked up vectorised (no Python loop), scalars return
        a single value as before.
        
        Parameters
        ----------
        key : str
            Name of the progenitor quantity to retrieve.
        interpolate : bool, optional
            Whether to interpolate the gridpoints or use the 
            nearest grid point.
        **axis_values : dict
            Axis values to retrieve the quantity at. The keys should 
            match the axis names of the grid. Each may be a scalar or
            an array; if arrays, all must be the same length.
        
        Returns
        -------
        unyt_quantity, or unyt_array
            Value(s) of the quantity at the specified axis values.
        """
 
        # Simple nearest neighbour lookup.
        if not interpolate:
            idx = self.nearest_index(**axis_values)
 
            return self.data[key][idx]
 
        # Otherwise, interpolate the grid.
        else:
 
            # Construct the interpolator if it doesn't already exist.
            if key not in self._interpolators:
                axis_points = [self.axes[name].value for name in self.axis_names]
                self._interpolators[key] = RegularGridInterpolator(
                    axis_points, self.data[key].value,
                    bounds_error=False, fill_value=np.nan,
                )
 
            scalar = self._check_axis_value_shapes(axis_values)
 
            # Get the n-dimensional points to interpolate at.
            columns = []
            for name in self.axis_names:
                val = self._to_axis_units(name, axis_values[name])
                val = np.atleast_1d(val.value if hasattr(val, "value") else val)
                columns.append(val)
 
            points = np.column_stack(columns)
 
            # Interpolate the grid at the specified points.
            result = self._interpolators[key](points)
            if scalar:
                result = result[0]
 
            return result * self.data[key].units
 
    def plot(self, quantity='N_mean', dimensions=("mass", "redshift"),
             x_log=False, y_log=False, c_log=False, show=True):
        """Plot a 2D grid of the specified quantity.
 
        Parameters
        ----------
        quantity : str
            Name of the quantity to plot.
        dimensions : tuple
            Names of the axes to plot against.
        x_log : bool
            Whether to log-scale the x-axis.
        y_log : bool
            Whether to log-scale the y-axis.
        c_log : bool
            Whether to log-scale the colour axis.
        show : bool
            Whether to show the plot immediately.
 
        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        ax : matplotlib.axes.Axes
            The matplotlib axes object.
        """
 
        fig, ax = plt.subplots(figsize=(7.5, 5))
 
        # Get the data to plot.
        X = self.axes[dimensions[0]].value
        Y = self.axes[dimensions[1]].value
 
        # Ensure we have the axes the right way round.
        x_idx = self.axis_names.index(dimensions[0])
        y_idx = self.axis_names.index(dimensions[1])
        Z = np.transpose(self.data[quantity].value, axes=(y_idx, x_idx))
 
        # Plot the heatmap.
        norm = LogNorm() if c_log else None
        im = ax.pcolormesh(X, Y, Z, shading="nearest", cmap="viridis", norm=norm)
        fig.colorbar(im, ax=ax, label=quantity)
 
        ax.set_xlabel(f"{dimensions[0]} [{self.axes[dimensions[0]].units}]")
        ax.set_ylabel(f"{dimensions[1]} [{self.axes[dimensions[1]].units}]")
        ax.set_title(f"{quantity}")
 
        if x_log:
            ax.set_xscale("log")
        if y_log:
            ax.set_yscale("log")
 
        fig.tight_layout()
        if show:
            plt.show()
 
        return fig, ax
    
def tng_query(path, api_key, params=None):
    """Get data from the TNG API.

    Parameters
    ----------
    path : str
        The API endpoint to query.
    api_key : dict
        The API key to use for authentication.
    params : dict, optional
        Additional parameters to pass to the API.
    
    Returns
    -------
    dict or requests.Response
        The JSON response from the API, or the raw response if not JSON.
    """

    header = {"api-key": api_key}

    r = requests.get(path, params=params, headers=header)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json()
    return r

def construct_tng_merger_grid(base_path, simulation, mass_bin_edges, api_key, tree_name='SubLink', 
                              part_type=4, min_particles=20, n_per_bin=None, seed=None, 
                              grid_path=None):
    """Construct a merger grid from TNG merger trees.
    
    Assumes that data is stored in the standard TNG format.
    See https://www.tng-project.org/data/docs/scripts/.
 
    Parameters
    ----------
    base_path : str
        Path to the base TNG output directory.
    simulation : str
        Name of the simulation to use.
    mass_bin_edges : unyt_array
        Edges of the mass bins to use.
    api_key : str
        API key for accessing the TNG data.
    tree_name : str
        Name of the merger tree to use.
    part_type : str
        Type of particles to use. Default 4 for stars.
    min_particles : int
        Minimum number of particles for galaxies to be considered.
    n_per_bin : int, None
        Number of galaxies to include per mass bin. 
        If None, all galaxies are used.
    seed : int, optional
        Random seed to use.
    grid_path : str, None
        Path to save the constructed merger grid to.
        If None, use a default path.
 
    Returns
    -------
    grid_path : str
        Path to the saved merger grid.
    """
 
    import illustris_python as il
 
    # Location of the TNG API.    
    api_url = 'http://www.tng-project.org/api/'
 
    # Procees the inputs.
    if grid_path is None:
        grid_path = f'merger_grid_{tree_name}.hdf5'
 
    rng = np.random.default_rng(seed)
 
    mass_bin_edges = mass_bin_edges.to("Msun")
    n_bin = len(mass_bin_edges) - 1
 
    print(f"Getting simulation information from {api_url} ...")
 
    # Get the list of simulations.
    r = tng_query(api_url, api_key)
    names = [sim['name'] for sim in r['simulations']]

    # Find the target simulation and extract the hubble parameter.
    i = names.index(simulation)
 
    sim = tng_query(r['simulations'][i]['url'], api_key)
    h = sim['hubble']
 
    # Now get the redshift information.
    snaps = tng_query(sim['snapshots'], api_key)
    snap_info = {entry['number']: entry['redshift'] for entry in snaps}
 
    # Get the snapshot closest to z=0.
    snap_z0 = min(snap_info, key=lambda k: abs(snap_info[k] - 0.0))
 
    print(f"Using snapshot {snap_z0} for z=0 (z={snap_info[snap_z0]:.3f})")
 
    # Get the tree file paths.
    chunk_pattern = il.sublink.treePath(base_path, tree_name, "*")
    chunk_paths = sorted(glob.glob(chunk_pattern))
    if not chunk_paths:
        raise FileNotFoundError(f"No tree chunk files found matching '{chunk_pattern}'")
 
    # Collect the z = 0 galaxies and their masses.
    subfind_ids, masses = [], []
    for path in chunk_paths:
        with h5py.File(path, "r") as f:
            at_z0 = f["SnapNum"][:] == snap_z0
            if not np.any(at_z0):
                continue
            subfind_ids.append(f["SubfindID"][at_z0])
            masses.append(f["SubhaloMassType"][at_z0, part_type])
 
    subfind_ids = np.concatenate(subfind_ids)
    masses = np.concatenate(masses)
 
    # Convert masses to Msun.
    masses = (masses * 1e10 / h) * Msun
 
    # Determine which galaxies fall into each mass bin.
    sample = {}
    for b in range(n_bin):
 
        in_bin = np.where((masses >= mass_bin_edges[b]) & (masses < mass_bin_edges[b + 1]))[0]
 
        # No galaxies avilable.
        if len(in_bin) == 0:
            sample[b] = []
 
        # Add everything.
        elif n_per_bin is None or len(in_bin) <= n_per_bin:
            sample[b] = [int(subfind_ids[i]) for i in in_bin]
 
        # Randomly sample up to the chosen limit.
        else:
            chosen = rng.choice(in_bin, size=n_per_bin, replace=False)
            sample[b] = [int(subfind_ids[i]) for i in chosen]

        # Report the number of galaxies in this bin.
        lo, hi = mass_bin_edges[b], mass_bin_edges[b + 1]
        print(f"bin [{lo.to('Msun'):.2e}, {hi.to('Msun'):.2e}): {len(sample[b])} galaxies")
 
    # Construct arrays to store progenitor numbers and mass fractions.
    snaps = sorted(snap_info.keys())
    n_snap = len(snaps)
 
    N_mean = np.full((n_bin, n_snap), np.nan)
    N_median = np.full((n_bin, n_snap), np.nan)
    N_std = np.full((n_bin, n_snap), np.nan)
    n_galaxies = np.zeros((n_bin, n_snap), dtype=int)
    n_galaxies_total = np.zeros(n_bin, dtype=int)
 
    frac1_mean = np.full((n_bin, n_snap), np.nan)
    frac1_median = np.full((n_bin, n_snap), np.nan)
    frac1_std = np.full((n_bin, n_snap), np.nan)
    frac2_mean = np.full((n_bin, n_snap), np.nan)
    frac2_median = np.full((n_bin, n_snap), np.nan)
    frac2_std = np.full((n_bin, n_snap), np.nan)
 
    # We need these fields for each galaxy.
    fields = ["SubfindID", "SnapNum", "SubhaloMassType", "SubhaloLenType",
            "SubhaloID", "FirstProgenitorID", "NextProgenitorID",
            "DescendantID", "LastProgenitorID"]

    # For each mass bin.
    for b in range(n_bin):
 
        # Record the total number of galaxies.
        n_galaxies_total[b] = len(sample[b])
        if not sample[b]:
            continue
 
        N_func = []
        frac1_func = []
        frac2_func = []

        # Load each subhalo tree.
        for sid in sample[b]:
            tree = il.sublink.loadTree(base_path, snap_z0, sid, fields=fields,
                                        onlyMPB=False, treeName=tree_name)
            if tree is None:
                continue
 
            # For each snapshot, count the number of progenitors meeting
            # the particle threshold, and the mass fraction held by the
            # two most massive.
            curve = {}
            frac1_curve = {}
            frac2_curve = {}
            for s in snaps:
                at_snap = tree["SnapNum"] == s
 
                if not np.any(at_snap):
                    curve[int(s)] = None
                    frac1_curve[int(s)] = None
                    frac2_curve[int(s)] = None
                    continue
 
                n_part = tree["SubhaloLenType"][:, part_type]
                resolved = n_part >= min_particles
                mask = at_snap & resolved
                n_resolved = int(np.sum(mask))
 
                # There has to be at least one progenitor.
                curve[int(s)] = max(n_resolved, 1)
 
                masses_here = tree["SubhaloMassType"][mask, part_type]
                masses_sorted = np.sort(masses_here)[::-1]
                total_mass = masses_sorted.sum()
 
                if (total_mass <= 0) or (n_resolved == 0):
                    frac1_curve[int(s)] = None
                    frac2_curve[int(s)] = None
                    continue

                # Get the mass fraction of the two most massive 
                # progenitors.
                frac1_curve[int(s)] = float(masses_sorted[0] / total_mass)
                frac2_curve[int(s)] = float(masses_sorted[1] / total_mass) if n_resolved >= 2 else None
 
            N_func.append(curve)
            frac1_func.append(frac1_curve)
            frac2_func.append(frac2_curve)

        # Compute the mean, median, and std of each quantity 
        # at each snapshot.
        for j, s in enumerate(snaps):
            vals = [c[int(s)] for c in N_func if c[int(s)] is not None]
            n_galaxies[b, j] = len(vals)
            if vals:
                N_mean[b, j] = np.mean(vals)
                N_median[b, j] = np.median(vals)
                N_std[b, j] = np.std(vals)
 
            frac1_vals = [c[int(s)] for c in frac1_func if c[int(s)] is not None]
            if frac1_vals:
                frac1_mean[b, j] = np.mean(frac1_vals)
                frac1_median[b, j] = np.median(frac1_vals)
                frac1_std[b, j] = np.std(frac1_vals)
 
            frac2_vals = [c[int(s)] for c in frac2_func if c[int(s)] is not None]
            if frac2_vals:
                frac2_mean[b, j] = np.mean(frac2_vals)
                frac2_median[b, j] = np.median(frac2_vals)
                frac2_std[b, j] = np.std(frac2_vals)

    # Convert snapshot number to redshift.
    z = np.array([snap_info[int(s)] for s in snaps])
 
    # Get the mass bin centres.
    mass_bin_centers = np.sqrt(mass_bin_edges[:-1] * mass_bin_edges[1:])
 
    with h5py.File(grid_path, "w") as hf:
        hf.attrs["axes"] = ["mass", "redshift"]
        hf.attrs["snaps"] = snaps
        hf.attrs["date_created"] = str(np.datetime64("now"))

        # Save the axes gridpoints.
        axes_group = hf.create_group("axes")
        dset = axes_group.create_dataset("mass", data=mass_bin_centers.to("Msun").value)
        dset.attrs["Units"] = "Msun"
        dset.attrs['Description'] = "Grid axes masses."
        dset = axes_group.create_dataset("redshift", data=z)
        dset.attrs["Units"] = "dimensionless"
        dset.attrs['Description'] = "Grid axes redshifts."

        # Save the progenitor counts.
        prog_group = hf.create_group("progenitors")
        for key, value in {"N_mean": N_mean, "N_median": N_median, "N_std": N_std}.items():
            dset = prog_group.create_dataset(key, data=value)
            dset.attrs["Units"] = "dimensionless"
            dset.attrs['Description'] = f"{key.split('_')[1]} of the progenitors at each grid point."
 
        count_group = hf.create_group("counts")
        for key, value in {"n_galaxies": n_galaxies}.items():
            dset = count_group.create_dataset(key, data=value)
            dset.attrs['Description'] = "Number of galaxies in each mass bin."
            dset.attrs["Units"] = "dimensionless"
 
        # Save the mass fractions.
        frac_group = hf.create_group("mass_fractions")
        frac_descriptions = {
            "frac1_mean": "Mean fraction of total resolved progenitor mass held by the most massive progenitor.",
            "frac1_median": "Median fraction of total resolved progenitor mass held by the most massive progenitor.",
            "frac1_std": "Std of the fraction of total resolved progenitor mass held by the most massive progenitor.",
            "frac2_mean": "Mean fraction of total resolved progenitor mass held by the second most massive progenitor.",
            "frac2_median": "Median fraction of total resolved progenitor mass held by the second most massive progenitor.",
            "frac2_std": "Std of the fraction of total resolved progenitor mass held by the second most massive progenitor.",
        }
        frac_values = {
            "frac1_mean": frac1_mean, "frac1_median": frac1_median, "frac1_std": frac1_std,
            "frac2_mean": frac2_mean, "frac2_median": frac2_median, "frac2_std": frac2_std,
        }
        for key, value in frac_values.items():
            dset = frac_group.create_dataset(key, data=value)
            dset.attrs["Units"] = "dimensionless"
            dset.attrs["Description"] = frac_descriptions[key]

        # Store information about the grid construction.
        model_group = hf.create_group("model")
        model_group.attrs["simulation"] = simulation
        model_group.attrs["tree_name"] = tree_name
        model_group.attrs["part_type"] = part_type
        model_group.attrs["min_particles"] = min_particles
        model_group.attrs["mass_bin_edges_Msun"] = mass_bin_edges.to("Msun").value
        model_group.attrs["n_per_bin"] = n_per_bin if n_per_bin is not None else -1
        model_group.attrs["n_galaxies_total"] = n_galaxies_total
 
    return grid_path