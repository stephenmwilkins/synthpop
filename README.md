# synthpop

`synthpop` is a Python package for generating realistic mock galaxy populations and synthetic surveys. Built to work seamlessly with the accompanying `synthesizer` package, it enables the creation of physically motivated galaxy catalogues together with synthetic observations across a wide range of wavelengths and observing facilities.

The package can be used to generate self-consistent galaxy populations at a single epoch or across multiple cosmic epochs, making it suitable for studies of galaxy evolution, survey design, and observational forecasting. `synthpop` also supports the construction of mock lightcones, allowing users to create synthetic survey volumes that capture the evolving galaxy population along the observer's line of sight.

## Purpose

`synthpop` is not intended to replace detailed galaxy formation models, hydrodynamical simulations, or semi-analytic models. Instead, it provides a flexible and computationally efficient framework for populating the Universe with galaxies whose properties are drawn from observational constraints, empirical relations, or user-defined prescriptions.

This approach allows researchers to rapidly explore how different assumptions about galaxy populations translate into observable quantities, without the computational expense of running a full galaxy formation simulation.

## Applications

`synthpop` can be used for:

- Generating mock galaxy catalogues for survey planning and optimisation.
- Creating synthetic observations using `synthesizer`.
- Producing lightcones for wide-area and deep-field surveys.
- Testing source extraction, photometric redshift, and catalogue analysis pipelines.
- Forecasting the scientific performance of future observatories and surveys.
- Exploring the impact of alternative galaxy population models and empirical scaling relations.
- Quantifying selection effects, completeness, and observational biases.

## Key Features

- Self-consistent generation of galaxy populations across cosmic time.
- Support for both single-epoch and multi-epoch catalogues.
- Mock lightcone generation.
- Integration with `synthesizer` for producing realistic synthetic observations.
- Modular and extensible architecture for custom population models.
- Fast generation of large mock survey datasets.
- Designed for reproducible survey forecasting and interpretation.

Whether you are designing a new survey, testing an analysis pipeline, or investigating the observational consequences of different galaxy population models, `synthpop` provides a lightweight and flexible platform for creating realistic mock universes.