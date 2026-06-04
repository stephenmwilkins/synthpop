from setuptools import find_packages, setup

# Setup configuration
setup(
    # --- add these lines ---
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
]
)
