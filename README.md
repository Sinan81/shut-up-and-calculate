[![Pytest CI](https://github.com/Sinan81/shut-up-and-calculate/actions/workflows/pytest_ci.yml/badge.svg)](https://github.com/Sinan81/shut-up-and-calculate/actions/workflows/pytest_ci.yml)

This project is about plotting various standard properties of 2-dimensional tight-binding hamiltonians.

Keywords: computational condensed matter physics, Fermi surface, tight binding approximation, energy bands, cuprates, two-dimensional systems, many body physics

Our guiding principles are:
* "Keep it simple stupid!" (KISS) Avoid clever programming stuff, and reduce the barrier to get started and contribute!
* "Shut up and calculate!" Richard Feynman?

# Create a virtual python environment

Due to tkinter dependency for plotting, using a conda environment is preferred. Install miniconda: https://docs.conda.io/en/latest/miniconda.html Then create a conda environment like:
```
conda env create -f ./conda_environment.yml
conda env activate fermi
```

# Usage

```python
import warnings
warnings.filterwarnings('ignore')
from tba import *
x = CuprateSingleBand()
x.plot_bands()
x.plot_Fermi_surface_contour()
x.filling_vs_energy()
# calculate charge susceptibility \chi_c
x.chic.plot_vs_q()
```

# References
The following references are recommended for topics discussed in this repository:
* Piers Coleman, Introduction to Many Body Physics, 2015, Cambridge University Press
* Ashcroft & Mermin, Solid State Physics

# List of Topics

[Notation basics](docs/basics.md)

[Energy bands, Fermi Surface, Density of States](docs/bands.md)

[Susceptibility](docs/susceptibility.md)

[Random Phase Approximation (RPA)](docs/rpa.md)

Generalized Random Phase Approximation (gRPA)

Cuprate Single Band

Cuprate Three Band

Cuprate Four Band LCO

[D-density-wave (DDW)](docs/ddw.md)

Tetra Single Band Superconductor

Hexa Single Band

# To-Do
- Multiband susceptibility
- Extended RPA (with ladder diagrams and non-local interactions)
- Adaptive susceptibility plots using python adaptive or scikit-optimize
- Mean field and broken symmetry calculations, and coexisting orders
- pytests etc
- speed up integration via [low-level callback functions](https://docs.scipy.org/doc/scipy/tutorial/integrate.html#faster-integration-using-low-level-callback-functions)


# Example Visuals

<p float='left'>
  <img src='images/tetra/cuprate_three_band_energy_bands.png', width=400>
  <img src='images/tetra/cuprate_three_band_energy_band_cuts.png', width=400>
  <br>
  <img src='images/hexa/hexa_single_band_fermi_surface.png', width=400>
  <img src="images/tetra/cuprate_single_band_susceptibility.png" width="400" />
</p>
