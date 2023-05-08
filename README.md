This project is about plotting various standard properties of 2-dimensional tight-binding hamiltonians.

Keywords: computational condensed matter physics, Fermi surface, tight binding approximation, energy bands, cuprates, two-dimensional systems, many body physics

"Shut up and calculate!" Richard Feynman

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
print(list_of_models)
x = System(model=cuprate_single_band)
x.plot_bands1()
x.plot_Fermi_surface_contour()
x.filling_vs_E1()
x.plot_chi_vs_q()
```

# References
The following references are recommended for topics discussed in this repository:
* Piers Coleman, Introduction to Many Body Physics, 2015, Cambridge University Press
* Ashcroft & Mermin, Solid State Physics

# List of Topics

[Notation basics](docs/basics.md)

[Energy bands, Fermi Surface etc](docs/bands.md)

[Susceptibility](docs/susceptibility.md)

# Coming Soon
- A procedural version of this code will also be provided so that users can choose between OOP vs otherwise.
- pytests
- Random Phase approximation (RPA)
- Multiband susceptibility
- Extended RPA (with ladder diagrams and non-local interactions)


# Example Visuals

<p float='left'>
  <img src='images/tetra/cuprate_three_band_energy_bands.png', width=400>
  <img src='images/tetra/cuprate_three_band_energy_band_cuts.png', width=400>
  <br>
  <img src='images/hexa/hexa_single_band_fermi_surface.png', width=400>
  <img src="images/tetra/cuprate_single_band_susceptibility.png" width="400" />
</p>
