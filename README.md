This project is about plotting various standard properties of 2-dimensional tight-binding hamiltonians.

Keywords: computational condensed matter physics, Fermi surface, tight binding approximation, energy bands, cuprates, two-dimensional systems, many body physics

# Sample Run

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
# Performance

Plotting single band bare static charge susceptibility, $`\chi(q,\omega=0)`$ for a (qx,qy) grid of 32x32 takes about 3 minutes on an average laptop (with the procedural version of the code). This is largely thanks to python numba module. Otherwise the same calculation takes 30 times longer! Object oriented version is slower due to numba complications, requiring to turn off certain optimisations.

# TO-DO
- A procedural version of this code will also be provided so that users can choose between OOP vs otherwise.
- pytests will be added.
- energy band cuts
- Random Phase approximation (RPA)
- Multiband susceptibility
- Non-local interactions in RPA


# Example Visuals: Three Band Cuprate Model

![Energy bands](images/tetra/cuprate_three_band_energy_bands.png)

# Example Visuals: Single Band Cuprate Model

<p float="left">
  <img src="images/tetra/cuprate_single_band_energy_bands.png" width="400" />
  <img src="images/tetra/cuprate_single_band_filling_vs_fermi_level.png" width="400" />
  <br>
  <img src="images/tetra/cuprate_single_band_fermi_surface.png" width="400" />
  <img src="images/tetra/cuprate_single_band_susceptibility.png" width="400" />
</p>


# Example Visuals: Single Band Hexa System

![Fermi "surface"](images/hexa/hexa_single_band_fermi_surface.png)

