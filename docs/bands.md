# Energy Bands Etc.

Building on the definitions provided in [Notation Basics](/docs/basics.md) section, a non-interating Hamiltonian is given by:

$$ \hat H_0 = \sum_{i,\sigma} \epsilon_i \hat n_{i\sigma} + \sum_{ij\sigma} t_{ij} \hat c_{i\sigma} ^\dagger \hat c_{j\sigma}^{} $$

where $\epsilon_i$ is the orbital energy, $t_{ij}$ is the tunneling matrix elements between sites $i$ and $j$, and $\sigma$ is the spin label. Spin label, and the corresponding summation will be dropped from now on, unless when it is necessary to include. In the multiorbital case, one also needs to sum over orbitals, and hence label $i$ should be replaced with $i\alpha$ where $\alpha$ is the orbital label.

In translationally invariant systems, it's convenient to work in momentum space. Accordingly, we'll first Fourier transform $\hat H_0$ starting with the orbital energy term.

```math
\sum \epsilon_i \hat n_i = \frac{1}{N} \sum_i \epsilon_i \sum_k \sum_{k'} e^{(\mathbf k - \mathbf k')\cdot \mathbf R_i} \hat c_{\mathbf k}^\dagger \hat c_{\mathbf k'}^{} = \epsilon \sum_{\bf k} \hat c_{\mathbf k}^\dagger \hat c_{\mathbf k}^{}
```
since
```math
\sum_i e^{(\mathbf k - \mathbf k')\cdot \mathbf R_i} = N \delta_{k,k'}
```

Next we transform the tunneling term:
```math
\sum_{ij} t_{ij} \hat c_i^\dagger \hat c_j^{} = \sum_{ij} t_{ij} \frac{1}{N} \sum_k \sum_{k'} e^{-i \mathbf k\cdot R_i} e^{i \mathbf k'\cdot R_j } \hat c_{\mathbf k}^\dagger \hat c_{\mathbf k'}^{}
```
Since $t_{ij}$ doesn't explicitly depend on $i$ or $j$ rather their relative difference: $t_{ij} = t(\mathbf R_j - \mathbf R_i) = t(\mathbf R)$ where we define $\mathbf R \equiv \mathbf R_j - \mathbf R_i$. Hence
```math
\sum_{ij} t_{ij} \hat c_i^\dagger \hat c_j^{} = \sum_{\mathbf R} t(\mathbf R) \frac{1}{N} \sum_k \sum_{k'} \hat c_{\mathbf k}^\dagger \hat c_{\mathbf k'}^{} e^{i\mathbf k' \cdot \mathbf R} \sum_i e^{-i(\mathbf k - \mathbf k')\cdot \mathbf R_i}
```
Since $\sum_i e^{(\mathbf k - \mathbf k')\cdot \mathbf R_i} = N \delta_{k,k'}$ and defining
```math
t(\mathbf k) \equiv \sum_{\mathbf R} t(\mathbf R) e^{i\mathbf k \cdot \mathbf R}
```

we obtain
```math
\sum_{ij} t_{ij} \hat c_i^\dagger \hat c_j^{} =  \sum_{\mathbf k} \hat c_{\mathbf k}^\dagger \hat c_{\mathbf k}^{} t(\mathbf k)
```

It's instructive to calculate $t(\mathbf k)$ for the simple case of single-band system with nearest neighbour only tunneling. Along $x$ dimension, for a given site a particle can tunnel to the left ($R = \hat x$ ) and right ($R= - \hat x$) with magnitude $-t$. Accordingly:
```math
t_x(\mathbf k) =  \sum_{\mathbf R} t(\mathbf R) e^{i\mathbf k \cdot \mathbf R} = -te^{i \mathbf k \cdot \hat x } -te^{i \mathbf k \cdot (-\hat x)} = -t(e^{ik_x} + e^{-ik_x}) = -2t\cos k_x
```
Similarly tunneling matrix along $y$ direction is $-2t\cos k_y$. As a result
```math
H_0(\mathbf k) = \epsilon - 2t\left( \cos k_x + \cos k_y\right)
```

# Plotting Energy Bands and related properties

The energy band(s) can be plotted via:
```python
from tba import *
x = CuprateSingleBand()
x.plot_bands()
```
giving

![Single band cuprate](/images/tetra/cuprate_single_band_energy_bands.png)

It's often convenient to instead look at energy bands along certain 3D cuts:
```python
from tba import *
x = CuprateThreeBand()
x.plot_bands_along_sym_cuts()
```

![](/images/tetra/cuprate_three_band_energy_band_cuts.png)

One can also do: "x.plot_bands_along_sym_cuts(withhos=True)" to plot density of states along side the energy band cuts plot.

Many properties of a solid state system is determined by the so called Fermi surface, hence one can plot this as:
```python
from tba import *
x = CuprateSingleBand()
x.plot_Fermi_surface_contour()
```
where default filling is number of orbitals minus 0.4. Other filling can be specified by passing filling option. Hence, for single band cuprate system the Fermi surface is:

![](/images/tetra/cuprate_single_band_fermi_surface.png)

By default, the Fermi surface is plotted in the extended zone with First Brilloin Zone is indicated. These can customized by passing relevant options as can been seen in the source code.

# Density of States (DoS)

DoS is simply a histogram of energy levels: number of energy levels in a given $dE$ interval. DoS appears in many physical properties of a system, and can be measured experimentally. Below we show literally a histogram plot of energy levels in single band cuprate system:
```python
from tba import *
x = CuprateSingleBand()
x.histogram_of_states()
```
![](/images/tetra/cuprate_single_band_histogram_of_states.png)

More precisely, DoS is calculated using a delta function such as of Lorenz'ian form:
```math
N(\omega,k) = \pi \delta(\omega,{\mathbf k}) = \frac{\gamma}{(\omega-E_{\mathbf k})^2 + \gamma^2}
```
where $\gamma$ is a small broadenning parameter, such as $\gamma=0.05$.
Hence DoS can be plotted as:
```python
from tba import *
x = CuprateSingleBand()
x.density_of_states()
```
![](/images/tetra/cuprate_single_band_density_of_states.png)

In multiorbital systems, sometimes it's of interest how much each orbital contributes to DoS.
This property is calculated via eigenvectors associated with energy values, i.e. eigenvalues.
For a given vector, the norm of the i'th element yields contribution from i'th orbital in the Hamiltonian matrix basis.
This property is plotted as
```python
from tba import *
x = CuprateThreeBand()
# currently orbital weight plot is very slow
x.density_of_states(orb_wgt=True,fast=False)
```
![](/images/tetra/cuprate_three_band_density_of_states_orbital_resolved.png)

# Van-Hove Singularities

A Van-Hove singularity (VHS) is basically  where DoS exhibits a spike or a kink.
A spike can be due to an energy band being flat with respect to momenta, where a lot of energy levels exist in a narrow energy interval.
This is easily visualized by plotting DoS along side energy band cuts along symmetry points:
```python
from tba import *
x = CuprateSingleBand()
x.plot_bands_along_sym_cuts(withhos=True)
```
![](/images/tetra/cuprate_single_band_energy_band_cuts_with_hos.png)
