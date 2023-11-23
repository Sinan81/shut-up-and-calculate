# Susceptibility


Susceptibility, $\chi$, determines how a given system responds to an external perturbations:
```math
\chi = \frac{\text{change in observable}}{\text{perturbation}}
```
A sharp divergence of $\chi$ at a given set of parameters signals a possible instability towards a phase transition. Theoretically, susceptibilities are usually calculated via a two-particle Green's function. In the following discussions, we set fundamental physical constants to 1 for convenience (such as $\hbar=1$).


# Non-interacting (Bare) Charge Susceptibility

Charge susceptibility determines how much electron density changes due to varying chemical potential:
```math
\chi_c = \frac{\delta n}{\delta\mu}
```
For a tight binding system, changing chemical potential would be equivalent to changing site energies, $\epsilon$.
For a single band system, non-interacting (bare) charge susceptibility is given by:


```math
\chi_{0}(q,\omega) = - \frac{1}{N_k}\sum_k \frac{f(\epsilon_k) - f(\epsilon_{k+q})}{\omega + \epsilon_k - \epsilon_{k+q} + i0^+}
```


# Static Bare Charge Susceptibility


In the static limit of ${\omega \rightarrow 0}$, the imaginary part goes to zero. Hence the static bare susceptibility is:


```math
\chi_{0}(q,\omega=0) = - \frac{1}{N_k}\sum_k \frac{f(\epsilon_k) - f(\epsilon_{k+q})}{\epsilon_k - \epsilon_{k+q}}
```
A significant qualitative insight into this equation can be gained as follows:


1. At low $T$, the Fermi function $f(\epsilon)$ is like a step or heaviside function. Accordingly, the expression $f(\epsilon)-f(\epsilon')$ is non zero only if either $\epsilon$ is above Fermi level while the other is below.
2. The expression $1/(\epsilon-\epsilon')$ will be the maximum when both energies are very close, so that denominator is almost zero.
3. Putting the above two items together, we conclude that the high intensity points in susceptibility will originate from energies near Fermi energy.
4. Furthermore, if there are somewhat parallel regions on the Fermi surface, also known as nesting, the intensity will be even higher.


Bare susceptibility can be calculated as:
```python
x = System()
x.chi.plot_chi_vs_q(Nq=10,style='topview') # plot res Nq by Nq
# surface plot option is also available
```
<p float='left'>
 <img src='/images/tetra/cuprate_single_band_susceptibility_fill040_64x64.png'/>
</p>


which takes about 30 minutes on a laptop with 16 CPUs (the purely procedural version takes only 3 minutes).
In fact, due to symmetry considerations it's sufficient to plot $\chi_{0}$ only for a fraction of the First Brillouin Zone: 1/8th of the entire zone for a single band cuprate system.
Note that in the presence of non-local (that's $q$ dependent) interactions, symmetry considerations might need to be revisited in calculating interacting susceptibility.
Often, it's convenient to plot $\chi_{0}$ along symmetry cuts only:


```python
x  = System()
# 20 data points per cut (80 points in total)
# hence 10x faster than a full calculation.
x.chi.plot_chi_along_sym_cuts(num=20)
```


giving


<p float='left'>
 <img src='/images/tetra/cuprate_single_band_susceptibility_cuts.png' />
</p>

As shown above, $\chi_0$ peak occurs along a diagonal cut from $X\rightarrow M$ or ($Y\rightarrow M$). This is a result of (i) nesting and (ii) the fact that electron density or filling is below the so-called van-Hove filling $p_{vH}$ where high-intensity arcs cross along $X\rightarrow M$ or equivalent cuts.

# Origin of High-Intensity Arcs in Bare Susceptibility

The properties of $\chi_0$ integrand is discussed above.
In connection with these, the high-intensity points in $\chi_0(q)$ originate from the opposite parts of the Fermi surface, also called **nesting**.
This amounts to a high-intensity curvature that's in the same shape as the Fermi surface that's 2-times enlarged.
When different pieces of high-intesity curvature in the extended zone are folded back to the first brilloine zone, the qualitative structure of $\chi_0(q)$ is obtained along with high-intensity peaks where the arcs are crossing.
These qualitative considerations are depicted in the below animation (refresh page or click to replay).

<p float='left'>
 <img src='/images/tetra/origin_of_susceptibility_arcs.gif' width="400" />
 <img src='/images/tetra/origin_of_susceptibility_arcs_final.png' width="400" />
</p>

# Bare Current Susceptibility

Current operator is defined as:
```math
\hat J_{ij} = - \hat i t_{ij}\left (  \hat c_{i}^\dagger \hat c_{j}^{} - \hat c_{j}^\dagger \hat c_{i}^{}  \right )
```
which is associated with a given bond $ij$ by definition.
Unlike density or spin operators that are local in nature, Fourier transormation of non-local bond terms where a particle is destroyed and created on nearest neighbour site pairs introduces so called phase factors along side $\hat c_{\mathbf k}^\dagger \hat c_{\mathbf k'}^{}$ terms.
As a result, the current susceptibility involves some momentum dependent factors on top of the usual charge susceptbility integrand:
```math
\Lambda_{0,\alpha\beta}(q,\omega) = - \frac{1}{N_k}\sum_k h_{\alpha\beta}(\mathbf k, \mathbf q)\frac{f(\epsilon_k) - f(\epsilon_{k+q})}{\omega + \epsilon_k - \epsilon_{k+q} + i0^+}
```
where $h$ is the extra factor due to fourier transformation of bond terms, and $\alpha\beta \in \{xx,yy,xy,yx}$ are the possible directions along which current can flow.
For example, for a single band cuprate system:
```math
\begin{align}
h_{xx}(\mathbf k, \mathbf q) &= 4 t^2 {\sin(k_x + q_x/2)}^2 \\
h_{yy}(\mathbf k, \mathbf q) &= 4 t^2 {\sin(k_y + q_y/2)}^2 \\
h_{xy}(\mathbf k, \mathbf q) &= 4 t_{nn}^2 {\sin(k_x + k_y + q_x/2 + q_y/2)}^2
\end{align}
```
where $t_{nn}$ is the next nearest neighbour hopping term.
Here we note that this agrees with the litereture, for example see the discussion in Scalettar et al 1993.
Static current susceptibility is calculated as
```python
from tba import *
x  = System()
x.chi.calc_chi_vs_q(sus_type='current',plot_zone='Q1', shiftPlot=0,Nq=4,recalc=True)
```
Detailed discussions and derivations will be provided at a later time.

# References

1. [Scalapino etal 1993, Phys. Rev. B 47, 7995, "Insulator, metal, or superconductor: The criteria"](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.47.7995)
