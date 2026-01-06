# Random Phase Approximation (RPA)

Within RPA, interacting $\chi$ is calculated via infinite Dyson summation of a subset of Feynman diagrams:
```math
\begin{split}
\chi & = \chi_0 + \chi_0 U \chi_0 + \chi_0 U \chi_0 U \chi_0 + ... \\
     & = \chi_0 + \chi_0 U \chi
\end{split}
```
where $\chi_0$ is the bare susceptibility, $U$ is the interaction.
Solving for $\chi$ gives:
```math
\chi= \frac{\chi_0}{1-U\chi_0}
```
which is the RPA approximation, a so-called weak coupling approach. A phase transtion occurs when $\chi$ diverges.
Hence, one can calculate phase boundaries for a given parameter space.
A diverging $\chi$, that is when $\chi \rightarrow \infty$, implies that the denominator $1-U\chi_0$ goes to zero.
Solving $1-U(\mathbf q) \chi_0(\mathbf q) = 0$ yields critical $U$, or $U_c$, marking a phase boundary.

# Direct Interaction

Density-density interactions, or direct interaction, can simply be a constant or it can be dependent on external momentum $\mathbf q$ in the presence of non-local interactions.
For example, for a single band system, the Forier transformed interactions are:
```math
U + V ( \cos q_x + \cos q_y ) + 2 V' \cos q_x \cos q_y
```
where $U$, $V$, and $V'$ are the local, nearest neighbour, and next-nearest neighbour interaction terms respectively.
Since $U$ term is simply a constant, it makes $\chi$ diverge where $\chi(q)$ is the maximum first, which occurs around $(\pi,\pi)$.
On the other hand cosine terms in interaction function have tendency to emphesize $\chi$ values near $q=0$.

# Calculating RPA

```python
from tba import *
x = System()
x.U=1.65
x.V=0
x.Vnn=0 # Vprime
x.chis.calc_rpa_vs_q(Nq=10, plot_zone='Q1', recalc=True, shiftPlot=0)
x.chis.plot_vs_q(chi_type='rpa')
```
giving:

![RPA Susceptibility](/images/tetra/cuprate_single_band_susceptibility_rpa.png)

In this calculation we note that RPA susceptibility is about an order of magnitude bigger than the underlying bare susceptibility which is in the order of 1.
This is because for the selected parameter values, system is near critical: in fact $\chi$ will diverge at $U=1.74$ with all other parameters kept the same.
$\chi$ maximum and corresponding momenta, to be denote by $q^*$, can be obtained via
```python
qstar = get_max_3D(mysystem.chis.rpa)
```
For a given $\mathbf q^*$, a critical parameter value can be determined by plotting inverse RPA susceptibility, and finding zero crossing:
```python
from tba import * ; x = System()
# assume qstar is pi,pi
qstar = (np.pi, np.pi)
x.chis.rpa_get_critical_value(qstar,param='U',plot=True)
```
giving

![RPA Susceptibility](/images/rpa_find_critical_value.png)

# Generalized RPA (gRPA) with exchange terms

In addition to the usual bubble type diagrams, one can also infinitely some so called ladder diagrams originating from exchange interactions, to be labeled by $V_X$. A given exchange vertex of form $f(k_1-k_2)$ can be expressed as $g(k_1)h(k_2)$ so that $k_1$ and $k_2$ integrations are decoupled and can be performed. This decoupling leads to a function basis over which bare susceptibility, and interactions are projected, so that the infinite sum of ladder diagrams can performed. This is the generalized RPA (gRPA), of which details is as follows.

We shall explain details of gRPA using the example of a single-band tetra system. In the presence of direct interaction $U$ and near neighbour interaction $V$, decoupling of $V\left (\cos(k_{1x} -k_{2x}) + \cos(k_{1y} - k_{2y}) \right )$ using trigonometric identities leads to a function basis of
```math
     \cos(k_x), \cos(k_y), \sin(k_x), \sin(k_y)
```
. in order to account for onsite U, identity term is also added to this basis. From now on we shall call it *g-basis*. As a result interacting susceptibility can be calculated as:
```math
\chi(q) = \chi_0(q) + \sum_{ij} A_i \tilde \Gamma_{ij} A_j
```
where
```math
\tilde {\mathbf \Gamma} = [ 1 - \tilde V_\rho \tilde \chi_0 (q) ]^{-1} \tilde V_\rho
```
is the effective interaction matrix in the g-basis,
```math
\tilde V_\rho = \tilde V_X -2 \tilde V_D
```
is the interaction vertex in g-basis,
```math
\tilde V_X =
\begin{bmatrix}
V & 0 & 0 & 0 & 0 \\
0 & V & 0 & 0 & 0 \\
0 & 0 & V & 0 & 0 \\
0 & 0 & 0 & V & 0 \\
0 & 0 & 0 & 0 & U
\end{bmatrix}
```
is the ladder (exchange interaction) vertex in g-basis,
```math
\tilde V_D =
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & U(q)
\end{bmatrix}
```
is the bubble (direct interaction) vertex in g-basis, $U(q) = U + V( \cos q_x + \cos q_y )$
```math
\tilde \chi_0^{ij} (q) = \frac{1}{N_k} \sum_k g_i(k) \chi_0(k,q) g_j(k)
```
is the matrix of bare susceptibility projected to g-basis,
```math
\tilde A_i(q) = \frac{1}{N_k} \sum_k g_i(k) \chi_0(k,q)
```
is the array of bare susceptibility partially projected to g-basis.

For completeness, we shall also define the usual forms of interaction functions:
```math
\begin{align}
V(k_1-k_2)  & = &   \sum_{ij} g_i(k_1) \tilde V_X^{ij} g_j(k_2) \\
V(q)  & = &  \sum_{ij} g_i \tilde V_D^{ij}(q) g_j \\
\Gamma(k_1,k_2,q) & = & \sum_{ij} g_i(k_1) \tilde \Gamma_{ij}(q) g_j(k_2)
\end{align}
```


Infinite sum of ladder/exchange diagrams as well as bubbles has been shown to be important in charge order or superconductivity etc calculations.

An example numerical calculation of interacting $\chi$ in gRPA is as follows:
```python
from tba import *
x = System()
x.U=0.5
x.V=0.5
x.Vnn=0 # not implemented in gbasis yet, hence set to zero.
x.chic.plot_vs_q(chi_type='grpa', Nq=8, style='topview')
```

Note that in the case of spin susceptibility, only exchange terms contribute. Hence $\tilde V_\rho$ is replaced with $\tilde V_\sigma = \tilde V_X$ 

It's instructive to compare RPA vs GRPA vs bare susceptibility along symmetry cuts:
```python
from tba import *
x = System()
x.U=0.5
x.V=0.5
x.Vnn=0 # not implemented in gbasis yet, hence set to zero.
x.chic.plot_along_sym_cuts(num=10,rpa=True,grpa=True)
```
![Compare](/images/tetra/cuprate_single_band_chi_cuts.png)

Refs:

- [Collective excitations in the normal state of Cu-O-based superconductors, Littlewood etal, 1989](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.39.12371)
- [Spatially Modulated Electronic Nematicity in the Three-Band Model of Cuprate Superconductors, Bulut et al, 2013](https://arxiv.org/abs/1305.3301)
