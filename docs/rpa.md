# Random Phase Approximation (RPA)

Within RPA, interacting $\chi$ is calculated via infinite Dyson summation of a subset of Feynman diagrams:
```math
\chi = \chi_0 + \chi_0 U \chi_0 + \chi_0 U \chi_0 U \chi_0 + ... = \chi_0 + \chi_0 U \chi
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
x.model.U=1.65
x.model.V=0
x.model.Vnn=0 # Vprime
x.chi.calc_rpa_vs_q(Nq=10, plot_zone='Q1', recalc=True, shiftPlot=0)
x.chi.plot_vs_q(chi_type='charge_rpa')
```
giving:

![RPA Susceptibility](/images/tetra/cuprate_single_band_susceptibility_rpa.png)

In this calculation we note that RPA susceptibility is about an order of magnitude bigger than the underlying bare susceptibility which is in the order of 1.
This is because for the selected parameter values, system is near critical: in fact $\chi$ will diverge at $U=1.74$ with all other parameters kept the same.
$\chi$ maximum and corresponding momenta, to be denote by $q^*$, can be obtained via
```python
qstar = get_max_3D(mysystem.chi.rpa)
```
For a given $\mathbf q^*$, a critical parameter value can be determined by plotting inverse RPA susceptibility, and finding zero crossing:
```python
from tba import * ; x = System()
# assume qstar is pi,pi
qstar = (np.pi, np.pi)
x.chi.rpa_get_critical_value(qstar,param='U',plot=True)
```
giving

![RPA Susceptibility](/images/rpa_find_critical_value.png)

# TO-DO

- A through discussion of RPA theory
- generilized RPA with exchange interactions
