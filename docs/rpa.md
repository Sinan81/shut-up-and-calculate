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
Solving $1-U(\mathbf q) \chi_0(\mathbf q)$ yields critical $U$, or $U_c$, marking a phase boundary.

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
x.model.U=1.75
x.model.V=0
x.model.Vnn=0 # Vprime
x.calc_chi_rpa_vs_q(Nq=10, plot_zone='Q1', recalc=True, shiftPlot=0)
x.plot_chi_vs_q(chi_type='charge_rpa')
```

# TO-DO

- A through discussion of RPA theory
- generilized RPA with exchange interactions
