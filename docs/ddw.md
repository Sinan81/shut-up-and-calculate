# D-Density Wave

D-density wave involves staggered loop currents with $Q=(\pi,\pi)$, and was used in 2000s as a phenomenological model for Pseudogap.

![](/images/tetra/ddw_cartoon.png)

Correpsonding Hamiltonian matrix is given by:

$$ \mathbf{H} = \left [
\begin{matrix}
E(\mathbf{k}) & iW(\mathbf{k}) \\
-i W(\mathbf{k}) & E(\mathbf{k+Q})
\end{matrix}
\right ]
$$

where the basis is $\{c^\dagger_k, c^\dagger_{k+Q}\}$,

$$
E(\mathbf{k}) = -2 t \left ( \cos k_x + \cos k_y \right) + 4t\prime\cos k_x \cos k_y
$$

$t$ is nearest neighbout tunneling, $t\prime$ is the next nearest neighbour tunneling, and $W(\mathbf{k}) = W_0 \left ( \cos k_x - \cos k_y \right)$.

DDW opens a gap around $X=(\pi,0)$ or $Y=(0,\pi)$ points as shown below.
```python
from tba import *
ddw = System(model=tetra_single_band_ddw)
ddw.plot_bands_along_sym_cuts(plot_Emin=-1,plot_Emax=2,num=200)
```

![](/images/tetra/tetra_single_band_ddw_energy_band_cuts.png)

It would be instructive, to replot this figure for symettry points of AF reduced Brilloin Zone.

Accordingly, the Fermi surface is modified by DDW in a way similar to the pseudogap phase, where the above mentioned gap extends to certain regions along AF RBZ resulting in electron and hole pockets.
![](/images/tetra/tetra_single_band_ddw_fermi_surface.png)

# References
- [Hidden Order in Cuprates, Chakravarty et al 2000](https://arxiv.org/abs/cond-mat/0005443)
- [Time-reversal symmetry breaking by a (d+id) density-wave state in underdoped cuprate superconductors, Tewari et al 2007](https://www.arxiv.org/abs/0711.2329)
- [Spin and Current Correlation Functions in the \bf d-density Wave State of the Cuprates, Tewari et al 2001](https://arxiv.org/abs/cond-mat/0101027)

