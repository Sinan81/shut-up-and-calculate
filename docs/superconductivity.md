# Superconductivity

In the presence of an effective attractive interaction, electrons below Fermi level can pair up with the holes above with opposite spin, resulting in bosonic quasiparticles called Cooper pairs. 
Upon lowering the temperature, these bosons can condense into a single wave function, resulting in a gap around Fermi level, and superconductivity with zero electricial resistence. 
The Cooper paring of electrons and holes can assume certain symmetries and the associated form factors for the so-called gap function. 

As an example, below is a tight-binding Hamiltonian with a d-wave pairing:

$$ \mathbf{H} = \left [
\begin{matrix}
E(\mathbf{k}) - \mu & \Delta (\mathbf{k}) \\
\Delta(\mathbf{k}) & \mu - E(\mathbf{k})
\end{matrix}
\right ]
$$

where the basis is $\{c^\dagger_{k\uparrow}, c_{k\downarrow}\}$ corresponding to electron and hole creation operators respectively, 

$$
E(\mathbf{k}) = -2 t \left ( \cos k_x + \cos k_y \right) + 4t\prime\cos k_x \cos k_y
$$

$t$ is nearest neighbout tunneling, $t\prime$ is the next nearest neighbour tunneling, $\mu$ is chemical potential determining filling, and $\Delta(\mathbf{k}) = \frac{1}{2} \Delta_0 \left ( \cos k_x - \cos k_y \right)$ is the pairing, or gap, function with a d-wave form factor.

Unlike an anisotropic s-wave gap function (such as where the gap function is simply a constant), d-wave gap is maximum in certain parts of k-space while it's zero in others. 

Below is the plot of the band structure for this simple mean field SC Hamiltonian:

![](/images/tetra/sc_dwave.png)

This Hamiltonian can be understood in terms of an electron band coupled to a replica hole band (which's the same band mirrored with respect to Fermi level).

# References
- [Spin and Current Correlation Functions in the d-density Wave State of the Cuprates, Tewari et al 2001](https://arxiv.org/abs/cond-mat/0101027)
