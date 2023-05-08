# Basic Defiitions and Formal Matters

Here, basic notation and derivations will be provided.

Many body physics and tight-binding approximation usually experessed in the language of second quantization.

* $\hat c_{i}^\dagger$ creates an electron at site $i$
* $\hat c_{i}$ annihilates an electron at site $i$
* $\hat n_{i}$ counts number of electrons at site $i$.

In many body physics, one usually works in $k$-space. Hence, we'll define Fourier transformation pairs:

$$\hat c_{i} = \frac{1}{\sqrt{N}} \sum_{\mathbf k} e^{\mathbf k\cdot \mathbf R_{i} } \hat c_{\mathbf k}$$

$$\hat c_{\mathbf k} = \frac{1}{\sqrt{N}} \sum_{i} e^{-\mathbf k\cdot \mathbf R_{i} } \hat c_{i}$$

$$\hat c_{i}^\dagger = \frac{1}{\sqrt{N}} \sum_{\mathbf k} e^{-\mathbf k\cdot \mathbf R_{i} } \hat c_{\mathbf k}^\dagger$$

$$\hat c_{\mathbf k}^\dagger = \frac{1}{\sqrt{N}} \sum_{i} e^{\mathbf k\cdot \mathbf R_{i} } \hat c_{i}^\dagger$$

where $\mathbf k = k_x \hat x + k_y \hat y$ is the momentum vector, $\mathbf R_i = x \hat x + y \hat y$ is the position vector of *i* th unit cell, and $N$ is the number of unit-cells in the system.

