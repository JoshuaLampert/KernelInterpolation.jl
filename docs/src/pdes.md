# Solving PDEs by collocation

Kernel methods are also suitable to solve partial differential equations (PDEs), which is also sometimes known as Hermite-Birkhoff
interpolation, a special case of generalized interpolation. In an abstract setting generalized interpolation deals with the following
problem: Given a Hilbert space $H$ and a set of functionals $\{\lambda_i\}_{i = 1}^N\subset H^*$ ($H^*$ being the dual space), find a function $s\in H$ such that
$\lambda_i(s) = f_i$ for $i = 1,\ldots,N$ for given function values $f_i$. [Classical interpolation](@ref classical_interpolation), discussed in the previous section,
corresponds to the case where $H$ is a reproducing kernel Hilbert space (RKHS) and $\lambda_i(s) = s(x_i)$ are point evaluations at the nodes $x_i$ for
$X = \{x_i\}_{i = 1}^N$. In the case of Hermite-Birkhoff interpolation, the functionals $\lambda_i$ are not point evaluations but differential operators, which
are applied to the function $s$ and then evaluated at the nodes $x_i$.

## Stationary PDEs

Consider the following general stationary PDE in a domain $\Omega\subset\mathbb{R}^d$:

```math
\mathcal{L}u = f,
```

where $\mathcal{L}$ is a linear differential operator of order $m$, $u$ is the unknown function and $f$ is a given function. The operator $\mathcal{L}$ can be
written as

```math
\mathcal{L}u = \sum_{|\alpha|\leq m} a_\alpha D^\alpha u,
```

where $D^\alpha = \partial_{x_1}^{\alpha_1}\cdots\partial_{x_d}^{\alpha_d}$ is a partial derivative of order $|\alpha| = \alpha_1 + \cdots + \alpha_d$.
Note that in the context of PDEs we usually use the notation $u$ for the unknown function instead of $s$ as in the general interpolation problem.
For a complete description of the PDE, we also need to specify boundary conditions on the boundary $\partial\Omega$, which can be written with a
boundary operator $\mathcal{B}$ as

```math
\mathcal{B}u = g,
```

where $g$ is a given function. As boundary operator $\mathcal{B}$ we usually consider the identity operator, which corresponds to Dirichlet boundary conditions.
Like in the case of classical interpolation, we pick a set of nodes $X_I = \{x_i\}_{i = 1}^{N_I}\subset\Omega$. Due to the additional boundary conditions, we also
pick a set of nodes $X_B = \{x_i\}_{i = N_I + 1}^N\subset\partial\Omega$. Let $N = N_I + N_B$ and $X = X_I\cup X_B$. We again formulate an ansatz function $u$
as a linear combination of basis functions. In the simplest case, we use the same linear combination, i.e.

```math
u(x) = \sum_{i = 1}^N c_iK(x, x_i),
```

where $K(x, x_i)$ are the kernel functions. This approach is also non as non-symmetric collocation or Kansa's method.

TODO:

* Explain basic setup and basics of collocation for stationary and time-dependent PDEs
* Define custom differential operators and PDEs and solve them
* Stationary (example Laplace in L shape) and time-dependent PDEs
* AD vs analytic derivatives
