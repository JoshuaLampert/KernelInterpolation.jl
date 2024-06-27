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
as a linear combination of basis functions. In the simplest case, we use the same linear combination (neglecting polynomial augmentation for simplicity), i.e.

```math
u(x) = \sum_{j = 1}^N c_iK(x, x_j),
```

where $K$ is the kernel function. This approach is also non as non-symmetric collocation or Kansa's method. By enforcing the conditions $\mathcal{L}u(x_i) = f(x_i)$
for $i = 1,\ldots,N_I$ and $\mathcal{B}u(x_i) = g(x_i)$ for $i = N_I + 1,\ldots,N$ we obtain a linear system of equations for the coefficients $c_i$, which can be
written as

```math
\begin{pmatrix}
\tilde{A}_I \\ \tilde{A}_B
\end{pmatrix}
c = \begin{pmatrix}
f_{X_I} \\ g_{X_B}
\end{pmatrix},
```

where $\tilde{A}_I\in\mathbb{R}^{N_I\times N}$ and $\tilde{A}_B\in\mathbb{R}^{N_I\times N}$ are the matrices corresponding to the conditions at the interior and boundary nodes,
respectively, i.e.

```math
(\tilde{A}_I)_{ij} = \mathcal{L}K(x_i, x_j), i = 1, \ldots, N_I, j = 1, \ldots, N \\
(\tilde{A}_B)_{ij} = \mathcal{B}K(x_i, x_j), i = 1, \ldots, N_B, j = 1, \ldots, N.
```

Since the kernel function is known and differentiable, we can compute the derivatives of $K$ analytically. Note, however, that the system matrix
$A = \begin{pmatrix} \tilde{A}_I \\ \tilde{A}_B \end{pmatrix}$ is not invertible in general because it not symmetric anymore as it was the case in the classical interpolation.
Thus, this approach is also called non-symmetric collocation.
Let us see how this can be implemented in KernelInterpolation.jl by solving the Poisson equation ``-\Delta u = f`` in an L-shaped domain. We start by defining the equation
(thus the differential operator) and the right-hand side. KernelInterpolation.jl already provides a set of predefined differential operators and equations.

```@example poisson
using KernelInterpolation

# right-hand-side of Poisson equation
f(x, equations) = 5 / 4 * pi^2 * sinpi(x[1]) * cospi(x[2] / 2)
pde = PoissonEquation(f)

# analytical solution of equation
u(x, equations) = sinpi(x[1]) * cospi(x[2] / 2)
```

Next, we define the domain and the boundary of the L-shaped domain. We use a homogeneous grid for the nodes and filter the inner and boundary nodes in two separate
[`NodeSet`](@ref)s.

```@example poisson
function create_L_shape(N)
    x_min1 = (0.0, 0.0)
    x_max1 = (1 * pi, 1.0)
    x_min2 = (1 * pi, 0.0)
    x_max2 = (2 * pi, 1.0)
    x_min3 = (0.0, 1.0)
    x_max3 = (1 * pi, 2.0)
    nodeset1 = homogeneous_hypercube(N, x_min1, x_max1)
    nodeset2 = homogeneous_hypercube(N, x_min2, x_max2)
    nodeset3 = homogeneous_hypercube(N, x_min3, x_max3)
    nodeset = merge(nodeset1, nodeset2, nodeset3)
    unique!(nodeset)
    nodeset_inner = empty_nodeset(2)
    nodeset_boundary = empty_nodeset(2)
    for x in nodeset
        if x[1] == 0.0 || x[2] == 0.0 || x[2] == 2.0 || x[1] == 2.0 * pi || (x[1] == 1.0 * pi && x[2] >= 1.0) || (x[2] == 1.0 && x[1] >= pi)
            push!(nodeset_boundary, x)
        else
            push!(nodeset_inner, x)
        end
    end
    return nodeset_inner, nodeset_boundary
end
nodeset_inner, nodeset_boundary = create_L_shape(6)
```

Finally, we define the boundary condition, the kernel, and collect all necessary information in a [`SpatialDiscretization`](@ref), which can be solved by calling the
[`solve_stationary`](@ref) function.

```@example poisson
# Dirichlet boundary condition (here taken from analytical solution)
g(x) = u(x, pde)

kernel = WendlandKernel{2}(3, shape_parameter = 0.3)
sd = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary, kernel)
itp = solve_stationary(sd)
```

The result `itp` is an [`Interpolation`](@ref) object, which can be used to evaluate the solution at arbitrary points. We can save the solution on a finer grid
to a VTK file and visualize it.

```@example poisson
many_nodes_inner, many_nodes_boundary = create_L_shape(20)
many_nodes = merge(many_nodes_inner, many_nodes_boundary)
OUT = "out"
ispath(OUT) || mkpath(OUT)
vtk_save(joinpath(OUT, "poisson_2d_L_shape"), many_nodes, itp, x -> u(x, pde);
         keys = ["numerical", "analytical"])
```

The resulting VTK file can be visualized with a tool like ParaView. After applying the filter "Warp by Scalar", setting the coloring accordingly, and changing the
"Representation" to "Point Gaussian", we obtain the following visualization:

![Poisson equation in an L shape domain](poisson_L_shape.png)

TODO:

* Explain basic setup and basics of collocation for stationary and time-dependent PDEs
* Define custom differential operators and PDEs and solve them
* Stationary (example Laplace in L shape) and time-dependent PDEs
* AD vs analytic derivatives
