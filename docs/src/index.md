# KernelInterpolation.jl

[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JoshuaLampert.github.io/KernelInterpolation.jl/dev/)
[![Build Status](https://github.com/JoshuaLampert/KernelInterpolation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JoshuaLampert/KernelInterpolation.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/JoshuaLampert/KernelInterpolation.jl/graph/badge.svg)](https://codecov.io/gh/JoshuaLampert/KernelInterpolation.jl)
[![Coveralls](https://coveralls.io/repos/github/JoshuaLampert/KernelInterpolation.jl/badge.svg?branch=main)](https://coveralls.io/github/JoshuaLampert/KernelInterpolation.jl?branch=main)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

[**KernelInterpolation.jl**](https://github.com/JoshuaLampert/KernelInterpolation.jl) is a [Julia](https://julialang.org/) package that
implements methods for multivariate interpolation in arbitrary dimension based on symmetric (conditionally) positive-definite kernels
with a focus on radial-basis functions. It can be used for classical interpolation of scattered data, as well as for generalized
(Hermite-Birkhoff) interpolation by using a meshfree collocation approach. This can be used to solve partial differential equations both
stationary ones and time-dependent ones by using some time integration method from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl).

## Installation

If you have not yet installed Julia, then you first need to [download Julia](https://julialang.org/downloads/). Please
[follow the instructions for your operating system](https://julialang.org/downloads/platform/). KernelInterpolation.jl
works with Julia v1.10 and newer. You can install KernelInterpolation.jl by executing the following commands from the Julia REPL

```julia
julia> using Pkg

julia> Pkg.add(url="https://github.com/JoshuaLampert/KernelInterpolation.jl")
```

For visualizing the results, additionally you need to install [Plots.jl](https://github.com/JuliaPlots/Plots.jl), which can be done by

```julia
julia> using Pkg

julia> Pkg.add("Plots")
```

To create special node sets, you might also want to install [QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl) and
for solving time-dependent partial differential equations [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) in a
similar way as above for Plots.jl. See the documentation for more examples on how to use these packages in combination with
KernelInterpolation.jl.

## Usage

In the Julia REPL, first load the package KernelInterpolation.jl

```julia
julia> using KernelInterpolation
```

In order to interpolate discrete function values of a (potentially) multivariate function $f: \mathbb{R}^d\to \mathbb{R}$, we
first need a set of nodes $X = \{x_1,\ldots,x_n\}\subset\mathbb{R}^d$, where the function values of $f$ are known. In KernelInterpolation.jl
we can, e.g., construct a homogeneous grid on a hypercube in 2 dimensions by calling

```julia
julia> nodeset = homogeneous_hypercube(5, (-2, -1), (2, 1))
```

Here, we specified that the hypercube has 5 nodes along each of the 2 dimensions (i.e. in total we have $5^2 = 25$ nodes) and that the
boundaries of the cube are given by the lower left corner located at $(-2, -1)$ and the upper right corner at $(2, 1)$. Similarly,
[`NodeSet`](@ref)s can be constructed by the functions [`random_hypercube`](@ref), [`random_hypercube_boundary`](@ref),
[`homogeneous_hypercube_boundary`](@ref), [`random_hypersphere`](@ref) or [`random_hypersphere_boundary`](@ref) or by directly passing
a set of nodes to the constructor of [`NodeSet`](@ref). Besides the `nodeset`, we need the function values at the nodes. Let's say, we
want to reconstruct the function $f(x) = \sin(x_1\cdot x_2)$. Then, we can create the vector of function values by

```julia
julia> f(x) = sin(x[1]*x[2])
julia> ff = f.(nodeset)
```

Finally, we obtain the [`Interpolation`](@ref) object by calling [`interpolate`](@ref), where we specify the kernel function that is used
for the reconstruction. Here, we take a Gaussian $\phi(r) = \exp(-(\varepsilon r)^2)$ with shape parameter $\varepsilon = 1/2$ as
radial-symmetric basis function:

```julia
julia> kernel = GaussKernel{dim(nodeset)}(shape_parameter = 0.5)
julia> itp = interpolate(nodeset, ff, kernel)
```

If the `kernel` is only conditionally positive definite, the interpolant will be augmented by a polynomial of the corresponding order of
the kernel. Another order can also be passed explicitly with the keyword argument `m` of [`interpolate`](@ref). The result `itp` is an
object that is callable on any point $x\in\mathbb{R}^d$, e.g.,

```julia
julia> itp([-1.3, 0.26])
-0.34096946394940986

julia> f([-1.3, 0.26])
-0.33160091709280176
```

For more sophisticated examples also involving solving stationary or time-dependent partial differential equations, see the
[documentation](https://joshualampert.github.io/KernelInterpolation.jl/dev/pdes).
More examples can be found in the [`examples/`](https://github.com/JoshuaLampert/KernelInterpolation.jl/tree/main/examples) subdirectory.

More examples can be found in the [`examples/`](https://github.com/JoshuaLampert/KernelInterpolation.jl/tree/main/examples) subdirectory.

### Visualization

In order to visualize the results, you need to have [Plots.jl](https://github.com/JuliaPlots/Plots.jl) installed and loaded

```julia
julia> using Plots
```

A [`NodeSet`](@ref) can simply be plotted by calling

```julia
julia> plot(nodeset)
```

An [`Interpolation`](@ref) object can be plotted by providing a [`NodeSet`](@ref) at which the interpolation is evaluated. Continuing
the example from above, we can visualize the resulting interpolant on a finer grid

```julia
julia> nodeset_fine = homogeneous_hypercube(20, 2, (-2, -1), (2, 1))
julia> plot(nodeset_fine, itp)
```

To visualize the true solution `f` in the same plot as a surface plot we can call

```julia
julia> plot!(nodeset_fine, f, st = :surface)
```

KernelInterpolation.jl also supports exporting (and importing) VTK files, which can be visualized using tools
such as [ParaView](https://www.paraview.org/) or [VisIt](https://visit-dav.github.io/visit-website/). See the documentation
for more details.

## Authors

The package is developed and maintained by Joshua Lampert (University of Hamburg).

## License and contributing

KernelInterpolation.jl is published under the MIT license (see [License](https://github.com/JoshuaLampert/KernelInterpolation.jl/blob/main/LICENSE)).
We are pleased to accept contributions from everyone, preferably in the form of a PR.
