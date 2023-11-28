# KernelInterpolation.jl

[![Build Status](https://github.com/JoshuaLampert/KernelInterpolation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JoshuaLampert/KernelInterpolation.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

**KernelInterpolation.jl** is a [Julia](https://julialang.org/) package that implements methods for multivariate interpolation in arbitrary dimension based on symmetric (conditionally) positive-definite kernels with a focus on radial-basis functions.

## Installation
If you have not yet installed Julia, then you first need to [download Julia](https://julialang.org/downloads/). Please [follow the instructions for your operating system](https://julialang.org/downloads/platform/). KernelInterpolation.jl works with Julia v1.8 and newer. You can install KernelInterpolation.jl by executing the following commands from the Julia REPL
```julia
julia> using Pkg

julia> Pkg.add("https://github.com/JoshuaLampert/KernelInterpolation.jl")
```
For visualizing the results, additionally you need to install [Plots.jl](https://github.com/JuliaPlots/Plots.jl), which can be done by
```julia
julia> using Pkg

julia> Pkg.add("Plots")
```

## Usage
In the Julia REPL, first load the package KernelInterpolation.jl
```julia
julia> using KernelInterpolation
```
In order to interpolate discrete function values of a (potentially) multivariate function $f: \mathbb{R}^d\to \mathbb{R}$, we first need a set of nodes $X = \\{x_1,\ldots,x_n\\}\subset\mathbb{R}^d$, where the function values of $f$ are known. In KernelInterpolation.jl we can, e.g., construct a homogeneous grid on a hypercube in 2 dimensions by calling
```julia
julia> nodeset = homogeneous_hypercube(5, 2, (-2, -1), (2, 1))
```
Here, we specified that the hypercube has 5 nodes along each of the 2 dimensions (i.e. in total we have $5^2 = 25$) nodes and the boundaries of the cube are given by the lower left corner located at $(-2, -1)$ and the upper right corner at $(2, 1)$. Similarly, `NodeSet`s can be constructed by the functions `random_hypercube`, `random_hypercube_boundary`, `homogeneous_hypercube_boundary`, `random_hypersphere` or `random_hypersphere_boundary` or by directly passing a set of nodes to the constructor of `NodeSet`. Besides the `nodeset`, we need the function values at the nodes. Let's say, we want to reconstruct the function $f(x) = \sin(x_1\cdot x_2)$. Then, we can create the vector of function values by
```julia
julia> f(x) = sin(x[1]*x[2])
julia> ff = f.(nodeset)
```
Finally, we obtain the `Interpolation` object by calling `interpolate`, where we specify the kernel function that is used for the reconstruction. Here, we take a Gaussian $\phi(r) = \exp(-(\varepsilon r)^2)$ with shape parameter $\varepsilon = 1/2$ as radial-symmetric basis function:
```julia
julia> kernel = GaussKernel{dim(nodeset)}(0.5)
julia> itp = interpolate(nodeset, ff, kernel)
```
If the `kernel` is only conditionally positive definite, the interpolant will be augmented by a polynomial of the corresponding order of the kernel. Another order can also be passed explicitly with the keyword argument `m` of `intepolate`. The result `itp` is an object that is callable on any point $x\in\mathbb{R}^d$, e.g.,
```julia
julia> itp([-1.3, 0.26])
-0.34096946394940986

julia> f([-1.3, 0.26])
-0.33160091709280176
```
More examples can be found in the [`examples/`](https://github.com/JoshuaLampert/KernelInterpolation.jl/tree/main/examples) subdirectory.

### Visualization
In order to visualize the results, you need to have [Plots.jl](https://github.com/JuliaPlots/Plots.jl) installed and loaded
```julia
julia> using Plots
```
A `NodeSet` can simply be plotted by calling
```julia
julia> plot(nodeset)
```
An `Interpolation` object can be plotted by providing a `NodeSet` at which the interpolation is evaluated. Continuing the example from above, we can visualize the resulting interpolant on a finer grid
```julia
julia> nodeset_fine = homogeneous_hypercube(20, 2, (-2, -1), (2, 1))
julia> plot(nodeset_fine, itp)
```
To visualize the true solution `f` in the same plot as a surface plot we can call
```julia
julia> plot!(nodeset_fine, f, st = :surface)
```

## Authors

The package is developed and maintained by Joshua Lampert (University of Hamburg).

## License and contributing

KernelInterpolation.jl is published under the MIT license (see [License](https://github.com/JoshuaLampert/KernelInterpolation.jl/blob/main/LICENSE)). We are pleased to accept contributions from everyone, preferably in the form of a PR.
