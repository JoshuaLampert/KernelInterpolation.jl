module KernelInterpolation

using Distances
using LinearAlgebra: norm, factorize
using RecipesBase
using StaticArrays

include("kernels.jl")
include("nodes.jl")
include("interpolation.jl")
include("visualization.jl")
include("util.jl")

export GaussKernel, metric, phi
export NodeSet, dim, values_along_dim, random_hypercube, homogeneous_hypercube,
       random_hypersphere, random_hypersphere_boundary
export kernel, nodeset, coefficients, kernel_matrix, interpolate
export examples_dir, get_examples, default_example, include_example

end
