module KernelInterpolation

using LinearAlgebra: norm, factorize
using RecipesBase
using StaticArrays
using TypedPolynomials: Variable, monomials, degree

include("kernels.jl")
include("nodes.jl")
include("interpolation.jl")
include("visualization.jl")
include("util.jl")

export GaussKernel, MultiquadricKernel, InverseMultiquadricKernel,
       RadialCharacteristicKernel, PolyharmonicSplineKernel, ThinPlateSplineKernel
export phi, order
export NodeSet, separation_distance, dim, values_along_dim, random_hypercube,
       homogeneous_hypercube, random_hypersphere, random_hypersphere_boundary
export kernel, nodeset, coefficients, kernel_coefficients, polynomial_coefficients,
       polynomial_basis, polyvars, system_matrix, interpolate
export examples_dir, get_examples, default_example, include_example

end
