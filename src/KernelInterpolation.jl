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
       PolyharmonicSplineKernel, ThinPlateSplineKernel, WendlandKernel,
       RadialCharacteristicKernel
export phi, Phi, order
export NodeSet, separation_distance, dim, values_along_dim, random_hypercube,
       random_hypercube_boundary, homogeneous_hypercube, homogeneous_hypercube_boundary,
       random_hypersphere, random_hypersphere_boundary
export interpolation_kernel, nodeset, coefficients, kernel_coefficients,
       polynomial_coefficients,
       polynomial_basis, polyvars, system_matrix, interpolate
export examples_dir, get_examples, default_example, include_example

end
