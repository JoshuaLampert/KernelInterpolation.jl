module KernelInterpolation

using ForwardDiff: ForwardDiff
using LinearAlgebra: norm, Symmetric, tr
using RecipesBase: RecipesBase, @recipe, @series
using SpecialFunctions: besselk, loggamma
using StaticArrays: StaticArrays, MVector
using TypedPolynomials: Variable, monomials, degree
using WriteVTK: vtk_grid, MeshCell, VTKCellTypes

include("kernels/kernels.jl")
include("nodes.jl")
include("differential_operators.jl")
include("pdes.jl")
include("interpolation.jl")
include("visualization.jl")
include("io.jl")
include("util.jl")

export get_name
export GaussKernel, MultiquadricKernel, InverseMultiquadricKernel,
       PolyharmonicSplineKernel, ThinPlateSplineKernel, WendlandKernel,
       RadialCharacteristicKernel, MaternKernel, Matern12Kernel, Matern32Kernel,
       Matern52Kernel, Matern72Kernel, RieszKernel,
       TransformationKernel, ProductKernel, SumKernel
export phi, Phi, order
export Laplacian
export PoissonEquation
export NodeSet, separation_distance, dim, eachdim, values_along_dim, random_hypercube,
       random_hypercube_boundary, homogeneous_hypercube, homogeneous_hypercube_boundary,
       random_hypersphere, random_hypersphere_boundary
export interpolation_kernel, nodeset, coefficients, kernel_coefficients,
       polynomial_coefficients, polynomial_basis, polyvars, system_matrix,
       interpolate, solve, kernel_inner_product, kernel_norm
export vtk_save
export examples_dir, get_examples, default_example, include_example

end
