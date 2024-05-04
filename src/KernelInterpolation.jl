module KernelInterpolation

using DiffEqCallbacks: PeriodicCallback, PeriodicCallbackAffect
using ForwardDiff: ForwardDiff
using LinearAlgebra: Symmetric, norm, tr, muladd
using Printf: @sprintf
using ReadVTK: VTKFile, get_points, get_point_data, get_data
using RecipesBase: RecipesBase, @recipe, @series
using SciMLBase: ODEFunction, ODEProblem, ODESolution, DiscreteCallback, u_modified!
using SimpleUnPack: @unpack
using SpecialFunctions: besselk, loggamma
using StaticArrays: StaticArrays, MVector
using TypedPolynomials: Variable, monomials, degree
using WriteVTK: WriteVTK, vtk_grid, paraview_collection, MeshCell, VTKCellTypes, CollectionFile

include("kernels/kernels.jl")
include("nodes.jl")
include("differential_operators.jl")
include("equations.jl")
include("kernel_matrices.jl")
include("interpolation.jl")
include("discretization.jl")
include("callbacks_step/callbacks_step.jl")
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
export PoissonEquation, HeatEquation
export SpatialDiscretization, Semidiscretization, semidiscretize
export NodeSet, separation_distance, dim, eachdim, values_along_dim, distance_matrix,
       random_hypercube, random_hypercube_boundary, homogeneous_hypercube,
       homogeneous_hypercube_boundary, random_hypersphere, random_hypersphere_boundary
export interpolation_kernel, nodeset, coefficients, kernel_coefficients,
       polynomial_coefficients, polynomial_basis, polyvars, system_matrix,
       interpolate, solve_stationary, kernel_inner_product, kernel_norm,
       TemporalInterpolation
export SaveSolutionCallback
export vtk_save, vtk_read
export examples_dir, get_examples, default_example, include_example

end
