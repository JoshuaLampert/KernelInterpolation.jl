"""
    KernelInterpolation

**KernelInterpolation.jl** is a Julia package that implements methods for multivariate interpolation in arbitrary
dimension based on symmetric (conditionally) positive-definite kernels with a focus on radial basis functions.
It can be used for classical interpolation of scattered data, as well as for generalized (Hermite-Birkhoff)
interpolation by using a meshfree collocation approach. This can be used to solve partial differential equations
both stationary ones and time-dependent ones by using some time integration method from OrdinaryDiffEq.jl.

See also: [KernelInterpolation.jl](https://github.com/JoshuaLampert/KernelInterpolation.jl)
"""
module KernelInterpolation

using DiffEqCallbacks: PeriodicCallback, PeriodicCallbackAffect
using ForwardDiff: ForwardDiff
using LinearAlgebra: Symmetric, I, norm, tr, dot, diagind
using Printf: @sprintf
using Random: Random
using ReadVTK: VTKFile, get_points, get_point_data, get_data
using RecipesBase: RecipesBase, @recipe, @series
using SciMLBase: ODEFunction, ODEProblem, ODESolution, DiscreteCallback, u_modified!
using SimpleUnPack: @unpack
using SpecialFunctions: besselk, loggamma
using StaticArrays: StaticArrays, MVector, SVector
using Reexport: @reexport
using TimerOutputs: TimerOutputs, print_timer, reset_timer!
@reexport using TrixiBase: trixi_include
using TrixiBase: @trixi_timeit, timer
using TypedPolynomials: Variable, monomials, degree
using WriteVTK: WriteVTK, vtk_grid, paraview_collection, MeshCell, VTKCellTypes,
                CollectionFile

# Define the AbstractInterpolation already here because they are needed in basis.jl
abstract type AbstractInterpolation{Basis, Dim, RealT} end

include("kernels/kernels.jl")
include("nodes.jl")
include("basis.jl")
include("regularization.jl")
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
       PolyharmonicSplineKernel, ThinPlateSplineKernel, WendlandKernel, WuKernel,
       RadialCharacteristicKernel, MaternKernel, Matern12Kernel, Matern32Kernel,
       Matern52Kernel, Matern72Kernel, RieszKernel,
       TransformationKernel, ProductKernel, SumKernel
export StandardBasis, LagrangeBasis
export phi, Phi, order
export PartialDerivative, Gradient, Laplacian, EllipticOperator
export PoissonEquation, EllipticEquation, AdvectionEquation, HeatEquation,
       AdvectionDiffusionEquation
export SpatialDiscretization, Semidiscretization, semidiscretize
export NoRegularization, L2Regularization
export NodeSet, empty_nodeset, separation_distance, dim, eachdim, values_along_dim,
       distance_matrix, random_hypercube, random_hypercube_boundary, homogeneous_hypercube,
       homogeneous_hypercube_boundary, random_hypersphere, random_hypersphere_boundary
export interpolation_kernel, nodeset, coefficients, kernel_coefficients,
       polynomial_coefficients, polynomial_basis, polyvars, system_matrix,
       interpolate, solve_stationary, kernel_inner_product, kernel_norm,
       kernel_matrix, operator_matrix
export Interpolation, TemporalInterpolation
export AliveCallback, SaveSolutionCallback, SummaryCallback
export vtk_save, vtk_read
export examples_dir, get_examples, default_example

end
