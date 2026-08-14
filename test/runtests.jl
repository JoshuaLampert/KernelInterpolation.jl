using TestItems
using TestItemRunner

@run_package_tests

@testsnippet Setup begin
    include("test_util.jl")
end

@testsnippet AdditionalImports begin
    using ForwardDiff: ForwardDiff
    using LinearAlgebra: LinearAlgebra, norm, dot, cholesky, lu, qr, Symmetric, Cholesky, I
    using LinearSolve: LUFactorization, QRFactorization, KrylovJL_GMRES
    using OrdinaryDiffEqRosenbrock: solve, Rodas5P
    import OrdinaryDiffEqNonlinearSolve
    using StaticArrays: SVector, MVector
    using Meshes: Meshes, Sphere, Point, PointSet, RegularSampling
end
