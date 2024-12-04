using TestItems
using TestItemRunner

@run_package_tests

@testsnippet Setup begin
    include("test_util.jl")
end

@testsnippet AdditionalImports begin
    using LinearAlgebra: norm, Symmetric, I
    using OrdinaryDiffEqRosenbrock: solve, Rodas5P
    import OrdinaryDiffEqNonlinearSolve
    using StaticArrays: MVector
    using Meshes: Meshes, Sphere, Point, PointSet, RegularSampling
end
