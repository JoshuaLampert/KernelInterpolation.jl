module TestExamples

using Test
using KernelInterpolation
# To obtain reproducible results in CI
using Random
Random.seed!(1)

include("test_util.jl")

EXAMPLES_DIR = examples_dir()

@testset "Examples" begin
    @ki_testset "interpolation_1d.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_1d.jl"))
        values_test = itp.(nodeset)
        @test isapprox(norm(values .- values_test, Inf), 0; atol = 1e-12)
    end

    @ki_testset "interpolation_2d.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d.jl"))
        values_test = itp.(nodeset)
        @test isapprox(norm(values .- values_test, Inf), 0; atol = 1e-7)
    end

    @ki_testset "interpolation_2d_sphere.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d_sphere.jl"))
        values_test = itp.(nodeset)
        @test isapprox(norm(values .- values_test, Inf), 0; atol = 1e-12)
    end

    @ki_testset "interpolation_2d_L_shape.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d_L_shape.jl"))
        values_test = itp.(nodeset)
        @test isapprox(norm(values .- values_test, Inf), 0; atol = 1e-12)
    end
end

end # module
