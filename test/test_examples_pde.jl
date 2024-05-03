module TestExamples

using Test
using KernelInterpolation
# To obtain reproducible results in CI
using Random
Random.seed!(1)

include("test_util.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "PDEs")

@testset "PDE Examples" begin
    @ki_testset "poisson_2d_basic.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "poisson_2d_basic.jl"),
                              l2=0.051892875031473, linf=0.009623271947010141,
                              pde_test=true)
    end

    @ki_testset "heat_2d_basic.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "heat_2d_basic.jl"),
                              l2=0.8163804598948964, linf=0.0751084002569955,
                              pde_test=true)
    end

    @ki_testset "heat_2d_manufactured.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "heat_2d_manufactured.jl"),
                              l2=0.051463425182303034, linf=0.005234201685725859,
                              pde_test=true)
    end
end

end # module
