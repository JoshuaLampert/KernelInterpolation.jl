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

    @ki_testset "laplace_2d_annulus.jl" begin
        # No analytical solution available (don't compare l2 and linf norms)
        @test_include_example(joinpath(EXAMPLES_DIR, "laplace_2d_annulus.jl"),
                              pde_test=true, atol=1e-11)
    end

    @ki_testset "anisotropic_elliptic_2d_basic.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "anisotropic_elliptic_2d_basic.jl"),
                              l2=0.3680733486177618, linf=0.09329545570900866,
                              pde_test=true)
    end

    @ki_testset "heat_2d_basic.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "heat_2d_basic.jl"),
                              l2=0.8163804598948964, linf=0.0751084002569955,
                              pde_test=true)
    end

    @ki_testset "heat_2d_manufactured.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "heat_2d_manufactured.jl"),
                              l2=0.051463428227268404, linf=0.005234201701416197,
                              pde_test=true)
    end

    @ki_testset "advection_1d_basic.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "advection_1d_basic.jl"),
                              l2=0.029243246170576723, linf=0.005575735203507293,
                              pde_test=true, atol=2e-4) # stability issues
    end

    @ki_testset "advection_3d_basic.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "advection_3d_basic.jl"),
                              l2=0.05532511824096249, linf=0.004383701006257845,
                              pde_test=true, tspan=(0.0, 0.1))
    end

    @ki_testset "advection_diffusion_2d_basic.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "advection_diffusion_2d_basic.jl"),
                              l2=1.5864983808899238, linf=0.5647130804141758,
                              pde_test=true, tspan=(0.0, 0.1),
                              atol=1e-5)
    end
end

rm("out"; force = true, recursive = true)
end # module
