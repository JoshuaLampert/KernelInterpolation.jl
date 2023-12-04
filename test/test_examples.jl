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
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_1d.jl"),
                              l2=0.08077225063031548, linf=0.006631451019030288)
    end

    @ki_testset "interpolation_1d_discontinuous.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_1d_discontinuous.jl"),
                              l2=2.0447270988707347, linf=0.8668905364406964)
    end

    @ki_testset "interpolation_2d.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d.jl"),
                              l2=0.38083912498369643, linf=0.09758611050523225, atol=1e-7)
    end

    @ki_testset "interpolation_2d_sphere.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d_sphere.jl"),
                              l2=1.5830667022001008, linf=0.3644408775720256)
    end

    @ki_testset "interpolation_2d_L_shape.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d_L_shape.jl"),
                              l2=2.116432713457176, linf=0.3412754879687297)
    end

    @ki_testset "interpolation_2d_polynomials.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d_polynomials.jl"),
                              l2=0.05394435588953249, linf=0.028279879132924693)
    end

    @ki_testset "interpolation_2d_convergence.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d_convergence.jl"), ns = 5:10)
    end

    @ki_testset "interpolation_2d_condition.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d_condition.jl"), ns = 5:10)
    end
end

end # module
