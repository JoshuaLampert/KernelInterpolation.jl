module TestExamples

using Test
using KernelInterpolation
# To obtain reproducible results in CI
using Random
Random.seed!(1)

include("test_util.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "interpolation")

@testset "Interpolation Examples" begin
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
                              l2=0.541153763525397, linf=0.2257866545662589)
    end

    @ki_testset "interpolation_2d_exact.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d_exact.jl"),
                              l2=0.0, linf=0.0)
    end

    @ki_testset "interpolation_2d_sphere.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d_sphere.jl"),
                              l2=1.0227617845926844, linf=0.2456367616554671)
    end

    @ki_testset "interpolation_2d_L_shape.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d_L_shape.jl"),
                              l2=2.116432713457176, linf=0.3412754879687297)
    end

    @ki_testset "interpolation_2d_polynomials.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d_polynomials.jl"),
                              l2=0.05394435588953249, linf=0.028279879132924693)
    end

    @ki_testset "interpolation_2d_transformation.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d_transformation.jl"),
                              l2=0.8382891350633075, linf=0.3927098382304266)
    end

    @ki_testset "interpolation_2d_product_kernel.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d_product_kernel.jl"),
                              l2=1.6308820716679489, linf=0.4234550933525697)
    end

    @ki_testset "interpolation_2d_sum_kernel.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d_sum_kernel.jl"),
                              l2=0.684013657976209, linf=0.16498541680450884)
    end

    @ki_testset "interpolation_5d.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_5d.jl"),
                              l2=0.4308925377778874, linf=0.06402624845465965)
    end

    @ki_testset "interpolation_2d_Riesz_kernel.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d_Riesz_kernel.jl"),
                              l2=0.11005379282198649, linf=0.028705574254708766)
    end

    @ki_testset "interpolation_2d_convergence.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d_convergence.jl"),
                              ns=5:10)
    end

    @ki_testset "interpolation_2d_condition.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "interpolation_2d_condition.jl"),
                              ns=5:10)
    end
end

end # module
