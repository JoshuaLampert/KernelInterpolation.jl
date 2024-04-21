module TestExamples

using Test
using KernelInterpolation
# To obtain reproducible results in CI
using Random
Random.seed!(1)

include("test_util.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "PDEs")

@testset "Examples" begin
    @ki_testset "poisson_2d.jl" begin
        @test_include_example(joinpath(EXAMPLES_DIR, "poisson_2d.jl"),
                              l2=0.04230093553749744, linf=0.009668612754467136,
                              pde_test=true, atol=1e-5)
    end
end

end # module
