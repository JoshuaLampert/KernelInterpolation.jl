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
                              l2=0.051892875031473, linf=0.009623271947010141,
                              pde_test=true)
    end
end

end # module
