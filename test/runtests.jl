using KernelInterpolation
using Test

@testset "KernelInterpolation.jl" begin
    include("test_aqua.jl")
    include("test_unit.jl")
    include("test_examples_interpolation.jl")
    include("test_examples_pde.jl")
end
