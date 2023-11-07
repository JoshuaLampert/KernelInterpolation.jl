using KernelInterpolation
using Test

@testset "KernelInterpolation.jl" begin
    include("test_aqua.jl")
    include("test_unit.jl")
    include("test_examples.jl")
end
