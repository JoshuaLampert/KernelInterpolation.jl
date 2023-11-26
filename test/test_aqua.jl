module TestAqua

using Aqua
using Test
using KernelInterpolation

@testset "Aqua.jl" begin
    Aqua.test_all(ambiguities = false, KernelInterpolation)
end

end #module
