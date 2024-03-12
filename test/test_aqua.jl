module TestAqua

using Aqua
using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports
using Test
using KernelInterpolation

@testset "Aqua.jl" begin
    Aqua.test_all(ambiguities = false, KernelInterpolation)
    @test isnothing(check_no_implicit_imports(KernelInterpolation))
    @test isnothing(check_no_stale_explicit_imports(KernelInterpolation))
end

end #module
