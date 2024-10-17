@testitem "Aqua.jl" begin
    import Aqua
    using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports
    Aqua.test_all(ambiguities = false, KernelInterpolation)
    @test isnothing(check_no_implicit_imports(KernelInterpolation))
    @test isnothing(check_no_stale_explicit_imports(KernelInterpolation;
                                                    ignore = (:trixi_include,)))
end
