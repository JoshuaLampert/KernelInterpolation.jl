@testitem "Aqua.jl" begin
    import Aqua
    using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports,
                           check_all_explicit_imports_via_owners,
                           check_all_qualified_accesses_via_owners,
                           check_no_self_qualified_accesses
    # We need `unbound_args = false` due to https://github.com/JuliaTesting/Aqua.jl/issues/265#issuecomment-2173168334
    Aqua.test_all(KernelInterpolation, ambiguities = false, unbound_args = false)
    @test isnothing(check_no_implicit_imports(KernelInterpolation))
    @test isnothing(check_no_stale_explicit_imports(KernelInterpolation;
                                                    ignore = (:trixi_include,)))
    @test isnothing(check_all_explicit_imports_via_owners(KernelInterpolation))
    @test isnothing(check_all_qualified_accesses_via_owners(KernelInterpolation))
    @test isnothing(check_no_self_qualified_accesses(KernelInterpolation))
end
