@testitem "Aqua.jl" begin
    import Aqua
    using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports,
                           check_all_explicit_imports_via_owners,
                           check_all_qualified_accesses_via_owners,
                           check_no_self_qualified_accesses
    # We need `unbound_args = false` due to https://github.com/JuliaTesting/Aqua.jl/issues/265#issuecomment-2173168334
    Aqua.test_all(KernelInterpolation, ambiguities = false, unbound_args = false)
    # `Aqua.test_ambiguities` is disabled since there are 12 known ambiguities between the call
    # overloads of differential operators/equations applied to kernels and to interpolations,
    # e.g., `op(kernel, x, y)` and `op(s, itp, x)`, which cannot be reached by meaningful calls.
    # Instead, check that their number does not grow. Only methods from this package, Base, and
    # Core are considered, since the result must not depend on other packages loaded by other
    # test items (e.g., plot recipes of Plots.jl).
    own_or_base(method) = Base.moduleroot(method.module) in (KernelInterpolation, Base,
                                                             Core)
    ambiguities = filter(((m1, m2),) -> own_or_base(m1) && own_or_base(m2),
                         Test.detect_ambiguities(KernelInterpolation))
    @test length(ambiguities) == 12
    @test isnothing(check_no_implicit_imports(KernelInterpolation))
    @test isnothing(check_no_stale_explicit_imports(KernelInterpolation;
                                                    ignore = (:trixi_include,
                                                              :derivative_discontinuity!)))
    @test isnothing(check_all_qualified_accesses_via_owners(KernelInterpolation))
    @test isnothing(check_no_self_qualified_accesses(KernelInterpolation))
end
