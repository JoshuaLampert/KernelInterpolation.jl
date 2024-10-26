@testsnippet PDEExamples begin
    # To obtain reproducible results in CI
    using Random
    Random.seed!(1)
    EXAMPLES_DIR = joinpath(examples_dir(), "PDEs")
end

@testitem "poisson_2d_basic.jl" setup=[Setup, AdditionalImports, PDEExamples] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "poisson_2d_basic.jl"),
                          l2=0.051892875031473, linf=0.009623271947010141,
                          pde_test=true)
end

@testitem "laplace_2d_annulus.jl" setup=[Setup, AdditionalImports, PDEExamples] begin
    # No analytical solution available (don't compare l2 and linf norms)
    @test_include_example(joinpath(EXAMPLES_DIR, "laplace_2d_annulus.jl"),
                          pde_test=true, atol=1e-11)
end

@testitem "anisotropic_elliptic_2d_basic.jl" setup=[Setup, AdditionalImports, PDEExamples] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "anisotropic_elliptic_2d_basic.jl"),
                          l2=0.6820834994466024, linf=0.10747754142379007,
                          pde_test=true, least_square_test=true)
end

@testitem "heat_2d_basic.jl" setup=[Setup, AdditionalImports, PDEExamples] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "heat_2d_basic.jl"),
                          l2=0.8163804581267793, linf=0.07510840007130493,
                          pde_test=true)
end

@testitem "heat_2d_manufactured.jl" setup=[Setup, AdditionalImports, PDEExamples] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "heat_2d_manufactured.jl"),
                          l2=0.05146343652866822, linf=0.005234201747677858,
                          pde_test=true)
end

@testitem "advection_1d_basic.jl" setup=[Setup, AdditionalImports, PDEExamples] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "advection_1d_basic.jl"),
                          l2=0.026436141175788297, linf=0.004194649552604207,
                          pde_test=true, atol=1e-6) # stability issues
end

@testitem "advection_3d_basic.jl" setup=[Setup, AdditionalImports, PDEExamples] begin
    if !Sys.iswindows() # Windows CI suddenly takes much smaller time steps for some reason
        @test_include_example(joinpath(EXAMPLES_DIR, "advection_3d_basic.jl"),
                              l2=0.055338785034078526, linf=0.004385483831323006,
                              pde_test=true, tspan=(0.0, 0.1))
    end
end

@testitem "advection_diffusion_2d_basic.jl" setup=[Setup, AdditionalImports, PDEExamples] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "advection_diffusion_2d_basic.jl"),
                          l2=1.5864821617681693, linf=0.56470989589118,
                          pde_test=true, tspan=(0.0, 0.1),
                          atol=1e-11) # stability issues
end
