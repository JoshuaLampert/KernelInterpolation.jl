@testsnippet PDEExamples begin
    # To obtain reproducible results in CI
    using Random
    Random.seed!(1)
    EXAMPLES_DIR = joinpath(examples_dir(), "PDEs")
end

@testitem "poisson_2d_basic.jl" setup=[Setup, AdditionalImports, PDEExamples] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "poisson_2d_basic.jl"),
                          l2=0.049326883009727776, linf=0.00884841202228337,
                          pde_test=true)
end

@testitem "poisson_2d_basic.jl with polynomial augmentation" setup=[
    Setup,
    AdditionalImports,
    PDEExamples
] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "poisson_2d_basic.jl"),
                          kernel=PolyharmonicSplineKernel{2}(3),
                          l2=0.21521434341247836, linf=0.038429436830987575,
                          pde_test=true)
end

@testitem "poisson_2d_lagrange_basis.jl" setup=[Setup, AdditionalImports, PDEExamples] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "poisson_2d_lagrange_basis.jl"),
                          l2=0.04932687666512765, linf=0.008848421428649026,
                          pde_test=true, atol=1e-7)
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

@testitem "poisson_3d_ball.jl" setup=[Setup, AdditionalImports, PDEExamples] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "poisson_3d_ball.jl"),
                          l2=1.1842841086211155, linf=0.14341093728770962,
                          pde_test=true, atol=1e-11)
end

@testitem "heat_2d_basic.jl" setup=[Setup, AdditionalImports, PDEExamples] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "heat_2d_basic.jl"),
                          l2=0.8166184454477932, linf=0.07519677063240593,
                          pde_test=true)
end

@testitem "heat_2d_manufactured.jl" setup=[Setup, AdditionalImports, PDEExamples] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "heat_2d_manufactured.jl"),
                          l2=0.0312855604318702, linf=0.0029173794750327886,
                          pde_test=true)
end

@testitem "heat_2d_lagrange_basis.jl" setup=[Setup, AdditionalImports, PDEExamples] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "heat_2d_lagrange_basis.jl"),
                          l2=0.03128571950518645, linf=0.002917382363885124,
                          pde_test=true, atol=1e-8)
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
                          l2=1.4971059347717564, linf=0.4610303242043753,
                          pde_test=true, tspan=(0.0, 0.1),
                          atol=1e-7) # stability issues
end

@testitem "rbf_fd_poisson_2d_basic.jl" setup=[Setup, AdditionalImports, PDEExamples] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "rbf_fd_poisson_2d_basic.jl"),
                          l2=0.3091145670601108, linf=0.03593545182808036,
                          pde_test=true)
end

# (almost) same values as for the `RBFFDLagrangeBasis`
@testitem "rbf_fd_poisson_2d_basic.jl with RBFFDStandardBasis" setup=[
    Setup,
    AdditionalImports,
    PDEExamples
] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "rbf_fd_poisson_2d_basic.jl"),
                          local_basis=RBFFDStandardBasis(),
                          l2=0.3091145670599337, linf=0.035935451828227466,
                          pde_test=true)
end

@testitem "rbf_fd_poisson_2d_basic.jl with RadiusSearch" setup=[
    Setup,
    AdditionalImports,
    PDEExamples
] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "rbf_fd_poisson_2d_basic.jl"),
                          stencil_selection=RadiusSearch(0.3),
                          l2=0.2631144635952774, linf=0.03741207775337395,
                          pde_test=true)
end

@testitem "rbf_fd_poisson_2d_least_squares.jl" setup=[Setup, AdditionalImports, PDEExamples] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "rbf_fd_poisson_2d_least_squares.jl"),
                          l2=0.08017627795452234, linf=0.011301965835894934,
                          pde_test=true, least_square_test=true,
                          atol=1e-2) # stability issues
end

@testitem "rbf_fd_advection_2d_basic.jl" setup=[Setup, AdditionalImports, PDEExamples] begin
    @test_include_example(joinpath(EXAMPLES_DIR, "rbf_fd_advection_2d_basic.jl"),
                          l2=0.24648672491853177, linf=0.05116013173606859,
                          pde_test=true)
end
