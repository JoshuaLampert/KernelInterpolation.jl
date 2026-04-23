@testitem "RBF-FD: stencil selection and local weights" setup=[Setup, AdditionalImports] begin
    nodeset = NodeSet([0.0, 0.25, 0.5, 0.75, 1.0])
    kernel = GaussKernel{1}(shape_parameter = 1.0)

    knn = KNearestNeighbors(3)
    neigh = select_neighbors(3, nodeset, knn)
    @test length(neigh.indices) == 3

    rad = RadiusSearch(0.3)
    neigh_rad = select_neighbors(3, nodeset, rad)
    @test length(neigh_rad.indices) ≥ 2

    weights, info = rbf_fd_weights(Laplacian(), nodeset[3], neigh.nodes, kernel)
    @test length(weights) == length(neigh.nodes)
    @test all(isfinite, weights)
    @test info.stencil_size == length(neigh.nodes)

    phs_kernel = PolyharmonicSplineKernel{1}(3)
    weights_poly, _ = rbf_fd_weights(Laplacian(), nodeset[3], neigh.nodes, phs_kernel;
                                     m = order(phs_kernel))
    @test length(weights_poly) == length(neigh.nodes)
    @test all(isfinite, weights_poly)
end

@testitem "RBF-FD: stationary discretization" setup=[Setup, AdditionalImports] begin
    nodeset_inner = NodeSet([0.25, 0.5, 0.75])
    nodeset_boundary = NodeSet([0.0, 1.0])

    f(x, eq) = 0.0
    equation = PoissonEquation(f)
    boundary_condition(x) = 0.0

    disc = SpatialDiscretization(equation, nodeset_inner,
                                 boundary_condition, nodeset_boundary,
                                 RBFFD(),
                                 GaussKernel{1}(shape_parameter = 2.0);
                                 stencil_selection = KNearestNeighbors(4))

    A = rbf_fd_pde_boundary_matrix(equation, nodeset_inner, nodeset_boundary, disc.basis)
    A_dispatch = pde_boundary_matrix(equation, nodeset_inner, nodeset_boundary, disc.basis)
    b = [rhs(nodeset_inner, equation); boundary_condition.(nodeset_boundary)]
    u = A \ b

    @test A_dispatch == A
    @test length(u) == length(nodeset_inner) + length(nodeset_boundary)
    @test norm(u) ≤ 1.0e-8

    L = operator_matrix(equation, nodeset_inner, nodeset_boundary, disc.basis)
    @test size(L) == (length(nodeset_inner) + length(nodeset_boundary),
                      length(nodeset_inner) + length(nodeset_boundary))

    itp = solve_stationary(disc)
    @test itp isa Interpolation
end

@testitem "RBF-FD: semidiscretization setup" setup=[Setup, AdditionalImports] begin
    nodeset_inner = NodeSet([0.25, 0.5, 0.75])
    nodeset_boundary = NodeSet([0.0, 1.0])

    f(t, x, eq) = 0.0
    equation = HeatEquation(0.1, f)
    boundary_condition(t, x) = 0.0
    initial_condition(t, x, eq) = sin(pi * x[1])

    disc = SpatialDiscretization(equation, nodeset_inner,
                                 boundary_condition, nodeset_boundary,
                                 RBFFD(),
                                 GaussKernel{1}(shape_parameter = 2.0);
                                 stencil_selection = KNearestNeighbors(4))

    semi = Semidiscretization(disc, initial_condition)
    prob = semidiscretize(semi, (0.0, 0.1))

    @test hasproperty(prob, :u0)
    @test length(prob.u0) == length(nodeset_inner) + length(nodeset_boundary)
end
