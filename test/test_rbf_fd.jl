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

    weights_cardinal, _ = rbf_fd_weights(Laplacian(), nodeset[3], neigh.nodes, kernel;
                                         m = 0,
                                         local_basis = RBFDCardinalBasis())
    @test length(weights_cardinal) == length(neigh.nodes)
    @test all(isfinite, weights_cardinal)
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

@testitem "RBF-FD: cardinal local basis discretization" setup=[Setup, AdditionalImports] begin
    nodeset_inner = NodeSet([0.25, 0.5, 0.75])
    nodeset_boundary = NodeSet([0.0, 1.0])

    f(x, eq) = 0.0
    equation = PoissonEquation(f)
    boundary_condition(x) = 0.0

    disc = SpatialDiscretization(equation, nodeset_inner,
                                 boundary_condition, nodeset_boundary,
                                 RBFFD(),
                                 GaussKernel{1}(shape_parameter = 2.0);
                                 stencil_selection = KNearestNeighbors(4),
                                 m = 0,
                                 local_basis = RBFDCardinalBasis())

    @test disc.method isa RBFFD
    @test disc.basis.local_basis isa RBFDCardinalBasis

    itp = solve_stationary(disc)
    @test itp isa Interpolation
end

@testitem "RBF-FD: basis indexing API" setup=[Setup, AdditionalImports] begin
    nodeset = NodeSet([0.0, 0.25, 0.5, 0.75, 1.0])
    kernel = GaussKernel{1}(shape_parameter = 1.0)
    stencil = KNearestNeighbors(3)

    basis_std = RBFFDBasis(nodeset, kernel, stencil; m = 0,
                           local_basis = RBFDStandardBasis())
    neigh = select_neighbors(nodeset[3], nodeset, stencil)

    b_std = basis_std[3, 2]
    @test b_std(nodeset[1]) ≈ kernel(nodeset[1], neigh.nodes[2])

    basis_card = RBFFDBasis(nodeset, kernel, stencil; m = 0,
                            local_basis = RBFDCardinalBasis())
    b_card = basis_card[3, 2]
    @test b_card(neigh.nodes[2]) ≈ 1.0 atol = 1.0e-10
    @test abs(b_card(neigh.nodes[1])) ≤ 1.0e-8
    @test abs(b_card(neigh.nodes[3])) ≤ 1.0e-8

    @test_throws BoundsError basis_std[0, 1]
    @test_throws BoundsError basis_std[3, 0]
    @test_throws BoundsError basis_std[3, 4]
end
