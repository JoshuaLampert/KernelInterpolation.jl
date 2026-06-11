@testitem "RBF-FD: stencil selection and local weights" setup=[Setup, AdditionalImports] begin
    nodeset = NodeSet([0.0, 0.25, 0.5, 0.75, 1.0])
    kernel = GaussKernel{1}(shape_parameter = 1.0)

    knn = KNearestNeighbors(3)
    neigh = select_neighbors(3, nodeset, knn)
    @test length(neigh.indices) == 3

    rad = RadiusSearch(0.3)
    neigh_rad = select_neighbors(3, nodeset, rad)
    @test length(neigh_rad.indices) ≥ 2

    basis = RBFFDBasis(nodeset, kernel, knn)
    weights, info = rbf_fd_weights(Laplacian(), 3, basis, RBFFDStandardBasis())
    @test length(weights) == length(neigh.nodes)
    @test all(isfinite, weights)
    @test info.stencil_size == length(neigh.nodes)

    phs_kernel = PolyharmonicSplineKernel{1}(3)
    basis_phs = RBFFDBasis(nodeset, phs_kernel, knn; m = order(phs_kernel))
    weights_poly, _ = rbf_fd_weights(Laplacian(), 3, basis_phs, RBFFDStandardBasis())
    @test length(weights_poly) == length(neigh.nodes)
    @test all(isfinite, weights_poly)

    weights_cardinal, _ = rbf_fd_weights(Laplacian(), 3, basis, RBFFDLagrangeBasis())
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
    b = [f.(nodeset_inner, Ref(equation)); boundary_condition.(nodeset_boundary)]
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
                                 RBFFD(RBFFDLagrangeBasis()),
                                 GaussKernel{1}(shape_parameter = 2.0);
                                 stencil_selection = KNearestNeighbors(4),
                                 m = 0)

    @test disc.method isa RBFFD
    @test disc.method.local_basis isa RBFFDLagrangeBasis

    itp = solve_stationary(disc)
    @test itp isa Interpolation
end

@testitem "RBF-FD: PDE and operator evaluation at interpolation" setup=[Setup, AdditionalImports] begin
    nodeset_inner = NodeSet([0.25 0.25
                             0.5 0.25
                             0.75 0.25
                             0.25 0.5
                             0.5 0.5
                             0.75 0.5
                             0.25 0.75
                             0.5 0.75
                             0.75 0.75])
    nodeset_boundary = NodeSet([0.0 0.0
                                0.5 0.0
                                1.0 0.0
                                0.0 0.5
                                1.0 0.5
                                0.0 1.0
                                0.5 1.0
                                1.0 1.0])
    u(x) = x[1] * (x[1] - 1.0) + (x[2] - 1.0) * x[2]
    f(x, eq) = -4.0 # -Δu
    g(x) = u(x)
    kernel = GaussKernel{2}(shape_parameter = 0.5)
    pde = PoissonEquation(f)

    # Lagrange local basis: the evaluation acts on the same local cardinal functions that
    # are used to assemble the operator matrix, so the PDE and operator are reproduced at
    # the inner nodes up to the linear-solve residual.
    disc = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary,
                                 RBFFD(RBFFDLagrangeBasis()), kernel;
                                 stencil_selection = KNearestNeighbors(5))
    itp = solve_stationary(disc)
    @test itp isa KernelInterpolation.RBFFDInterpolation
    for node in nodeset_inner
        @test isapprox(pde(itp, node), f(node, pde), atol = 1e-11)
        @test isapprox(Laplacian()(itp, node), -f(node, pde), atol = 1e-11)
    end

    # Curried (one-argument) form returns a callable consistent with the two-argument form
    pde_itp = @test_nowarn pde(itp)
    @test isapprox(pde_itp(first(nodeset_inner)), pde(itp, first(nodeset_inner)))
    laplacian_itp = @test_nowarn Laplacian()(itp)
    @test isapprox(laplacian_itp(first(nodeset_inner)),
                   Laplacian()(itp, first(nodeset_inner)))

    # Gradient returns a vector (covers the vector-valued `s` initialization path) and
    # agrees componentwise with the partial derivatives.
    node = first(nodeset_inner)
    grad = Gradient()(itp, node)
    @test grad isa AbstractVector
    @test length(grad) == 2
    @test isapprox(grad[1], PartialDerivative(1)(itp, node))
    @test isapprox(grad[2], PartialDerivative(2)(itp, node))

    # At an arbitrary point (not a node) the operator uses the stencil of the nearest node,
    # consistent with the local cardinal expansion of the interpolant.
    x = [0.42, 0.6]
    j = nearest_node_index(x, nodeset(itp))
    bas = KernelInterpolation.basis(itp)
    c = coefficients(itp)
    expected = sum(c[bas.stencil_indices[j][k]] * Laplacian()(bas.local_funcs[j][k], x)
                   for k in eachindex(bas.stencil_indices[j]))
    @test isapprox(Laplacian()(itp, x), expected)

    # Standard local basis: assembly and evaluation use different numerical routes for the
    # same mathematical weights, so the PDE is reproduced only approximately.
    disc_std = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary,
                                     RBFFD(RBFFDStandardBasis()), kernel;
                                     stencil_selection = KNearestNeighbors(5))
    itp_std = solve_stationary(disc_std)
    @test itp_std isa KernelInterpolation.RBFFDInterpolation
    for node in nodeset_inner
        @test isapprox(pde(itp_std, node), f(node, pde), atol = 1e-10)
    end
end

@testitem "RBF-FD: basis indexing API" setup=[Setup, AdditionalImports] begin
    nodeset = NodeSet([0.0, 0.25, 0.5, 0.75, 1.0])
    kernel = GaussKernel{1}(shape_parameter = 1.0)
    stencil = KNearestNeighbors(3)

    basis = RBFFDBasis(nodeset, kernel, stencil; m = 0)
    neigh = select_neighbors(nodeset[3], nodeset, stencil)

    # local_funcs always holds Lagrange cardinal functions
    b = basis[3, 2]
    @test b(neigh.nodes[2])≈1.0 atol=1.0e-10
    @test abs(b(neigh.nodes[1])) ≤ 1.0e-8
    @test abs(b(neigh.nodes[3])) ≤ 1.0e-8

    @test_throws BoundsError basis[0, 1]
    @test_throws BoundsError basis[3, 0]
    @test_throws BoundsError basis[3, 4]
end

@testitem "RBF-FD: kernel_matrix with RBFFDBasis" setup=[Setup, AdditionalImports] begin
    using SparseArrays: findnz
    X = NodeSet([0.0, 0.25, 0.5, 0.75, 1.0])
    Y = NodeSet([0.1, 0.3, 0.6, 0.9, 1.0, 0.0])
    kernel = GaussKernel{1}(shape_parameter = 1.0)
    stencil = KNearestNeighbors(3)

    basis = RBFFDBasis(X, kernel, stencil; m = 0)
    C = kernel_matrix(basis, Y)
    @test size(C) == (length(Y), length(X))

    nz_rows, nz_cols, _ = findnz(C)
    for j in eachindex(Y)
        y_j = Y[j]
        i = nearest_node_index(y_j, X)
        neigh = select_neighbors(X[i], X, stencil)
        local_basis = LagrangeBasis(neigh.nodes, kernel; m = 0)
        stored_cols = nz_cols[nz_rows .== j]
        @test Set(stored_cols) == Set(neigh.indices)

        for (k, global_idx) in enumerate(neigh.indices)
            @test C[j, global_idx] ≈ local_basis[k](y_j)
        end
    end
end

@testitem "RBF-FD: operator_matrix with RBFFDBasis" setup=[Setup, AdditionalImports] begin
    X = NodeSet([0.0, 0.25, 0.5, 0.75, 1.0])
    Y = NodeSet([0.1, 0.3, 0.6, 0.9, 1.0, 0.0])
    kernel = GaussKernel{1}(shape_parameter = 1.0)
    stencil = KNearestNeighbors(3)

    basis = RBFFDBasis(X, kernel, stencil; m = 0)
    L = operator_matrix(Laplacian(), basis, Y)
    @test size(L) == (length(Y), length(X))

    for j in eachindex(Y)
        y_j = Y[j]
        i = nearest_node_index(y_j, X)
        neigh = select_neighbors(X[i], X, stencil)
        local_basis = LagrangeBasis(neigh.nodes, kernel; m = 0)
        nz_cols = findall(!iszero, L[j, :])
        @test Set(nz_cols) == Set(neigh.indices)

        for (k, global_idx) in enumerate(neigh.indices)
            @test L[j, global_idx] ≈ Laplacian()(local_basis[k], y_j)
        end
    end
end
