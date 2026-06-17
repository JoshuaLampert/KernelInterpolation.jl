@testitem "RBF-FD: stencil selection and local weights" setup=[Setup, AdditionalImports] begin
    nodeset = NodeSet([0.0, 0.25, 0.5, 0.75, 1.0])
    kernel = GaussKernel{1}(shape_parameter = 1.0)

    knn = KNearestNeighbors(3)
    @test_nowarn display(knn)
    neigh = select_neighbors(3, nodeset, knn)
    @test length(neigh.indices) == 3

    rad = RadiusSearch(0.3)
    @test_nowarn display(rad)
    neigh_rad = select_neighbors(3, nodeset, rad)
    @test length(neigh_rad.indices) ≥ 2

    basis = RBFFDBasis(nodeset, kernel, knn)
    weights = rbf_fd_weights(Laplacian(), 3, basis, RBFFDStandardBasis())
    @test length(weights) == length(neigh.nodes)
    @test all(isfinite, weights)

    phs_kernel = PolyharmonicSplineKernel{1}(3)
    basis_phs = RBFFDBasis(nodeset, phs_kernel, knn; m = order(phs_kernel))
    weights_poly = rbf_fd_weights(Laplacian(), 3, basis_phs, RBFFDStandardBasis())
    @test length(weights_poly) == length(neigh.nodes)
    @test all(isfinite, weights_poly)

    weights_cardinal = rbf_fd_weights(Laplacian(), 3, basis, RBFFDLagrangeBasis())
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

    A = pde_boundary_matrix(equation, nodeset_inner, nodeset_boundary, disc.basis)
    b = [f.(nodeset_inner, Ref(equation)); boundary_condition.(nodeset_boundary)]
    u = A \ b

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
                                 local_basis = RBFFDLagrangeBasis())

    @test disc.method isa RBFFD
    @test disc.basis.local_basis isa RBFFDLagrangeBasis

    itp = solve_stationary(disc)
    @test itp isa Interpolation
end

@testitem "RBF-FD: PDE and operator evaluation at interpolation" setup=[
    Setup,
    AdditionalImports
] begin
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
                                 RBFFD(), kernel;
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
    j = nearest_node_index(x, centers(itp))
    bas = KernelInterpolation.basis(itp)
    c = coefficients(itp)
    expected = sum(c[bas.stencil_indices[j][k]] * Laplacian()(bas.local_funcs[j][k], x)
                   for k in eachindex(bas.stencil_indices[j]))
    @test isapprox(Laplacian()(itp, x), expected)

    # Three-argument form `op(itp, x, j)` evaluates with the stencil of center `j`
    # explicitly (mirroring `itp(x, j)`). With the nearest index it matches the
    # two-argument form, both for scalar operators/PDEs and the vector-valued gradient.
    @test isapprox(Laplacian()(itp, x, j), Laplacian()(itp, x))
    @test isapprox(pde(itp, x, j), pde(itp, x))
    @test isapprox(Laplacian()(itp, x, j), expected)
    grad_x = Gradient()(itp, x, j)
    @test grad_x isa AbstractVector
    @test length(grad_x) == 2
    @test isapprox(grad_x, Gradient()(itp, x))
    grad_expected = sum(c[bas.stencil_indices[j][k]] * Gradient()(bas.local_funcs[j][k], x)
                        for k in eachindex(bas.stencil_indices[j]))
    @test isapprox(grad_x, grad_expected)
    # At an inner node the matching stencil index (global index of the node) reproduces the
    # right-hand side, for the scalar PDE/operator and consistently for the gradient.
    for (i, inner_node) in enumerate(nodeset_inner)
        @test isapprox(pde(itp, inner_node, i), f(inner_node, pde), atol = 1e-11)
        @test isapprox(Laplacian()(itp, inner_node, i), -f(inner_node, pde), atol = 1e-11)
        @test isapprox(Gradient()(itp, inner_node, i), Gradient()(itp, inner_node))
    end

    # Standard local basis: assembly and evaluation use different numerical routes for the
    # same mathematical weights, so the PDE is reproduced only approximately.
    disc_std = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary,
                                     RBFFD(), kernel;
                                     stencil_selection = KNearestNeighbors(5),
                                     local_basis = RBFFDStandardBasis())
    itp_std = solve_stationary(disc_std)
    @test itp_std isa KernelInterpolation.RBFFDInterpolation
    for node in nodeset_inner
        @test isapprox(pde(itp_std, node), f(node, pde), atol = 1e-10)
    end

    # Least-squares RBF-FD: basis built on a strict subset of merge(ni, nb).
    basis_ls = RBFFDBasis(nodeset_inner, kernel, KNearestNeighbors(3))
    disc_ls = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary, basis_ls)
    itp_ls = @test_nowarn solve_stationary(disc_ls)
    @test nodeset(itp_ls) == merge(nodeset_inner, nodeset_boundary)
end

@testitem "RBF-FD: basis indexing API" setup=[Setup, AdditionalImports] begin
    nodeset = NodeSet([0.0, 0.25, 0.5, 0.75, 1.0])
    kernel = GaussKernel{1}(shape_parameter = 1.0)
    stencil = KNearestNeighbors(3)

    basis = RBFFDBasis(nodeset, kernel, stencil; m = 0)
    neigh = select_neighbors(3, nodeset, stencil)

    # local_funcs always holds Lagrange cardinal functions
    b = basis[3, 2]
    @test b(neigh.nodes[2])≈1.0 atol=1.0e-10
    @test abs(b(neigh.nodes[1])) ≤ 1.0e-12
    @test abs(b(neigh.nodes[3])) ≤ 1.0e-12

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
        neigh = select_neighbors(i, X, stencil)
        local_basis = LagrangeBasis(neigh.nodes, kernel; m = 0)
        stored_cols = nz_cols[nz_rows .== j]
        @test Set(stored_cols) == Set(neigh.indices)

        for (k, global_idx) in enumerate(neigh.indices)
            @test C[j, global_idx] ≈ local_basis[k](y_j)
        end
    end
end

@testitem "RBF-FD: differentiation_matrix with RBFFDBasis" setup=[Setup, AdditionalImports] begin
    X = NodeSet([0.0, 0.25, 0.5, 0.75, 1.0])
    Y = NodeSet([0.1, 0.3, 0.6, 0.9, 1.0, 0.0])
    kernel = GaussKernel{1}(shape_parameter = 1.0)
    stencil = KNearestNeighbors(3)

    basis = RBFFDBasis(X, kernel, stencil; m = 0)
    L = differentiation_matrix(Laplacian(), basis, Y)
    @test size(L) == (length(Y), length(X))

    for j in eachindex(Y)
        y_j = Y[j]
        i = nearest_node_index(y_j, X)
        neigh = select_neighbors(i, X, stencil)
        local_basis = LagrangeBasis(neigh.nodes, kernel; m = 0)
        nz_cols = findall(!iszero, L[j, :])
        @test Set(nz_cols) == Set(neigh.indices)

        for (k, global_idx) in enumerate(neigh.indices)
            @test L[j, global_idx] ≈ Laplacian()(local_basis[k], y_j)
        end
    end

    # PartialDerivative(1) in 1D is d/dx. Weights per row equal PartialDerivative(1)(ℓ_k, y_j).
    D1_1d = differentiation_matrix(PartialDerivative(1), basis, Y)
    @test size(D1_1d) == (length(Y), length(X))
    for j in eachindex(Y)
        y_j = Y[j]
        i = nearest_node_index(y_j, X)
        neigh = select_neighbors(i, X, stencil)
        local_basis_j = LagrangeBasis(neigh.nodes, kernel; m = 0)
        for (k, global_idx) in enumerate(neigh.indices)
            @test D1_1d[j, global_idx] ≈ PartialDerivative(1)(local_basis_j[k], y_j)
        end
    end

    # PartialDerivative(1) with polynomial augmentation m=2 ({1, x} in 1D) is exact on
    # linear functions: D * x_vals = ones, D * ones = zeros.
    basis_m2 = RBFFDBasis(X, kernel, stencil; m = 2)
    D1_m2 = differentiation_matrix(PartialDerivative(1), basis_m2)
    @test isapprox(D1_m2 * ones(length(X)), zeros(length(X)), atol = 1e-14)
    @test isapprox(D1_m2 * first.(X), ones(length(X)), atol = 1e-13)

    # 2D: PartialDerivative(1) and (2) are exact on linear functions with order(kernel_2d)=m=2
    # (monomials of degree ≤ 1 in 2D: {1, x₁, x₂}).
    nodes_2d = homogeneous_hypercube(4, (0.0, 0.0), (1.0, 1.0))
    kernel_2d = PolyharmonicSplineKernel{2}(3)
    basis_2d = RBFFDBasis(nodes_2d, kernel_2d, KNearestNeighbors(6))
    N_2d = length(nodes_2d)
    x1_vals = first.(nodes_2d)
    x2_vals = last.(nodes_2d)
    D1_2d = differentiation_matrix(PartialDerivative(1), basis_2d)
    D2_2d = differentiation_matrix(PartialDerivative(2), basis_2d)
    @test isapprox(D1_2d * ones(N_2d), zeros(N_2d), atol = 1e-13)
    @test isapprox(D1_2d * x1_vals, ones(N_2d), atol = 1e-13)
    @test isapprox(D1_2d * x2_vals, zeros(N_2d), atol = 1e-13)
    @test isapprox(D2_2d * x1_vals, zeros(N_2d), atol = 1e-13)
    @test isapprox(D2_2d * x2_vals, ones(N_2d), atol = 1e-13)

    # 2D kernel translate exactness: weights per row equal PartialDerivative applied to
    # the local Lagrange basis functions.
    basis_2d_m0 = RBFFDBasis(nodes_2d, kernel_2d, KNearestNeighbors(6); m = 0)
    D1_2d_m0 = differentiation_matrix(PartialDerivative(1), basis_2d_m0)
    D2_2d_m0 = differentiation_matrix(PartialDerivative(2), basis_2d_m0)
    stencil_2d = KNearestNeighbors(6)
    for j in eachindex(nodes_2d)
        x_j = nodes_2d[j]
        neigh = select_neighbors(j, nodes_2d, stencil_2d)
        local_basis_j = LagrangeBasis(neigh.nodes, kernel_2d; m = 0)
        for (k, global_idx) in enumerate(neigh.indices)
            @test D1_2d_m0[j, global_idx] ≈ PartialDerivative(1)(local_basis_j[k], x_j)
            @test D2_2d_m0[j, global_idx] ≈ PartialDerivative(2)(local_basis_j[k], x_j)
        end
    end
end

@testitem "RBF-FD: least-squares differentiation_matrix" setup=[
    Setup,
    AdditionalImports
] begin
    # 1D: more evaluation nodes Y than basis centers X → overdetermined, rectangular matrix.
    X = NodeSet([0.0, 0.25, 0.5, 0.75, 1.0])
    Y = NodeSet(LinRange(0.0, 1.0, 15))
    kernel = GaussKernel{1}(shape_parameter = 1.0)
    stencil = KNearestNeighbors(3)

    basis = RBFFDBasis(X, kernel, stencil; m = 0)
    D = differentiation_matrix(PartialDerivative(1), basis, Y)

    # Size is (|Y|, |X|): rectangular and overdetermined.
    @test size(D) == (length(Y), length(X))

    # Each row j uses the stencil of the nearest center in X to Y[j].
    for j in eachindex(Y)
        y_j = Y[j]
        i = nearest_node_index(y_j, X)
        neigh = select_neighbors(i, X, stencil)
        local_basis_j = LagrangeBasis(neigh.nodes, kernel; m = 0)
        nz_cols = findall(!iszero, D[j, :])
        @test Set(nz_cols) == Set(neigh.indices)
        for (k, global_idx) in enumerate(neigh.indices)
            @test D[j, global_idx] ≈ PartialDerivative(1)(local_basis_j[k], y_j)
        end
    end

    # With m=2 ({1, x} in 1D): D * p.(X) = p'.(Y) exactly for polynomials of degree ≤ 1.
    basis_m2 = RBFFDBasis(X, kernel, stencil; m = 2)
    D_m2 = differentiation_matrix(PartialDerivative(1), basis_m2, Y)
    @test size(D_m2) == (length(Y), length(X))
    x_vals = first.(X)
    @test isapprox(D_m2 * ones(length(X)), zeros(length(Y)), atol = 1e-12)
    @test isapprox(D_m2 * x_vals, ones(length(Y)), atol = 1e-12)

    # With m=3 ({1, x, x²} in 1D): Laplacian * p.(X) = p''.(Y) for degree ≤ 2.
    basis_m3 = RBFFDBasis(X, kernel, stencil; m = 3)
    Lap = differentiation_matrix(Laplacian(), basis_m3, Y)
    @test size(Lap) == (length(Y), length(X))
    @test isapprox(Lap * x_vals, zeros(length(Y)), atol = 1e-11)
    @test isapprox(Lap * x_vals .^ 2, 2 * ones(length(Y)), atol = 1e-11)

    # 2D: same rectangular structure and exactness on linear functions with m=2.
    nodes_2d = homogeneous_hypercube(4, (0.0, 0.0), (1.0, 1.0))
    eval_2d = homogeneous_hypercube(6, (0.0, 0.0), (1.0, 1.0))
    kernel_2d = PolyharmonicSplineKernel{2}(3)
    basis_2d = RBFFDBasis(nodes_2d, kernel_2d, KNearestNeighbors(6))
    D1_2d = differentiation_matrix(PartialDerivative(1), basis_2d, eval_2d)
    D2_2d = differentiation_matrix(PartialDerivative(2), basis_2d, eval_2d)

    @test size(D1_2d) == (length(eval_2d), length(nodes_2d))

    x1_at_X = first.(nodes_2d)
    x2_at_X = last.(nodes_2d)
    @test isapprox(D1_2d * ones(length(nodes_2d)), zeros(length(eval_2d)), atol = 1e-12)
    @test isapprox(D1_2d * x1_at_X, ones(length(eval_2d)), atol = 1e-12)
    @test isapprox(D1_2d * x2_at_X, zeros(length(eval_2d)), atol = 1e-12)
    @test isapprox(D2_2d * x1_at_X, zeros(length(eval_2d)), atol = 1e-12)
    @test isapprox(D2_2d * x2_at_X, ones(length(eval_2d)), atol = 1e-12)
end
