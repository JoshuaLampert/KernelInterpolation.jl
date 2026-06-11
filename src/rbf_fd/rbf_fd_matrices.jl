# Assemble a sparse RBF-FD matrix from local stencil contributions. For each target node
# `y_j` in `nodeset`, the stencil of the nearest center is located and row `j` is filled
# with `entry(ℓ_k, y_j)` for every local cardinal function `ℓ_k` on that stencil. This is
# the common pattern behind the evaluation ([`kernel_matrix`](@ref)) and differentiation
# ([`operator_matrix`](@ref)) matrices.
function _rbf_fd_sparse_matrix(entry, basis::RBFFDBasis, nodeset::NodeSet)
    n = length(nodeset)
    m = length(basis)
    rows = Int[]
    cols = Int[]
    vals = eltype(nodeset)[]
    x = centers(basis)

    for j in 1:n
        y_j = nodeset[j]
        i = nearest_node_index(y_j, x)
        for (k, global_idx) in enumerate(basis.stencil_indices[i])
            push!(rows, j)
            push!(cols, global_idx)
            push!(vals, entry(basis.local_funcs[i][k], y_j))
        end
    end

    return sparse(rows, cols, vals, n, m)
end

# Evaluation (resampling) matrix: row `j` maps the nodal values to the interpolant value
# at `nodeset[j]` using the nearest stencil's local cardinal functions. See the generic
# [`kernel_matrix`](@ref) for the meaning shared with the other bases.
function kernel_matrix(basis::RBFFDBasis, nodeset::NodeSet = centers(basis))
    return _rbf_fd_sparse_matrix(basis, nodeset) do f, y
        return f(y)
    end
end

# Sparse differentiation matrix: row `j` maps the nodal values to `𝓛u` evaluated at
# `nodeset[j]` using the nearest stencil's local cardinal functions. This always uses the
# cardinal route (the only one well defined at arbitrary points); the `basis.local_basis`
# weight algorithm only affects the collocation assembly in [`pde_matrix`](@ref). See the
# generic [`differentiation_matrix`](@ref) for the meaning shared with the other bases.
function differentiation_matrix(diff_op_or_pde, basis::RBFFDBasis,
                                nodeset::NodeSet = centers(basis))
    return _rbf_fd_sparse_matrix(basis, nodeset) do f, y
        value = diff_op_or_pde(f, y)
        value isa Number ||
            throw(ArgumentError("differentiation_matrix for an RBFFDBasis expects scalar-valued operators"))
        return value
    end
end

@doc raw"""
    pde_matrix(diff_op_or_pde, nodeset_inner, basis::RBFFDBasis)

Assemble the sparse RBF-FD operator matrix for the inner nodes. Each row corresponds to
one inner node and contains the local stencil weights ``\mathcal{L}\ell_k(x_i)`` of the
differential operator (or PDE) `diff_op_or_pde`. The weights are computed via
[`rbf_fd_weights`](@ref) using the weight algorithm stored in `basis`
(`basis.local_basis`). The inner nodes are assumed to be the first `length(nodeset_inner)`
centers of `basis`.

See also [`pde_boundary_matrix`](@ref).
"""
function pde_matrix(diff_op_or_pde, nodeset_inner::NodeSet, basis::RBFFDBasis)
    n_inner = length(nodeset_inner)
    n_total = length(basis)
    rows = Int[]
    cols = Int[]
    vals = eltype(basis.nodeset)[]

    for i in eachindex(nodeset_inner)
        weights, _ = rbf_fd_weights(diff_op_or_pde, i, basis)
        weights isa AbstractVector ||
            throw(ArgumentError("RBF-FD PDE assembly expects scalar operator values"))
        for (j, global_idx) in enumerate(basis.stencil_indices[i])
            push!(rows, i)
            push!(cols, global_idx)
            push!(vals, weights[j])
        end
    end

    return sparse(rows, cols, vals, n_inner, n_total)
end

# Boundary selection matrix for Dirichlet conditions. The unknowns are the nodal values
# ordered as `merge(nodeset_inner, nodeset_boundary)`, so the boundary block is the exact
# identity selecting the boundary unknowns.
function _rbf_fd_boundary_selection(nodeset_inner::NodeSet, nodeset_boundary::NodeSet)
    n_inner = length(nodeset_inner)
    n_boundary = length(nodeset_boundary)
    n_total = n_inner + n_boundary

    rows = collect(1:n_boundary)
    cols = collect((n_inner + 1):n_total)
    vals = fill(one(eltype(nodeset_inner)), n_boundary)

    return sparse(rows, cols, vals, n_boundary, n_total)
end

@doc raw"""
    pde_boundary_matrix(diff_op_or_pde, nodeset_inner, nodeset_boundary, basis::RBFFDBasis)

Assemble the full sparse RBF-FD PDE + boundary matrix
```math
    A = \begin{pmatrix}L\\B\end{pmatrix},
```
where ``L`` is the RBF-FD operator matrix on the inner nodes ([`pde_matrix`](@ref)) and
``B`` enforces the Dirichlet constraints by selecting the boundary nodal values. The
unknowns are ordered as `merge(nodeset_inner, nodeset_boundary)`.

See also [`pde_matrix`](@ref) and [`operator_matrix`](@ref).
"""
function pde_boundary_matrix(diff_op_or_pde, nodeset_inner::NodeSet,
                             nodeset_boundary::NodeSet, basis::RBFFDBasis)
    L = pde_matrix(diff_op_or_pde, nodeset_inner, basis)
    B = _rbf_fd_boundary_selection(nodeset_inner, nodeset_boundary)
    return [L
            B]
end

# RBF-FD handles polynomial augmentation locally via the stencils, so no global polynomial
# blocks are added. This method only exists so callers can pass `ps` uniformly; it must be
# empty.
function pde_boundary_matrix(diff_op_or_pde, nodeset_inner::NodeSet,
                             nodeset_boundary::NodeSet, basis::RBFFDBasis, ps)
    @assert isempty(ps) "RBF-FD augments polynomials per stencil; pass an empty `ps`."
    return pde_boundary_matrix(diff_op_or_pde, nodeset_inner, nodeset_boundary, basis)
end

# For an `RBFFDBasis` the local stencil weights already are the operator matrix in
# nodal-value space, so the system operator matrix is just the PDE + boundary matrix (no
# division by a kernel matrix as in the collocation case).
function operator_matrix(diff_op_or_pde, nodeset_inner::NodeSet,
                         nodeset_boundary::NodeSet, basis::RBFFDBasis)
    return pde_boundary_matrix(diff_op_or_pde, nodeset_inner, nodeset_boundary, basis)
end
