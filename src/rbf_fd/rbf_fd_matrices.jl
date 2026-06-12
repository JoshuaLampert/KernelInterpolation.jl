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
    X = centers(basis)

    for j in 1:n
        y_j = nodeset[j]
        i = nearest_node_index(y_j, X)
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

@doc raw"""
    pde_matrix(diff_op_or_pde, nodeset, basis::RBFFDBasis)

Assemble the sparse RBF-FD operator matrix. Each row `j` corresponds to one node in
`nodeset` and contains the local stencil weights ``\mathcal{L}\ell_k(y_j)`` of the
differential operator (or PDE) `diff_op_or_pde`, evaluated at `y_j` using the cardinal
functions of the nearest center in `basis`. The result is an `|nodeset| × |basis|` sparse
matrix, so `nodeset` may be any set of evaluation points and need not equal the centers of
`basis`. When `|nodeset| > |basis|`, the system assembled by [`pde_boundary_matrix`](@ref)
is overdetermined and solved in the least-squares sense.

See also [`differentiation_matrix`](@ref), [`pde_boundary_matrix`](@ref).
"""
function pde_matrix(diff_op_or_pde, nodeset::NodeSet, basis::RBFFDBasis)
    return _rbf_fd_sparse_matrix(basis, nodeset) do f, y
        value = diff_op_or_pde(f, y)
        value isa Number ||
            throw(ArgumentError("RBF-FD PDE assembly expects scalar operator values"))
        return value
    end
end

# For `RBFFDBasis` the cardinal functions are always used, so
# `pde_matrix` and `differentiation_matrix` coincide. See the generic
# [`differentiation_matrix`](@ref) for the meaning shared with the other bases.
function differentiation_matrix(diff_op_or_pde, basis::RBFFDBasis,
                                nodeset::NodeSet = centers(basis))
    return pde_matrix(diff_op_or_pde, nodeset, basis)
end

# `pde_boundary_matrix` for `RBFFDBasis` falls through to the generic `AbstractBasis` method
# in `kernel_matrices.jl`: `[pde_matrix(op, ni, basis); kernel_matrix(basis, nb)]`.
#
# `operator_matrix` is specialized to avoid the spurious `A_L / A` division in the generic
# `AbstractBasis` path (which would also incorrectly try to add global polynomial rows via
# `order(basis) > 0`, whereas RBF-FD polynomial augmentation is local per stencil).
function operator_matrix(diff_op_or_pde, nodeset_inner::NodeSet,
                         nodeset_boundary::NodeSet, basis::RBFFDBasis)
    return pde_boundary_matrix(diff_op_or_pde, nodeset_inner, nodeset_boundary, basis)
end
