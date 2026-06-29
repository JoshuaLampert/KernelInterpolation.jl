# Assemble a sparse RBF-FD matrix from local stencil contributions. For each target node
# `y_j` in `nodeset`, the stencil of the nearest center is located and row `j` is filled
# with the local weights `local_weights(basis, i, y_j, op)` scattered into the columns given
# by the stencil's global indices. `op = Identity()` gives the evaluation (resampling) matrix
# ([`kernel_matrix`](@ref)); a differential operator/equation gives the differentiation
# ([`operator_matrix`](@ref)) matrix. The local-basis policy stored in `basis` selects the
# numerical route (see [`local_weights`](@ref)).
function _rbf_fd_sparse_matrix(op, basis::RBFFDBasis, nodeset::NodeSet)
    n = length(nodeset)
    m = length(basis)
    X = centers(basis)
    stencils = basis.stencil_indices

    # Each evaluation node uses the stencil of its nearest center, and the rows are mutually
    # independent. Find the nearest center for every node first (threaded).
    nearest = Vector{Int}(undef, n)
    Threads.@threads for j in 1:n
        nearest[j] = nearest_node_index(nodeset[j], X)
    end

    # Preallocate the COO buffers: each row contributes as many nonzeros as its stencil has
    # entries, and the prefix sums give every row a disjoint block to fill in parallel.
    row_nnz = [length(stencils[nearest[j]]) for j in 1:n]
    row_start = cumsum(row_nnz) .- row_nnz # 0-based start offset of each row's block
    total = isempty(row_nnz) ? 0 : row_start[n] + row_nnz[n]
    rows = Vector{Int}(undef, total)
    cols = Vector{Int}(undef, total)
    vals = Vector{eltype(nodeset)}(undef, total)

    Threads.@threads for j in 1:n
        i = nearest[j]
        w = local_weights(basis, i, nodeset[j], op)
        w isa AbstractVector ||
            throw(ArgumentError("RBF-FD matrix assembly expects scalar operator values"))
        indices = stencils[i]
        base = row_start[j]
        for k in eachindex(indices)
            p = base + k
            rows[p] = j
            cols[p] = indices[k]
            vals[p] = w[k]
        end
    end

    return sparse(rows, cols, vals, n, m)
end

# Evaluation (resampling) matrix: row `j` maps the nodal values to the interpolant value
# at `nodeset[j]` using the nearest stencil. See the generic [`kernel_matrix`](@ref) for the
# meaning shared with the other bases.
function kernel_matrix(basis::RBFFDBasis, nodeset::NodeSet = centers(basis))
    return _rbf_fd_sparse_matrix(Identity(), basis, nodeset)
end

@doc raw"""
    pde_matrix(diff_op_or_pde, nodeset, basis::RBFFDBasis)

Assemble the sparse RBF-FD operator matrix. Each row `j` corresponds to one node in
`nodeset` and contains the local stencil weights of the differential operator (or PDE)
`diff_op_or_pde`, evaluated at `nodeset[j]` using the nearest center's stencil in `basis`.
The result is an `|nodeset| × |basis|` sparse matrix, so `nodeset` may be any set of
evaluation points and need not equal the centers of the [`RBFFDBasis`](@ref) `basis`.

See also [`differentiation_matrix`](@ref), [`pde_boundary_matrix`](@ref).
"""
function pde_matrix(diff_op_or_pde, nodeset::NodeSet, basis::RBFFDBasis)
    return _rbf_fd_sparse_matrix(diff_op_or_pde, basis, nodeset)
end

# For `RBFFDBasis` `pde_matrix` and `differentiation_matrix` coincide. See the generic
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
