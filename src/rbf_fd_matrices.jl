"""
    rbf_fd_pde_matrix(diff_op_or_pde, nodeset_inner, basis)

Assemble sparse RBF-FD operator matrix for inner nodes.
Each row corresponds to one inner node and contains local stencil weights.
"""
function rbf_fd_pde_matrix(diff_op_or_pde, nodeset_inner::NodeSet,
                           basis::RBFFDBasis)
    @unpack nodeset, kernel, stencil_selection, m, local_basis = basis

    n_inner = length(nodeset_inner)
    n_total = length(nodeset)
    rows = Int[]
    cols = Int[]
    vals = eltype(nodeset)[]

    for i in eachindex(nodeset_inner)
        x_i = nodeset_inner[i]
        neighbor_info = select_neighbors(x_i, nodeset, stencil_selection)
        weights, _ = rbf_fd_weights(diff_op_or_pde, x_i, neighbor_info.nodes, kernel;
                                    m, local_basis)

        weights isa AbstractVector ||
            throw(ArgumentError("RBF-FD PDE assembly expects scalar operator values"))

        for (j, global_idx) in enumerate(neighbor_info.indices)
            push!(rows, i)
            push!(cols, global_idx)
            push!(vals, weights[j])
        end
    end

    return sparse(rows, cols, vals, n_inner, n_total)
end

"""
    rbf_fd_boundary_matrix(nodeset_inner, nodeset_boundary)

Assemble sparse boundary constraint matrix for Dirichlet boundary conditions.
Unknowns are nodal values ordered as `merge(nodeset_inner, nodeset_boundary)`.
"""
function rbf_fd_boundary_matrix(nodeset_inner::NodeSet, nodeset_boundary::NodeSet)
    n_inner = length(nodeset_inner)
    n_boundary = length(nodeset_boundary)
    n_total = n_inner + n_boundary

    rows = collect(1:n_boundary)
    cols = collect((n_inner + 1):n_total)
    vals = fill(one(eltype(nodeset_inner)), n_boundary)

    return sparse(rows, cols, vals, n_boundary, n_total)
end

"""
    rbf_fd_pde_boundary_matrix(diff_op_or_pde, nodeset_inner, nodeset_boundary, basis)

Assemble full sparse RBF-FD PDE + boundary matrix:
```math
A = \begin{pmatrix}L\\B\end{pmatrix}
```
where `L` is the RBF-FD operator matrix on inner nodes and `B` enforces Dirichlet constraints.
"""
function rbf_fd_pde_boundary_matrix(diff_op_or_pde, nodeset_inner::NodeSet,
                                    nodeset_boundary::NodeSet, basis::RBFFDBasis)
    L = rbf_fd_pde_matrix(diff_op_or_pde, nodeset_inner, basis)
    B = rbf_fd_boundary_matrix(nodeset_inner, nodeset_boundary)
    return [L
            B]
end

function pde_boundary_matrix(diff_op_or_pde, nodeset_inner::NodeSet,
                             nodeset_boundary::NodeSet, basis::RBFFDBasis)
    return rbf_fd_pde_boundary_matrix(diff_op_or_pde, nodeset_inner,
                                      nodeset_boundary, basis)
end

function operator_matrix(diff_op_or_pde, nodeset_inner::NodeSet,
                         nodeset_boundary::NodeSet, basis::RBFFDBasis)
    return pde_boundary_matrix(diff_op_or_pde, nodeset_inner, nodeset_boundary, basis)
end
