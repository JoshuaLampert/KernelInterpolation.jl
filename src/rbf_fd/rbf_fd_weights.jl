# ==================== Local Weight Computation ====================

function _collect_operator_rhs(diff_op_or_pde, kernel, x_i, neighbor_nodes)
    first_value = diff_op_or_pde(kernel, x_i, neighbor_nodes[1])
    if first_value isa Number
        rhs = Vector{typeof(first_value)}(undef, length(neighbor_nodes))
        rhs[1] = first_value
        for j in 2:length(neighbor_nodes)
            rhs[j] = diff_op_or_pde(kernel, x_i, neighbor_nodes[j])
        end
        return rhs
    elseif first_value isa AbstractVector
        d = length(first_value)
        rhs = Matrix{eltype(first_value)}(undef, length(neighbor_nodes), d)
        rhs[1, :] .= first_value
        for j in 2:length(neighbor_nodes)
            rhs[j, :] .= diff_op_or_pde(kernel, x_i, neighbor_nodes[j])
        end
        return rhs
    else
        throw(ArgumentError("Unsupported operator output type $(typeof(first_value)). Expected Number or AbstractVector."))
    end
end

function _polynomial_rhs(diff_op_or_pde, ps, x_i)
    first_value = diff_op_or_pde(ps[1], x_i)
    if first_value isa Number
        rhs_p = Vector{typeof(first_value)}(undef, length(ps))
        rhs_p[1] = first_value
        for j in 2:length(ps)
            rhs_p[j] = diff_op_or_pde(ps[j], x_i)
        end
        return rhs_p
    elseif first_value isa AbstractVector
        d = length(first_value)
        rhs_p = Matrix{eltype(first_value)}(undef, length(ps), d)
        rhs_p[1, :] .= first_value
        for j in 2:length(ps)
            rhs_p[j, :] .= diff_op_or_pde(ps[j], x_i)
        end
        return rhs_p
    else
        throw(ArgumentError("Unsupported polynomial operator output type $(typeof(first_value))."))
    end
end

function _rbf_fd_cardinal_weights(diff_op_or_pde, x_i::AbstractVector,
                                  local_funcs::AbstractVector)
    first_value = diff_op_or_pde(local_funcs[1], x_i)
    if first_value isa Number
        weights = Vector{typeof(first_value)}(undef, length(local_funcs))
        weights[1] = first_value
        for j in 2:length(local_funcs)
            weights[j] = diff_op_or_pde(local_funcs[j], x_i)
        end
        return weights
    elseif first_value isa AbstractVector
        d = length(first_value)
        weights = Matrix{eltype(first_value)}(undef, length(local_funcs), d)
        weights[1, :] .= first_value
        for j in 2:length(local_funcs)
            weights[j, :] .= diff_op_or_pde(local_funcs[j], x_i)
        end
        return weights
    else
        throw(ArgumentError("Unsupported operator output type $(typeof(first_value)) for cardinal RBF-FD weights."))
    end
end

@doc raw"""
    rbf_fd_weights(diff_op_or_pde, i, basis::RBFFDBasis, local_basis = basis.local_basis)

Compute RBF-FD weights for node index `i` using precomputed stencil data from `basis`.
Dispatches on the local basis type stored in `basis`.
If `local_basis` is [`RBFFDLagrangeBasis`](@ref), weights are computed via the cardinal
function approach. If `local_basis` is [`RBFFDStandardBasis`](@ref), the local
kernel/polynomial system is solved directly.

Returns the weight vector (scalar operator) or matrix (vector operator).
"""
function rbf_fd_weights(diff_op_or_pde, i::Integer, basis::RBFFDBasis,
                        local_basis::AbstractRBFFDLocalBasis = basis.local_basis)
    return _rbf_fd_weights(diff_op_or_pde, i, basis, local_basis)
end

function _rbf_fd_weights(diff_op_or_pde, i, basis::RBFFDBasis,
                         ::RBFFDLagrangeBasis)
    x_i = basis.nodeset[i]
    return _rbf_fd_cardinal_weights(diff_op_or_pde, x_i, basis.local_funcs[i])
end

function _rbf_fd_weights(diff_op_or_pde, i, basis::RBFFDBasis,
                         ::RBFFDStandardBasis)
    x_i = basis.nodeset[i]
    indices = basis.stencil_indices[i]
    neighbor_nodes = NodeSet(basis.nodeset.nodes[indices])
    A = kernel_matrix(neighbor_nodes, basis.kernel)
    rhs_vec = _collect_operator_rhs(diff_op_or_pde, basis.kernel, x_i, neighbor_nodes)

    if basis.m > 0
        xx = polyvars(dim(neighbor_nodes))
        ps = monomials(xx, 0:(basis.m - 1))
        q = length(ps)
        P = polynomial_matrix(neighbor_nodes, ps)
        A_aug = [A P
                 P' zeros(eltype(A), q, q)]
        rhs_poly = _polynomial_rhs(diff_op_or_pde, ps, x_i)
        sol = A_aug \ [rhs_vec; rhs_poly]
        return sol isa AbstractVector ? sol[1:length(neighbor_nodes)] :
               sol[1:length(neighbor_nodes), :]
    else
        return A \ rhs_vec
    end
end

"""
    rbf_fd_weights_at_node(diff_op_or_pde, x_i, basis::RBFFDBasis)

Compute RBF-FD weights at an arbitrary point `x_i` using the stencil of the nearest
node in `basis`.

See [`rbf_fd_weights`](@ref) for more details.
"""
function rbf_fd_weights_at_node(diff_op_or_pde, x_i::AbstractVector, basis::RBFFDBasis,
                                local_basis::AbstractRBFFDLocalBasis = basis.local_basis)
    i = nearest_node_index(x_i, basis.nodeset)
    return rbf_fd_weights(diff_op_or_pde, i, basis, local_basis)
end

# ==================== Batch Weight Computation ====================

"""
    rbf_fd_weights_all_nodes(diff_op_or_pde, basis::RBFFDBasis)

Compute RBF-FD weights for all nodes in `basis`. Returns a `Dict` mapping each node
index to its weight vector (or matrix for vector operators).

See [`rbf_fd_weights`](@ref) for more details.
"""
function rbf_fd_weights_all_nodes(diff_op_or_pde, basis::RBFFDBasis,
                                  local_basis::AbstractRBFFDLocalBasis = basis.local_basis)
    return Dict(i => rbf_fd_weights(diff_op_or_pde, i, basis, local_basis)
                for i in eachindex(basis.nodeset))
end
