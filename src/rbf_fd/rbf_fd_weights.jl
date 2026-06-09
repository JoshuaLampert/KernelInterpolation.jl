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

function _operator_on_polynomial(op::PartialDerivative, p, xx, x)
    return ForwardDiff.gradient(y -> p(xx => y), x)[op.i]
end

function _operator_on_polynomial(::Laplacian, p, xx, x)
    return tr(ForwardDiff.hessian(y -> p(xx => y), x))
end

function _operator_on_polynomial(op::EllipticOperator, p, xx, x)
    @unpack A, b, c = op
    AA = A(x)
    bb = b(x)
    cc = c(x)
    H = ForwardDiff.hessian(y -> p(xx => y), x)
    gr = ForwardDiff.gradient(y -> p(xx => y), x)
    return sum(-AA[i, j] * H[i, j] for i in eachindex(gr), j in eachindex(gr)) +
           sum(bb[i] * gr[i] for i in eachindex(gr)) +
           cc * p(xx => x)
end

function _operator_on_polynomial(op::PoissonEquation, p, xx, x)
    return -_operator_on_polynomial(Laplacian(), p, xx, x)
end
function _operator_on_polynomial(op::EllipticEquation, p, xx, x)
    return _operator_on_polynomial(op.op, p, xx, x)
end
function _operator_on_polynomial(op::AdvectionEquation, p, xx, x)
    return dot(op.advection_velocity, ForwardDiff.gradient(y -> p(xx => y), x))
end
function _operator_on_polynomial(op::HeatEquation, p, xx, x)
    return -op.diffusivity * _operator_on_polynomial(Laplacian(), p, xx, x)
end
function _operator_on_polynomial(op::AdvectionDiffusionEquation, p, xx, x)
    gr = ForwardDiff.gradient(y -> p(xx => y), x)
    return dot(op.advection_velocity, gr) -
           op.diffusivity * _operator_on_polynomial(Laplacian(), p, xx, x)
end

function _operator_on_polynomial(op, p, xx, x)
    throw(ArgumentError("Polynomial augmentation is not implemented for operator type $(typeof(op))."))
end

function _polynomial_rhs(diff_op_or_pde, ps, xx, x_i)
    first_value = _operator_on_polynomial(diff_op_or_pde, ps[1], xx, x_i)
    if first_value isa Number
        rhs_p = Vector{typeof(first_value)}(undef, length(ps))
        rhs_p[1] = first_value
        for j in 2:length(ps)
            rhs_p[j] = _operator_on_polynomial(diff_op_or_pde, ps[j], xx, x_i)
        end
        return rhs_p
    elseif first_value isa AbstractVector
        d = length(first_value)
        rhs_p = Matrix{eltype(first_value)}(undef, length(ps), d)
        rhs_p[1, :] .= first_value
        for j in 2:length(ps)
            rhs_p[j, :] .= _operator_on_polynomial(diff_op_or_pde, ps[j], xx, x_i)
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
    rbf_fd_weights(diff_op_or_pde, i, basis::RBFFDBasis)

Compute RBF-FD finite difference weights for node index `i` using precomputed stencil
data from `basis`. Dispatches on the local basis type stored in `basis`.

# Returns
- `weights`: Finite difference weights (vector for scalar operators, matrix for vector operators)
- `info::NamedTuple`: Diagnostic information (stencil size, condition number, rank, SVD values)
"""
function rbf_fd_weights(diff_op_or_pde, i::Integer, basis::RBFFDBasis)
    return _rbf_fd_weights(diff_op_or_pde, i, basis, basis.local_basis)
end

function _rbf_fd_weights(diff_op_or_pde, i, basis::RBFFDBasis, ::RBFFDLagrangeBasis)
    x_i = basis.nodeset[i]
    indices = basis.stencil_indices[i]
    neighbor_nodes = NodeSet(basis.nodeset.nodes[indices])
    weights = _rbf_fd_cardinal_weights(diff_op_or_pde, x_i, basis.local_funcs[i])
    k_matrix = kernel_matrix(neighbor_nodes, basis.kernel)
    svals = svdvals(k_matrix)
    rank_tol = eps(eltype(svals)) * maximum(svals)
    rank_est = count(>(rank_tol), svals)
    info = (stencil_size = length(neighbor_nodes),
            condition_number = cond(k_matrix),
            rank = rank_est,
            singular_values = svals,
            x_i = x_i,
            local_basis = basis.local_basis)
    return weights, info
end

function _rbf_fd_weights(diff_op_or_pde, i, basis::RBFFDBasis, ::RBFFDStandardBasis)
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
        rhs_poly = _polynomial_rhs(diff_op_or_pde, ps, xx, x_i)
        sol = A_aug \ [rhs_vec; rhs_poly]
        weights = sol isa AbstractVector ? sol[1:length(neighbor_nodes)] :
                  sol[1:length(neighbor_nodes), :]
    else
        weights = A \ rhs_vec
    end

    svals = svdvals(A)
    rank_tol = eps(eltype(svals)) * maximum(svals)
    rank_est = count(>(rank_tol), svals)
    info = (stencil_size = length(neighbor_nodes),
            condition_number = cond(A),
            rank = rank_est,
            singular_values = svals,
            x_i = x_i,
            local_basis = basis.local_basis)
    return weights, info
end

"""
    rbf_fd_weights_at_node(diff_op_or_pde, x_i, basis::RBFFDBasis)

Compute RBF-FD weights at an arbitrary point `x_i` using the stencil of the nearest
node in `basis`. Returns `(weights, info)`.
"""
function rbf_fd_weights_at_node(diff_op_or_pde, x_i::AbstractVector, basis::RBFFDBasis)
    i = nearest_node_index(x_i, basis.nodeset)
    return rbf_fd_weights(diff_op_or_pde, i, basis)
end

# ==================== Batch Weight Computation ====================

"""
    rbf_fd_weights_all_nodes(diff_op_or_pde, basis::RBFFDBasis)

Compute RBF-FD weights for all nodes in `basis`. Returns a `Dict` mapping each node
index to its `(weights, info)` tuple.
"""
function rbf_fd_weights_all_nodes(diff_op_or_pde, basis::RBFFDBasis)
    return Dict(i => rbf_fd_weights(diff_op_or_pde, i, basis)
                for i in eachindex(basis.nodeset))
end
