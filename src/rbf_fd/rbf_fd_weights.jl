# ==================== Local Weight Computation ====================

# Build a weight vector (scalar operator output) or matrix (vector operator output, one
# column per component) by stacking the per-row values produced by `f`. `f(j)` must return
# either a `Number` or an `AbstractVector`, consistently across `j`.
function _stack_rows(f, n::Integer)
    first_value = f(1)
    if first_value isa Number
        out = Vector{typeof(first_value)}(undef, n)
        out[1] = first_value
        for j in 2:n
            out[j] = f(j)
        end
        return out
    elseif first_value isa AbstractVector
        d = length(first_value)
        out = Matrix{eltype(first_value)}(undef, n, d)
        out[1, :] .= first_value
        for j in 2:n
            out[j, :] .= f(j)
        end
        return out
    else
        throw(ArgumentError("Unsupported operator output type $(typeof(first_value)). Expected Number or AbstractVector."))
    end
end

"""
    local_weights(basis::RBFFDBasis, i, point, op)

Compute the local RBF-FD weights for stencil `i`, i.e. the coefficients that map the nodal
values on that stencil to `op` applied to the local interpolant, evaluated at `point`. Use
[`Identity`](@ref) for plain evaluation, or any differential operator/equation otherwise.

Dispatches on `basis.local_basis`: for [`RBFFDLagrangeBasis`](@ref) the (operator applied
to the) precomputed cardinal functions are evaluated at `point`; for
[`RBFFDStandardBasis`](@ref) the cached local system is solved with the corresponding
right-hand side.

Returns a weight vector (scalar operator) or matrix with one column per component (vector
operator).
"""
function local_weights(basis::RBFFDBasis, i::Integer, point::AbstractVector, op)
    return _local_weights(basis, basis.local_basis, i, point, op)
end

function _local_weights(basis::RBFFDBasis, ::RBFFDLagrangeBasis, i, point, op)
    funcs = basis.cache[i]
    return _stack_rows(k -> op(funcs[k], point), length(funcs))
end

function _local_weights(basis::RBFFDBasis, ::RBFFDStandardBasis, i, point, op)
    indices = basis.stencil_indices[i]
    X = centers(basis)
    kernel = basis.kernel
    ps = basis.ps
    nk = length(indices)
    q = length(ps)

    rhs_kernel = _stack_rows(k -> op(kernel, point, X[indices[k]]), nk)
    if q > 0
        rhs_poly = _stack_rows(l -> op(ps[l], point), q)
        rhs = vcat(rhs_kernel, rhs_poly)
    else
        rhs = rhs_kernel
    end

    sol = basis.cache[i] \ rhs
    return sol isa AbstractVector ? sol[1:nk] : sol[1:nk, :]
end

@doc raw"""
    rbf_fd_weights(diff_op_or_pde, i, basis::RBFFDBasis)

Compute RBF-FD weights for node index `i` using the precomputed stencil data from `basis`.
The algorithm is selected by `basis.local_basis` (see [`RBFFDLagrangeBasis`](@ref) and
[`RBFFDStandardBasis`](@ref)); both yield the same weights mathematically.

Returns the weight vector (scalar operator) or matrix (vector operator).

See also [`local_weights`](@ref).
"""
function rbf_fd_weights(diff_op_or_pde, i::Integer, basis::RBFFDBasis)
    return local_weights(basis, i, centers(basis)[i], diff_op_or_pde)
end
