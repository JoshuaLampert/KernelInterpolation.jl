"""
    AbstractStencilSelection

Abstract type for stencil selection strategies in RBF-FD. Defines how neighboring points
are selected for each local finite difference stencil.
"""
abstract type AbstractStencilSelection end

"""
    AbstractRBFFDLocalBasis

Abstract type selecting the local basis representation used to compute RBF-FD weights.
"""
abstract type AbstractRBFFDLocalBasis end

"""
    RBFFDStandardBasis()

Compute local RBF-FD weights by solving the local kernel (or kernel+polynomial) system.
"""
struct RBFFDStandardBasis <: AbstractRBFFDLocalBasis end
Base.show(io::IO, ::RBFFDStandardBasis) = print(io, "RBFFDStandardBasis")

"""
    RBFFDLagrangeBasis()

Compute local RBF-FD weights from local cardinal (Lagrange) basis functions
on each stencil, i.e., `w_j = 𝓛 ℓ_j(x_i)`.
"""
struct RBFFDLagrangeBasis <: AbstractRBFFDLocalBasis end
Base.show(io::IO, ::RBFFDLagrangeBasis) = print(io, "RBFFDLagrangeBasis")

"""
    KNearestNeighbors(k::Int)

Stencil selection strategy using k-nearest neighbors. Each interior point uses its `k` nearest
neighbors (by Euclidean distance) to form the local FD stencil. This ensures uniform sparsity.

# Arguments
- `k::Int`: Number of nearest neighbors to use (default: 25)
"""
struct KNearestNeighbors <: AbstractStencilSelection
    k::Int
    function KNearestNeighbors(k::Int)
        k ≥ 1 || throw(ArgumentError("k must be ≥ 1, got $k"))
        return new(k)
    end
end

KNearestNeighbors() = KNearestNeighbors(25)

Base.show(io::IO, ss::KNearestNeighbors) = print(io, "KNearestNeighbors(k=$(ss.k))")

"""
    RadiusSearch(radius::Float64)

Stencil selection strategy using fixed radius search. Each interior point uses all neighbors
within a given Euclidean distance `radius` to form the local FD stencil. This allows variable
stencil sizes but adapts to local point density.

# Arguments
- `radius::Float64`: Search radius for neighbor selection
"""
struct RadiusSearch <: AbstractStencilSelection
    radius::Float64
    function RadiusSearch(radius::Float64)
        radius > 0.0 || throw(ArgumentError("radius must be > 0, got $radius"))
        return new(radius)
    end
end

Base.show(io::IO, ss::RadiusSearch) = print(io, "RadiusSearch(r=$(ss.radius))")

# ==================== Neighbor Selection ====================

"""
    select_neighbors(i, nodeset, stencil_selection::KNearestNeighbors)

Select k nearest neighbors of `nodeset[i]` from `nodeset` using Euclidean distance.
Returns a tuple of (neighbor_indices, neighbor_nodes).
"""
function select_neighbors(i::Int, nodeset::NodeSet, stencil::KNearestNeighbors)
    k = stencil.k
    n = length(nodeset)
    k ≤ n || throw(ArgumentError("k=$(k) exceeds nodeset size $(n)"))
    x_i = nodeset[i]

    # Compute Euclidean distances to all nodes and keep center point in stencil.
    distances = [norm(x_i .- nodeset[j]) for j in 1:n]
    neighbor_indices = sortperm(distances)[1:k]

    neighbor_nodes = NodeSet([nodeset[j] for j in neighbor_indices])
    return (indices = neighbor_indices, nodes = neighbor_nodes)
end

"""
    select_neighbors(i, nodeset, stencil_selection::RadiusSearch)

Select all neighbors of `nodeset[i]` from `nodeset` within the search radius.
Returns a tuple of (neighbor_indices, neighbor_nodes).
"""
function select_neighbors(i::Int, nodeset::NodeSet, stencil::RadiusSearch)
    radius = stencil.radius
    x_i = nodeset[i]
    neighbor_indices = Int[]
    neighbor_nodes_list = []

    for j in eachindex(nodeset)
        dist = norm(x_i .- nodeset[j])
        if dist <= radius
            push!(neighbor_indices, j)
            push!(neighbor_nodes_list, nodeset[j])
        end
    end

    isempty(neighbor_indices) && throw(ArgumentError("No neighbors found within radius $(radius) for point index $(i)"))

    neighbor_nodes = NodeSet(neighbor_nodes_list)
    return (indices = neighbor_indices, nodes = neighbor_nodes)
end

function select_neighbors(x_i::AbstractVector, nodeset::NodeSet,
                          stencil::KNearestNeighbors)
    distances = [norm(x_i .- nodeset[j]) for j in eachindex(nodeset)]
    k = stencil.k
    k ≤ length(nodeset) || throw(ArgumentError("k=$(k) exceeds nodeset size $(length(nodeset))"))
    neighbor_indices = sortperm(distances)[1:k]
    neighbor_nodes = NodeSet([nodeset[j] for j in neighbor_indices])
    return (indices = neighbor_indices, nodes = neighbor_nodes)
end

function select_neighbors(x_i::AbstractVector, nodeset::NodeSet,
                          stencil::RadiusSearch)
    neighbor_indices = Int[]
    neighbor_nodes_list = []
    for j in eachindex(nodeset)
        if norm(x_i .- nodeset[j]) <= stencil.radius
            push!(neighbor_indices, j)
            push!(neighbor_nodes_list, nodeset[j])
        end
    end
    isempty(neighbor_indices) && throw(ArgumentError("No neighbors found within radius $(stencil.radius)"))
    neighbor_nodes = NodeSet(neighbor_nodes_list)
    return (indices = neighbor_indices, nodes = neighbor_nodes)
end

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

_operator_on_polynomial(op::PoissonEquation, p, xx, x) = -_operator_on_polynomial(Laplacian(), p, xx, x)
_operator_on_polynomial(op::EllipticEquation, p, xx, x) = _operator_on_polynomial(op.op, p, xx, x)
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
                                  neighbor_nodes::NodeSet, kernel::AbstractKernel;
                                  m::Int = order(kernel))
    local_basis = LagrangeBasis(neighbor_nodes, kernel; m)
    first_value = diff_op_or_pde(local_basis[1], x_i)
    if first_value isa Number
        weights = Vector{typeof(first_value)}(undef, length(local_basis))
        weights[1] = first_value
        for j in 2:length(local_basis)
            weights[j] = diff_op_or_pde(local_basis[j], x_i)
        end
        return weights
    elseif first_value isa AbstractVector
        d = length(first_value)
        weights = Matrix{eltype(first_value)}(undef, length(local_basis), d)
        weights[1, :] .= first_value
        for j in 2:length(local_basis)
            weights[j, :] .= diff_op_or_pde(local_basis[j], x_i)
        end
        return weights
    else
        throw(ArgumentError("Unsupported operator output type $(typeof(first_value)) for cardinal RBF-FD weights."))
    end
end

"""
    rbf_fd_weights(diff_op_or_pde, x_i, neighbor_nodes, kernel;
                   m = order(kernel), local_basis = RBFFDStandardBasis())

Compute RBF-FD finite difference weights for a given interior point and its stencil.

This function solves a local system to find weights `w` such that the finite difference
approximation \$\\mathcal{L}u_h(x_i) ≈ \\sum_{j ∈ stencil} w_j u_j\$ holds when:
- \$u_h(x) = \\sum_{j ∈ stencil} c_j \\phi(\\|x - center_j\\|)\$ is the local kernel interpolant
- \$\\mathcal{L}\$ is the differential operator

# Arguments
- `diff_op_or_pde`: Differential operator or equation callable like `op(kernel, x, y)`
- `x_i::AbstractVector`: Target interior point where operator is approximated
- `neighbor_nodes::NodeSet`: Nodes in the local stencil
- `kernel::AbstractKernel`: Kernel used for local interpolation
- `m::Int`: Polynomial order parameter. If `m = 0`, no polynomial is added. If `m > 0`,
  monomials up to degree `m - 1` are included.
- `local_basis::AbstractRBFFDLocalBasis`: Local basis strategy (`RBFFDStandardBasis()` or
    `RBFFDLagrangeBasis()`).

# Returns
- `weights`: Finite difference weights, vector for scalar operators and matrix for vector operators
- `info::NamedTuple`: Diagnostic information

# Notes
- For standard RBF-FD: uses direct kernel evaluations only
- For polyharmonic RBF-FD: augments with low-degree polynomials for better conditioning
- The order of polynomial augmentation is determined by the kernel's `order()` property
"""
function rbf_fd_weights(diff_op_or_pde, x_i::AbstractVector,
                        neighbor_nodes::NodeSet, kernel::AbstractKernel;
                        m::Int = order(kernel),
                        local_basis::AbstractRBFFDLocalBasis = RBFFDStandardBasis())
    m >= 0 || throw(ArgumentError("m must be >= 0, got $m"))

    if local_basis isa RBFFDLagrangeBasis
        weights = _rbf_fd_cardinal_weights(diff_op_or_pde, x_i, neighbor_nodes, kernel; m)
        k_matrix = kernel_matrix(neighbor_nodes, kernel)
        svals = svdvals(k_matrix)
        rank_tol = eps(eltype(svals)) * maximum(svals)
        rank_est = count(>(rank_tol), svals)
        info = (
            stencil_size = length(neighbor_nodes),
            condition_number = cond(k_matrix),
            rank = rank_est,
            singular_values = svals,
            x_i = x_i,
            local_basis = local_basis
        )
        return weights, info
    end

    A = kernel_matrix(neighbor_nodes, kernel)
    rhs = _collect_operator_rhs(diff_op_or_pde, kernel, x_i, neighbor_nodes)

    if m > 0
        xx = polyvars(dim(neighbor_nodes))
        ps = monomials(xx, 0:(m - 1))
        q = length(ps)
        P = polynomial_matrix(neighbor_nodes, ps)
        A_aug = [A P
                 P' zeros(eltype(A), q, q)]
        rhs_poly = _polynomial_rhs(diff_op_or_pde, ps, xx, x_i)
        rhs_aug = [rhs
                   rhs_poly]
        sol = A_aug \ rhs_aug
        if sol isa AbstractVector
            weights = sol[1:length(neighbor_nodes)]
        else
            weights = sol[1:length(neighbor_nodes), :]
        end
    else
        weights = A \ rhs
    end

    svals = svdvals(A)
    rank_tol = eps(eltype(svals)) * maximum(svals)
    rank_est = count(>(rank_tol), svals)
    info = (
        stencil_size = length(neighbor_nodes),
        condition_number = cond(A),
        rank = rank_est,
        singular_values = svals,
        x_i = x_i,
        local_basis = local_basis
    )
    return weights, info
end

"""
    rbf_fd_weights_at_node(kernel, operator, x_i, nodeset, stencil_selection::AbstractStencilSelection;
                           center_nodes = nothing, m = order(kernel),
                           local_basis = RBFFDStandardBasis())

Convenience wrapper to compute FD weights at a single interior point with automatic neighbor selection.

# Arguments
- `kernel::AbstractKernel`: Radial basis kernel
- `operator::AbstractDifferentialOperator`: Differential operator
- `x_i::AbstractVector`: Interior point where operator is approximated
- `nodeset::NodeSet`: Full set of nodes
- `stencil_selection::AbstractStencilSelection`: Strategy for selecting neighbors (KNearestNeighbors or RadiusSearch)
- `center_nodes::NodeSet`: Nodes to use as kernel centers (default: neighbor nodes)
- `m::Int`: Polynomial order parameter. If `m = 0`, no polynomial is added.
- `local_basis::AbstractRBFFDLocalBasis`: Local basis strategy.

# Returns
- `weights::Vector`: Finite difference weights indexed by neighbor position in stencil
- `neighbor_info::NamedTuple`: Information about the neighbor selection
- `weight_info::NamedTuple`: Diagnostic information from weight computation
"""
function rbf_fd_weights_at_node(kernel::AbstractKernel,
                               operator,
                               x_i::AbstractVector, nodeset::NodeSet,
                               stencil_selection::AbstractStencilSelection;
                               center_nodes::NodeSet = nothing,
                               m::Int = order(kernel),
                               local_basis::AbstractRBFFDLocalBasis = RBFFDStandardBasis())

    # Select neighbors
    neighbor_info = select_neighbors(x_i, nodeset, stencil_selection)

    center_nodes === nothing || throw(ArgumentError("`center_nodes` is not supported in RBF-FD and should be omitted."))

    # Compute weights
    weights, weight_info = rbf_fd_weights(operator, x_i, neighbor_info.nodes, kernel;
                                          m, local_basis)

    return weights, neighbor_info, weight_info
end

# ==================== Batch Weight Computation ====================

"""
    rbf_fd_weights_all_nodes(kernel, operator, nodeset_interior, nodeset_centers,
                            stencil_selection; m = order(kernel),
                            local_basis = RBFFDStandardBasis())

Compute RBF-FD weights for all interior points. Returns a dictionary mapping each
interior node index to its (weights, neighbor_info, diagnostics).

This is useful for inspection and debugging, but for assembly of the global matrix,
the `rbf_fd_pde_matrix()` function is more suitable.

# Arguments
- All arguments same as `rbf_fd_weights_at_node`

# Returns
- `weight_dict::Dict`: Dictionary with interior node indices as keys and
  (weights, neighbor_info, weight_info) tuples as values
"""
function rbf_fd_weights_all_nodes(kernel::AbstractKernel,
                                 operator,
                                 nodeset_interior::NodeSet,
                                 nodeset_centers::NodeSet,
                                 stencil_selection::AbstractStencilSelection;
                                 m::Int = order(kernel),
                                 local_basis::AbstractRBFFDLocalBasis = RBFFDStandardBasis())

    weight_dict = Dict()

    for i in eachindex(nodeset_interior)
        x_i = nodeset_interior[i]
        weights, neighbor_info, weight_info = rbf_fd_weights_at_node(
            kernel, operator, x_i, nodeset_centers, stencil_selection;
            m, local_basis
        )
        weight_dict[i] = (weights, neighbor_info, weight_info)
    end

    return weight_dict
end
