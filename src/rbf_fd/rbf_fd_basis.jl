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
    RBFFDBasis(nodeset, kernel, stencil_selection;
               m = order(kernel), local_basis = RBFFDStandardBasis())

Configuration object for RBF-FD discretizations.

`RBFFDBasis` stores the global node set, kernel, and stencil strategy used to build
local RBF-FD weights. All stencils and local basis functions are precomputed at
construction time and stored for reuse in weight computation, matrix assembly, and
interpolant evaluation.

The `local_funcs` field always stores Lagrange cardinal functions (one per stencil
node). The weight computation algorithm is chosen separately via the `local_basis`
argument to [`rbf_fd_weights`](@ref) or the [`RBFFD`](@ref) method object.

Use `basis[i, k]` to access the `k`-th local cardinal function on the stencil around
center index `i`.
"""
struct RBFFDBasis{Dim, RealT, Kernel, Stencil <: AbstractStencilSelection, F}
    nodeset::NodeSet{Dim, RealT}
    kernel::Kernel
    stencil_selection::Stencil
    m::Int
    local_funcs::Vector{Vector{F}}
    stencil_indices::Vector{Vector{Int}}

    function RBFFDBasis(nodeset::NodeSet{Dim, RealT}, kernel::Kernel,
                        stencil_selection::Stencil;
                        m::Int = order(kernel)) where {
                                                       Dim, RealT,
                                                       Kernel <: AbstractKernel{Dim},
                                                       Stencil <: AbstractStencilSelection
                                                       }
        m >= 0 || throw(ArgumentError("m must be >= 0, got $m"))
        n = length(nodeset)

        neigh1 = select_neighbors(1, nodeset, stencil_selection)
        funcs1 = _build_local_funcs(kernel, neigh1.nodes, m)
        F = eltype(funcs1)

        stencil_indices_vec = Vector{Vector{Int}}(undef, n)
        local_funcs_vec = Vector{Vector{F}}(undef, n)
        stencil_indices_vec[1] = neigh1.indices
        local_funcs_vec[1] = funcs1

        for i in 2:n
            neigh = select_neighbors(i, nodeset, stencil_selection)
            stencil_indices_vec[i] = neigh.indices
            local_funcs_vec[i] = _build_local_funcs(kernel, neigh.nodes, m)
        end

        return new{Dim, RealT, Kernel, Stencil, F}(nodeset, kernel, stencil_selection, m,
                                                   local_funcs_vec, stencil_indices_vec)
    end
end

interpolation_kernel(basis::RBFFDBasis) = basis.kernel
centers(basis::RBFFDBasis) = basis.nodeset
dim(::RBFFDBasis{Dim}) where {Dim} = Dim
Base.eltype(::RBFFDBasis{Dim, RealT}) where {Dim, RealT} = RealT
Base.length(basis::RBFFDBasis) = length(basis.nodeset)
order(basis::RBFFDBasis) = basis.m

function _build_local_funcs(kernel::AbstractKernel, stencil_nodes::NodeSet, m::Int)
    return collect(LagrangeBasis(stencil_nodes, kernel; m))
end

"""
    getindex(basis::RBFFDBasis, i::Integer, k::Integer)

Return the `k`-th local stencil basis function associated with center index `i`.
The returned object is callable.
"""
function Base.getindex(basis::RBFFDBasis, i::Integer, k::Integer)
    (1 <= i <= length(basis.nodeset)) || throw(BoundsError(basis, (i, k)))
    (1 <= k <= length(basis.local_funcs[i])) || throw(BoundsError(basis, (i, k)))
    return basis.local_funcs[i][k]
end

function Base.show(io::IO, basis::RBFFDBasis)
    print(io,
          "RBFFDBasis with $(length(basis.nodeset)) nodes, kernel $(basis.kernel), stencil $(basis.stencil_selection)")
    return nothing
end
