"""
    RBFFDBasis(nodeset, kernel, stencil_selection;
               m = order(kernel), local_basis = RBFFDStandardBasis())

Configuration object for RBF-FD discretizations.

`RBFFDBasis` stores the global node set, kernel, and stencil strategy used to build
local RBF-FD weights.

Use `basis[i, k]` to access the `k`-th local basis function on the stencil around
center index `i`.
"""
struct RBFFDBasis{Dim, RealT, Kernel, Stencil <: AbstractStencilSelection,
                  LocalBasis <: AbstractRBFFDLocalBasis}
    nodeset::NodeSet{Dim, RealT}
    kernel::Kernel
    stencil_selection::Stencil
    m::Int
    local_basis::LocalBasis

    function RBFFDBasis(nodeset::NodeSet{Dim, RealT}, kernel::Kernel,
                        stencil_selection::Stencil;
                        m::Int = order(kernel),
                        local_basis::LocalBasis = RBFFDStandardBasis()) where {
                                                                               Dim,
                                                                               RealT,
                                                                               Kernel <:
                                                                               AbstractKernel{Dim},
                                                                               Stencil <:
                                                                               AbstractStencilSelection,
                                                                               LocalBasis <:
                                                                               AbstractRBFFDLocalBasis
                                                                               }
        m >= 0 || throw(ArgumentError("m must be >= 0, got $m"))
        return new{Dim, RealT, Kernel, Stencil, LocalBasis}(nodeset, kernel,
                                                            stencil_selection, m,
                                                            local_basis)
    end
end

interpolation_kernel(basis::RBFFDBasis) = basis.kernel
centers(basis::RBFFDBasis) = basis.nodeset
dim(::RBFFDBasis{Dim}) where {Dim} = Dim
Base.eltype(::RBFFDBasis{Dim, RealT}) where {Dim, RealT} = RealT
Base.length(basis::RBFFDBasis) = length(basis.nodeset)

function _rbffd_local_basis_function(basis::RBFFDBasis,
                                     stencil_nodes::NodeSet,
                                     k::Int,
                                     ::RBFFDStandardBasis)
    return x -> basis.kernel(x, stencil_nodes[k])
end

function _rbffd_local_basis_function(basis::RBFFDBasis,
                                     stencil_nodes::NodeSet,
                                     k::Int,
                                     ::RBFFDLagrangeBasis)
    local_basis = LagrangeBasis(stencil_nodes, basis.kernel; m = basis.m)
    return local_basis[k]
end

"""
    getindex(basis::RBFFDBasis, i::Integer, k::Integer)

Return the `k`-th local stencil basis function associated with center index `i`.
The returned object is callable.
"""
function Base.getindex(basis::RBFFDBasis, i::Integer, k::Integer)
    (1 <= i <= length(basis.nodeset)) || throw(BoundsError(basis, (i, k)))
    x_i = basis.nodeset[i]
    neighbor_info = select_neighbors(x_i, basis.nodeset, basis.stencil_selection)
    (1 <= k <= length(neighbor_info.nodes)) || throw(BoundsError(basis, (i, k)))
    return _rbffd_local_basis_function(basis, neighbor_info.nodes, k, basis.local_basis)
end

function Base.show(io::IO, basis::RBFFDBasis)
    print(io,
          "RBFFDBasis with $(length(basis.nodeset)) nodes, kernel $(basis.kernel), stencil $(basis.stencil_selection), local basis $(basis.local_basis)")
    return nothing
end
