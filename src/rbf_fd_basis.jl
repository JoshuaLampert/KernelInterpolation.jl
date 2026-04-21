"""
    RBFFDBasis(nodeset, kernel, stencil_selection = KNearestNeighbors();
               m = order(kernel))

Configuration object for RBF-FD discretizations.

`RBFFDBasis` stores the global node set, kernel, and stencil strategy used to build
local RBF-FD weights.
"""
struct RBFFDBasis{Dim, RealT, Kernel, Stencil <: AbstractStencilSelection}
    nodeset::NodeSet{Dim, RealT}
    kernel::Kernel
    stencil_selection::Stencil
    m::Int

    function RBFFDBasis(nodeset::NodeSet{Dim, RealT}, kernel::Kernel,
                        stencil_selection::Stencil = KNearestNeighbors();
                        m::Int = order(kernel)) where {
                                                                                         Dim,
                                                                                         RealT,
                                                                                         Kernel <: AbstractKernel{Dim},
                                                                                         Stencil <: AbstractStencilSelection
                                                                                         }
        m >= 0 || throw(ArgumentError("m must be >= 0, got $m"))
        return new{Dim, RealT, Kernel, Stencil}(nodeset, kernel, stencil_selection, m)
    end
end

interpolation_kernel(basis::RBFFDBasis) = basis.kernel
centers(basis::RBFFDBasis) = basis.nodeset
dim(::RBFFDBasis{Dim}) where {Dim} = Dim
Base.eltype(::RBFFDBasis{Dim, RealT}) where {Dim, RealT} = RealT
Base.length(basis::RBFFDBasis) = length(basis.nodeset)

function Base.show(io::IO, basis::RBFFDBasis)
    print(io,
          "RBFFDBasis with $(length(basis.nodeset)) nodes, kernel $(basis.kernel), stencil $(basis.stencil_selection)")
    return nothing
end
