"""
    AbstractRBFFDLocalBasis

Abstract type selecting the local basis representation used to compute RBF-FD weights.

See also [`RBFFDStandardBasis`](@ref) and [`RBFFDLagrangeBasis`](@ref).
"""
abstract type AbstractRBFFDLocalBasis end

"""
    RBFFDStandardBasis()

Compute local RBF-FD weights by solving the local kernel (or kernel+polynomial) system.
The per-stencil augmented system matrix is factorized once at construction and reused for
weight computation, matrix assembly, and interpolant evaluation.

See also [`AbstractRBFFDLocalBasis`](@ref) and [`RBFFDBasis`](@ref).
"""
struct RBFFDStandardBasis <: AbstractRBFFDLocalBasis end

"""
    RBFFDLagrangeBasis()

Compute local RBF-FD weights from local cardinal (Lagrange) basis functions
on each stencil, i.e., `w_j = 𝓛 ℓ_j(x_i)`. The cardinal functions are built once at
construction and reused for weight computation, matrix assembly, and interpolant
evaluation.

See also [`AbstractRBFFDLocalBasis`](@ref) and [`RBFFDBasis`](@ref).
"""
struct RBFFDLagrangeBasis <: AbstractRBFFDLocalBasis end

"""
    RBFFDBasis(nodeset, kernel, stencil_selection;
               m = order(kernel), local_basis = RBFFDLagrangeBasis())

Configuration object for RBF-FD discretizations.

`RBFFDBasis` stores the global node set, kernel, and stencil strategy used to build
local RBF-FD weights. All stencils are precomputed at construction time. What else is
precomputed is selected by `local_basis`:

- [`RBFFDLagrangeBasis`](@ref) (default): the Lagrange cardinal functions on each stencil
  are precomputed and stored. Weight computation, matrix assembly, and interpolant
  evaluation evaluate (operators applied to) these cardinal functions.
- [`RBFFDStandardBasis`](@ref): the factorization of each stencil's augmented kernel/
  polynomial system is precomputed and stored. The same operations are obtained by solving
  the local system with the appropriate right-hand side.

Both choices produce the same weights mathematically; they differ only in the numerical
route and in what `basis[i, k]` returns (see [`getindex`](@ref)).

Use `basis[i, k]` to access the `k`-th local basis function on the stencil around center
index `i`.

The polynomial augmentation is local to each stencil; its order is given by
[`local_order`](@ref) (and the monomials by [`polynomial_basis`](@ref)). No *global*
polynomials are added, so [`order`](@ref) of an `RBFFDBasis` is always `0`, consistent with
[`LagrangeBasis`](@ref).
"""
struct RBFFDBasis{Dim, RealT, Kernel, Stencil <: AbstractStencilSelection,
                  LocalBasis <: AbstractRBFFDLocalBasis, Polys, Cache} <: AbstractBasis
    centers::NodeSet{Dim, RealT}
    kernel::Kernel
    stencil_selection::Stencil
    stencil_indices::Vector{Vector{Int}}
    ps::Polys
    local_basis::LocalBasis
    cache::Cache

    function RBFFDBasis(centers::NodeSet{Dim, RealT}, kernel::Kernel,
                        stencil_selection::Stencil;
                        m::Int = order(kernel),
                        local_basis::LocalBasis = RBFFDLagrangeBasis()) where {
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
        n = length(centers)
        ps = _local_polynomial_basis(local_basis, polyvars(Dim), m)

        stencil_indices = Vector{Vector{Int}}(undef, n)
        neighbor_nodesets = Vector{NodeSet{Dim, RealT}}(undef, n)
        for i in 1:n
            neigh = select_neighbors(i, centers, stencil_selection)
            stencil_indices[i] = neigh.indices
            neighbor_nodesets[i] = neigh.nodes
        end

        cache = _build_local_cache(local_basis, kernel, neighbor_nodesets, ps, m)

        return new{Dim, RealT, Kernel, Stencil, LocalBasis, typeof(ps),
                   typeof(cache)}(centers, kernel, stencil_selection, stencil_indices,
                                  ps, local_basis, cache)
    end
end

# RBF-FD adds no *global* polynomial augmentation (it is handled locally per stencil, see
# `polynomials(::RBFFDBasis, xx)` in `discretization.jl`), so `order` is 0 just like for
# `LagrangeBasis`. The local augmentation degree is given by [`local_order`](@ref).
order(::RBFFDBasis) = 0

"""
    local_order(basis::RBFFDBasis)

Return the order `m` of the polynomial basis applied *separately* on each RBF-FD stencil,
of an [`RBFFDBasis`](@ref), i.e., the polynomial degree plus 1 (`0` if no separate
polynomials are used). This is distinct from [`order`](@ref), which is the (always zero)
global augmentation.

Like the global [`StandardBasis`](@ref)/[`LagrangeBasis`](@ref) distinction, this depends on
the local-basis policy: for [`RBFFDStandardBasis`](@ref) the polynomials are an explicit part
of the local system, so `local_order` is their order `m`; for [`RBFFDLagrangeBasis`](@ref)
they are baked into the cardinal functions, so there is no separate polynomial basis and
`local_order` is `0`.
"""
local_order(basis::RBFFDBasis) = maximum(degree.(basis.ps), init = -1) + 1

# The polynomial basis stored on the `RBFFDBasis`. For `RBFFDStandardBasis` the monomials of
# degree `< m` are an explicit part of the local system (used to build the factorization and
# the right-hand sides). For `RBFFDLagrangeBasis` the polynomials are baked into the cardinal
# functions (built from `m` directly), so no separate polynomial basis is stored — matching
# the global `LagrangeBasis`, whose `order` is `0` and which adds no explicit polynomials.
_local_polynomial_basis(::RBFFDStandardBasis, xx, m::Int) = monomials(xx, 0:(m - 1))
_local_polynomial_basis(::RBFFDLagrangeBasis, xx, m::Int) = monomials(xx, 0:-1)

# RBFFDLagrangeBasis: cache the Lagrange cardinal functions on each stencil.
function _build_local_cache(::RBFFDLagrangeBasis, kernel::AbstractKernel,
                            neighbor_nodesets, ps, m::Int)
    return [collect(LagrangeBasis(nodes, kernel; m)) for nodes in neighbor_nodesets]
end

# RBFFDStandardBasis: cache the factorization of each stencil's augmented system matrix
# `M = [A P; P' 0]`, reused for every right-hand side (weights and evaluation).
function _build_local_cache(::RBFFDStandardBasis, kernel::AbstractKernel,
                            neighbor_nodesets, ps, m::Int)
    return [_local_factorization(kernel, nodes, ps) for nodes in neighbor_nodesets]
end

function _local_factorization(kernel::AbstractKernel, neighbor_nodes::NodeSet, ps)
    A = kernel_matrix(neighbor_nodes, kernel)
    q = length(ps)
    if q > 0
        P = polynomial_matrix(neighbor_nodes, ps)
        M = [A P
             P' zeros(eltype(A), q, q)]
    else
        M = A
    end
    return factorize(M)
end

# Note: Indexing with one index is not supported for `RBFFDBasis` and so `iterate` and `collect` also do not work.
"""
    getindex(basis::RBFFDBasis, i::Integer, k::Integer)

Return the `k`-th local stencil basis function associated with center index `i`.
The returned object is callable.

In both cases `k` ranges over the stencil size. For [`RBFFDLagrangeBasis`](@ref) it is the
`k`-th Lagrange cardinal function (with polynomial augmentation baked in, like the global
[`LagrangeBasis`](@ref)); for [`RBFFDStandardBasis`](@ref) it is the `k`-th kernel translate
`x -> kernel(x, x_{i_k})`, matching the global [`StandardBasis`](@ref). The polynomial part
is not exposed through indexing (see [`polynomial_basis`](@ref)); only for the Lagrange basis
do the returned functions carry the cardinal (nodal-value) semantics.
"""
function Base.getindex(basis::RBFFDBasis, i::Integer, k::Integer)
    return _getindex(basis, basis.local_basis, i, k)
end

function _getindex(basis::RBFFDBasis, ::RBFFDLagrangeBasis, i::Integer, k::Integer)
    (1 <= i <= length(centers(basis))) || throw(BoundsError(basis, (i, k)))
    funcs = basis.cache[i]
    (1 <= k <= length(funcs)) || throw(BoundsError(basis, (i, k)))
    return funcs[k]
end

function _getindex(basis::RBFFDBasis, ::RBFFDStandardBasis, i::Integer, k::Integer)
    (1 <= i <= length(centers(basis))) || throw(BoundsError(basis, (i, k)))
    indices = basis.stencil_indices[i]
    (1 <= k <= length(indices)) || throw(BoundsError(basis, (i, k)))
    kernel = basis.kernel
    c = centers(basis)[indices[k]]
    return x -> kernel(x, c)
end

"""
    polynomial_basis(basis::RBFFDBasis)

Return the (stencil-independent) polynomial basis functions used for the local polynomial
augmentation of the RBF-FD stencils. Empty if `local_order(basis) == 0`.
"""
polynomial_basis(basis::RBFFDBasis) = basis.ps

function Base.show(io::IO, basis::RBFFDBasis)
    print(io,
          "RBFFDBasis with $(length(centers(basis))) nodes, kernel $(basis.kernel), stencil $(basis.stencil_selection)")
    return nothing
end
