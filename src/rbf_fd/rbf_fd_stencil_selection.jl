"""
    AbstractStencilSelection

Abstract type for stencil selection strategies in RBF-FD. Defines how neighboring points
are selected for each local finite difference stencil.
"""
abstract type AbstractStencilSelection end

"""
    KNearestNeighbors(k::Int)

Stencil selection strategy using k-nearest neighbors. Each interior point uses its `k` nearest
neighbors (by Euclidean distance) to form the local FD stencil. This ensures uniform sparsity.

See also [`AbstractStencilSelection`](@ref) and [`RadiusSearch`](@ref).
"""
struct KNearestNeighbors <: AbstractStencilSelection
    k::Int
    function KNearestNeighbors(k::Int)
        k ≥ 1 || throw(ArgumentError("k must be ≥ 1, got $k"))
        return new(k)
    end
end

Base.show(io::IO, ss::KNearestNeighbors) = print(io, "KNearestNeighbors(k=$(ss.k))")

"""
    select_neighbors(nodeset, stencil_selection)

Select the stencils for *all* nodes of `nodeset` at once using `stencil_selection`. A single
spatial search structure (a [`KDTree`](https://github.com/KristofferC/NearestNeighbors.jl))
is built over the nodes and reused for every query, which makes the overall neighborhood
search scale like `O(N log N)` instead of the `O(N^2)` of a per-node brute-force distance scan.
Returns a `Vector` whose `i`-th entry holds the global neighbor indices of `nodeset[i]`.

See also [`select_neighbors(i, nodeset, stencil_selection)`](@ref).
"""
function select_neighbors(nodeset::NodeSet, stencil::KNearestNeighbors)
    k = stencil.k
    n = length(nodeset)
    k ≤ n || throw(ArgumentError("k=$(k) exceeds nodeset size $(n)"))
    tree = KDTree(nodeset.nodes)
    # `knn` returns the `k` nearest neighbors sorted by distance; the center node itself has
    # distance 0 and is therefore always included in its own stencil.
    indices, _ = knn(tree, nodeset.nodes, k, true)
    return indices
end

"""
    select_neighbors(i, nodeset, stencil_selection)

Selects neighbors of `nodeset[i]` from `nodeset` using `stencil_selection`.
Returns a tuple of `(indices, nodes)`. To select the stencils of all nodes at once
(reusing a single search structure), use [`select_neighbors(nodeset, stencil_selection)`](@ref).
"""
function select_neighbors(i::Int, nodeset::NodeSet, stencil::KNearestNeighbors)
    k = stencil.k
    n = length(nodeset)
    k ≤ n || throw(ArgumentError("k=$(k) exceeds nodeset size $(n)"))
    tree = KDTree(nodeset.nodes)
    # `knn` returns the `k` nearest neighbors sorted by distance; the center node itself has
    # distance 0 and is therefore always included in its own stencil.
    neighbor_indices, _ = knn(tree, nodeset[i], k, true)

    neighbor_nodes = NodeSet(nodeset.nodes[neighbor_indices])
    return (indices = neighbor_indices, nodes = neighbor_nodes)
end

"""
    RadiusSearch(radius)

Stencil selection strategy using fixed radius search with search `radius`. Each interior point
uses all neighbors within a given Euclidean distance `radius` to form the local FD stencil.
This allows variable stencil sizes but adapts to local point density.

See also [`AbstractStencilSelection`](@ref) and [`KNearestNeighbors`](@ref).
"""
struct RadiusSearch{RealT <: Real} <: AbstractStencilSelection
    radius::RealT
    function RadiusSearch(radius::Real)
        radius > 0 || throw(ArgumentError("radius must be > 0, got $radius"))
        return new{typeof(radius)}(radius)
    end
end

Base.show(io::IO, ss::RadiusSearch) = print(io, "RadiusSearch(r=$(ss.radius))")

function select_neighbors(i::Int, nodeset::NodeSet, stencil::RadiusSearch)
    radius = stencil.radius
    tree = KDTree(nodeset.nodes)
    neighbor_indices = inrange(tree, nodeset[i], radius)

    isempty(neighbor_indices) &&
        throw(ArgumentError("No neighbors found within radius $(radius) for point index $(i)"))

    neighbor_nodes = NodeSet(nodeset.nodes[neighbor_indices])
    return (indices = neighbor_indices, nodes = neighbor_nodes)
end

function select_neighbors(nodeset::NodeSet, stencil::RadiusSearch)
    radius = stencil.radius
    tree = KDTree(nodeset.nodes)
    indices = inrange(tree, nodeset.nodes, radius)
    for (i, ind) in enumerate(indices)
        isempty(ind) &&
            throw(ArgumentError("No neighbors found within radius $(radius) for point index $(i)"))
    end
    return indices
end
