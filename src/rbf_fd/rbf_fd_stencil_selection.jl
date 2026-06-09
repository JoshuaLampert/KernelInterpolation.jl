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

# Arguments
- `k::Int`: Number of nearest neighbors to use
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
    RadiusSearch(radius)

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
    select_neighbors(i, nodeset, stencil_selection)

Selects neighbors of `nodeset[i]` from `nodeset` using `stencil_selection`.
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

    neighbor_nodes = NodeSet(nodeset.nodes[neighbor_indices])
    return (indices = neighbor_indices, nodes = neighbor_nodes)
end

function select_neighbors(i::Int, nodeset::NodeSet, stencil::RadiusSearch)
    radius = stencil.radius
    x_i = nodeset[i]
    neighbor_indices = Int[]

    for j in eachindex(nodeset)
        dist = norm(x_i .- nodeset[j])
        if dist <= radius
            push!(neighbor_indices, j)
        end
    end

    isempty(neighbor_indices) &&
        throw(ArgumentError("No neighbors found within radius $(radius) for point index $(i)"))

    neighbor_nodes = NodeSet(nodeset.nodes[neighbor_indices])
    return (indices = neighbor_indices, nodes = neighbor_nodes)
end

function select_neighbors(x_i::AbstractVector, nodeset::NodeSet,
                          stencil::KNearestNeighbors)
    distances = [norm(x_i .- nodeset[j]) for j in eachindex(nodeset)]
    k = stencil.k
    k ≤ length(nodeset) ||
        throw(ArgumentError("k=$(k) exceeds nodeset size $(length(nodeset))"))
    neighbor_indices = sortperm(distances)[1:k]
    neighbor_nodes = NodeSet(nodeset.nodes[neighbor_indices])
    return (indices = neighbor_indices, nodes = neighbor_nodes)
end

function select_neighbors(x_i::AbstractVector, nodeset::NodeSet,
                          stencil::RadiusSearch)
    neighbor_indices = Int[]
    for j in eachindex(nodeset)
        if norm(x_i .- nodeset[j]) <= stencil.radius
            push!(neighbor_indices, j)
        end
    end
    isempty(neighbor_indices) &&
        throw(ArgumentError("No neighbors found within radius $(stencil.radius)"))
    neighbor_nodes = NodeSet(nodeset.nodes[neighbor_indices])
    return (indices = neighbor_indices, nodes = neighbor_nodes)
end
