"""
    NodeSet(nodes)

Set of interpolation nodes.
"""
struct NodeSet{Dim, RealT}
    nodes::Vector{SVector{Dim, RealT}}
end

function Base.show(io::IO, nodeset::NodeSet)
    print(io, "NodeSet{", dim(nodeset), ", ", eltype(nodeset), "} with ", length(nodeset),
          " nodes")
end

function Base.show(io::IO, ::MIME"text/plain", nodeset::NodeSet)
    if get(io, :compact, false)
        show(io, semi)
    else
        println(io, "NodeSet{", dim(nodeset), ", ", eltype(nodeset), "} with ",
                length(nodeset), " nodes:")
        for i in 1:min(20, length(nodeset))
            if isassigned(nodeset, i)
                println(io, "  ", nodeset[i])
            else
                println(io, "  ", "#undef")
            end
        end
        if length(nodeset) > 20
            println("  ⋮")
        end
    end
end

# Constructors
function NodeSet(nodes::AbstractVector{Vector{RealT}}) where {RealT}
    n = length(nodes)
    @assert n > 0
    ndims = length(nodes[1])
    @assert ndims > 0
    data = [SVector{ndims, RealT}(nodes[i]) for i in 1:n]
    return NodeSet{ndims, RealT}(data)
end
function NodeSet(nodes::AbstractMatrix{RealT}) where {RealT}
    NodeSet(Vector{eltype(nodes)}[eachrow(nodes)...])
end
function NodeSet(nodes::AbstractVector{RealT}) where {RealT}
    @assert length(nodes) > 0
    return NodeSet([[node] for node in nodes])
end

dim(nodeset::NodeSet{Dim, RealT}) where {Dim, RealT} = Dim
# Functions to treat NodeSet as array
Base.eltype(nodeset::NodeSet{Dim, RealT}) where {Dim, RealT} = RealT
Base.length(nodeset::NodeSet) = length(nodeset.nodes)
Base.size(nodeset::NodeSet) = (length(nodeset), dim(nodeset))
Base.iterate(nodeset::NodeSet, state = 1) = iterate(nodeset.nodes, state)
Base.collect(nodeset::NodeSet) = collect(nodeset.nodes)
Base.axes(nodeset::NodeSet) = axes(nodeset.nodes)
Base.isassigned(nodeset::NodeSet, i::Int) = isassigned(nodeset.nodes, i)
function Base.similar(nodeset::NodeSet{Dim, RealT}) where {Dim, RealT}
    NodeSet{Dim, RealT}(similar(nodeset.nodes))
end
function Base.similar(nodeset::NodeSet{Dim, RealT}, ::Type{T}) where {Dim, RealT, T}
    NodeSet{Dim, T}(similar(nodeset.nodes, SVector{Dim, T}))
end
function Base.similar(nodeset::NodeSet{Dim, RealT}, n::Int) where {Dim, RealT}
    NodeSet{Dim, RealT}(similar(nodeset.nodes, n))
end
function Base.similar(nodeset::NodeSet{Dim, RealT}, ::Type{T}, n::Int) where {Dim, RealT, T}
    NodeSet{Dim, T}(similar(nodeset.nodes, SVector{Dim, T}, n))
end
Base.getindex(nodeset::NodeSet, i::Int) = getindex(nodeset.nodes, i)
Base.getindex(nodeset::NodeSet, is::UnitRange) = nodeset.nodes[is]
Base.lastindex(nodeset::NodeSet) = lastindex(nodeset.nodes)
function Base.setindex!(nodeset::NodeSet{Dim, RealT}, v::SVector{Dim, RealT},
                        i::Int) where {Dim, RealT}
    nodeset.nodes[i] = v
end
function Base.setindex!(nodeset::NodeSet{Dim, RealT}, v::Vector{RealT},
                        i::Int) where {Dim, RealT}
    @assert length(v) == dim(nodeset)
    nodeset.nodes[i] = v
end
function Base.setindex!(nodeset::NodeSet{RealT}, v::RealT, i::Int) where {RealT}
    @assert dim(nodeset) == 1
    nodeset.nodes[i] = [v]
end
function Base.push!(nodeset::NodeSet{Dim, RealT}, v::SVector{Dim, RealT}) where {Dim, RealT}
    push!(nodeset.nodes, v)
end
function Base.push!(nodeset::NodeSet{Dim, RealT}, v::Vector{RealT}) where {Dim, RealT}
    @assert length(v) == dim(nodeset)
    push!(nodeset.nodes, v)
end
function Base.merge(nodeset::NodeSet{Dim, RealT},
                    others::NodeSet{Dim, RealT}...) where {Dim, RealT}
    nodes_merged = similar(nodeset, 0)
    merge!(nodes_merged, nodeset, others...)
    return nodes_merged
end
function Base.merge!(nodeset::NodeSet{Dim, RealT},
                     others::NodeSet{Dim, RealT}...) where {Dim, RealT}
    foreach(other -> append!(nodeset.nodes, other.nodes), others)
    #     return NodeSet(merge(nodeset.nodes, foreach(other -> other.nodes, others)...))
end
Base.unique(nodeset::NodeSet) = NodeSet(unique(nodeset.nodes))
Base.unique!(nodeset::NodeSet) = unique!(nodeset.nodes)

"""
    values_along_dim(nodeset::NodeSet, i::Int)

Convenience function to return all ``x_i``-values of the nodes, i.e. the `i`-th component of each node.
Supported for `nodeset` with `dim(nodeset) >= i`.
"""
function values_along_dim(nodeset::NodeSet, i::Int)
    @assert dim(nodeset) >= i
    x_i = Vector{eltype(nodeset)}(undef, length(nodeset))
    for (j, node) in enumerate(nodeset)
        x_i[j] = node[i]
    end
    return x_i
end

"""
    random_hypercube(n, dim, x_min = 0.0, x_max = 1.0)

Create a `NodeSet` with `n` random nodes each of dimension `dim` inside a hypercube defined by
the bounds `x_min` and `x_max`. If the bounds are given as single values, they are applied for
each dimension. If they are `Tuple`s of size `dim` the hypercube has the according bounds.
"""
function random_hypercube(n::Int, dim::Int, x_min, x_max)
    nodes = x_min .+ (x_max - x_min) .* rand(n, dim)
    return NodeSet(nodes)
end

function random_hypercube(n::Int, dim::Int, x_min::NTuple{Dim} = ntuple(i -> 0.0, dim),
                          x_max::NTuple{Dim} = ntuple(i -> 1.0, dim)) where {Dim}
    @assert dim == Dim
    nodes = rand(n, dim)
    for i in 1:dim
        nodes[:, i] = x_min[i] .+ (x_max[i] - x_min[i]) .* view(nodes, :, i)
    end
    return NodeSet(nodes)
end

"""
    homogeneous_hypercube(n, dim, x_min = 0.0, x_max = 1.0)

Create a `NodeSet` with `n` homogeneously distributed nodes in every dimension each of dimension
`dim` inside a hypercube defined by the bounds `x_min` and `x_max`. The resulting `NodeSet` will have
``n^dims`` points. If the bounds are given as single values, they are applied for each dimension.
If they are `Tuple`s of size `dim` the hypercube has the according bounds.
"""
function homogeneous_hypercube(n::Int, dim::Int, x_min, x_max)
    return homogeneous_hypercube(n, dim, ntuple(i -> x_min, dim), ntuple(i -> x_max, dim))
end

function homogeneous_hypercube(n::Int, dim::Int, x_min::NTuple{Dim} = ntuple(i -> 0.0, dim),
                               x_max::NTuple{Dim} = ntuple(i -> 1.0, dim)) where {Dim}
    @assert dim == Dim
    nodes = Vector{SVector{dim, Float64}}(undef, n^dim)
    for (i, indices) in enumerate(Iterators.product(ntuple(i -> 1:n, dim)...))
        node = Vector(undef, dim)
        for j in 1:dim
            node[j] = x_min[j] + (x_max[j] - x_min[j]) * (indices[j] - 1) / (n - 1)
        end
        nodes[i] = node
    end
    return NodeSet(nodes)
end

"""
    random_hypersphere(n, dim, r = 1.0, center = zeros(dim))

Create a `NodeSet` with `n` random nodes each of dimension `dim` inside a hypersphere with
radius `r`.
"""
function random_hypersphere(n::Int, dim::Int, r = 1.0, center = zeros(dim))
    nodes = randn(n, dim)
    for i in 1:n
        nodes[i, :] .= center .+ r .* nodes[i, :] ./ norm(nodes[i, :]) * rand()^(1 / dim)
    end
    return NodeSet(nodes)
end

"""
    random_hypersphere_boundary(n, dim, r = 1.0, center = zeros(dim))

Create a `NodeSet` with `n` random nodes each of dimension `dim` at the boundary of a
hypersphere with radius `r`.
"""
function random_hypersphere_boundary(n::Int, dim::Int, r = 1.0, center = zeros(dim))
    if dim == 1 && n >= 2
        @warn "For one dimension the boundary of the hypersphere consists only of 2 points"
        return NodeSet([-r, r])
    end
    nodes = randn(n, dim)
    for i in 1:n
        nodes[i, :] .= center .+ r .* nodes[i, :] ./ norm(nodes[i, :])
    end
    return NodeSet(nodes)
end
