module KernelInterpolationMeshesExt

using Meshes: Meshes, Point, PointSet, to, ustrip
using KernelInterpolation: KernelInterpolation, NodeSet

# Meshes.jl uses Unitful.jl for units, which is not available in KernelInterpolation.jl
# Thus, we need to remove the units from the Point struct
uto(p::Point) = ustrip(to(p))

function KernelInterpolation.NodeSet(points::Vector{P}) where {P <: Point}
    return NodeSet(uto.(points))
end

function KernelInterpolation.NodeSet(points::PointSet)
    return NodeSet(parent(points))
end

function Meshes.PointSet(nodes::KernelInterpolation.NodeSet)
    return PointSet(Tuple.(nodes.nodes))
end

end
