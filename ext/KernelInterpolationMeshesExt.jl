module KernelInterpolationMeshesExt

using Meshes: Meshes, Point, PointSet, to
using KernelInterpolation: KernelInterpolation

# Meshes.jl uses Unitful.jl for units, which is not available in KernelInterpolation.jl
# Thus, we need to remove the units from the Point struct
val(u) = u.val
val(p::Point) = val.(to(p))

function KernelInterpolation.NodeSet(points::Vector{P}) where P <: Point
    return KernelInterpolation.NodeSet(val.(points))
end

function KernelInterpolation.NodeSet(points::PointSet)
    return KernelInterpolation.NodeSet(points.geoms)
end

function Meshes.PointSet(nodes::KernelInterpolation.NodeSet)
    return Meshes.PointSet(Tuple.(nodes.nodes))
end

end
