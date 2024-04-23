"""
    vtk_save(filename, nodeset::NodeSet, functions...;
             keys = "value_" .* string.(eachindex(functions)))

Save a `NodeSet` to a VTK file. You can optionally pass a list of functions to save
the values of the functions at the nodes. The functions can also be passed as `Interpolation`.
The optional keyword argument `keys` is used to specify the names of the data arrays in the VTK file.
"""
function vtk_save(filename, nodeset::NodeSet, functions...;
                  keys = "value_" .* string.(eachindex(functions)))
    @assert dim(nodeset)<=3 "Only 1D, 2D, and 3D data can be saved to VTK files."
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in eachindex(nodeset)]
    points = values_along_dim.(Ref(nodeset), eachdim(nodeset))
    vtk_grid(filename, points..., cells, append = false) do vtk
        for (i, func) in enumerate(functions)
            vtk["$(keys[i])"] = func.(nodeset)
        end
    end
end
