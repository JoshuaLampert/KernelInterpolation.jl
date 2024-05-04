"""
    vtk_save(filename, nodeset::NodeSet, functions_or_vectors...;
             keys = "value_" .* string.(eachindex(functions_or_vectors)))

Save a [`NodeSet`](@ref) to a VTK file. You can optionally pass a list of functions or functions_or_vectors to save
the values of the functions at the nodes. The functions can also be passed as
[`KernelInterpolation.Interpolation`](@ref) or directly as vectors. The optional keyword argument `keys` is used to
specify the names of the data arrays in the VTK file.

See [`add_to_pvd`](@ref)
"""
function vtk_save(filename, nodeset::NodeSet, functions_or_vectors...;
                  keys = "value_" .* string.(eachindex(functions_or_vectors)))
    @assert dim(nodeset)<=3 "Only 1D, 2D, and 3D data can be saved to VTK files."
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in eachindex(nodeset)]
    points = values_along_dim.(Ref(nodeset), eachdim(nodeset))
    vtk_grid(filename, points..., cells, append = false) do vtk
        for (i, fun_or_vec) in enumerate(functions_or_vectors)
            if fun_or_vec isa AbstractArray
                vec = fun_or_vec
            else
                vec = fun_or_vec.(Ref(time), nodeset)
            end
            vtk["$(keys[i])"] = vec
        end
    end
end

"""
    add_to_pvd(filename, pvd, time, nodeset::NodeSet, functions_or_vectors...;
               keys = "value_" .* string.(eachindex(functions_or_vectors)))

Same as [`vtk_save`](@ref), but appends the data to a Paraview collection file `pvd` at time `time`.
"""
function add_to_pvd(filename, pvd, time, nodeset::NodeSet, functions_or_vectors...;
                    keys = "value_" .* string.(eachindex(functions_or_vectors)))
    @assert dim(nodeset)<=3 "Only 1D, 2D, and 3D data can be saved to VTK files."
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in eachindex(nodeset)]
    points = values_along_dim.(Ref(nodeset), eachdim(nodeset))
    vtk_grid(filename, points..., cells, append = false) do vtk
        for (i, fun_or_vec) in enumerate(functions_or_vectors)
            if fun_or_vec isa AbstractArray
                vec = fun_or_vec
            else
                vec = fun_or_vec.(Ref(time), nodeset)
            end
            vtk["$(keys[i])"] = vec
        end
        pvd[time] = vtk
    end
end

"""
    vtk_read(filename)

Read a set of nodes from a VTK file and return them as a [`NodeSet`](@ref). Note that the data
will always be returned as a 3D [`NodeSet`](@ref), even if the data is 1D or 2D. The point data
will be returned as a dictionary with the keys being the names of the data arrays in the VTK file.
"""
function vtk_read(filename)
    vtk = VTKFile(filename)
    nodeset = NodeSet(transpose(get_points(vtk)))
    point_data = Dict{String, AbstractArray}()
    # If there are no point_data `get_point_data` will throw an error
    # I don't see a way to check for point_data without catching the error
    try
        for (key, data) in get_point_data(vtk)
            point_data[key] = get_data(data)
        end
    catch
    end
    return nodeset, point_data
end
