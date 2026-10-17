"""
    vtk_save(filename, nodeset::NodeSet, functions_or_vectors...;
             keys = "value_" .* string.(eachindex(functions_or_vectors)))

Save a [`NodeSet`](@ref) to a VTK file. You can optionally pass a list of space-dependent functions
or vectors to save the values of the functions at the nodes. The functions can also be passed as
[`Interpolation`](@ref) or directly as vectors. The optional keyword argument `keys` is used to
specify the names of the data arrays in the VTK file.

See also [`add_to_pvd`](@ref), [`vtk_read`](@ref).
"""
function vtk_save(filename, nodeset::NodeSet, functions_or_vectors...;
                  keys = "value_" .* string.(eachindex(functions_or_vectors)))
    if dim(nodeset) > 3
        throw(ArgumentError("Only 1D, 2D, and 3D data can be saved to VTK files, got " *
                            "dimension $(dim(nodeset))."))
    end
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in eachindex(nodeset)]
    points = Base.getindex.(Ref(nodeset), :, eachdim(nodeset))
    vtk_grid(filename, points..., cells, append = false) do vtk
        for (i, fun_or_vec) in enumerate(functions_or_vectors)
            if fun_or_vec isa AbstractArray
                vec = fun_or_vec
            else
                vec = fun_or_vec.(nodeset)
            end
            vtk["$(keys[i])"] = vec
        end
    end
end

"""
    add_to_pvd(filename, pvd, time, nodeset::NodeSet, functions_or_vectors...;
               keys = "value_" .* string.(eachindex(functions_or_vectors)))

Same as [`vtk_save`](@ref), but appends the data to a Paraview collection file `pvd` at time `time`.
In contrast to [`vtk_save`](@ref), the functions are time- and space-dependent.
"""
function add_to_pvd(filename, pvd, time, nodeset::NodeSet, functions_or_vectors...;
                    keys = "value_" .* string.(eachindex(functions_or_vectors)))
    if dim(nodeset) > 3
        throw(ArgumentError("Only 1D, 2D, and 3D data can be saved to VTK files, got " *
                            "dimension $(dim(nodeset))."))
    end
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in eachindex(nodeset)]
    points = Base.getindex.(Ref(nodeset), :, eachdim(nodeset))
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
        return nothing
    end
end

"""
    vtk_read(filename)

Read a set of nodes from a VTK file and return them as a [`NodeSet`](@ref). Note that the data
will always be returned as a 3D [`NodeSet`](@ref), even if the data is 1D or 2D. The point data
will be returned as a dictionary with the keys being the names of the data arrays in the VTK file.

See also [`vtk_save`](@ref).
"""
function vtk_read(filename)
    vtk = VTKFile(filename)
    nodeset = NodeSet(transpose(get_points(vtk)))
    point_data = Dict{String, AbstractArray}()
    # If the file contains no point data, `get_point_data` throws a `BoundsError` (there is no
    # public way to check for point data beforehand). Only this error is caught, such that
    # other errors, e.g., when reading the data arrays, are not silently ignored.
    vtk_point_data = try
        get_point_data(vtk)
    catch e
        e isa BoundsError || rethrow()
        nothing
    end
    if !isnothing(vtk_point_data)
        for (key, data) in vtk_point_data
            point_data[key] = get_data(data)
        end
    end
    return nodeset, point_data
end
