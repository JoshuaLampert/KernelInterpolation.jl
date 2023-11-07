@recipe function f(nodeset::NodeSet)
    if dim(nodeset) == 1
        x = values_along_dim(nodeset, 1)
        seriestype := :scatter
        label --> "points"
        ylim --> (-0.1, 0.1)
        x, zeros(length(x))
    elseif dim(nodeset) == 2
        x = values_along_dim(nodeset, 1)
        y = values_along_dim(nodeset, 2)
        seriestype := :scatter
        label --> "points"
        x, y
    elseif dim(nodeset) == 3
        x = values_along_dim(nodeset, 1)
        y = values_along_dim(nodeset, 2)
        z = values_along_dim(nodeset, 3)
        seriestype := :scatter
        label --> "points"
        x, y, z
    else
        @error("Plotting a NodeSet is only supported for dimension up to 3, but the set has dimension $(dim(nodeset))")
    end
end

@recipe function f(nodeset::NodeSet, itp::Interpolation)
    if dim(nodeset) == 1
        @series begin
            x = values_along_dim(itp.nodeset, 1)
            seriestype := :scatter
            markershape --> :star
            label --> "training nodes"
            x, itp.(x)
        end
        @series begin
            x = values_along_dim(nodeset, 1)
            perm = sortperm(x)
            label --> "interpolation"
            xguide --> "x"
            yguide --> "f"
            x[perm], itp.(x[perm])
        end
    elseif dim(nodeset) == 2
        @series begin
            x = values_along_dim(itp.nodeset, 1)
            y = values_along_dim(itp.nodeset, 2)
            markershape --> :star
            markersize --> 10
            seriestype := :scatter
            label --> "training nodes"
            x, y, itp.(itp.nodeset)
        end
        @series begin
            x = values_along_dim(nodeset, 1)
            y = values_along_dim(nodeset, 2)
            seriestype --> :scatter
            label --> "interpolation"
            xguide --> "x"
            yguide --> "y"
            zguide --> "f"
            x, y, itp.(nodeset)
        end
    else
        @error("Plotting an interpolation is only supported for dimension up to 2, but the interpolation has dimension $(dim(nodeset))")
    end
end

@recipe function f(nodeset::NodeSet, f::Function)
    if dim(nodeset) == 1
        @series begin
            x = values_along_dim(nodeset, 1)
            perm = sortperm(x)
            label --> "f"
            xguide --> "x"
            yguide --> "f"
            x[perm], f.(x[perm])
        end
    elseif dim(nodeset) == 2
        @series begin
            x = values_along_dim(nodeset, 1)
            y = values_along_dim(nodeset, 2)
            seriestype --> :scatter
            label --> "f"
            xguide --> "x"
            yguide --> "y"
            zguide --> "f"
            x, y, f.(nodeset)
        end
    else
        @error("Plotting an interpolation is only supported for dimension up to 2, but the interpolation has dimension $(dim(nodeset))")
    end
end
