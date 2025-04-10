"""
    examples_dir()

Return the directory where the example files provided with KernelInterpolation.jl are located.

# Examples
```@example
readdir(examples_dir())
```
"""
examples_dir() = pkgdir(KernelInterpolation, "examples")

"""
    get_examples()

Return a list of all examples that are provided by KernelInterpolation.jl. See also
[`examples_dir`](@ref) and [`default_example`](@ref).
"""
function get_examples()
    examples = String[]
    for (root, dirs, files) in walkdir(examples_dir())
        for f in files
            if endswith(f, ".jl")
                push!(examples, joinpath(root, f))
            end
        end
    end

    return examples
end

"""
    default_example()

Return the path to an example that can be used to quickly see KernelInterpolation.jl in action.
See also [`examples_dir`](@ref) and [`get_examples`](@ref).
"""
function default_example()
    joinpath(examples_dir(), "interpolation", "interpolation_2d.jl")
end

# Create `d` polyvars from `TypedPolynomials.jl`, don't use `@polyvars` because of
# https://github.com/JuliaAlgebra/TypedPolynomials.jl/issues/51, instead use the
# workaround from there
polyvars(d) = ntuple(i -> Variable{Symbol("x[", i, "]")}(), d)
# The function above is not type stable.
# Therefore, we define some common special cases for performance reasons.
polyvars(::Val{1}) = (Variable{Symbol("x[1]")}(),)
polyvars(::Val{2}) = (Variable{Symbol("x[1]")}(), Variable{Symbol("x[2]")}())
function polyvars(::Val{3})
    (Variable{Symbol("x[1]")}(), Variable{Symbol("x[2]")}(),
     Variable{Symbol("x[3]")}())
end
