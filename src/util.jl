"""
    examples_dir()

Return the directory where the example files provided with KernelInterpolation.jl are located.

# Examples
```@example
readdir(examples_dir())
```
"""
examples_dir() = pkgdir(KernelInterpolation, "examples")::String

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
    return joinpath(examples_dir(), "interpolation", "interpolation_2d.jl")
end

# Element type of an `n × m` matrix that is assembled entrywise, where `entry` returns one of
# its entries and `fallback` is the element type to use if the matrix is empty. Deriving the
# element type from an actual entry instead of from the node coordinates keeps the assembly
# correct if the kernel carries parameters whose element type differs from the one of the
# nodes. This is the case when differentiating with respect to, e.g., a shape parameter with
# forward-mode automatic differentiation, where the kernel holds dual numbers, while the nodes
# stay real.
function matrix_eltype(entry, n, m, fallback)
    (n == 0 || m == 0) && return fallback
    return promote_type(typeof(entry()), fallback)
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
    return (Variable{Symbol("x[1]")}(), Variable{Symbol("x[2]")}(),
            Variable{Symbol("x[3]")}())
end
