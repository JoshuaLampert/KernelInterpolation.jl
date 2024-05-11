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

# Note: We can't call the method below `KernelInterpolation.include` since that is created automatically
# inside `module KernelInterpolation` to `include` source files and evaluate them within the global scope
# of `KernelInterpolation`. However, users will want to evaluate in the global scope of `Main` or something
# similar to manage dependencies on their own.
"""
    include_example([mod::Module=Main,] example::AbstractString; kwargs...)

`include` the file `example` and evaluate its content in the global scope of module `mod`.
You can override specific assignments in `example` by supplying keyword arguments.
It's basic purpose is to make it easier to modify some parameters while running KernelInterpolation from the
REPL. Additionally, this is used in tests to reduce the computational burden for CI while still
providing examples with sensible default values for users.

Copied from [Trixi.jl](https://github.com/trixi-framework/Trixi.jl).
"""
function include_example(mod::Module, example::AbstractString; kwargs...)
    Base.include(ex -> replace_assignments(ex; kwargs...), mod, example)
end

function include_example(example::AbstractString; kwargs...)
    include_example(Main, example; kwargs...)
end

# Apply the function `f` to `expr` and all sub-expressions recursively.
walkexpr(f, expr::Expr) = f(Expr(expr.head, (walkexpr(f, arg) for arg in expr.args)...))
walkexpr(f, x) = f(x)

# Replace assignments to `key` in `expr` by `key = val` for all `(key,val)` in `kwargs`.
function replace_assignments(expr; kwargs...)
    # replace explicit and keyword assignments
    expr = walkexpr(expr) do x
        if x isa Expr
            for (key, val) in kwargs
                if (x.head === Symbol("=") || x.head === :kw) && x.args[1] === Symbol(key)
                    x.args[2] = :($val)
                    # dump(x)
                end
            end
        end
        return x
    end

    return expr
end

# Create `d` polyvars from `TypedPolynomials.jl`, don't use `@polyvars` because of
# https://github.com/JuliaAlgebra/TypedPolynomials.jl/issues/51, instead use the
# workaround from there
polyvars(d) = ntuple(i -> Variable{Symbol("x[", i, "]")}(), d)

# Store main timer for global timing of functions
const main_timer = TimerOutput()

# Always call timer() to hide implementation details
timer() = main_timer
