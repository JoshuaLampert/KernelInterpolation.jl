"""
    test_include_example(example; args...)

Test by calling `include_example(example; parameters...)`.
By default, only the absence of error output is checked.
"""
macro test_include_example(example, args...)
    local kwargs = Pair{Symbol, Any}[]
    for arg in args
        if arg.head == :(=)
            push!(kwargs, Pair(arg.args...))
        end
    end
    quote
        println("═"^100)
        println($example)

        # evaluate examples in the scope of the module they're called from
        @test_nowarn include_example(@__MODULE__, $example; $kwargs...)
        println("═"^100)
    end
end

"""
    @ki_testset "name of the testset" #= code to test #=

Similar to `@testset`, but wraps the code inside a temporary module to avoid
namespace pollution.
"""
macro ki_testset(name, expr)
    @assert name isa String
    mod = gensym(name)
    quote
        local time_start = time_ns()
        @eval module $mod
        using Test
        using KernelInterpolation
        include(@__FILE__)
        # We define `EXAMPLES_DIR` in (nearly) all test modules and use it to
        # get the path to the examples to be tested. However, that's not required
        # and we want to fail gracefully if it's not defined.
        try
            import ..EXAMPLES_DIR
        catch
            nothing
        end
        @testset $name $expr
        end
        local time_stop = time_ns()
        flush(stdout)
        @info("Testset "*$name*" finished in "
              *string(1.0e-9 * (time_stop - time_start))*" seconds.\n")
        nothing
    end
end
