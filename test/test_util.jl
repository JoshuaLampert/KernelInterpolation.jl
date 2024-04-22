"""
    test_include_example(example; l2=nothing, linf=nothing,
                         atol=1e-12, rtol=sqrt(eps()),
                         args...)

Test by calling `include_example(example; parameters...)`.
By default, only the absence of error output is checked.
"""
macro test_include_example(example, args...)
    local l2 = get_kwarg(args, :l2, nothing)
    local linf = get_kwarg(args, :linf, nothing)
    local atol = get_kwarg(args, :atol, 1e-12)
    local rtol = get_kwarg(args, :rtol, sqrt(eps()))
    local pde_test = get_kwarg(args, :pde_test, false)
    local kwargs = Pair{Symbol, Any}[]
    for arg in args
        if (arg.head == :(=) &&
            !(arg.args[1] in (:l2, :linf, :atol, :rtol)))
            push!(kwargs, Pair(arg.args...))
        end
    end
    quote
        println("═"^100)
        println($example)

        # evaluate examples in the scope of the module they're called from
        @test_nowarn include_example(@__MODULE__, $example; $kwargs...)
        # if present, compare l2 and linf against reference values
        if !isnothing($l2) || !isnothing($linf)
            if !$pde_test
                values_test = itp.(nodeset)
                # Check interpolation at interpolation nodes
                @test isapprox(norm(values .- values_test, Inf), 0;
                               atol = $atol, rtol = $rtol)
                many_values = f.(many_nodes)
                many_values_test = itp.(many_nodes)
                @test isapprox(norm(many_values .- many_values_test), $l2;
                               atol = $atol, rtol = $rtol)
                @test isapprox(norm(many_values .- many_values_test, Inf), $linf;
                               atol = $atol, rtol = $rtol)
            else
                rhs_values = KernelInterpolation.rhs(pde, nodeset_inner)
                for i in eachindex(nodeset_inner)
                    @test isapprox(pde(itp, nodeset_inner[i]), rhs_values[i],
                                   atol = $atol, rtol = $rtol)
                end
                for (node, value) in zip(nodeset_boundary, values_boundary)
                    @test isapprox(itp(node), value, atol = $atol, rtol = $rtol)
                end
                many_values = u.(many_nodes)
                many_values_test = itp.(many_nodes)
                @test isapprox(norm(many_values .- many_values_test), $l2;
                               atol = $atol, rtol = $rtol)
                @test isapprox(norm(many_values .- many_values_test, Inf), $linf;
                               atol = $atol, rtol = $rtol)
            end
        end
        println("═"^100)
    end
end

# Get the first value assigned to `keyword` in `args` and return `default_value`
# if there are no assignments to `keyword` in `args`.
function get_kwarg(args, keyword, default_value)
    val = default_value
    for arg in args
        if arg.head == :(=) && arg.args[1] == keyword
            val = arg.args[2]
            break
        end
    end
    return val
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
        using LinearAlgebra: norm # We use `norm` is all `@ki_testset`s
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
