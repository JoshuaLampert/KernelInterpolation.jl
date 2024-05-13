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
            !(arg.args[1] in (:l2, :linf, :atol, :rtol, :pde_test)))
            push!(kwargs, Pair(arg.args...))
        end
    end
    quote
        println("═"^100)
        println($example)

        # evaluate examples in the scope of the module they're called from
        @test_nowarn_mod trixi_include(@__MODULE__, $example; $kwargs...)
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
                if pde isa KernelInterpolation.AbstractStationaryEquation
                    rhs_values = KernelInterpolation.rhs(nodeset_inner, pde)
                    for i in eachindex(nodeset_inner)
                        @test isapprox(pde(itp, nodeset_inner[i]), rhs_values[i],
                                       atol = $atol, rtol = $rtol)
                    end
                    values_boundary = g.(nodeset_boundary)
                    # Because of some namespace issues
                    itp2 = itp

                    many_values = u.(many_nodes, Ref(pde))
                elseif pde isa KernelInterpolation.AbstractTimeDependentEquation
                    t = last(tspan)
                    values_boundary = g.(t, nodeset_boundary)
                    itp2 = titp(t)

                    many_values = u.(Ref(t), many_nodes, Ref(pde))
                else
                    error("Unknown PDE type")
                end

                for (node, value) in zip(nodeset_boundary, values_boundary)
                    @test isapprox(itp2(node), value, atol = $atol, rtol = $rtol)
                end

                many_values_test = itp2.(many_nodes)
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

# Copied from TrixiBase.jl. See https://github.com/trixi-framework/TrixiBase.jl/issues/9.
"""
    @test_nowarn_mod expr
Modified version of `@test_nowarn expr` that prints the content of `stderr` when
it is not empty and ignores some common info statements printed in KernelInterpolation.jl
uses.
"""
macro test_nowarn_mod(expr, additional_ignore_content = String[])
    quote
        let fname = tempname()
            try
                ret = open(fname, "w") do f
                    redirect_stderr(f) do
                        $(esc(expr))
                    end
                end
                stderr_content = read(fname, String)
                if !isempty(stderr_content)
                    println("Content of `stderr`:\n", stderr_content)
                end

                # Patterns matching the following ones will be ignored. Additional patterns
                # passed as arguments can also be regular expressions, so we just use the
                # type `Any` for `ignore_content`.
                ignore_content = Any["[ Info: You just called `trixi_include`. Julia may now compile the code, please be patient.\n"]
                append!(ignore_content, $additional_ignore_content)
                for pattern in ignore_content
                    stderr_content = replace(stderr_content, pattern => "")
                end

                # We also ignore simple module redefinitions for convenience. Thus, we
                # check whether every line of `stderr_content` is of the form of a
                # module replacement warning.
                @test occursin(r"^(WARNING: replacing module .+\.\n)*$", stderr_content)
                ret
            finally
                rm(fname, force = true)
            end
        end
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
        using LinearAlgebra: norm # We use `norm` in all `@ki_testset`s
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
