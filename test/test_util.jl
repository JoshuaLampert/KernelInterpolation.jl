using TrixiTest: @trixi_test_nowarn, get_kwarg

"""
    test_include_example(example; l2=nothing, linf=nothing,
                         atol=1e-12, rtol=sqrt(eps()),
                         kwargs...)

Test by calling `include_example(example; kwargs...)`.
By default, only the absence of error output is checked.
"""
macro test_include_example(example, args...)
    local l2 = get_kwarg(args, :l2, nothing)
    local linf = get_kwarg(args, :linf, nothing)
    local l2_ls = get_kwarg(args, :l2_ls, nothing)
    local linf_ls = get_kwarg(args, :linf_ls, nothing)
    local l2_reg = get_kwarg(args, :l2_reg, nothing)
    local linf_reg = get_kwarg(args, :linf_reg, nothing)
    local atol = get_kwarg(args, :atol, 1e-12)
    local rtol = get_kwarg(args, :rtol, sqrt(eps()))
    local interpolation_test = get_kwarg(args, :interpolation_test, true)
    local least_square_test = get_kwarg(args, :least_square_test, false)
    local regularization_test = get_kwarg(args, :regularization_test, false)
    local pde_test = get_kwarg(args, :pde_test, false)
    local kwargs = Pair{Symbol, Any}[]
    for arg in args
        if (arg.head == :(=) &&
            !(arg.args[1] in (:l2, :linf, :l2_ls, :linf_ls, :l2_reg, :linf_reg,
                              :atol, :rtol,
                              :interpolation_test, :least_square_test, :regularization_test,
                              :pde_test)))
            push!(kwargs, Pair(arg.args...))
        end
    end
    quote
        println("═"^100)
        println($example)

        # evaluate examples in the scope of the module they're called from
        @trixi_test_nowarn trixi_include(@__MODULE__, $example; $kwargs...)
        # if present, compare l2 and linf against reference values
        if !isnothing($l2) || !isnothing($linf)
            if !$pde_test
                # interpolation test
                # assumes `many_nodes` and `values` are defined in the example
                values_test = itp.(nodeset)
                # Check interpolation at interpolation nodes
                if $interpolation_test
                    @test isapprox(norm(values .- values_test, Inf), 0;
                                   atol = $atol, rtol = $rtol)
                end
                many_values = f.(many_nodes)
                many_values_test = itp.(many_nodes)
                @test isapprox(norm(many_values .- many_values_test), $l2;
                               atol = $atol, rtol = $rtol)
                @test isapprox(norm(many_values .- many_values_test, Inf), $linf;
                               atol = $atol, rtol = $rtol)
                if $least_square_test
                    many_values_ls = ls.(many_nodes)
                    @test isapprox(norm(many_values .- many_values_ls), $l2_ls;
                                   atol = $atol, rtol = $rtol)
                    @test isapprox(norm(many_values .- many_values_ls, Inf), $linf_ls;
                                   atol = $atol, rtol = $rtol)
                end
                if $regularization_test
                    many_values_reg = itp_reg.(many_nodes)
                    @test isapprox(norm(many_values .- many_values_reg), $l2_reg;
                                   atol = $atol, rtol = $rtol)
                    @test isapprox(norm(many_values .- many_values_reg, Inf), $linf_reg;
                                   atol = $atol, rtol = $rtol)
                end
            else
                # PDE test
                # assumes `many_nodes`, `nodes_inner` and `nodeset_boundary` are defined in the example
                # if `u` is defined, it is used to compare the solution (analytical solution or initial condition) using the l2 and linf norms
                if pde isa KernelInterpolation.AbstractStationaryEquation
                    if !$least_square_test
                        rhs_values = KernelInterpolation.rhs(nodeset_inner, pde)
                        for i in eachindex(nodeset_inner)
                            @test isapprox(pde(itp, nodeset_inner[i]), rhs_values[i],
                                           atol = $atol, rtol = $rtol)
                        end
                        values_boundary = g.(nodeset_boundary)
                    end
                    # Because of some namespace issues
                    itp2 = itp

                    if @isdefined u
                        many_values = u.(many_nodes, Ref(pde))
                    end
                elseif pde isa KernelInterpolation.AbstractTimeDependentEquation
                    t = last(tspan)
                    values_boundary = g.(t, nodeset_boundary)
                    itp2 = titp(t)

                    if @isdefined u
                        many_values = u.(Ref(t), many_nodes, Ref(pde))
                    end
                else
                    error("Unknown PDE type")
                end

                if !$least_square_test
                    for (node, value) in zip(nodeset_boundary, values_boundary)
                        @test isapprox(itp2(node), value, atol = $atol, rtol = $rtol)
                    end
                end

                if @isdefined many_values
                    many_values_test = itp2.(many_nodes)
                    @test isapprox(norm(many_values .- many_values_test), $l2;
                                   atol = $atol, rtol = $rtol)
                    @test isapprox(norm(many_values .- many_values_test, Inf), $linf;
                                   atol = $atol, rtol = $rtol)
                end
            end
        end
        println("═"^100)
        # Clean up
        rm("out"; force = true, recursive = true)
    end
end
