# Adapted from Trixi.jl
# https://github.com/trixi-framework/Trixi.jl/blob/cd097fc9d1fe80fb4d7824968d54c99bf3bd5281/src/callbacks_step/save_solution.jl

"""
SaveSolutionCallback(; interval::Integer=0,
                       dt=nothing,
                       save_initial_solution=true,
                       save_final_solution=true,
                       output_directory="out",
                       extra_functions=(),
                       keys=append!(["itp"], "value_" .* string.(eachindex(extra_functions))))

Save the current numerical solution in regular intervals in VTK format as a
Paraview Collection (.pvd). Either pass `interval` to save every `interval` time steps
or pass `dt` to save in intervals of `dt` in terms of integration time by adding additional
(shortened) time steps where necessary (note that this may change the solution).
The interpolation object will always be saved at the inner and boundary nodes of the corresponding
[`Semidiscretization`](@ref). You can pass extra functions (time- and space-dependent) or
vectors to save at these nodes via `extra_functions`. The corresponding keys in the `.vtu` files can
be specified by `keys`.

See also [`add_to_pvd`](@ref), [`vtk_save`](@ref).
"""
mutable struct SaveSolutionCallback{IntervalType, ExtraFunctions}
    interval_or_dt::IntervalType
    save_initial_solution::Bool
    save_final_solution::Bool
    output_directory::String
    extra_functions::ExtraFunctions
    keys::Vector{String}
    pvd::CollectionFile
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:SaveSolutionCallback})
    @nospecialize cb # reduce precompilation time

    save_solution_callback = cb.affect!
    print(io, "SaveSolutionCallback(interval=", save_solution_callback.interval_or_dt, ")")
end

function Base.show(io::IO,
               cb::DiscreteCallback{<:Any,
                                    <:PeriodicCallbackAffect{<:SaveSolutionCallback}})
    @nospecialize cb # reduce precompilation time

    save_solution_callback = cb.affect!.affect!
    print(io, "SaveSolutionCallback(dt=", save_solution_callback.interval_or_dt, ")")
end

function SaveSolutionCallback(; interval::Integer = 0,
                              dt = nothing,
                              save_initial_solution = true,
                              save_final_solution = true,
                              output_directory = "out",
                              extra_functions = (),
                              keys = append!(["itp"], "value_" .* string.(eachindex(extra_functions))))
    if !isnothing(dt) && interval > 0
        throw(ArgumentError("You can either set the number of steps between output (using `interval`) or the time between outputs (using `dt`) but not both simultaneously"))
    end

    # Expected most frequent behavior comes first
    if isnothing(dt)
        interval_or_dt = interval
    else # !isnothing(dt)
        interval_or_dt = dt
    end

    pvd = paraview_collection(joinpath(output_directory, "solution"))
    solution_callback = SaveSolutionCallback(interval_or_dt,
                                             save_initial_solution, save_final_solution,
                                             output_directory, extra_functions, keys, pvd)

    # Expected most frequent behavior comes first
    if isnothing(dt)
        # Save every `interval` (accepted) time steps
        # The first one is the condition, the second the affect!
        return DiscreteCallback(solution_callback, solution_callback,
                                save_positions = (false, false),
                                initialize = initialize_save_cb,
                                finalize = finalize_save_cb)
    else
        # Add a `tstop` every `dt`, and save the final solution.
        return PeriodicCallback(solution_callback, dt,
                                save_positions = (false, false),
                                initialize = initialize_save_cb,
                                finalize = finalize_save_cb,
                                final_affect = save_final_solution)
    end
end

function initialize_save_cb(cb, u, t, integrator)
    # The SaveSolutionCallback is either cb.affect! (with DiscreteCallback)
    # or cb.affect!.affect! (with PeriodicCallback).
    # Let recursive dispatch handle this.
    initialize_save_cb(cb.affect!, u, t, integrator)
end

function initialize_save_cb(solution_callback::SaveSolutionCallback, u, t, integrator)
    mkpath(solution_callback.output_directory)

    if solution_callback.save_initial_solution
        solution_callback(integrator)
    end

    return nothing
end

# this method is called to determine whether the callback should be activated
function (solution_callback::SaveSolutionCallback)(u, t, integrator)
    @unpack interval_or_dt, save_final_solution = solution_callback

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return interval_or_dt > 0 && (((integrator.stats.naccept % interval_or_dt == 0) &&
            !(integrator.stats.naccept == 0 && integrator.iter > 0)) ||
            (save_final_solution && isfinished(integrator)))
end

# this method is called when the callback is activated
function (solution_callback::SaveSolutionCallback)(integrator)
    @unpack pvd, output_directory, extra_functions, keys = solution_callback
    semi = integrator.p
    @unpack nodeset_inner, nodeset_boundary = semi.spatial_discretization
    nodeset = merge(nodeset_inner, nodeset_boundary)
    A = semi.cache.kernel_matrix
    u = A * integrator.u
    t = integrator.t
    iter = integrator.stats.naccept
    filename = joinpath(solution_callback.output_directory, @sprintf("solution_%06d", iter))
    add_to_pvd(filename, pvd, t, nodeset, u, extra_functions...; keys = keys)

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)
    return nothing
end

function finalize_save_cb(cb, u, t, integrator)
    # The SaveSolutionCallback is either cb.affect! (with DiscreteCallback)
    # or cb.affect!.affect! (with PeriodicCallback).
    # Let recursive dispatch handle this.
    finalize_save_cb(cb.affect!, u, t, integrator)
end

function finalize_save_cb(solution_callback::SaveSolutionCallback, u, t, integrator)
    WriteVTK.vtk_save(solution_callback.pvd)
    return nothing
end
