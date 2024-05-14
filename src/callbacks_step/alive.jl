# Adapted from Trixi.jl
# https://github.com/trixi-framework/Trixi.jl/blob/c221bca89b38d416fb49137b1b266cecd1646b52/src/callbacks_step/alive.jl
"""
    AliveCallback(io::IO = stdout; interval::Integer=0, dt=nothing)

Inexpensive callback showing that a simulation is still running by printing
some information such as the current time to the screen every `interval`
time steps or after a time of `dt` in terms of integration time by adding additional
(shortened) time steps where necessary (note that this may change the solution).
"""
mutable struct AliveCallback{IntervalType}
    start_time::Float64
    io::IO
    interval_or_dt::IntervalType
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:AliveCallback})
    @nospecialize cb # reduce precompilation time

    alive_callback = cb.affect!
    print(io, "AliveCallback(interval=", alive_callback.interval_or_dt, ")")
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:AliveCallback}})
    @nospecialize cb # reduce precompilation time

    alive_callback = cb.affect!.affect!
    print(io, "AliveCallback(dt=", alive_callback.interval_or_dt, ")")
end

function AliveCallback(io::IO = stdout; interval::Integer = 0,
                       dt = nothing)
    if !isnothing(dt) && interval > 0
        throw(ArgumentError("You can either set the number of steps between output (using `interval`) or the time between outputs (using `dt`) but not both simultaneously"))
    end

    # Expected most frequent behavior comes first
    if isnothing(dt)
        interval_or_dt = interval
    else # !isnothing(dt)
        interval_or_dt = dt
    end

    alive_callback = AliveCallback(0.0, io, interval_or_dt)

    # Expected most frequent behavior comes first
    if isnothing(dt)
        # Save every `interval` (accepted) time steps
        # The first one is the condition, the second the affect!
        return DiscreteCallback(alive_callback, alive_callback,
                                save_positions = (false, false),
                                finalize = finalize,
                                initialize = initialize!)
    else
        # Add a `tstop` every `dt`, and save the final solution.
        return PeriodicCallback(alive_callback, dt,
                                save_positions = (false, false),
                                initialize = initialize!,
                                finalize = finalize,
                                final_affect = true)
    end
end

function initialize!(cb, u, t, integrator)
    # The AliveCallback is either cb.affect! (with DiscreteCallback)
    # or cb.affect!.affect! (with PeriodicCallback).
    # Let recursive dispatch handle this.
    initialize!(cb.affect!, u, t, integrator)
end

function initialize!(alive_callback::AliveCallback, u, t, integrator)
    alive_callback.start_time = time_ns()

    return nothing
end

# this method is called to determine whether the callback should be activated
function (alive_callback::AliveCallback)(u, t, integrator)
    @unpack interval_or_dt = alive_callback

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return interval_or_dt > 0 && (((integrator.stats.naccept % interval_or_dt == 0) &&
             !(integrator.stats.naccept == 0 && integrator.iter > 0)))
end

# this method is called when the callback is activated
function (alive_callback::AliveCallback)(integrator)
    @unpack io = alive_callback
    t = integrator.t
    t_initial = first(integrator.sol.prob.tspan)
    t_final = last(integrator.sol.prob.tspan)
    sim_time_percentage = (t - t_initial) / (t_final - t_initial) * 100
    runtime_absolute = 1.0e-9 * (time_ns() - alive_callback.start_time)
    println(io,
            rpad(@sprintf("#timesteps: %6d │ Δt: %.4e │ sim. time: %.4e (%5.3f%%)",
                          integrator.stats.naccept, integrator.dt, t,
                          sim_time_percentage), 71) *
            @sprintf("│ run time: %.4e s", runtime_absolute))

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)
    return nothing
end

function finalize(cb, u, t, integrator)
    # The AliveCallback is either cb.affect! (with DiscreteCallback)
    # or cb.affect!.affect! (with PeriodicCallback).
    # Let recursive dispatch handle this.
    finalize(cb.affect!, u, t, integrator)
end

function finalize(alive_callback::AliveCallback, u, t, integrator)
    io = alive_callback.io
    println(io, "─"^100)
    println(io, "Simulation finished.  Final time: ", integrator.t,
            "  Time steps: ", integrator.stats.naccept, " (accepted), ",
            integrator.iter, " (total)")
    println(io, "─"^100)
    println(io)
    return nothing
end
