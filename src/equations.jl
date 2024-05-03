abstract type AbstractEquation end

abstract type AbstractStationaryEquation <: AbstractEquation end

function rhs(nodeset::NodeSet, equations::AbstractStationaryEquation)
    if equations.f isa AbstractVector
        return equations.f
    else
        return equations.f.(nodeset, Ref(equations))
    end
end

@doc raw"""
    PoissonEquation(f)

Poisson equation with right-hand side `f`, which can be a space-dependent function or a vector. The Poisson equation is
defined as
```math
    -\Delta u = f
```
"""
struct PoissonEquation{F} <: AbstractStationaryEquation where {F}
    f::F

    # accept either a function or a vector
    function PoissonEquation(f)
        return new{typeof(f)}(f)
    end
end

function Base.show(io::IO, ::PoissonEquation)
    print(io, "-Δu = f")
end

function (::PoissonEquation)(kernel::RadialSymmetricKernel, x, y)
    return -Laplacian()(kernel, x, y)
end

abstract type AbstractTimeDependentEquation <: AbstractEquation end

function rhs(t, nodeset::NodeSet, equations::AbstractTimeDependentEquation)
    if equations.f isa AbstractVector
        return equations.f
    else
        return equations.f.(Ref(t), nodeset, Ref(equations))
    end
end

@doc raw"""
    HeatEquation(diffusivity, f)

Heat equation with thermal diffusivity `diffusivity`. The heat equation is defined as
```math
    \partial_t u = \kappa\Delta u + f,
```
where ``\kappa`` is the thermal diffusivity and ``f`` is the right-hand side, which can be a time- and space-dependent function or a vector.
"""
struct HeatEquation{RealT, F} <: AbstractTimeDependentEquation
    diffusivity::RealT
    f::F

    function HeatEquation(diffusivity, f)
        return new{typeof(diffusivity), typeof(f)}(diffusivity, f)
    end
end

function Base.show(io::IO, ::HeatEquation)
    print(io, "∂_t u = κΔu + f")
end

function (equations::HeatEquation)(kernel::RadialSymmetricKernel, x, y)
    return -equations.diffusivity * Laplacian()(kernel, x, y)
end
