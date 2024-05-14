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

See also [`Laplacian`](@ref).
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

@doc raw"""
    EllipticEquation(A, b, c, f)

Libear second-order elliptic equation with matrix `A`, vector `b`, and scalar `c` and right-hand side `f`.
The elliptic equation is defined as
```math
    \mathcal{L}u = \sum_{i,j = 1}^d a_{ij}(x)\partial_{x_i,x_j}^2u + \sum_{i = 1}^db_i(x)\partial_{x_i}u + c(x)u = f,
```
where `A`, `b` and `c` are space-dependent functions returning a matrix, a vector and a scalar, respectively.

See also [`EllipticOperator`](@ref).
"""
struct EllipticEquation{AType, BType, CType, F} <:
       AbstractStationaryEquation where {AType, BType, CType, F}
    op::EllipticOperator{AType, BType, CType}
    f::F

    function EllipticEquation(A, b, c, f)
        return new{typeof(A), typeof(b), typeof(c), typeof(f)}(EllipticOperator(A, b, c), f)
    end
end

function Base.show(io::IO, ::EllipticEquation)
    print(io,
          "-∑_{i,j = 1}^d aᵢⱼ (x)∂_{x_i,x_j}^2u + ∑_{i = 1}^d bᵢ(x)∂_{x_i}u + c(x)u = f")
end

function (equations::EllipticEquation)(kernel::RadialSymmetricKernel, x, y)
    return equations.op(kernel, x, y)
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
    AdvectionEquation(advection_velocity)

Advection equation with advection velocity `advection_velocity`. The advection equation is defined as
```math
    \partial_t u + \mathbf{a}\cdot\nabla u = f,
```
where ``\mathbf{a}`` is the advection velocity and ``f`` a source term.
"""
struct AdvectionEquation{RealT, F} <: AbstractTimeDependentEquation where {RealT, F}
    advection_velocity::Vector{RealT}
    f::F

    function AdvectionEquation(advection_velocity::Vector{RealT}, f) where {RealT}
        return new{RealT, typeof(f)}(advection_velocity, f)
    end

    function AdvectionEquation(advection_velocity::NTuple, f)
        return new{eltype(advection_velocity), typeof(f)}(collect(advection_velocity), f)
    end
end

function Base.show(io::IO, ::AdvectionEquation)
    print(io, "∂_t u + a⋅∇u = f")
end

function (equations::AdvectionEquation)(kernel::RadialSymmetricKernel, x, y)
    return dot(equations.advection_velocity, Gradient()(kernel, x, y))
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

@doc raw"""
    AdvectionDiffusionEquation(diffusivity, advection_velocity, f)

Advection-diffusion equation with diffusivity `diffusivity` and advection velocity `advection_velocity`.
The advection-diffusion equation is defined as
```math
    \partial_t u + \mathbf{a}\cdot\nabla u = \kappa\Delta u + f,
```
where ``\mathbf{a}`` is the advection velocity, ``\kappa`` is the diffusivity, and ``f`` is the right-hand side,
which can be a time- and space-dependent function or a vector.
"""
struct AdvectionDiffusionEquation{RealT, F} <:
       AbstractTimeDependentEquation where {RealT, F}
    diffusivity::RealT
    advection_velocity::Vector{RealT}
    f::F

    function AdvectionDiffusionEquation(diffusivity::RealT,
                                        advection_velocity::Vector{RealT}, f) where {RealT}
        return new{typeof(diffusivity), typeof(f)}(diffusivity, advection_velocity, f)
    end

    function AdvectionDiffusionEquation(diffusivity::RealT, advection_velocity::NTuple,
                                        f) where {RealT}
        return new{typeof(diffusivity), typeof(f)}(diffusivity, collect(advection_velocity),
                                                   f)
    end
end

function Base.show(io::IO, ::AdvectionDiffusionEquation)
    print(io, "∂_t u + a⋅∇u = κΔu + f")
end

function (equations::AdvectionDiffusionEquation)(kernel::RadialSymmetricKernel, x, y)
    return dot(equations.advection_velocity, Gradient()(kernel, x, y)) -
           equations.diffusivity * Laplacian()(kernel, x, y)
end
