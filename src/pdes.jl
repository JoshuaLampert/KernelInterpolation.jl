abstract type AbstractPDE end

@doc raw"""
    PoissonEquation(f)

Poisson equation with right-hand side `f`, which can be a function or a vector. The Poisson equation is
defined as
```math
    -Δu = f
```
"""
struct PoissonEquation{F} <: AbstractPDE where {F}
    f::F

    # accept either a function or a vector
    function PoissonEquation(f)
        return new{typeof(f)}(f)
    end
end

function Base.show(io::IO, ::PoissonEquation)
    print(io, "-Δu = f")
end

function (pde::PoissonEquation)(kernel::RadialSymmetricKernel, x, y)
    return -Laplacian()(kernel, x, y)
end

function rhs(pde::PoissonEquation, nodeset::NodeSet)
    return pde.f.(nodeset)
end

function rhs(pde::PoissonEquation{Vector{RealT}}, nodeset::NodeSet) where {RealT}
    return pde.f
end
