"""
    AbstractBasis

Abstract type for a basis of a kernel function space. Every basis represents a
set of functions, which can be obtained by indexing the basis object. Every basis
object holds a kernel function and a [`NodeSet`](@ref) of centers and potentially
more fields depending on the concrete basis type.
"""
abstract type AbstractBasis end

function (basis::AbstractBasis)(x)
    return [basis[i](x) for i in eachindex(basis)]
end

"""
    interpolation_kernel(basis)

Return the kernel from a basis.
"""
interpolation_kernel(basis::AbstractBasis) = basis.kernel

"""
	centers(basis)

Return the centers from a basis object.
"""
centers(basis::AbstractBasis) = basis.centers

"""
    order(basis)

Return the order ``m`` of the polynomial, which is needed by this `basis` for
the interpolation, i.e., the polynomial degree plus 1. If ``m = 0``,
no polynomial is added.
"""
order(basis::AbstractBasis) = order(interpolation_kernel(basis))
dim(basis::AbstractBasis) = dim(basis.centers)
Base.length(basis::AbstractBasis) = length(centers(basis))
Base.eachindex(basis::AbstractBasis) = Base.OneTo(length(basis))
function Base.iterate(basis::AbstractBasis, state = 1)
    state > length(basis) ? nothing : (basis[state], state + 1)
end
Base.collect(basis::AbstractBasis) = Function[basis[i] for i in 1:length(basis)]

function Base.show(io::IO, basis::AbstractBasis)
    return print(io,
                 "$(nameof(typeof(basis))) with $(length(centers(basis))) centers and kernel $(interpolation_kernel(basis)).")
end

@doc raw"""
    StandardBasis(centers, kernel)

The standard basis for a function space defined by a kernel and a [`NodeSet`](@ref) of `centers`.
The basis functions are given by

```math
    b_j(x) = K(x, x_j)
```

where `K` is the kernel and `x_j` are the nodes in `centers`.
"""
struct StandardBasis{Kernel} <: AbstractBasis
    centers::NodeSet
    kernel::Kernel
    function StandardBasis(centers::NodeSet, kernel::Kernel) where {Kernel}
        if dim(kernel) != dim(centers)
            throw(DimensionMismatch("The dimension of the kernel and the centers must be the same"))
        end
        new{typeof(kernel)}(centers, kernel)
    end
end

Base.getindex(basis::StandardBasis, i) = x -> basis.kernel(x, centers(basis)[i])

@doc raw"""
    LagrangeBasis(centers, kernel, m = order(kernel))

The Lagrange (or cardinal) basis with respect to a kernel and a [`NodeSet`](@ref) of `centers`. This basis
already includes polynomial augmentation of degree `m` defaulting to `order(kernel)`. The basis functions are given such that

```math
    b_j(x_i) = \delta_{ij},
```

which means that the [`kernel_matrix`](@ref) of this basis is the identity matrix making it suitable for interpolation. Since the
basis already includes polynomials no additional polynomial augmentation is needed for interpolation with this basis.
"""
struct LagrangeBasis{Kernel, I <: AbstractInterpolation} <: AbstractBasis
    centers::NodeSet
    kernel::Kernel
    basis_functions::Vector{I}
    function LagrangeBasis(centers::NodeSet, kernel::Kernel;
                           m = order(kernel)) where {Kernel}
        if dim(kernel) != dim(centers)
            throw(DimensionMismatch("The dimension of the kernel and the centers must be the same"))
        end
        K = length(centers)
        values = zeros(K)
        values[1] = 1.0
        b = interpolate(centers, values, kernel; m = m)
        basis_functions = Vector{typeof(b)}(undef, K)
        basis_functions[1] = b
        for i in 2:K
            values[i - 1] = 0.0
            values[i] = 1.0
            basis_functions[i] = interpolate(centers, values, kernel; m = m)
        end
        new{typeof(kernel), eltype(basis_functions)}(centers, kernel, basis_functions)
    end
end

Base.getindex(basis::LagrangeBasis, i) = x -> basis.basis_functions[i](x)
Base.collect(basis::LagrangeBasis) = basis.basis_functions
# Polynomials are already inherently defined included in the basis
order(::LagrangeBasis) = 0
