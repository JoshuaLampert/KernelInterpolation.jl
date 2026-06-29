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
    return state > length(basis) ? nothing : (basis[state], state + 1)
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
struct StandardBasis{Dim, RealT, Kernel} <: AbstractBasis
    centers::NodeSet{Dim, RealT}
    kernel::Kernel
    function StandardBasis(centers::NodeSet, kernel::Kernel) where {Kernel}
        if dim(kernel) != dim(centers)
            throw(DimensionMismatch("The dimension of the kernel and the centers must be the same"))
        end
        return new{dim(centers), eltype(centers), typeof(kernel)}(centers, kernel)
    end
end

Base.getindex(basis::StandardBasis, i) = x -> basis.kernel(x, centers(basis)[i])

@doc raw"""
    LagrangeBasis(centers, kernel, m = order(kernel))

The Lagrange (or cardinal) basis with respect to a kernel and a [`NodeSet`](@ref) of `centers`. This basis
already includes polynomial augmentation of order `m` defaulting to `order(kernel)`. The basis functions are given such that

```math
    b_j(x_i) = \delta_{ij},
```

which means that the [`kernel_matrix`](@ref) of this basis is the identity matrix making it suitable if multiple interpolations
with the same `centers` of the basis and the same `kernel`, but with different right-hand sides or nodesets are performed.
Since the basis already includes polynomials no additional polynomial augmentation is needed for interpolation with this basis.
"""
struct LagrangeBasis{Dim, RealT, Kernel, I <: AbstractInterpolation} <: AbstractBasis
    centers::NodeSet{Dim, RealT}
    kernel::Kernel
    basis_functions::Vector{I}
    function LagrangeBasis(centers::NodeSet{Dim, RealT},
                           kernel::Kernel;
                           m::Integer = order(kernel)) where {Dim, RealT,
                                                              Kernel <: AbstractKernel}
        if dim(kernel) != dim(centers)
            throw(DimensionMismatch("The dimension of the kernel and the centers must be the same"))
        end
        K = length(centers)
        std_basis = StandardBasis(centers, kernel)
        xx = polyvars(Val(Dim))
        ps = monomials(xx, 0:(m - 1))
        q = length(ps)
        # All `K` cardinal functions share the same augmented interpolation matrix and differ
        # only in their right-hand side (the `i`-th unit vector). Assemble and factorize that
        # matrix once and solve for all cardinal functions simultaneously with a multiple
        # right-hand side, instead of re-assembling and re-factorizing it for each one.
        system_matrix = interpolation_matrix(std_basis, ps)
        rhs = [Matrix{RealT}(I, K, K); zeros(RealT, q, K)]
        coefficients = factorize(system_matrix) \ rhs
        # Each cardinal function is an `Interpolation` sharing the same `system_matrix`. The
        # polynomials are baked into the cardinal functions (each carries its own polynomial
        # basis), so no separate basis-level polynomials are stored; cf.
        # `polynomial_basis(basis_functions[i])`.
        Itp = Interpolation{typeof(std_basis), Dim, RealT, typeof(system_matrix),
                            typeof(ps), typeof(xx)}
        basis_functions = [Itp(std_basis, centers, coefficients[:, i], system_matrix, ps,
                               xx)
                           for i in 1:K]
        return new{Dim, RealT, Kernel, eltype(basis_functions)}(centers, kernel,
                                                                basis_functions)
    end
end

Base.getindex(basis::LagrangeBasis, i) = basis.basis_functions[i]
Base.collect(basis::LagrangeBasis) = basis.basis_functions
# Polynomials are already inherently defined included in the basis
order(::LagrangeBasis) = 0
