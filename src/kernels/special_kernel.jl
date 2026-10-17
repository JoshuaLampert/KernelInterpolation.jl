@doc raw"""
    TransformationKernel{Dim}(kernel, transformation)

Given a base `kernel` and a bijective `transformation` function, construct
a new kernel that applies the transformation to both arguments ``x`` and ``y``,
i.e., the new kernel ``K_T`` is given by
```math
    K_T(x, y) = K(Tx, Ty),
```
where ``K`` is the base `kernel` and ``T`` the transformation, i.e. if ``K``
is a kernel of dimension ``d``, ``T`` is a function from dimension `Dim` to ``d``,
where `Dim` is the dimension of the new kernel.

The [`order`](@ref) of the new kernel is the order ``m`` of the base kernel. If the base kernel is
conditionally positive definite of order ``m > 0``, ``K_T`` is conditionally positive definite with
respect to the polynomials ``p\circ T`` of order ``m``. The interpolation, however, is augmented by
polynomials of order ``m`` in ``x``, which span the same space only if ``T`` is affine. Hence, ``T``
has to be affine (and injective) in this case.
"""
struct TransformationKernel{Dim, Kernel, Transformation} <: AbstractKernel{Dim}
    kernel::Kernel
    trafo::Transformation
end

function TransformationKernel{Dim}(kernel, transformation) where {Dim}
    return TransformationKernel{Dim, typeof(kernel), typeof(transformation)}(kernel,
                                                                             transformation)
end

function (kernel::TransformationKernel)(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    K = kernel.kernel
    T = kernel.trafo
    return K(T(x), T(y))
end

function Base.show(io::IO, kernel::TransformationKernel{Dim}) where {Dim}
    print(io, "TransformationKernel{", Dim, "}(kernel = ", kernel.kernel, ")")
    return nothing
end

order(kernel::TransformationKernel) = order(kernel.kernel)
# Assumes the transformation itself is smooth.
smoothness(kernel::TransformationKernel) = smoothness(kernel.kernel)

@doc raw"""
    ProductKernel{Dim}(kernels)

Given a vector of `kernels`, construct a new kernel that multiplies the
results of the component kernels, i.e., the new kernel ``K`` is given by
```math
    K(x, y) = \prod_{i = 1}^n K_i(x, y),
```
where ``K_i`` are the component kernels and ``n`` the number of kernels.
Note that all component kernels need to have the same [`dim`](@ref).
A `ProductKernel` can also be constructed using the `*` operator.

All component kernels have to be positive definite, i.e., of [`order`](@ref) 0, and an
`ArgumentError` is thrown otherwise. The product of positive definite kernels is positive definite
by the Schur product theorem, see Wendland (2004), Theorem 6.2 (6), p. 65, while the product of conditionally
positive definite kernels is in general not conditionally positive definite. For example, the
product of ``\Phi(x) = -\Vert x\Vert`` (order 1) with itself is ``\Vert x\Vert^2``, for which
``\sum_{i,j}c_ic_j\Vert x_i - x_j\Vert^2 = -2\Vert\sum_ic_ix_i\Vert^2\le 0`` if
``\sum_ic_i = 0``.

- Holger Wendland (2004)
  Scattered Data Approximation
  Cambridge University Press
  [DOI: 10.1017/CBO9780511617539](https://doi.org/10.1017/CBO9780511617539)
"""
struct ProductKernel{Dim} <: AbstractKernel{Dim}
    kernels::Vector{AbstractKernel}

    function ProductKernel{Dim}(kernels) where {Dim}
        @assert all(dim.(kernels) .== Dim)
        if any(k -> order(k) > 0, kernels)
            throw(ArgumentError("All component kernels of a `ProductKernel` have to be " *
                                "positive definite, i.e., of order 0, got the orders " *
                                "$(order.(kernels))."))
        end
        return new(kernels)
    end
end

function (kernel::ProductKernel)(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    res = eltype(x)(1.0)
    for k in kernel.kernels
        res *= k(x, y)
    end
    return res
end

function Base.show(io::IO, kernel::ProductKernel{Dim}) where {Dim}
    print(io, "ProductKernel{", Dim, "}(kernels = [")
    for (i, k) in enumerate(kernel.kernels)
        if i < length(kernel.kernels)
            print(io, k, ", ")
        else
            print(io, k)
        end
    end
    print(io, "])")
    return nothing
end

# All component kernels are positive definite, see the constructor
order(::ProductKernel) = 0
smoothness(kernel::ProductKernel) = minimum(smoothness.(kernel.kernels))

Base.:*(k1::AbstractKernel, k2::AbstractKernel) = ProductKernel{dim(k1)}([k1, k2])

@doc raw"""
    SumKernel{Dim}(kernels)

Given a vector of `kernels`, construct a new kernel that sums the
results of the component kernels, i.e., the new kernel ``K`` is given by
```math
    K(x, y) = \sum_{i = 1}^n K_i(x, y),
```
where ``K_i`` are the component kernels and ``n`` the number of kernels.
Note that all component kernels need to have the same [`dim`](@ref).
A `SumKernel` can also be constructed using the `+` operator.

If the component kernels are conditionally positive definite of orders ``m_i``, the sum is
conditionally positive definite of order ``\max_i m_i``, since every conditionally positive
definite kernel of order ``m`` is also conditionally positive definite of every order larger than
``m``. Hence, the [`order`](@ref) of a `SumKernel` is the maximum of the orders of its components.
"""
struct SumKernel{Dim} <: AbstractKernel{Dim}
    kernels::Vector{AbstractKernel}

    function SumKernel{Dim}(kernels) where {Dim}
        @assert all(dim.(kernels) .== Dim)
        return new(kernels)
    end
end

function (kernel::SumKernel)(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    res = zero(eltype(x))
    for k in kernel.kernels
        res += k(x, y)
    end
    return res
end

function Base.show(io::IO, kernel::SumKernel{Dim}) where {Dim}
    print(io, "SumKernel{", Dim, "}(kernels = [")
    for (i, k) in enumerate(kernel.kernels)
        if i < length(kernel.kernels)
            print(io, k, ", ")
        else
            print(io, k)
        end
    end
    print(io, "])")
    return nothing
end

order(kernel::SumKernel) = maximum(order.(kernel.kernels))
smoothness(kernel::SumKernel) = minimum(smoothness.(kernel.kernels))

Base.:+(k1::AbstractKernel, k2::AbstractKernel) = SumKernel{dim(k2)}([k1, k2])
