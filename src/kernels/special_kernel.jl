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
"""
struct TransformationKernel{Dim, Kernel, Transformation} <: AbstractKernel{Dim}
    kernel::Kernel
    trafo::Transformation
end

function TransformationKernel{Dim}(kernel, transformation) where {Dim}
    return TransformationKernel{Dim, typeof(kernel), typeof(transformation)}(kernel,
                                                                             transformation)
end

function (kernel::TransformationKernel)(x, y)
    @assert length(x) == length(y)
    K = kernel.kernel
    T = kernel.trafo
    return K(T(x), T(y))
end

function Base.show(io::IO, kernel::TransformationKernel{Dim}) where {Dim}
    print(io, "TransformationKernel{", Dim, "}(kernel = ", kernel.kernel, ")")
end

order(kernel::TransformationKernel) = order(kernel.kernel)

@doc raw"""
    ProductKernel{Dim}(kernels)

Given a vector of `kernels`, construct a new kernel that multiplies the
results of the component kernels, i.e., the new kernel ``K`` is given by
```math
    K(x, y) = \prod_{i = 1}^n K_i(x, y),
```
where ``K_i`` are the component kernels and ``n`` the number of kernels.
Note that all component kernels need to have the same [`dim`](@ref).
"""
struct ProductKernel{Dim} <: AbstractKernel{Dim}
    kernels::Vector{AbstractKernel}

    function ProductKernel{Dim}(kernels) where {Dim}
        @assert all(dim.(kernels) .== Dim)
        new(kernels)
    end
end

function (kernel::ProductKernel)(x, y)
    @assert length(x) == length(y)
    res = 1.0
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
    print("])")
end

# TODO: Is that correct in general?
order(kernel::ProductKernel) = maximum(order.(kernel.kernels))

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
"""
struct SumKernel{Dim} <: AbstractKernel{Dim}
    kernels::Vector{AbstractKernel}

    function SumKernel{Dim}(kernels) where {Dim}
        @assert all(dim.(kernels) .== Dim)
        new(kernels)
    end
end

function (kernel::SumKernel)(x, y)
    @assert length(x) == length(y)
    res = 0.0
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
    print("])")
end

# TODO: Is that correct in general?
order(kernel::SumKernel) = minimum(order.(kernel.kernels))

Base.:+(k1::AbstractKernel, k2::AbstractKernel) = SumKernel{dim(k2)}([k1, k2])
