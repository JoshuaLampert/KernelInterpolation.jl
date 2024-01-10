@doc raw"""
    TransformationKernel(kernel, transformation)

Given a base `kernel` and a bijective `transformation` function, construct
a new kernel that applies the transformation to both arguments ``x`` and ``y``,
i.e. the new kernel ``K_T`` is given by
```math
    K_T(x, y) = K(Tx, Ty),
```
where ``K`` is the base kernel and ``T`` the transformation.
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
    return print(io, "TransformationKernel{", Dim, "}(kernel = ", kernel.kernel, ")")
end

order(kernel::TransformationKernel) = order(kernel.kernel)
