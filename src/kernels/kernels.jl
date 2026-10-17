"""
    AbstractKernel

An abstract supertype of kernels.
"""
abstract type AbstractKernel{Dim} end

"""
    dim(kernel)

Return the dimension of a kernel, i.e. the size of the input vector.
"""
dim(::AbstractKernel{Dim}) where {Dim} = Dim

"""
    get_name(kernel::AbstractKernel)

Return a human-readable name of the kernel, which is, e.g., used as the title of plots.
"""
get_name(kernel::AbstractKernel) = string(nameof(typeof(kernel))) * "{" *
                                   string(dim(kernel)) * "}"

function (kernel::AbstractKernel)(x)
    return kernel(x, zero(x))
end

# The error is thrown in a separate function, such that building the error message does not
# prevent inlining of the (hot) kernel evaluations, in which the lengths are checked.
@noinline function throw_length_mismatch(x, y)
    throw(DimensionMismatch("x and y must have the same length, got lengths $(length(x)) and " *
                            "$(length(y))."))
end

# This allows to evaluate 1D kernels at a scalar, which is sometimes more convenient
# For multidimensional kernels, this is needed, such that evaluating the kernel at one scalar
# evaluates the basic function, which is exploited, e.g., for plotting the kernel.
function (kernel::AbstractKernel)(x::Real, y::AbstractVector)
    length(y) == 1 || throw_length_mismatch(x, y)
    return kernel(SVector(x), y)
end
function (kernel::AbstractKernel)(x::AbstractVector, y::Real)
    length(x) == 1 || throw_length_mismatch(x, y)
    return kernel(x, SVector(y))
end
function (kernel::AbstractKernel)(x::Real, y::Real)
    return kernel(SVector(x), SVector(y))
end

include("radialsymmetric_kernel.jl")
include("special_kernel.jl")
