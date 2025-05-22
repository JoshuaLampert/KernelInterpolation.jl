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

Returns the canonical, human-readable name for the given system of equations.
"""
get_name(kernel::AbstractKernel) = string(nameof(typeof(kernel))) * "{" *
                                   string(dim(kernel)) * "}"

function (kernel::AbstractKernel)(x)
    return kernel(x, zero(x))
end

# This allows to evaluate 1D kernels at a scalar, which is sometimes more convenient
function (kernel::AbstractKernel{1})(x::Real, y::AbstractVector)
    @assert length(y) == 1
    return kernel(SVector(x), y)
end
function (kernel::AbstractKernel{1})(x::AbstractVector, y::Real)
    @assert length(x) == 1
    return kernel(x, SVector(y))
end
function (kernel::AbstractKernel{1})(x::Real, y::Real)
    return kernel(SVector(x), SVector(y))
end

include("radialsymmetric_kernel.jl")
include("special_kernel.jl")
