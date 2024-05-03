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

include("radialsymmetric_kernel.jl")
include("special_kernel.jl")
