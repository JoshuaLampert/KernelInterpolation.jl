"""
    AbstractRegularization

An abstract supertype of regularizations. A regularization implements a function
[`regularize!`](@ref) that takes a matrix and returns a regularized version of it.
"""
abstract type AbstractRegularization end

"""
    regularize!(A, reg::AbstractRegularization)

Apply the regularization `reg` to the matrix `A` in place.
"""
function regularize!(A, ::AbstractRegularization) end

"""
    NoRegularization()

A regularization that does nothing.
"""
struct NoRegularization <: AbstractRegularization end

function regularize!(A, ::NoRegularization)
    return nothing
end

"""
    L2Regularization(regularization_parameter::Real)

A regularization that adds a multiple of the identity matrix to the input matrix.
"""
struct L2Regularization{RealT <: Real} <: AbstractRegularization
    regularization_parameter::RealT
end

function regularize!(A, reg::L2Regularization)
    A[diagind(A)] .+= reg.regularization_parameter
    return nothing
end
