abstract type AbstractDifferentialOperator end

function (D::AbstractDifferentialOperator)(kernel::RadialSymmetricKernel, x, y)
    @assert length(x) == length(y)
    return save_call(D, kernel, x .- y)
end

# Workaround to avoid evaluating the derivative at zeros to allow automatic differentiation,
# see https://github.com/JuliaDiff/ForwardDiff.jl/issues/303
# the same issue appears with Zygote.jl
function save_call(D::AbstractDifferentialOperator, kernel::RadialSymmetricKernel, x)
    if all(iszero, x)
        x[1] = eps(typeof(x[1]))
    end
    # x .+= eps(typeof(x[1]))
    return D(kernel, x)
end

"""
    Laplacian()

The Laplacian operator. It can be called with an `RadialSymmetricKernel` and points
`x` and `y` to evaluate the Laplacian of the `kernel` at `x - y`.
"""
struct Laplacian <: AbstractDifferentialOperator
end

function Base.show(io::IO, ::Laplacian)
    print(io, "Δ")
end

function (::Laplacian)(kernel::RadialSymmetricKernel, x)
    H = ForwardDiff.hessian(x -> Phi(kernel, x), x)
    return tr(H)
end
