"""
    Interpolation

Interpolation object that can be evaluated at a node.
"""
struct Interpolation{K, A, Dim, RealT}
    k::K
    factorized_kernel_matrix::A
    nodeset::NodeSet{Dim, RealT}
    c::Vector{RealT}
end

function Base.show(io::IO, itp::Interpolation)
    return print(io,
                 "Interpolation with $(length(nodeset(itp))) nodes and kernel: $(kernel(itp))")
end

"""
    kernel(itp)

Return the kernel from an interpolation object.
"""
kernel(itp::Interpolation) = itp.k

"""
    nodeset(itp)

Return the node set from an interpolation object.
"""
nodeset(itp::Interpolation) = itp.nodeset

"""
    coefficients(itp)

Obtain the coefficients of the linear combination for the interpolant.
"""
coefficients(itp::Interpolation) = itp.c

"""
    kernel_matrix(itp)

Return the kernel matrix, i.e. the matrix with entries ``a_{ij} = K(x_i, x_j)``
for the kernel function `K` and nodes `x_i`.
"""
kernel_matrix(itp::Interpolation) = itp.factorized_kernel_matrix

"""
    interpolate(nodeset, values, k = GaussKernel())

Interpolate the `values` evaluated at the nodes in the `nodeset` to a function using the kernel `k`.
"""
function interpolate(nodeset::NodeSet{Dim, RealT}, values::Vector{RealT},
                     k = GaussKernel()) where {Dim, RealT}
    n = length(nodeset)
    @assert length(values) == n
    kernel_matrix = Matrix{RealT}(undef, n, n)
    for i in 1:n
        for j in 1:n
            kernel_matrix[i, j] = k(nodeset[i], nodeset[j])
        end
    end
    factorized_kernel_matrix = factorize(kernel_matrix)
    c = factorized_kernel_matrix \ values
    return Interpolation(k, factorized_kernel_matrix, nodeset, c)
end

function (itp::Interpolation)(x)
    s = 0
    c = coefficients(itp)
    k = kernel(itp)
    xs = nodeset(itp)
    for i in 1:length(c)
        s += c[i] * k(x, xs[i])
    end
    return s
end
