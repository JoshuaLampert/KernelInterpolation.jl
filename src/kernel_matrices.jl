@doc raw"""
kernel_matrix(nodeset1, nodeset2, kernel)

Return the kernel matrix for the nodeset and kernel. The kernel matrix is defined as
```math
A_{ij} = K(x_i, \xi_j),
```
where ``x_i`` are the nodes in the `nodeset1`, ``\xi_j`` are the nodes in the `nodeset2`,
and ``K`` the kernel.
"""
function kernel_matrix(nodeset1, nodeset2, kernel)
    n = length(nodeset1)
    m = length(nodeset2)
    A = Matrix{eltype(nodeset1)}(undef, n, m)
    for i in 1:n
        for j in 1:m
            A[i, j] = kernel(nodeset1[i], nodeset2[j])
        end
    end
    return A
end

"""
kernel_matrix(nodeset, kernel)

Return the kernel matrix for the nodeset and kernel. The kernel matrix is defined as
```math
A_{ij} = K(x_i, x_j),
```
where ``x_i`` are the nodes in the nodeset and ``K`` the kernel.
"""
function kernel_matrix(nodeset, kernel)
    kernel_matrix(nodeset, nodeset, kernel)
end

"""
polynomial_matrix(nodeset, ps)

Return the polynomial matrix for the nodeset and polynomials. The polynomial matrix is defined as
```math
A_{ij} = p_j(x_i),
```
where ``x_i`` are the nodes in the nodeset and ``p_j`` the polynomials.
"""
function polynomial_matrix(nodeset, ps)
    n = length(nodeset)
    q = length(ps)
    A = Matrix{eltype(nodeset)}(undef, n, q)
    xx = polyvars(dim(nodeset))
    for i in 1:n
        for j in 1:q
            A[i, j] = ps[j](xx => nodeset[i])
        end
    end
    return A
end

@doc raw"""
    pde_matrix(equations, nodeset_inner, nodeset, kernel)

Compute the matrix of a partial differential equation with a given kernel. The matrix is defined as
```math
    A_{ij} = equations(kernel, x_i, \xi_j),
```
where ``x_i`` are the nodes in `nodeset_inner` and ``\xi_j``` are the nodes in `nodeset`.
"""
function pde_matrix(equations, nodeset_inner, nodeset, kernel)
    n_i = length(nodeset_inner)
    n = length(nodeset)
    A = Matrix{eltype(nodeset)}(undef, n_i, n)
    for i in 1:n_i
        for j in 1:n
            A[i, j] = equations(kernel, nodeset_inner[i], nodeset[j])
        end
    end
    return A
end
