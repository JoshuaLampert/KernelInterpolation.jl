# Dealing with noisy data

This tutorial is based on Chapter 19 of [^Fasshauer2007] and will show how we can use regularization techniques and least-squares
fitting to deal with noisy data. Most of the code for this tutorial is also available in the two examples
[`interpolation/regularization_2d.jl`](https://github.com/JoshuaLampert/KernelInterpolation.jl/blob/main/examples/interpolation/regularization_2d.jl) and
[`interpolation/least_squares_2d.jl`](https://github.com/JoshuaLampert/KernelInterpolation.jl/blob/main/examples/interpolation/least_squares_2d.jl).

## Define problem setup and perform interpolation

We start by defining a simple two-dimensional interpolation problem. We will use the famous Franke function as the
target function and add some noise to the function values. The Franke function is defined as

```math
f(x, y) = \frac{3}{4}\exp\left(-\frac{(9x - 2)^2}{4} - \frac{(9y - 2)^2}{4}\right) + \frac{3}{4}\exp\left(-\frac{(9x + 1)^2}{49} - \frac{9y + 1}{10}\right) + \frac{1}{2}\exp\left(-\frac{(9x - 7)^2}{4} - \frac{(9y - 3)^2}{4}\right) - \frac{1}{5}\exp\left(-(9x - 4)^2 - (9y - 7)^2\right).
```

As nodes for the interpolation, we choose a random set of 1089 points in the unit square.

```@example noisy-itp
using KernelInterpolation
using Random # hide
Random.seed!(1234) # hide

function f(x)
    0.75 * exp(-0.25 * ((9 * x[1] - 2)^2 + (9 * x[2] - 2)^2)) +
    0.75 * exp(-(9 * x[1] + 1)^2 / 49 - (9 * x[2] + 1) / 10) +
    0.5 * exp(-0.25 * ((9 * x[1] - 7)^2 + (9 * x[2] - 3)^2)) -
    0.2 * exp(-(9 * x[1] - 4)^2 - (9 * x[2] - 7)^2)
end

N = 1089
nodeset = random_hypercube(N; dim = 2)
values = f.(nodeset)
values_noisy = values .+ 0.03 * randn(N)
```

As kernel, let's use the [`ThinPlateSplineKernel`](@ref), which uses linear augmentation. We start by performing the interpolation
based on the noisy data.

```@example noisy-itp
kernel = ThinPlateSplineKernel{dim(nodeset)}()
itp = interpolate(nodeset, values_noisy, kernel)
```

We plot the resulting interpolation and compare it to the original Franke function.

```@example noisy-itp
using Plots
p1 = surface(itp, colorbar = false)
p2 = surface(homogeneous_hypercube(40; dim = 2), f, colorbar = false)

plot(p1, p2, layout = (1, 2))
savefig("interpolation_noisy.png") # hide
nothing # hide
```

![Interpolation of noisy function values](interpolation_noisy.png)

We can see that the interpolation looks much rougher than the original Franke function. This is expected since we fit the noisy data too closely.
Therefore, we would like to find a way how to stabilize the approximation and reduce the influence of the noise.

## Use regularization to stabilize the approximation

The first possibility to stabilize the approximation is to use regularization. One of the simplest regularization techniques is the L2-regularization
(or also known as ridge regression).
One way to motivate the L2-regularization is to consider the interpolation problem as a minimization problem. We can write the interpolation problem as

```math
\min_{c \in \mathbb{R}^N} \frac{1}{2}c^TAc
```

subject to the constraint ``Ac = f``, where ``A`` is the interpolation matrix and `f` the function values. This problem can be solved with the help of Lagrange multipliers
and it turns out the solution simply is ``c = A^{-1}f`` as we already know. The idea of L2-regularization is to relax the condition ``Ac = f`` and instead of enforcing the
equality, we penalize the deviation from the equality by adding the L2-norm of the difference. This leads to the minimization problem

```math
\min_{c \in \mathbb{R}^N} \frac{1}{2}c^TAc + \frac{1}{2\lambda}\|Ac - f\|_2^2.
```

Computing the gradient of this expression with respect to ``c`` and setting it to zero, we obtain the regularized solution

```math
c = (A + \lambda I)^{-1}f
```

assuming the regularity and symmetry of the interpolation matrix ``A``. The parameter ``\lambda`` is a regularization parameter that controls the trade-off between
the interpolation error and the regularization term. The larger ``\lambda`` is, the more the interpolation is regularized, which leads to a smoother approximation.
In practice, this means that we only change the interpolation matrix by adding a constant to the diagonal. Note that the polynomial augmentation is not affected by
the regularization. In KernelInterpolation.jl, we can pass a regularizer to the [`interpolate`](@ref) function.

```@example noisy-itp
λ = 0.01
itp_reg = interpolate(nodeset, values_noisy, kernel, regularization = L2Regularization(λ))
```

Plotting the regularized interpolation, we can see that the approximation is much smoother than the unregularized interpolation and thus much closer to the underlying
target function.

```@example noisy-itp
surface(itp_reg, colorbar = false)
savefig("interpolation_noisy_regularized.png") # hide
nothing # hide
```

![Regularized interpolation of noisy function values](interpolation_noisy_regularized.png)

We compare the stability of the regularized and unregularized interpolation by looking a the condition numbers of the two system matrices.

```@example noisy-itp
using LinearAlgebra
A_itp = system_matrix(itp)
A_itp_reg = system_matrix(itp_reg)
cond(A_itp), cond(A_itp_reg)
```

We can see that the condition number is drastically reduced from around `1.5e8` to `1.5e4` by using regularization. This means that the regularized interpolation is much
more stable and less sensitive to the noise in the data.

## Use least-squares approximation to fit noisy data

As an alternative to using regularization, we can also use least-squares fitting to approximate the noisy data. The idea of least-squares approximation is to use
another set of nodes to construct the RBF basis than we use for the interpolation. This means we construct another `NodeSet` consisting of the `centers` for the
basis functions, which is smaller than the `nodeset` we use for the interpolation. If the nodeset is given by ``X = \{x_1, \ldots, x_N\}`` and the `centers` are
``\Xi = \{\xi_1, \ldots, \xi_M\}`` with ``M \le N``, we obtain a rectangular system matrix ``A\in\mathbb{R}^{N\times M}`` with ``A_{ij} = K(x_j, \xi_k)`` for ``j = 1, \ldots, N`` and
``k = 1, \ldots, M``. The overdetermined system ``Ac = f`` can be solved by the least-squares method. Again, only the kernel matrix part is affected by the least-squares
approximation and the polynomial augmentation is not changed. In KernelInterpolation.jl, we can pass `centers` to the [`interpolate`](@ref) function.

```@example noisy-itp
M = 81
centers = random_hypercube(M; dim = 2)
ls = interpolate(nodeset, centers, values_noisy, kernel)
```

We plot the least-squares approximation and, again, see a better fit to the underlying target function.

```@example noisy-itp
surface(ls, colorbar = false)
savefig("interpolation_noisy_least_squares.png") # hide
nothing # hide
```

![Least squares approximation of noisy function values](interpolation_noisy_least_squares.png)

Finally, we compare the error of the three methods to the true data without noise:

```@example noisy-itp
values_itp = itp.(nodeset)
values_itp_reg = itp_reg.(nodeset)
values_ls = ls.(nodeset)
norm(values_itp .- values), norm(values_itp_reg .- values), norm(values_ls .- values)
```

which confirms our findings above as the errors of the stabilized schemes are smaller than the error of the unregularized interpolation.
Note that we did not put much effort in optimizing the regularization parameter or the number of centers for the least-squares approximation
and that there is still room for improvement.

[^Fasshauer2007]:
    Fasshauer (2007):
    Meshfree Approximation Methods with Matlab,
    World Scientific,
    [DOI: 10.1142/6437](https://doi.org/10.1142/6437).
