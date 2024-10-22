# Dealing with noisy data

This tutorial is based on Chapter 19 of [^Fasshauer2007] and will show how we can use regularization techniques and least-squares
fitting to deal with noisy data. Most of the code for this tutorial is also available in the two examples
[`interpolation/regularization_2d.jl`](https://github.com/JoshuaLampert/KernelInterpolation.jl/blob/main/examples/interpolation/regularization_2d.jl) and
[`interpolation/least_squares_2d.jl`](https://github.com/JoshuaLampert/KernelInterpolation.jl/blob/main/examples/interpolation/least_squares_2d.jl).

## Define problem setup and perform interpolation

We start by defining a simple two-dimensional interpolation problem. We will use the famous Franke function as the
target function and add some noise to the function values. The Franke's function is defined as

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
values = f.(nodeset) .+ 0.03 * randn(N)
```

As kernel, let's use the [`ThinPlateSplineKernel`](@ref), which uses linear augmentation. We start by performing the interpolation
based on the noisy data.

```@example noisy-itp
kernel = ThinPlateSplineKernel{dim(nodeset)}()
itp = interpolate(nodeset, values, kernel)
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

We can see that the interpolation looks much rougher than the original Franke function. This is expected since we fit the noisy data too closely.
Therefore, we would like to find a way how to stabilize the approximation and reduce the influence of the noise.

## Use regularization to stabilize the approximation

The first possibility to stabilize the approximation is to use regularization. One of the simplest regularization techniques is the L2-regularization.
One way to motivate the L2-regularization is to consider the interpolation problem as a minimization problem. We can write the interpolation problem as

```math
\min_{c \in \mathbb{R}^N} \frac{1}{2}c^TAc
```

with constraints ``Ac = f``, where `A` is the interpolation matrix and `f` the function values.

## Use least-squares approximation to fit noisy data

[^Fasshauer2007]:
    Fasshauer (2007):
    Meshfree Approximation Methods with Matlab,
    World Scientific,
    [DOI: 10.1142/6437](https://doi.org/10.1142/6437).
