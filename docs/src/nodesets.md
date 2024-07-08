# [Sets of nodes](@id nodesets)

Numerical methods based on kernel functions are usually meshfree, i.e. they do not need not any information of
connectivity between the different points (a mesh). Instead, they usually solely use a(n) (unstructured) set of points in space
``X = \{x_1, \ldots, x_N\}\subset\mathbb{R}^d`` with ``N`` nodes ``x_i`` of any dimension ``d\in\mathbb{N}``. These vectors are also
often called, e.g., *points*, *nodes*, *centers*, or *data sites* and sets of nodes are also sometimes called *point clouds*. Since meshes
can sometimes be cumbersome to create and handle especially in higher space dimensions, kernel methods are often convenient and
flexible for high-dimensional problems.
In KernelInterpolation.jl, sets of points are called [`NodeSet`](@ref) and can be of any dimension. You can
create [`NodeSet`](@ref)s simply by passing a matrix, where the rows are the different points or by passing a `Vector` of `Vector`s:

```@example nodesets
using KernelInterpolation
nodes = NodeSet([0.0 0.0
                 0.0 1.0
                 1.0 0.0
                 1.0 1.0])
nodes2 = NodeSet([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
```

One characterization of how well distributed a set of nodes is, is the so-called *separation distance*, which is defined by

```math
q_X = \frac{1}{2}\min\limits_{x_i\neq x_j}\|x_i - x_j\|_2.
```

Geometrically, ``q_X`` is the radius of the largest ball that can be placed around every node in ``X`` such that no two balls
overlap. This quantity depends only on the choice of the nodes and is always computed by KernelInterpolation.jl. It can be accessed
by calling

```@example nodesets
q = separation_distance(nodes)
```

The separation distance usually plays a crucial role when estimating the stability properties of kernel methods. Another important
geometric property of node sets, often essential for the error analysis of kernel methods, is the so-called *fill distance* given by

```math
h_X = \sup_{x\in\Omega}\min_{x_j\in X}\|x - x_j\|_2,
```

which can be interpreted as the radius of the largest ball that can be placed in ``\Omega`` such that the ball does not contain any
point in ``X``. This quantity depends on the choice of a domain ``\Omega\subset\mathbb{R^d}`` that covers ``X`` and can therefore not
solely be computed by the `NodeSet`. However, it can be estimated by creating a fine grid of points inside ``\Omega``. Let's say we take
``\Omega = [0,1]^2``. We can conveniently create a set of equidistant points within any hypercube by calling [`homogeneous_hypercube`](@ref):

```@example nodesets
nodes_fine = homogeneous_hypercube(20, (0.0, 0.0), (1.0, 1.0))
```

This creates a `NodeSet` with 20 nodes equally spaced along both dimensions. The distance matrix of the two sets, i.e. the matrix
with entries ``D_{ij} = \|x_i - \xi_j\|_2`` for ``x_i\in X`` and ``\xi_j`` being the evaluation points in ``\Omega``, can be obtained by
calling the function [`distance_matrix`](@ref):

```@example nodesets
D = distance_matrix(nodes, nodes_fine)
```

Finally, the fill distance is approximated by

```@example nodesets
h = maximum(minimum(D, dims = 1))
```

Note that this is only an estimate. The true fill distance is ``\sqrt{2}/2\approx 0.707`` (and reached by placing ``x\in\Omega`` at
``(0.5, 0.5)^T``). The estimate can be improved by taking a finer evaluation grid.

Next to [`homogeneous_hypercube`](@ref), KernelInterpolation.jl provides additional convenience functions to create specific commonly
used [`NodeSet`](@ref)s. These are [`homogeneous_hypercube`](@ref) to create equally spaced nodes at the boundary of a hypercube,
[`random_hypercube`](@ref) and [`random_hypercube_boundary`](@ref) to create random uniformly distributed nodes inside or at the
boundary of a hypercube, and [`random_hypersphere`](@ref) and [`random_hypersphere_boundary`](@ref) for random uniformly distributed
nodes inside or at the boundary of a hypersphere. Note that the first argument `n` of the `homogeneous_*` functions denotes the number
of points __along each dimension__, while for the `random_*` function it denotes the __number of total generated nodes__.

Other sampling methods for hypercubes of any dimension can be obtained by using the package [QuasiMonteCarlo.jl](https://docs.sciml.ai/QuasiMonteCarlo/stable/).
We can simply pass the (transposed) result of any sampling algorithm from QuasiMonteCarlo.jl to the constructor of [`NodeSet`](@ref).
To create 500 Halton points in a box bounded by ``[-1.0, -1.0, -1.0]`` and ``[2.0, 2.0, 2.0]`` we can, e.g., run:

```@example nodesets
using QuasiMonteCarlo: sample, HaltonSample
nodes_matrix = sample(500, [-1.0, -1.0, -1.0], [2.0, 2.0, 2.0], HaltonSample())
nodes_halton = NodeSet(nodes_matrix')
```

For the available sampling algorithms in QuasiMonteCarlo.jl, see the [overview in the documentation](https://docs.sciml.ai/QuasiMonteCarlo/stable/samplers/).

Another possibility to create more advanced [`NodeSet`](@ref)s is by using the package [Meshes.jl](https://github.com/JuliaGeometry/Meshes.jl) and the sampling
algorithms defined therein. For example, we can create a regularly sampled set of nodes on the surface of a sphere by running:

```@example nodesets
using Meshes: Sphere, Point, RegularSampling, sample
sphere = Sphere(Point(0.0, 0.0, 0.0), 1.0)
sampler = RegularSampling(20, 30)
points = sample(sphere, sampler)
nodes = NodeSet(collect(points))
```

For more information on the available sampling algorithms in Meshes.jl, see the [documentation](https://juliageometry.github.io/Meshes.jl/stable/sampling/).
In the documentation of Meshes.jl, you can also find information on how to create more complex geometries like ellipsoids, tori, and many more.
In general, a `PointSet` from Meshes.jl or a `Vector` of `Point`s can be directly passed to the constructor of a [`NodeSet`](@ref) and vice versa can a
[`NodeSet`](@ref) be passed to the constructor of a `PointSet`.

More complicated [`NodeSet`](@ref)s consisting of different shapes can be created, e.g., by `merge`ing different [`NodeSet`](@ref)s.

## Visualizing [`NodeSet`](@ref)s

To visualize a [`NodeSet`](@ref), there are currently two possibilities. The first one uses [Plots.jl](https://docs.juliaplots.org/stable/).
After installing and loading Plots.jl, we can then simply call `plot` on any 1D, 2D, or 3D [`NodeSet`](@ref) to plot it.

```@example nodesets
using Plots
plot(nodes_halton)
savefig("nodes_halton.png") # hide
nothing # Avoid showing the path # hide
```

![Halton nodes](nodes_halton.png)

You might want to consider using other plotting backends, e.g. [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl) can be used by
additionally calling `pyplot()` before `plot` in the above code snippet. Refer to the [documentation of Plots.jl](https://docs.juliaplots.org/stable/backends/)
for the different available backends.
In order to color the nodes according to the values of a function (or an [`Interpolation`](@ref)) at the nodes,
you can additionally pass the vector of function values as keyword argument `zcolor` (note that you can treat a [`NodeSet`](@ref)
as a usual array, e.g., broadcasting works with the common dot syntax).

```@example nodesets
f(x) = sinpi(x[1])
plot(nodes_halton, zcolor = f.(nodes_halton))
savefig("nodes_halton_function.png") # hide
nothing # Avoid showing the path # hide
```

![Halton nodes with function values](nodes_halton_function.png)

For 1D or 2D [`NodeSet`](@ref)s you can also pass a function (or, again, an object of [`Interpolation`](@ref)),
which is then used to determine the values in the vertical direction. For a surface plot of a function based on a set of nodes,
you can, e.g., run the following

```@example nodesets
plot(nodes_fine, f, st = :surface)
savefig("nodes_fine.png") # hide
nothing # Avoid showing the path # hide
```

![Surface plot](nodes_fine.png)

As an alternative to plotting from within Julia, you can save [`NodeSet`](@ref)s to the commonly used [VTK files](https://vtk.org/)
and then view the result, e.g., in [ParaView](https://www.paraview.org/) or [VisIt](https://visit-dav.github.io/visit-website/). You can
save a [`NodeSet`](@ref) simply by using [`vtk_save`](@ref) and passing a filename as well as the [`NodeSet`](@ref):

```@example nodesets
vtk_save("nodes_halton", nodes_halton)
```

Again, you can additionally save node values by passing additional functions or vectors (of the same size as the [`NodeSet`](@ref)),
which can also be visualized with ParaView or VisIt.
Note that you can also read back in a [`NodeSet`](@ref) (and possibly the additional node values) by using [`vtk_read`](@ref):

```@example nodesets
nodes_halton2, _ = vtk_read("nodes_halton.vtu")
rm("nodes_halton.vtu") #clean up again # hide
all(nodes_halton2 .== nodes_halton)
```
