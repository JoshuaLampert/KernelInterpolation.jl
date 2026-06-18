# Development

If you have any suggestions or ideas for improvements or new features, we are pleased to accept and discuss
[issues](https://github.com/JoshuaLampert/KernelInterpolation.jl/issues) or if you are willing to contribute,
feel free to [open a pull request](https://github.com/JoshuaLampert/KernelInterpolation.jl/pulls), even if it
is only fixing a typo or improving the docs.

## Changing KernelInterpolation.jl and running it locally

If you plan to edit KernelInterpolation.jl, you first need to clone a local copy of the repository, which can
be done by using `git`. It is recommended that you create a project, e.g. call it `run`, inside the repository,
where you can add packages that you use during executing and testing KernelInterpolation.jl, but are not needed
by KernelInterpolation.jl. This way you can keep the Project.toml of the main repository clean. To do so, you
can execute the following lines in a terminal:

```sh
git clone https://github.com/JoshuaLampert/KernelInterpolation.jl.git
cd KernelInterpolation.jl
mkdir run
cd run
julia --project=. -e 'using Pkg; Pkg.develop(PackageSpec(path=".."))' # Install local KernelInterpolation.jl clone
julia --project=. -e 'using Pkg; Pkg.add(["Plots", "QuasiMonteCarlo", "Meshes", "OrdinaryDiffEqRosenbrock", "OrdinaryDiffEqNonlinearSolve"])' # Install additional packages
```

If you use other packages for executing KernelInterpolation.jl, you can add them to the project in the `run`
directory in an analogous way as above. To use the Julia project within `run`, be sure to start the Julia REPL
by

```sh
julia --project=.
```

if already inside the the `run` directory or `julia --project=run` if in the main directory of the repo.

## Contributing

KernelInterpolation.jl is an open-source project and we are very happy to accept contributions from the community. You can contribute by openinga an issue or a pull request (PR).
If you want to contribute code by opening a pull request, please make sure that your code follows the conventions of the codebase,
is well-documented, and meaningful tests are added. On each pull request, we run automated checks that verify that all tests pass,
the code is covered by tests, the documentation builds correctly, and that the code is properly formatted (see below).
When opening a PR prefer small, focused changes.

See also the [contributing guidelines](https://github.com/JoshuaLampert/KernelInterpolation.jl/blob/main/CONTRIBUTING.md) for more details also regarding LLM/AI usage.

## Preview of the documentation

If you want to build the documentation locally, you can run

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```

once from the KernelInterpolation.jl main directory to tell [Documenter.jl](https://documenter.juliadocs.org/stable/man/guide/)
to build the documentation of your local clone. To build the documentation, run

```sh
julia --project=docs --color=yes docs/make.jl
```

The resulting `.html` files can then be found in `docs/build/` and you can look at them by opening them in a browser.
For pull requests from the main repository (i.e. not from a fork), the documentation is automatically built and can
be previewed under `https://joshualampert.github.io/KernelInterpolation.jl/previews/PRXXX/` where `XXX` is the number
of the pull request.

## Automated source code formatting

To format the code, KernelInterpolation.jl uses the [SciML style](https://juliaeditorsupport.github.io/JuliaFormatter.jl/stable/sciml_style/) of [JuliaFormatter.jl](https://github.com/JuliaEditorSupport/JuliaFormatter.jl).
For more details you can have a look at the current [.JuliaFormatter.toml](https://github.com/JoshuaLampert/KernelInterpolation.jl/blob/main/.JuliaFormatter.toml)
file that holds the configuration options we use for JuliaFormatter.jl.

Note that we expect all contributions to KernelInterpolation.jl to be formatted with JuliaFormatter.jl before being
merged to the main branch. We ensure this by running an automated check on all PRs that verify that running JuliaFormatter.jl
again will not change the source code.

We currently use v1.0.62 of JuliaFormatter.jl, which can be installed by running

```sh
julia -e 'using Pkg; Pkg.add(PackageSpec(name = "JuliaFormatter", version="1.0.62")); Pkg.pin("JuliaFormatter")'
```

You can then recursively format the core Julia files in the KernelInterpolation.jl repo by executing

```sh
julia -e 'using JuliaFormatter; format(".")'
```
