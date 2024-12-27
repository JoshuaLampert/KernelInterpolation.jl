using KernelInterpolation
using Documenter
import Changelog

# Define module-wide setups such that the respective modules are available in doctests
DocMeta.setdocmeta!(KernelInterpolation, :DocTestSetup, :(using KernelInterpolation);
                    recursive = true)

# Create changelog
Changelog.generate(Changelog.Documenter(),                        # output type
                   joinpath(@__DIR__, "..", "CHANGELOG.md"),      # input file
                   joinpath(@__DIR__, "src", "changelog.md");     # output file
                   repo = "JoshuaLampert/KernelInterpolation.jl", # default repository for links
                   branch = "main",)

makedocs(;
         modules = [KernelInterpolation],
         authors = "Joshua Lampert <joshua.lampert@uni-hamburg.de> and contributors",
         repo = Remotes.GitHub("JoshuaLampert", "KernelInterpolation.jl"),
         sitename = "KernelInterpolation.jl",
         format = Documenter.HTML(;
                                  prettyurls = get(ENV, "CI", "false") == "true",
                                  canonical = "https://JoshuaLampert.github.io/KernelInterpolation.jl/stable",
                                  edit_link = "main",
                                  assets = String[],
                                  mathengine = Documenter.MathJax3()),
         pages = ["Home" => "index.md",
             "Guide" => [
                 "Sets of nodes" => "nodesets.md",
                 "Interpolation" => "interpolation.md",
                 "Solving PDEs by collocation" => "pdes.md"],
             "Tutorials" => [
                 "1D interpolation and differentiation" => "tutorial_differentiating_interpolation.md",
                 "Dealing with noisy data" => "tutorial_noisy_data.md"],
             "Development" => "development.md",
             "Reference" => "ref.md",
             "Changelog" => "changelog.md",
             "License" => "license.md"])

deploydocs(;
           repo = "github.com/JoshuaLampert/KernelInterpolation.jl",
           devbranch = "main",
           push_preview = true)
