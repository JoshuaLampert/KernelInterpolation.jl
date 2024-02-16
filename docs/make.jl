using KernelInterpolation
using Documenter

# Define module-wide setups such that the respective modules are available in doctests
DocMeta.setdocmeta!(KernelInterpolation, :DocTestSetup, :(using KernelInterpolation);
                    recursive = true)

makedocs(;
         modules = [KernelInterpolation],
         authors = "Joshua Lampert <joshua.lampert@uni-hamburg.de> and contributors",
         repo = Remotes.GitHub("JoshuaLampert", "KernelInterpolation.jl"),
         sitename = "KernelInterpolation.jl",
         format = Documenter.HTML(;
                                  prettyurls = get(ENV, "CI", "false") == "true",
                                  canonical = "https://JoshuaLampert.github.io/KernelInterpolation.jl/stable",
                                  edit_link = "main",
                                  assets = String[]),
         pages = ["Home" => "index.md",
             "Reference" => "ref.md",
             "License" => "license.md"])

deploydocs(;
           repo = "github.com/JoshuaLampert/KernelInterpolation.jl",
           devbranch = "main")
