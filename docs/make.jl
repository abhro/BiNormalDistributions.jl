using BiNormalDistributions
using Documenter

DocMeta.setdocmeta!(BiNormalDistributions, :DocTestSetup, :(using BiNormalDistributions); recursive=true)

makedocs(;
    modules=[BiNormalDistributions],
    authors="Abhro R. and contributors",
    sitename="BiNormalDistributions.jl",
    format=Documenter.HTML(;
        canonical="https://abhro.github.io/BiNormalDistributions.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/abhro/BiNormalDistributions.jl",
    devbranch="main",
)
