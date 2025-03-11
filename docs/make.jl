using BiNormalDistribution
using Documenter

DocMeta.setdocmeta!(BiNormalDistribution, :DocTestSetup, :(using BiNormalDistribution); recursive=true)

makedocs(;
    modules=[BiNormalDistribution],
    authors="Abhro R. and contributors",
    sitename="BiNormalDistribution.jl",
    format=Documenter.HTML(;
        canonical="https://abhro.github.io/BiNormalDistribution.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/abhro/BiNormalDistribution.jl",
    devbranch="main",
)
