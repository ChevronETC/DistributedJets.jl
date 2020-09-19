using Documenter, DistributedJets

makedocs(sitename="DistributedJets", modules=[DistributedJets])

deploydocs(
    repo = "github.com/ChevronETC/DistributedJets.jl.git",
)v
