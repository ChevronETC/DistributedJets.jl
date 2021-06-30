using Distributed
addprocs(6)
@everywhere using BenchmarkTools, DistributedArrays, DistributedJets, Jets, LinearAlgebra, Statistics

const SUITE = BenchmarkGroup()

@everywhere JopFoo_df!(d,m;diagonal,kwargs...) = d .= diagonal .* m
@everywhere function JopFoo(diag)
    spc = JetSpace(Float64, length(diag))
    JopLn(;df! = JopFoo_df!, dom = spc, rng = spc, s = (diagonal=diag,))
end

@everywhere JopBar_f!(d,m;kwargs...) = d .= m.^2
@everywhere JopBar_df!(δd,δm;mₒ,kwargs...) = δd .= 2 .* mₒ .* δm
@everywhere function JopBar(n)
    spc = JetSpace(Float64, n)
    JopNl(f! = JopBar_f!, df! = JopBar_df!, dom = spc, rng = spc)
end

_F = DArray(I->[JopBar(100) for i in I[1], j in I[2]], (50,10), workers()[1:3], [3,1])
F = @blockop _F
rangeF = range(F)
d₁ = rand(rangeF)
d₂ = rand(rangeF)
d₃ = rand(rangeF)
d₄ = zeros(rangeF)
α₁ = rand(Float64)
α₂ = rand(Float64)
α₃ = rand(Float64)
SUITE["DBArray"] = BenchmarkGroup()
SUITE["DBArray"]["construct"] = @benchmarkable rand($rangeF)
SUITE["DBArray"]["norm"] = @benchmarkable norm($d₁)
SUITE["DBArray"]["dot"] = @benchmarkable dot($d₁,$d₂)
SUITE["DBArray"]["mean"] = @benchmarkable mean($d₁)
SUITE["DBArray"]["var"] = @benchmarkable var($d₁)
SUITE["DBArray"]["var, mean"] = @benchmarkable var($d₁; mean=1.0)
SUITE["DBArray"]["extrema"] = @benchmarkable extrema($d₁)
SUITE["DBArray"]["broadcasting"] = @benchmarkable d₄ .= α₁*d₁ .+ α₂*d₂ .- α₃*d₃

_F = DArray(I->[JopBar(100) for i in I[1], j in I[2]], (50,10), workers()[1:3], [3,1])
F = @blockop _F
domainF = domain(F)
rangeF = range(F)
m = rand(domain(F))
d = rand(range(F))
J = jacobian!(F, m)
SUITE["DBlock, homogeneous"] = BenchmarkGroup()
SUITE["DBlock, homogeneous"]["construct"] = @benchmarkable @blockop $_F
SUITE["DBlock, homogeneous"]["mul!"] = @benchmarkable mul!($d, $F, $m)
SUITE["DBlock, homogeneous"]["mul"] = @benchmarkable $F * $m
SUITE["DBlock, homogeneous"]["jacobian"] = @benchmarkable jacobian!($F, $m)
SUITE["DBlock, homogeneous"]["mul!, linear"] = @benchmarkable mul!($d, $J, $m)
SUITE["DBlock, homogeneous"]["mul, linear"] = @benchmarkable $J * $m
SUITE["DBlock, homogeneous"]["mul!, adjoint"] = @benchmarkable mul!($m, ($J)', $d)
SUITE["DBlock, homogeneous"]["mul, adjoint"] = @benchmarkable ($J)' * $d
SUITE["DBlock, homogeneous"]["adjoint"] = @benchmarkable $(J)'
SUITE["DBlock, homogeneous"]["shape"] = @benchmarkable shape($F)
SUITE["DBlock, homogeneous"]["size"] = @benchmarkable size($F)
SUITE["DBlock, homogeneous"]["domain"] = @benchmarkable domain($F)
SUITE["DBlock, homogeneous"]["range"] = @benchmarkable range($F)
SUITE["DBlock, homogeneous"]["dom,block"] = @benchmarkable getblock($m, 3)
SUITE["DBlock, homogeneous"]["dom,block!"] = @benchmarkable setblock!($m, 3, $(rand(100)))
SUITE["DBlock, homogeneous"]["rng,block"] = @benchmarkable getblock($d, 10)
SUITE["DBlock, homogeneous"]["rng,block!"] = @benchmarkable setblock!($d, 10, $(rand(100)))

F = @blockop _F perfstatfile="stats.json"
J = jacobian(F, m)
SUITE["DBlock, perfstat, homogeneous"] = BenchmarkGroup()
SUITE["DBlock, perfstat, homogeneous"]["mul!"] = @benchmarkable mul!($d, $F, $m)
SUITE["DBlock, perfstat, homogeneous"]["mul"] = @benchmarkable $F * $m
SUITE["DBlock, perfstat, homogeneous"]["mul!, linear"] = @benchmarkable mul!($d, $J, $m)
SUITE["DBlock, perfstat, homogeneous"]["mul, linear"] = @benchmarkable $J * $m
SUITE["DBlock, perfstat, homogeneous"]["mul!, adjoint"] = @benchmarkable mul!($m, ($J)', $d)
SUITE["DBlock, perfstat, homogeneous"]["mul, adjoint"] = @benchmarkable ($J)' * $d
rm("stats.json", force=true)

@everywhere function myblock(i,j)
    if iseven(i) && isodd(j)
        return JopFoo(rand(100))
    else
        return JopBar(100)
    end
end

_F = DArray(I->[myblock(i,j) for i in I[1], j in I[2]], (50,10), workers()[1:3], [3,1])
F = @blockop _F
domainF = domain(F)
rangeF = range(F)
m = rand(domain(F))
d = rand(range(F))
J = jacobian!(F, m)
SUITE["DBlock, heterogeneous"] = BenchmarkGroup()
SUITE["DBlock, heterogeneous"]["construct"] = @benchmarkable @blockop $_F
SUITE["DBlock, heterogeneous"]["mul!"] = @benchmarkable mul!($d, $F, $m)
SUITE["DBlock, heterogeneous"]["mul"] = @benchmarkable $F * $m
SUITE["DBlock, heterogeneous"]["jacobian"] = @benchmarkable jacobian!($F, $m)
SUITE["DBlock, heterogeneous"]["mul!, linear"] = @benchmarkable mul!($d, $J, $m)
SUITE["DBlock, heterogeneous"]["mul, linear"] = @benchmarkable $J * $m
SUITE["DBlock, heterogeneous"]["mul!, adjoint"] = @benchmarkable mul!($m, ($J)', $d)
SUITE["DBlock, heterogeneous"]["mul, adjoint"] = @benchmarkable ($J)' * $d
SUITE["DBlock, heterogeneous"]["adjoint"] = @benchmarkable $(J)'
SUITE["DBlock, heterogeneous"]["shape"] = @benchmarkable shape($F)
SUITE["DBlock, heterogeneous"]["size"] = @benchmarkable size($F)
SUITE["DBlock, heterogeneous"]["domain"] = @benchmarkable domain($F)
SUITE["DBlock, heterogeneous"]["range"] = @benchmarkable range($F)
SUITE["DBlock, heterogeneous"]["dom,block"] = @benchmarkable getblock($m, 3)
SUITE["DBlock, heterogeneous"]["dom,block!"] = @benchmarkable setblock!($m, 3, $(rand(100)))
SUITE["DBlock, heterogeneous"]["rng,block"] = @benchmarkable getblock($d, 10)
SUITE["DBlock, heterogeneous"]["rng,block!"] = @benchmarkable setblock!($d, 10, $(rand(100)))

_F = DArray(I->[JopBar(100) for i in I[1], j in I[2]], (50,10), workers()[1:6], [3,2])
F = @blockop _F
domainF= domain(F)
rangeF = range(F)
m = rand(domain(F))
d = rand(range(F))
J = jacobian!(F, m)
SUITE["DBlock, homogeneous, distributed model"] = BenchmarkGroup()
SUITE["DBlock, homogeneous, distributed model"]["construct"] = @benchmarkable @blockop $_F
SUITE["DBlock, homogeneous, distributed model"]["mul!"] = @benchmarkable mul!($d, $F, $m)
SUITE["DBlock, homogeneous, distributed model"]["mul"] = @benchmarkable $F * $m
SUITE["DBlock, homogeneous, distributed model"]["jacobian"] = @benchmarkable jacobian!($F, $m)
SUITE["DBlock, homogeneous, distributed model"]["mul!, linear"] = @benchmarkable mul!($d, $J, $m)
SUITE["DBlock, homogeneous, distributed model"]["mul, linear"] = @benchmarkable $J * $m
SUITE["DBlock, homogeneous, distributed model"]["mul!, adjoint"] = @benchmarkable mul!($m, ($J)', $d)
SUITE["DBlock, homogeneous, distributed model"]["mul, adjoint"] = @benchmarkable ($J)' * $d
SUITE["DBlock, homogeneous, distributed model"]["adjoint"] = @benchmarkable $(J)'
SUITE["DBlock, homogeneous, distributed model"]["shape"] = @benchmarkable shape($F)
SUITE["DBlock, homogeneous, distributed model"]["size"] = @benchmarkable size($F)
SUITE["DBlock, homogeneous, distributed model"]["domain"] = @benchmarkable domain($F)
SUITE["DBlock, homogeneous, distributed model"]["range"] = @benchmarkable range($F)
SUITE["DBlock, homogeneous, distributed model"]["dom,block"] = @benchmarkable getblock($m, 3)
SUITE["DBlock, homogeneous, distributed model"]["dom,block!"] = @benchmarkable setblock!($m, 3, $(rand(100)))
SUITE["DBlock, homogeneous, distributed model"]["rng,block"] = @benchmarkable getblock($d, 10)
SUITE["DBlock, homogeneous, distributed model"]["rng,block!"] = @benchmarkable setblock!($d, 10, $(rand(100)))

_F = DArray(I->[i == j ? JopBar(100) : JopZeroBlock(JetSpace(Float64,100),JetSpace(Float64,100)) for i in I[1], j in I[2]], (20,20), workers()[1:3], [3,1])
F = @blockop _F isdiag=true
domainF= domain(F)
rangeF = range(F)
m = rand(domain(F))
d = rand(range(F))
J = jacobian!(F, m)
SUITE["DBlock, homogeneous, block diagonal"] = BenchmarkGroup()
SUITE["DBlock, homogeneous, block diagonal"]["construct"] = @benchmarkable @blockop $_F
SUITE["DBlock, homogeneous, block diagonal"]["mul!"] = @benchmarkable mul!($d, $F, $m)
SUITE["DBlock, homogeneous, block diagonal"]["mul"] = @benchmarkable $F * $m
SUITE["DBlock, homogeneous, block diagonal"]["jacobian"] = @benchmarkable jacobian!($F, $m)
SUITE["DBlock, homogeneous, block diagonal"]["mul!, linear"] = @benchmarkable mul!($d, $J, $m)
SUITE["DBlock, homogeneous, block diagonal"]["mul, linear"] = @benchmarkable $J * $m
SUITE["DBlock, homogeneous, block diagonal"]["mul!, adjoint"] = @benchmarkable mul!($m, ($J)', $d)
SUITE["DBlock, homogeneous, block diagonal"]["mul, adjoint"] = @benchmarkable ($J)' * $d
SUITE["DBlock, homogeneous, block diagonal"]["adjoint"] = @benchmarkable $(J)'
SUITE["DBlock, homogeneous, block diagonal"]["shape"] = @benchmarkable shape($F)
SUITE["DBlock, homogeneous, block diagonal"]["size"] = @benchmarkable size($F)
SUITE["DBlock, homogeneous, block diagonal"]["domain"] = @benchmarkable domain($F)
SUITE["DBlock, homogeneous, block diagonal"]["range"] = @benchmarkable range($F)
SUITE["DBlock, homogeneous, block diagonal"]["dom,block"] = @benchmarkable getblock($m, 3)
SUITE["DBlock, homogeneous, block diagonal"]["dom,block!"] = @benchmarkable setblock!($m, 3, $(rand(100)))
SUITE["DBlock, homogeneous, block diagonal"]["rng,block"] = @benchmarkable getblock($d, 10)
SUITE["DBlock, homogeneous, block diagonal"]["rng,block!"] = @benchmarkable setblock!($d, 10, $(rand(100)))
