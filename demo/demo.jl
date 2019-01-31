using DistributedJets

using Distributed
addprocs(2)
@everywhere using Distributed, DistributedArrays, DistributedJets, Jets

@everywhere function JopFoo(diag)
    df!(d,m;diagonal,kwargs...) = d .= diagonal .* m
    spc = JetSpace(Float64, length(diag))
    JopLn(;df! = df!, df′! = df!, dom = spc, rng = spc, s = (diagonal=diag,))
end

ops = distribute(reshape([JopFoo(rand(2)) ; JopFoo(rand(2))],2,1))

A = @blockop ops

domain(A)
range(A)

zeros(domain(A))
x=zeros(range(A))
y=ones(range(A))
z=rand(range(A))

x.cuts





x.indices



DistributedJets.indices(range(A))

DArray(I->ones(length(I[1])), [1:2,3:4])

B = A'
