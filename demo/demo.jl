using Revise
using Jets,DistributedJets

using Distributed
addprocs(2)
@everywhere using Revise
@everywhere using DistributedArrays,DistributedJets,Jets

@everywhere JopFoo_df!(d,m;diagonal,kwargs...) = d .= diagonal .* m
@everywhere function JopFoo(diag)
    spc = JetSpace(Float64, length(diag))
    JopLn(;df! = JopFoo_df!, df′! = JopFoo_df!, dom = spc, rng = spc, s = (diagonal=diag,))
end

@everywhere JopBar_f!(d,m) = d .= m.^2
@everywhere JopBar_df!(δd,δm;mₒ,kwargs...) = δd .= 2 .* mₒ .* δm
function JopBar(n)
    spc = JetSpace(Float64, n)
    JopNl(f! = JopBar_f!, df! = JopBar_df!, df′! = JopBar_df!, dom = spc, rng = spc)
end

_ops = reshape([JopFoo([1.0,2.0]) ; JopFoo([3.0,4.0])],2,1)
ops = distribute(_ops)

A = @blockop ops
B = @blockop _ops

domain(A)
R = range(A)

zeros(domain(A))
x=zeros(range(A))
y=ones(range(A))
z=rand(range(A))

m = ones(domain(A))
m = rand(domain(A))

d = A*m
f = B*m
_f = convert(Array,d)
_f ≈ f

a = A'*d
_a = B'*f

a ≈ _a

DistributedJets.blockproc(R,1)
DistributedJets.blockproc(R,2)

b=block(d, R, 1)
block!(d, R, 1, b)
block!(d, R, 2, b)

d

_ops = reshape([JopBar(2) ; JopBar(2)],2,1)
ops = distribute(_ops)

F = @blockop ops
G = @blockop _ops

m = rand(domain(F))
J = jacobian(F, m)
K = jacobian(G, m)

δ = rand(domain(F))
d = J*δ
_d = K*δ

convert(Array,d) ≈ _d

@everywhere foo(i,j) = JopFoo(rand(2))
ops = DArray(I->[foo(i,j) for i in I[1], j in I[2]], (2,1), workers(), [2,1])

@everywhere g(x) = x^2

fetch(@spawnat 2 typeof(g))

typeof(sum)

ops = [JopFoo(rand(2)) JopFoo(rand(2))]

typeof(ops)
