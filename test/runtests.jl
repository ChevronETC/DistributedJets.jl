using Distributed
addprocs(2)
@everywhere using DistributedArrays, DistributedJets, Jets, LinearAlgebra, Test

@everywhere JopFoo_df!(d,m;diagonal,kwargs...) = d .= diagonal .* m
@everywhere function JopFoo(diag)
    spc = JetSpace(Float64, size(diag))
    JopLn(;df! = JopFoo_df!, df′! = JopFoo_df!, dom = spc, rng = spc, s = (diagonal=diag,))
end

@everywhere JopBar_f!(d,m) = d .= m.^2
@everywhere JopBar_df!(δd,δm;mₒ,kwargs...) = δd .= 2 .* mₒ .* δm
@everywhere function JopBar(n)
    spc = JetSpace(Float64, n)
    JopNl(f! = JopBar_f!, df! = JopBar_df!, df′! = JopBar_df!, dom = spc, rng = spc)
end

@everywhere JopBaz_df!(d,m;A,kwargs...) = d .= A*m
@everywhere JopBaz_df′!(m,d;A,kwargs...) = m .= A'*d
@everywhere function JopBaz(A)
    dom = JetSpace(eltype(A), size(A,2))
    rng = JetSpace(eltype(A), size(A,1))
    JopLn(;df! = JopBaz_df!, df′! = JopBaz_df′!, dom = dom, rng = rng, s = (A=A,))
end

@testset "DArray irregular construction" for T in (Float32,Float64,Complex{Float32},Complex{Float64})
    A = DArray(I->myid()*ones(T,length(I[1]),length(I[2])), workers(), [1:2,3:10], [1:2])
    @test size(A) == (10,2)
    @test A.indices[1] == (1:2, 1:2)
    @test A.indices[2] == (3:10, 1:2)
    @test indices(A,1) == [1:2,3:10]
    @test indices(A,2) == [1:2]
    @test fetch(@spawnat procs(A)[1] all(a->a≈T(myid()), localpart(A)))
    @test fetch(@spawnat procs(A)[2] all(a->a≈T(myid()), localpart(A)))
    A = DArray(I->myid()*ones(length(I[1])), workers(), [1:2,3:10])
    @test size(A) == (10,)
    @test A.indices[1] == (1:2,)
    @test A.indices[2] == (3:10,)
    @test indices(A,1) == [1:2,3:10]
end

@testset "JetDSpace construction, 1D arrays" begin
    A = @blockop DArray(I->[JopFoo(rand(2)) for i in I[1], j in I[2]], (2,1))
    R = range(A)
    @test size(R) == (4,)
    @test length(R) == 4
    @test eltype(R) == Float64
    @test eltype(typeof(R)) == Float64
    @test ndims(R) == 1
    @test indices(R) == [1:2,3:4]
    @test procs(R) == workers()
    @test nblocks(R) == 2
end

@testset "JetDSpace construction, 2D arrays" begin
    A = @blockop DArray(I->[JopFoo(rand(2,3)) for i in I[1], j in I[2]], (2,1))
    R = range(A)
    @test size(R) == (12,)
    @test length(R) == 12
    @test eltype(R) == Float64
    @test eltype(typeof(R)) == Float64
    @test ndims(R) == 1
    @test indices(R) == [1:6,7:12]
    @test procs(R) == workers()
    @test nblocks(R) == 2
end

@testset "JetDSpace operations, 1D arrays" begin
    A = @blockop DArray(I->[JopFoo(rand(2)) for i in I[1], j in I[2]], (2,1))
    R = range(A)
    @test dzeros(4) ≈ zeros(R)
    @test dones(4) ≈ ones(R)
    d = rand(R)
    _d = drand(4)
    @test size(d) == size(_d)
    @test d.darray.cuts == _d.cuts
    @test d.darray.indices == _d.indices
    d = Array(R)
    @test size(d) == size(_d)
    @test d.darray.cuts == _d.cuts
    @test d.darray.indices == _d.indices

    x = getblock(d,1)
    x .= π
    setblock!(d,1,x)
    @test d[1:2] ≈ [π,π]
    x .= 0
    @test remotecall_fetch(getblock!, workers()[1], d, 1, x) ≈ [π,π]
end

@testset "JetDSpace operations, 2D arrays" begin
    A = @blockop DArray(I->[JopFoo(rand(2,3)) for i in I[1], j in I[2]], (4,1))
    R = range(A)
    @test dzeros(24) ≈ zeros(R)
    @test dones(24) ≈ ones(R)
    d = rand(R)
    _d = drand(24)
    @test size(d) == size(_d)
    @test d.darray.cuts == _d.cuts
    @test d.darray.indices == _d.indices
    d = Array(R)
    @test size(d) == size(_d)
    @test d.darray.cuts == _d.cuts
    @test d.darray.indices == _d.indices

    x = getblock(d,1)
    x .= π
    setblock!(d,1,x)
    @test d[1:6] ≈ [π, π, π, π, π, π]
    x .= 0
    @test remotecall_fetch(getblock!, workers()[1], d, 1, x) ≈ [π π π; π π π]
end

@testset "DBArray, inner product" begin
    A = @blockop DArray(I->[JopFoo(rand(2,3)) for i in I[1], j in I[2]], (4,1))
    d₁ = rand(range(A))
    d₂ = rand(range(A))
    @test dot(d₁, d₂) ≈ dot(convert(Array, d₁), convert(Array, d₂))
end

@testset "DBArray, norm" begin
    A = @blockop DArray(I->[JopFoo(rand(2,3)) for i in I[1], j in I[2]], (4,1))
    d = rand(range(A))
    @test norm(d) ≈ norm(convert(Array, d))
end

@testset "DBArray broadcasting, 1D arrays" begin
    A = @blockop DArray(I->[JopFoo(rand(2)) for i in I[1], j in I[2]], (7,1), workers(), [2,1])
    R = range(A)
    R.blkindices
    R.blkspaces[1]
    d₁ = ones(R)
    d₂ = ones(R)
    d₃ = ones(R)
    d = ones(R)

    α₁ = rand(Float64)
    α₂ = rand(Float64)
    α₃ = rand(Float64)

    d = α₃*d₃

    d .= α₁*d₁ .+ α₂*d₂ .+ α₃*d₃
    α = α₁ + α₂ + α₃

    for i = 1:length(d)
        @test d[i] ≈ α
    end
end

@testset "DBArray broadcasting, 2D arrays" begin
    A = @blockop DArray(I->[JopFoo(rand(2,3)) for i in I[1], j in I[2]], (7,1), workers(), [2,1])
    R = range(A)
    R.blkindices
    R.blkspaces[1]
    d₁ = ones(R)
    d₂ = ones(R)
    d₃ = ones(R)
    d = ones(R)

    α₁ = rand(Float64)
    α₂ = rand(Float64)
    α₃ = rand(Float64)

    d = α₃*d₃

    d .= α₁*d₁ .+ α₂*d₂ .+ α₃*d₃
    α = α₁ + α₂ + α₃

    for i = 1:length(d)
        @test d[i] ≈ α
    end
end

@everywhere function myblocks(i,j)
    if i ∈ (1,2,3) && j ∈ (2,3,1)
        return JopBar(10)
    elseif i == (2,3) && j == 4
        JopZeroBlock(JetSpace(Float64,10),JetSpace(Float64,10))
    elseif i == 2 && j == 4
        JopBaz(rand(10,10)) ∘ JopBar(10)
    else
        return JopBaz(rand(10,10))
    end
end

@testset "JopDBlock, heterogeneous" begin
    _F = DArray(I->[myblocks(i,j) for i in I[1], j in I[2]], (3,4), workers(), [2,1])
    F = @blockop _F

    @test isa(F, JopNl{<:Jet{<:Jets.JetBSpace,<:DistributedJets.JetDSpace,typeof(DistributedJets.JetDBlock_f!)}})

    F₁ = remotecall_fetch(localpart, workers()[1], F)
    F₂ = remotecall_fetch(localpart, workers()[2], F)

    @test isa(F₁, JopNl{<:Jet{<:Jets.JetBSpace,<:Jets.JetBSpace,typeof(Jets.JetBlock_f!)}})

    _G = [_F[i,j] for i in 1:3, j in 1:4]
    G = @blockop _G

    @test procs(F) == procs(state(F).ops)

    @test ones(range(F)) ≈ DArray(I->ones(length(I[1])), procs(_F), [1:20,21:30])
    @test ones(domain(F)) ≈ ones(40)
    @test zeros(range(F)) ≈ DArray(I->zeros(length(I[1])), procs(_F), [1:20,21:30])
    @test zeros(domain(F)) ≈ zeros(40)
    @test size(rand(domain(F))) == (40,)
    x = rand(range(F))
    @test size(x) == (30,)
    @test x.darray.cuts == [[1,21,31]]
    x = Array(range(F))
    @test size(x) == (30,)
    @test x.darray.cuts == [[1,21,31]]

    m = rand(domain(F))

    @test collect(F*m) ≈ G*m
    @test collect(F*m) ≈ [F₁*m ; F₂*m]

    J = jacobian!(F, m)
    _J = jacobian!(G, m)

    δm = rand(domain(J))
    @test collect(J*δm) ≈ _J*δm

    δd = rand(range(J))
    @test J'*δd ≈ _J'*collect(δd)

    F₃₂ = getblock(JopNl,F,3,2)
    G₃₂ = _G[3,2]
    @test F₃₂*m[21:30] ≈ G₃₂*m[21:30]
    A₁₄ = getblock(JopLn,F,1,4)
    B₁₄ = _G[1,4]
    @test A₁₄*m[31:40] ≈ B₁₄*m[31:40]
end

@testset "JopDBlock, homogeneous" begin
    _F = DArray(I->[JopBar(10) for i in I[1], j in I[2]], (3,4), workers(), [2,1])
    F = @blockop _F

    _G = [_F[i,j] for i in 1:3, j in 1:4]
    G = @blockop _G

    @test isa(F, JopNl{<:Jet{<:Jets.JetBSpace,<:DistributedJets.JetDSpace,typeof(DistributedJets.JetDBlock_f!)}})

    @test ones(range(F)) ≈ DArray(I->ones(length(I[1])), procs(_F), [1:20,21:30])
    @test ones(domain(F)) ≈ ones(40)
    @test zeros(range(F)) ≈ DArray(I->zeros(length(I[1])), procs(_F), [1:20,21:30])
    @test zeros(domain(F)) ≈ zeros(40)
    @test size(rand(domain(F))) == (40,)
    x = rand(range(F))
    @test size(x) == (30,)
    @test x.darray.cuts == [[1,21,31]]
    x = Array(range(F))
    @test size(x) == (30,)
    @test x.darray.cuts == [[1,21,31]]

    m = rand(domain(F))
    @test collect(F*m) ≈ G*m

    J = jacobian!(F, m)
    _J = jacobian!(G, m)

    δm = rand(domain(J))
    @test collect(J*δm) ≈ _J*δm

    δd = rand(range(J))
    @test J'*δd ≈ _J'*collect(δd)
end

rmprocs(workers())
