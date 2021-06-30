using Distributed

#Need to remove all procs before running these tests
addprocs(4)

@everywhere using DistributedArrays, DistributedJets, JSON, Jets, LinearAlgebra, Test

@everywhere JopFoo_df!(d,m;diagonal,kwargs...) = d .= diagonal .* m
@everywhere function JopFoo(diag)
    spc = JetSpace(Float64, size(diag))
    JopLn(;df! = JopFoo_df!, df′! = JopFoo_df!, dom = spc, rng = spc, s = (diagonal=diag,))
end

@everywhere JopClose_df!(d,m;diagonal,kwargs...) = d .= diagonal .* m
@everywhere function JopClose(diag)
    spc = JetSpace(Float64, size(diag))
    file = touch(tempname())
    JopLn(;df! = JopClose_df!, dom = spc, rng = spc, s = (diagonal=diag, file=file))
end
@everywhere Base.close(J::Jet{D,R,typeof(JopClose_df!)}) where {D,R} = rm(state(J).file)

@everywhere JopBar_f!(d,m) = d .= m.^2
@everywhere JopBar_df!(δd,δm;mₒ,kwargs...) = δd .= 2 .* mₒ .* δm
@everywhere function JopBar(n)
    spc = JetSpace(Float64, n)
    JopNl(f! = JopBar_f!, df! = JopBar_df!, df′! = JopBar_df!, dom = spc, rng = spc)
end
@everywhere Jets.perfstat(J::T) where {D,R,T<:Jets.Jet{D,R,typeof(JopBar_f!)}} = Float64(π)

@everywhere JopBaz_df!(d,m;A,kwargs...) = d .= A*m
@everywhere JopBaz_df′!(m,d;A,kwargs...) = m .= A'*d
@everywhere function JopBaz(A)
    dom = JetSpace(eltype(A), size(A,2))
    rng = JetSpace(eltype(A), size(A,1))
    JopLn(;df! = JopBaz_df!, df′! = JopBaz_df′!, dom = dom, rng = rng, s = (A=A,))
end

@testset "DArray irregular construction, T=$T" for T in (Float32,Float64,Complex{Float32},Complex{Float64})
    A = DArray(I->myid()*ones(T,length(I[1]),length(I[2])), workers()[1:2], [1:2,3:10], [1:2])
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
    @test indices(A) == indices(A,1)
end

@testset "JetDSpace construction from JetBSpace array" begin
    _blkspaces = [JetBSpace([JetSpace(Float64,2),JetSpace(Float64,2)]), JetBSpace([JetSpace(Float64,2),JetSpace(Float64,2)])]
    blkspaces = distribute(_blkspaces; procs=workers()[1:2])
    R = JetDSpace(blkspaces)
    @test length(R) == 8
    @test nblocks(R) == 4
    @test blockmap(R) == [1:2,3:4]
    @test R.indices == [1:4,5:8]
end

@testset "JetDSpace construction, 1D arrays" begin
    A = @blockop DArray(I->[JopFoo(rand(2)) for i in I[1], j in I[2]], (2,1), workers()[1:2])
    R = range(A)
    @test size(R) == (4,)
    @test length(R) == 4
    @test eltype(R) == Float64
    @test eltype(typeof(R)) == Float64
    @test ndims(R) == 1
    @test indices(R) == [1:2,3:4]
    @test procs(R) == workers()[1:2]
    @test nprocs(R) == 2
    @test nprocs(A) == 2
    @test nblocks(R) == 2
end

@testset "JetDSpace construction, 2D arrays" begin
    A = @blockop DArray(I->[JopFoo(rand(2,3)) for i in I[1], j in I[2]], (2,1), workers()[1:2])
    R = range(A)
    @test size(R) == (12,)
    @test length(R) == 12
    @test eltype(R) == Float64
    @test eltype(typeof(R)) == Float64
    @test ndims(R) == 1
    @test indices(R) == [1:6,7:12]
    @test procs(R) == workers()[1:2]
    @test nprocs(R) == 2
    @test nprocs(A) == 2
    @test nblocks(R) == 2
end

@testset "JetDSpace operations, 1D arrays" begin
    A = @blockop DArray(I->[JopFoo(rand(2)) for i in I[1], j in I[2]], (2,1), workers()[1:2])
    R = range(A)
    @test dzeros((4,), workers()[1:2]) ≈ zeros(R)
    @test dones((4,), workers()[1:2]) ≈ ones(R)
    d = rand(R)
    _d = drand((4,), workers()[1:2])
    @test size(d) == size(_d)
    @test d.darray.cuts == _d.cuts
    @test d.darray.indices == _d.indices
    d = Array(R)
    @test size(d) == size(_d)
    @test d.darray.cuts == _d.cuts
    @test d.darray.indices == _d.indices
    @test indices(d) == [1:2,3:4]

    y = similar(d)
    @test isa(y, DistributedJets.DBArray{Float64})
    z = similar(d, Float32)
    @test isa(z, DistributedJets.DBArray{Float32})
    _z = similar(d, Float32, length(y))
    @test isa(_z, DistributedJets.DBArray{Float32})
    __z = similar(d, Float32, (length(y),))
    @test isa(__z, DistributedJets.DBArray{Float32})

    @test remotecall_fetch(localindices, workers()[1], R) == 1:2
    @test remotecall_fetch(localindices, workers()[2], R) == 3:4
    @test remotecall_fetch(localblockindices, workers()[1], R) == 1:1
    @test remotecall_fetch(localblockindices, workers()[2], R) == 2:2

    x = getblock(d,1)
    x .= π
    setblock!(d,1,x)
    @test d[1:2] ≈ [π,π]
    x .= 0
    @test remotecall_fetch(getblock!, workers()[1], d, 1, x) ≈ [π,π]
end

@testset "JetDSpace operations, 2D arrays" begin
    A = @blockop DArray(I->[JopFoo(rand(2,3)) for i in I[1], j in I[2]], (4,1), workers()[1:2])
    R = range(A)
    @test dzeros((24,), workers()[1:2]) ≈ zeros(R)
    @test dones((24,), workers()[1:2]) ≈ ones(R)
    d = rand(R)
    _d = drand((24,), workers()[1:2])
    @test size(d) == size(_d)
    @test d.darray.cuts == _d.cuts
    @test d.darray.indices == _d.indices
    d = Array(R)
    @test size(d) == size(_d)
    @test d.darray.cuts == _d.cuts
    @test d.darray.indices == _d.indices
    @test blockmap(d) == [1:2,3:4]

    @test remotecall_fetch(localindices, workers()[1], R) == 1:12
    @test remotecall_fetch(localindices, workers()[2], R) == 13:24
    @test remotecall_fetch(localblockindices, workers()[1], R) == 1:2
    @test remotecall_fetch(localblockindices, workers()[2], R) == 3:4

    x = getblock(d,1)
    x .= π
    setblock!(d,1,x)
    @test d[1:6] ≈ [π, π, π, π, π, π]
    x .= 0
    @test remotecall_fetch(getblock!, workers()[1], d, 1, x) ≈ [π π π; π π π]
end

@testset "DBArray, construction" begin
    @everywhere foo(i) = rem(i,2) == 0 ? i*π*ones(2) : i*π*ones(3)
    nblks = 7
    A = DBArray(foo, (nblks,), workers(), [4])
    B = DBArray(foo, (nblks,), workers())
    C = DBArray(foo, (nblks,))
    D = DBArray(foo, blockmap(A))
    E = DBArray(foo, [1:1,2:2,3:3,4:7])

    blockmap(E)

    E.blkindices
    E.indices

    @test A ≈ B
    @test A ≈ C
    @test A ≈ D
    for i = 1:18
        @test A[i] ≈ E[i]
    end

    @test A.blkindices == B.blkindices
    @test A.blkindices == C.blkindices
    @test A.blkindices == D.blkindices
    @test A.blkindices != E.blkindices
    @test A.indices == B.indices
    @test A.indices == C.indices
    @test A.indices == D.indices
    @test A.indices != E.indices

    for iblock in 1:nblocks(A)
        a = getblock(A, iblock)
        b = getblock(B, iblock)
        c = getblock(C, iblock)
        d = getblock(D, iblock)
        e = getblock(E, iblock)
        n = length(a)
        m = rem(iblock,2) == 0 ? 2 : 3
        @test n == m
        @test length(b) == n
        @test length(c) == n
        @test length(d) == n
        @test length(e) == n
        @test a ≈ iblock*π*ones(m)
        @test b ≈ a
        @test c ≈ a
        @test d ≈ a
        @test e ≈ a
    end
end

@testset "DBArray, inner product" begin
    A = @blockop DArray(I->[JopFoo(rand(2,3)) for i in I[1], j in I[2]], (4,1), workers()[1:2])
    d₁ = rand(range(A))
    d₂ = rand(range(A))
    @test dot(d₁, d₂) ≈ dot(convert(Array, d₁), convert(Array, d₂))
end

@testset "DBArray, norm" begin
    A = @blockop DArray(I->[JopFoo(rand(2,3)) for i in I[1], j in I[2]], (4,1))
    d = rand(range(A))
    @test norm(d) ≈ norm(convert(Array, d))
    @test norm(d, 0) ≈ norm(convert(Array, d), 0)
    @test norm(d, Inf) ≈ norm(convert(Array, d), Inf)

    _d = similar(d, Float32)
    _d .= d
    @test typeof(norm(_d, 0.5)) == Float32
end

@testset "DBArray, extrema" begin
    A = @blockop DArray(I->[JopFoo(rand(2,3)) for i in I[1], j in I[2]], (4,1), workers()[1:2])
    d = rand(range(A))
    mn,mx = extrema(d)
    _mn,_mx = extrema(convert(Array,d))
    @test mn ≈ _mn
    @test mx ≈ _mx
end

@testset "DBArray broadcasting, 1D arrays" begin
    A = @blockop DArray(I->[JopFoo(rand(2)) for i in I[1], j in I[2]], (7,1), workers()[1:2], [2,1])
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

    d₁ .= 3.14

    for i = 1:length(d₁)
        @test d₁[i] ≈ 3.14
    end
end

@testset "DBArray broadcasting, 2D arrays" begin
    A = @blockop DArray(I->[JopFoo(rand(2,3)) for i in I[1], j in I[2]], (7,1), workers()[1:2], [2,1])
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

    d₁ .= 3.14
    for i = 1:length(d₁)
        @test d₁[i] ≈ 3.14
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

@testset "JopDBlock, heterogeneous, tall and skinny" begin
    _F = DArray(I->[myblocks(i,j) for i in I[1], j in I[2]], (3,4), workers()[1:2], [2,1])
    F = @blockop _F

    @test nblocks(F,1) == 3
    @test nblocks(F,2) == 4
    @test nblocks(F) == (3,4)
    @test blockmap(F)[1,1] == (1:2,1:4)
    @test blockmap(F)[2,1] == (3:3,1:4)

    @test isa(F, JopNl{<:Jet{<:JetBSpace,<:DistributedJets.JetDSpace,typeof(DistributedJets.JetDBlock_f!)}})

    F₁ = remotecall_fetch(localpart, workers()[1], F)
    F₂ = remotecall_fetch(localpart, workers()[2], F)

    @test remotecall_fetch(localblockindices, workers()[1], F) == (1:2,1:4)
    @test remotecall_fetch(localblockindices, workers()[2], F) == (3:3,1:4)
    @test remotecall_fetch(localblockindices, workers()[1], F, 1) == 1:2
    @test remotecall_fetch(localblockindices, workers()[1], F, 2) == 1:4

    @test isa(F₁, JopNl{<:Jet{<:JetBSpace,<:JetBSpace,typeof(Jets.JetBlock_f!)}})

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

@testset "JopDBlock, homogeneous, tall and skinny" begin
    _F = DArray(I->[JopBar(10) for i in I[1], j in I[2]], (3,4), workers()[1:2], [2,1])
    F = @blockop _F

    _G = [_F[i,j] for i in 1:3, j in 1:4]
    G = @blockop _G

    @test isa(F, JopNl{<:Jet{<:JetBSpace,<:DistributedJets.JetDSpace,typeof(DistributedJets.JetDBlock_f!)}})

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

@testset "JopDBlock, distributed->distributed, block diagonal, square blocks" begin
    _F = DArray(I->[i==j ? JopBar(10) : JopZeroBlock(JetSpace(Float64,10),JetSpace(Float64,10)) for i in I[1], j in I[2]], (4,4), workers(), [4,1])
    F = @blockop _F isdiag=true

    _G = [_F[i,j] for i in 1:4, j in 1:4]
    G = @blockop _G

    @test isa(F, JopNl{<:Jet{<:DistributedJets.JetDSpace,<:DistributedJets.JetDSpace,typeof(DistributedJets.JetDBlock_f!)}})

    @test ones(range(F)) ≈ DArray(I->ones(length(I[1])), procs(_F), [1:10,11:20,21:30,31:40])
    @test ones(domain(F)) ≈ DArray(I->ones(length(I[1])), procs(_F), [1:10,11:20,21:30,31:40])

    m = rand(domain(F))
    @test collect(F*m) ≈ G*collect(m)

    J = jacobian!(F, m)
    _J = jacobian!(G, collect(m))

    δm = rand(domain(F))
    @test collect(J*δm) ≈ _J*collect(δm)
    δd = rand(range(J))
    @test collect(J'*δd) ≈ _J'*collect(δd)
end

@testset "JopDBlock, distributed->distributed, block diagonal, tall blocks" begin
    _A = DArray(I->[i==j ? JopBaz(rand(10,5)) : JopZeroBlock(JetSpace(Float64,5),JetSpace(Float64,10)) for i in I[1], j in I[2]], (4,4), workers(), [4,1])
    A = @blockop _A isdiag=true

    _B = [_A[i,j] for i in 1:4, j in 1:4]
    B = @blockop _B

    @test isa(A, JopLn{<:Jet{<:DistributedJets.JetDSpace,<:DistributedJets.JetDSpace,typeof(DistributedJets.JetDBlock_f!)}})

    @test ones(range(A)) ≈ DArray(I->ones(length(I[1])), procs(_A), [1:10,11:20,21:30,31:40])
    @test ones(domain(A)) ≈ DArray(I->ones(length(I[1])), procs(_A), [1:5,6:10,11:15,16:20])

    m = rand(domain(A))
    @test collect(A*m) ≈ B*collect(m)
    d = rand(range(A))
    @test collect(A'*d) ≈ B'*collect(d)
end

@testset "JopDBlock, distributed->distributed, block diagonal, number of blocks different from number of procs" begin
    _A = DArray(I->[i==j ? JopBaz(rand(10,5)) : JopZeroBlock(JetSpace(Float64,5),JetSpace(Float64,10)) for i in I[1], j in I[2]], (10,10), workers(), [4,1])
    A = @blockop _A isdiag=true

    _B = [_A[i,j] for i in 1:10, j in 1:10]
    B = @blockop _B

    @test isa(A, JopLn{<:Jet{<:DistributedJets.JetDSpace,<:DistributedJets.JetDSpace,typeof(DistributedJets.JetDBlock_f!)}})

    m = rand(domain(A))
    @test collect(A*m) ≈ B*collect(m)
    d = rand(range(A))
    @test collect(A'*d) ≈ B'*collect(d)
end

@testset "JopDBlock, heterogeneous, distributed->distributed" begin
    _F = DArray(I->Jop[myblocks(i,j) for i in I[1], j in I[2]], (3,5), workers(), [2,2])
    F = @blockop _F

    @test nblocks(F,1) == 3
    @test nblocks(F,2) == 5
    @test nblocks(F) == (3,5)
    @test blockmap(F)[1,1] == (1:2,1:3)
    @test blockmap(F)[2,1] == (3:3,1:3)
    @test blockmap(F)[1,2] == (1:2,4:5)
    @test blockmap(F)[2,2] == (3:3,4:5)

    @test isa(F, JopNl{<:Jet{<:DistributedJets.JetDSpace,<:DistributedJets.JetDSpace,typeof(DistributedJets.JetDBlock_f!)}})

    @test procs(F) == [workers()[1] workers()[3]; workers()[2] workers()[4]]

    @test remotecall_fetch(localblockindices, workers()[1], F) == (1:2,1:3)
    @test remotecall_fetch(localblockindices, workers()[2], F) == (3:3,1:3)
    @test remotecall_fetch(localblockindices, workers()[3], F) == (1:2,4:5)
    @test remotecall_fetch(localblockindices, workers()[4], F) == (3:3,4:5)

    @test remotecall_fetch(localblockindices, workers()[1], F, 1) == 1:2
    @test remotecall_fetch(localblockindices, workers()[1], F, 2) == 1:3

    F₁₁ = remotecall_fetch(localpart, workers()[1], F)
    @test isa(F₁₁, JopNl{<:Jet{<:JetBSpace,<:JetBSpace,typeof(Jets.JetBlock_f!)}})

    _G = [_F[i,j] for i in 1:3, j in 1:5]
    G = @blockop _G

    @test procs(F) == procs(state(F).ops)

    @test ones(range(F)) ≈ DArray(I->ones(length(I[1])), procs(_F), [1:20,21:30])
    @test ones(domain(F)) ≈ DArray(I->ones(length(I[1])), procs(_F), [1:30,31:50])
    @test zeros(range(F)) ≈ DArray(I->zeros(length(I[1])), procs(_F), [1:20,21:30])
    @test zeros(domain(F)) ≈ DArray(I->zeros(length(I[1])), procs(_F), [1:30,31:50])
    @test size(rand(domain(F))) == (50,)
    x = rand(range(F))
    @test size(x) == (30,)
    @test x.darray.cuts == [[1,21,31]]
    x = Array(range(F))
    @test size(x) == (30,)
    @test x.darray.cuts == [[1,21,31]]

    m = rand(domain(F))
    @test collect(F*m) ≈ G*collect(m)

    J = jacobian!(F, m)
    _J = jacobian!(G, m)

    δm = rand(domain(J))
    @test collect(J*δm) ≈ _J*collect(δm)

    δd = rand(range(J))
    @test J'*δd ≈ _J'*collect(δd)

    F₃₂ = getblock(JopNl,F,3,2)
    G₃₂ = _G[3,2]
    @test F₃₂*m[21:30] ≈ G₃₂*m[21:30]
    A₁₄ = getblock(JopLn,F,1,4)
    B₁₄ = _G[1,4]
    @test A₁₄*m[31:40] ≈ B₁₄*m[31:40]
end

@testset "JetDBlock, localpart" begin
    F = @blockop DArray(I->[JopBar(10) for i in I[1], j in I[2]], (3,4), workers()[1:2], [2,1])
    m = rand(domain(F))
    J = jacobian!(F, m)
    _J = remotecall_fetch(localpart, workers()[1], J)
    @test isa(_J, JopLn)
    for jblock = 1:nblocks(_J,2), iblock = 1:nblocks(_J,1)
        @test isa(getblock(_J,iblock,jblock), JopLn)
    end

    __J = @blockop [jacobian(JopBar(10), m[1:10]) jacobian(JopBar(10), m[11:20]) jacobian(JopBar(10), m[21:30]) jacobian(JopBar(10), m[31:40]);
           jacobian(JopBar(10), m[1:10]) jacobian(JopBar(10), m[11:20]) jacobian(JopBar(10), m[21:30]) jacobian(JopBar(10), m[31:40])]
    @test __J*m ≈ _J*m

    JT = J'
    _JT = remotecall_fetch(localpart, workers()[1], JT)

    typeof(_JT)
    getblock(JT, 1, 1)

    for jblock = 1:nblocks(JT,2), iblock = 1:nblocks(JT,1)
        @test isa(getblock(JT,iblock,jblock), JopAdjoint)
    end
end

@testset "JopDBlock, statistics reporting" begin
    rm("stats.json", force=true)

    _F = DArray(I->[JopBar(10) for i in I[1], j in I[2]], (3,4), workers(), [2,1])
    F = @blockop _F perfstatfile="stats.json"

    m = rand(domain(F))
    d = F*m

    s = JSON.parse(read("stats.json", String))
    @test s["step"][1]["pid"][1]["localblock"] ≈ π*ones(8)
    @test s["step"][1]["pid"][2]["localblock"] ≈ π*ones(4)
    @test s["step"][1]["pid"][1]["operation"] == "f"
    @test s["step"][1]["pid"][2]["operation"] == "f"

    J = jacobian(F, m)
    d = J*m

    s = JSON.parse(read("stats.json", String))
    @test s["step"][2]["pid"][1]["localblock"] ≈ π*ones(8)
    @test s["step"][2]["pid"][2]["localblock"] ≈ π*ones(4)
    @test s["step"][2]["pid"][1]["operation"] == "df"
    @test s["step"][2]["pid"][2]["operation"] == "df"

    m = J'*d

    s = JSON.parse(read("stats.json", String))
    @test s["step"][3]["pid"][1]["localblock"] ≈ π*ones(8)
    @test s["step"][3]["pid"][2]["localblock"] ≈ π*ones(4)
    @test s["step"][3]["pid"][1]["operation"] == "df′"
    @test s["step"][3]["pid"][2]["operation"] == "df′"

    rm("stats.json", force=true)
end

@testset "block operator composed with operator" begin
    A₁ = JopFoo(rand(2))
    A₂ = @blockop DArray(I->[JopBar(2) for i in I[1], j in I[2]], (2,1))
    A = A₂ ∘ A₁

    m = rand(domain(A))
    d = A*m
    A₁₁ = getblock(A, 1, 1)
    A₂₁ = getblock(A, 2, 1)
    @test getblock(d,1) ≈ A₂₁ * m
    @test getblock(d,2) ≈ A₂₁ * m
end

@testset "block operators with close" begin
    A = @blockop DArray(I->[JopClose(2) for i in I[1], j in I[2]], (2,1))
    A₁ = getblock(A,1,1)
    A₂ = getblock(A,2,1)
    @test isfile(state(A₁).file)
    @test isfile(state(A₂).file)
    close(A)
    @test !isfile(state(A₁).file)
    @test !isfile(state(A₂).file)
end

@testset "localblockindices for composite operators" begin
    A = @blockop DArray(I->[JopBar(2) for i in I[1], j in I[2]], (2,1))
    B = JopBar(2)
    C = A ∘ B
    @test_throws RemoteException remotecall_fetch(localblockindices, workers()[1], C)
end

@testset "vectorized operators" begin
    A = @blockop DArray(I->[JopFoo(rand(2,3)) for i in I[1], j in I[2]], (2,1))
    x = rand(domain(A))

    @test size(vec(domain(A))) == (2*3,)
    @test size(vec(range(A))) == (2*2*3,)
    @test size(range(A)) == size(vec(range(A)))

    B = vec(A)
    @test isa(jet(B), Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(Jets.JetVec_f!)})

    d = A * x
    _d = vec(A) * vec(x)
    @test d ≈ _d

    a = A' * d
    _a = vec(A') * d
    __a = reshape(_a, domain(A))
    @test a ≈ __a
end

rmprocs(workers())
