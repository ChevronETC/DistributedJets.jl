using Distributed

addprocs(2)
@everywhere using DistributedArrays, DistributedJets, Jets, Test

@everywhere JopFoo_df!(d,m;diagonal,kwargs...) = d .= diagonal .* m
@everywhere function JopFoo(diag)
    spc = JetSpace(Float64, length(diag))
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

@testset "JetDSpace construction" begin
    A = @blockop DArray(I->[JopFoo(rand(2)) for i in I[1], j in I[2]], (2,1))
    R = range(A)
    @test size(R) == (4,)
    @test length(R) == 4
    @test eltype(R) == Float64
    @test eltype(typeof(R)) == Float64
    @test ndims(R) == 1
    @test indices(R) == [1:2,3:4]
    @test procs(R) == workers()
end

@testset "JetDSpace operations" begin
    A = @blockop DArray(I->[JopFoo(rand(2)) for i in I[1], j in I[2]], (2,1))
    R = range(A)
    @test dzeros(4) ≈ zeros(R)
    @test dones(4) ≈ ones(R)
    d = rand(R)
    _d = drand(4)
    @test size(d) == size(_d)
    @test d.cuts == _d.cuts
    @test d.indices == _d.indices
    d = Array(R)
    @test size(d) == size(_d)
    @test d.cuts == _d.cuts
    @test d.indices == _d.indices

    x = getblock(d,R,1)
    x .= π
    setblock!(d,R,1,x)
    @test d[1:2] ≈ [π,π]
    x .= 0
    remotecall_fetch(getblock!, workers()[1], d, R, 1, x) ≈ [π,π]
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
    @test x.cuts == [[1,21,31]]
    x = Array(range(F))
    @test size(x) == (30,)
    @test x.cuts == [[1,21,31]]

    m = rand(domain(F))
    @test collect(F*m) ≈ G*m

    J = jacobian(F, m)
    _J = jacobian(G, m)

    δm = rand(domain(J))
    @test collect(J*δm) ≈ _J*δm

    δd = rand(range(J))
    @test J'*δd ≈ _J'*collect(δd)
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
    @test x.cuts == [[1,21,31]]
    x = Array(range(F))
    @test size(x) == (30,)
    @test x.cuts == [[1,21,31]]

    m = rand(domain(F))
    @test collect(F*m) ≈ G*m

    J = jacobian(F, m)
    _J = jacobian(G, m)

    δm = rand(domain(J))
    @test collect(J*δm) ≈ _J*δm

    δd = rand(range(J))
    @test J'*δd ≈ _J'*collect(δd)
end
