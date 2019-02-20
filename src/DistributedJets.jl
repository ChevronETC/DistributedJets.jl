module DistributedJets

using Distributed, DistributedArrays, Jets, LinearAlgebra, ParallelOperations
import Jets:BlockArray, JetAbstractSpace, JetBSpace, JetBlock, JopAdjoint, df!, df′!, f!, point!

#
# DArray extensions
#
function DistributedArrays.DArray(init::Function, _pids::AbstractArray{Int}, rngs::Vector{UnitRange{Int}}...)
    pids = vec(_pids)
    np = mapreduce(length, *, rngs)
    @assert length(pids) >= np
    pids = reshape(pids[1:np], map(rng->length(rng), rngs))
    idxs,cuts = DistributedArrays.chunk_idxs(rngs...)
    dims = ntuple(i->idxs[end][i][end], length(rngs))
    id = DistributedArrays.next_did()
    DistributedArrays.DArray(id, init, dims, pids, idxs, cuts)
end

function DistributedArrays.chunk_idxs(rngs::Vector{UnitRange{Int}}...)
    chunks = map(rng->length(rng), rngs)
    cuts = [prescribedist(rng) for rng in rngs]
    ndim = length(rngs)
    idxs = Array{NTuple{ndim,UnitRange{Int}}}(undef, chunks...)
    for cidx in CartesianIndices(chunks)
        idxs[cidx] = ntuple(i->(cuts[i][cidx[i]]:cuts[i][cidx[i]+1]-1), ndim)
    end
    idxs, cuts
end

function prescribedist(rng::Vector{UnitRange{Int}})
    n = length(rng)
    cuts = Vector{Int}(undef, n+1)
    for j=1:n
        cuts[j] = rng[j][1]
    end
    cuts[n+1] = rng[n][end]+1
    cuts
end

Jets.indices(cuts::AbstractVector{Int}) = [cuts[i]:(cuts[i+1]-1) for i=1:(length(cuts)-1)]
Jets.indices(A::DArray{T}, i::Integer) where {T} = indices(A.cuts[i])

#
# Distributed block operators
#
struct JetDSpace{T,S<:JetAbstractSpace{T,1}} <: JetAbstractSpace{T,1}
    blkspaces::DArray{S,1,Array{S,1}}
    blkindices::Vector{UnitRange{Int}}
    indices::Vector{UnitRange{Int}}
end

Base.size(R::JetDSpace) = (R.indices[end][end],)
Base.eltype(R::Type{JetDSpace{T,S}}) where {T,S} = T
Base.eltype(R::Type{JetDSpace{T}}) where {T} = T

Jets.indices(R::JetDSpace) = R.indices
Jets.space(R::JetDSpace, iblock::Integer) where {T,S} = R.blkspaces[iblock]
Jets.nblocks(R::JetDSpace) = R.blkindices[end][end]

Distributed.procs(R::JetDSpace) = procs(R.blkspaces)
Distributed.nprocs(R::JetDSpace) = length(procs(R.blkspaces))

struct DBArray{T,A<:BlockArray{T}} <: AbstractArray{T,1}
    darray::DArray{T,1,A}
    blkindices::Vector{UnitRange{Int}}
end

Base.IndexStyle(::Type{T}) where {T<:DBArray} = IndexLinear()
Base.size(x::DBArray) = size(x.darray)
Base.getindex(x::DBArray, i::Int) = x.darray[i]

DistributedArrays.localpart(x::DBArray) = localpart(x.darray)
Distributed.procs(x::DBArray) = procs(x.darray)
Jets.nblocks(x::DBArray) = x.blkindices[end][end]

# I'm begin lazy here... probably better to figure out how to use broadcasting...
Base.isapprox(x::DBArray, y::DBArray; kwargs...) = isapprox(x.darray, y.darray; kwargs...)
Base.isapprox(x::DBArray, y::DArray; kwargs...) = isapprox(x.darray, y; kwargs...)
Base.isapprox(x::DArray, y::DBArray; kwargs...) = isapprox(x, y.darray; kwargs...)

function Base.collect(x::DBArray{T,A}) where {T,A}
    _x = A[]
    _indices = UnitRange{Int}[]
    n = 0
    for pid in procs(x)
        y = remotecall_fetch(localpart, pid, x.darray)
        _x = [_x; y.arrays]
        for i = 1:length(y.indices)
            push!(_indices, ((n+y.indices[i][1]):(n+y.indices[i][end])))
        end
        n = _indices[end][end]
    end
    BlockArray(_x, _indices)
end

Base.convert(::BlockArray, x::DBArray) = collect(x)
Base.convert(::Array, x::DBArray) = convert(Array, collect(x))
# end of being lazy

DistributedArrays.empty_localpart(T,N,::Type{A}) where {A<:BlockArray} = BlockArray([Array{T}(undef, ntuple(zero, N))], [0:0])
for f in (:Array, :ones, :rand, :zeros)
    @eval (Base.$f)(R::JetDSpace) = DBArray(DArray(i->($f)(localpart(R.blkspaces)[1]), procs(R), indices(R)), R.blkindices)
end

getblocklocal(x::DBArray, δblock) = getblock(localpart(x), δblock)
function Jets.getblock(x::DBArray{T,A}, iblock::Integer) where {T,B,A<:BlockArray{T,B}}
    ipid = findfirst(rng->iblock∈rng, x.blkindices)
    remotecall_fetch(getblocklocal, procs(x)[ipid], x, iblock - x.blkindices[ipid][1] + 1)::B
end
function Jets.getblock!(x::DBArray{T,A}, iblock::Integer, xblock::AbstractArray) where {T,B,A<:BlockArray{T,B}}
    ipid = findfirst(rng->iblock∈rng, x.blkindices)::Int
    xblock .= remotecall_fetch(getblocklocal, procs(x)[ipid], x, iblock - x.blkindices[ipid][1] + 1)::B
end

function setblocklocal!(x::DBArray, δblock::Integer, xblock)
    setblock!(localpart(x), δblock, xblock)
    nothing
end
function Jets.setblock!(x::DBArray, iblock::Integer, xblock)
    ipid = findfirst(rng->iblock∈rng, x.blkindices)::Int
    remotecall_fetch(setblocklocal!, procs(x)[ipid], x, iblock - x.blkindices[ipid][1] + 1, xblock)
end

function Jets.JopBlock(A::DArray{T,2}) where {T<:Jop}
    jets = DArray(I->[jet(A[irow,icol]) for irow in I[1], icol in I[2]], procs(A), indices(A,1), indices(A,2))
    JopNl(JetBlock(jets))
end

function Jets.JopBlock(A::DArray{T,2}) where {T<:JopLn}
    jets = DArray(I->[jet(A[irow,icol]) for irow in I[1], icol in I[2]], procs(A), indices(A,1), indices(A,2))
    JopLn(JetBlock(jets))
end

function Jets.JetBlock(jets::DArray{T,2}) where {T<:Jet}
    length(jets.cuts[2]) < 3 || error("Distributed support is across rows")

    function _domain(jets)
        _jets = localpart(jets)
        N = size(_jets,2)
        N == 1 && (return domain(_jets[1,1]))
        JetBSpace([domain(_jets[1,i]) for i=1:N])
    end
    dom = remotecall_fetch(_domain, procs(jets)[1], jets)

    pids = procs(jets)[:]

    function build(I, jets)
        irng = indices(jets, 1)[I[1][1]]
        jrng = indices(jets, 2)[I[2][1]]
        _jets = [jets[i,j] for i in irng, j in jrng]
        [JetBlock(_jets) for k=1:1, l=1:1]
    end
    n1,n2 = length(indices(jets, 1)),length(indices(jets, 2))
    _jets = DArray(I->build(I, jets), (n1,n2), pids, [n1,n2])

    blkspaces = DArray(I->[range(localpart(_jets)[1]) for i=1:1], (n1,), pids, [n1])

    countidxs = zeros(Int, n1)
    countblks = zeros(Int, n1)
    _countidxs(spaces) = length(localpart(spaces)[1])
    _countblks(spaces) = nblocks(localpart(spaces)[1])
    @sync for (i,pid) in enumerate(pids)
        @async begin
            countidxs[i] = remotecall_fetch(_countidxs, pid, blkspaces)
            countblks[i] = remotecall_fetch(_countblks, pid, blkspaces)
        end
    end
    idxs = Vector{UnitRange{Int}}(undef, length(pids))
    lst = 0
    for i = 1:length(idxs)
        frst = lst + 1
        lst = frst + countidxs[i] - 1
        idxs[i] = frst:lst
    end
    blkidxs = Vector{UnitRange{Int}}(undef, length(pids))
    lst = 0
    for i = 1:length(idxs)
        frst = lst + 1
        lst = frst + countblks[i] - 1
        blkidxs[i] = frst:lst
    end

    rng = JetDSpace(blkspaces, blkidxs, idxs)

    Jet(dom = dom, rng = rng, f! = JetDBlock_f!, df! = JetDBlock_df!, df′! = JetDBlock_df′!, s = (jets=_jets, dom=dom, blockmap=indices(jets, 1)))
end

function addmasterpid(pids)
    if myid() ∉ pids
        return [myid();pids]
    end
    pids
end

function JetDBlock_f!(d::DBArray, m::AbstractArray; jets, kwargs...)
    pids = procs(jets)
    _m = bcast(m, addmasterpid(pids))
    function _f!(d, _m, jets)
        jet = localpart(jets)[1]
        f!(localpart(d), jet, localpart(_m); jets=state(jet).jets, dom=domain(jet), rng=range(jet))
        nothing
    end
    @sync for pid in pids
        @async remotecall_fetch(_f!, pid, d, _m, jets)
    end
    d
end

function JetDBlock_df!(d::DBArray, m::AbstractArray; jets, kwargs...)
    pids = procs(jets)
    _m = bcast(m, addmasterpid(pids))
    function _df!(d, _m, jets)
        jet = localpart(jets)[1]
        df!(localpart(d), jet, localpart(_m); jets=state(jet).jets, dom=domain(jet), rng=range(jet))
        nothing
    end
    @sync for pid in pids
        @async remotecall_fetch(_df!, pid, d, _m, jets)
    end
    d
end

function JetDBlock_df′!(m::AbstractArray, d::DBArray; jets, dom, kwargs...)
    pids = procs(jets)
    _m = TypeFutures(m, zeros, dom)
    function _df′!(_m, d, jets)
        jet = localpart(jets)[1]
        df′!(localpart(_m), jet, localpart(d); jets=state(jet).jets, dom=domain(jet), rng=range(jet))
        nothing
    end
    @sync for pid in pids
        @async remotecall_fetch(_df′!, pid, _m, d, jets)
    end
    reduce!(_m)
    m
end

function Jets.point!(jet::Jet{D,R,typeof(JetDBlock_f!)}, mₒ::AbstractArray) where {D<:JetAbstractSpace,R<:JetAbstractSpace}
    jets = state(jet).jets
    pids = procs(jets)
    _mₒ = bcast(mₒ, addmasterpid(pids))
    function _point!(jets, _mₒ)
        jet = localpart(jets)[1]
        point!(jet, localpart(_mₒ))
        nothing
    end
    @sync for pid in pids
        @async remotecall_fetch(_point!, pid, jets, _mₒ)
    end
    jet
end

Distributed.procs(A::Jop{Jet{D,R,typeof(JetDBlock_f!)}}) where {D<:JetAbstractSpace, R<:JetAbstractSpace} = procs(state(A).jets)

function Jets.nblocks(jet::Jet{D,R,typeof(JetDBlock_f!)}, i::Integer) where {D,R}
    if i == 1 # range
        S = range(jet)
        return S.blkindices[end][end]
    end
    S = domain(jet)
    S.blkindices[end][end]
end
Jets.nblocks(jet::Jet{D,R,typeof(JetDBlock_f!)}) where {D<:JetAbstractSpace, R<:JetAbstractSpace} = (nblocks(jet, 1), nblocks(jet, 2))
Jets.nblocks(A::Jop{T}) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetDBlock_f!)}} = nblocks(jet(A))
Jets.nblocks(A::Jop{T}, i::Integer) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetDBlock_f!)}} = nblocks(jet(A), i)

function Jets.getblock(jet::Jet{D,R,typeof(JetDBlock_f!)}, i::Integer, j::Integer) where {D,R}
    jets = state(jet).jets
    blockmap = state(jet).blockmap
    ipid = findfirst(rng->i∈rng, blockmap)
    pid = procs(jets)[ipid]

    _getblock(jets, δi, j) = state(localpart(jets)[1]).jets[δi,j]
    remotecall_fetch(_getblock, pid, jets, i - blockmap[ipid][1] + 1, j)
end

Jets.getblock(A::JopLn{T}, i::Integer, j::Integer) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetDBlock_f!)}} = JopLn(getblock(jet(A), i, j))
Jets.getblock(::Type{JopLn}, A::Jop{T}, i::Integer, j::Integer) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetDBlock_f!)}} = JopLn(getblock(jet(A), i, j))
Jets.getblock(::Type{JopNl}, F::Jop{T}, i::Integer, j::Integer) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetDBlock_f!)}} = JopNl(getblock(jet(F), i, j))
Jets.getblock(A::JopAdjoint{T}, i::Integer, j::Integer) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetDBlock_df!)}} = JopAdjoint(getblock(A.op, j, i))

Jets.indices(jet::Jet{D,R,typeof(JetDBlock_f!)}, i::Integer) where {D,R} = indices(state(A).jets, i)
Jets.indices(A::Jop{T}, i::Integer) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetDBlock_f!)}} = indices(jet(A), i)
Jets.indices(A::JopAdjoint{Jet{D,R,typeof(JetDBlock_f!)}}, i::Integer) where {D,R} = indices(A.op, i == 1 ? 2 : 1)

export blockproc

end
