module DistributedJets

using Distributed, DistributedArrays, Jets, LinearAlgebra, ParallelOperations
import Jets:JetAbstractSpace, JetBSpace, JetBlock, JopAdjoint, df!, df′!, f!, point!

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
    blkindices::DArray{UnitRange{Int},1,Array{UnitRange{Int},1}}
    indices::Vector{UnitRange{Int}}
end

Base.size(R::JetDSpace) = (R.indices[end][end],)
Base.eltype(R::Type{JetDSpace{T,S}}) where {T,S} = T
Base.eltype(R::Type{JetDSpace{T}}) where {T} = T
Jets.indices(R::JetDSpace) = R.indices
Distributed.procs(R::JetDSpace) = procs(R.blkspaces)

Jets.space(R::JetDSpace, iblock::Integer) where {T,S} = R.blkspaces[iblock]

function blockproc(S::JetDSpace, iblock::Integer)
    rngs = indices(S.blkspaces, 1)
    S.blkspaces.pids[findfirst(rng->iblock∈rng, rngs)]
end

function getblocklocal!(x, R, iblock, xblock)
    for (i,j) in enumerate(R.blkindices[iblock])
        xblock[i] = x[j]
    end
    xblock
end
getblocklocal(x::DArray, R::JetDSpace, iblock::Integer) = getblocklocal!(x, R, iblock, Array(space(R, iblock)))
Jets.getblock!(x::DArray, R::JetDSpace, iblock::Integer, xblock::AbstractArray) = remotecall_fetch(getblocklocal!, blockproc(R, iblock), x, R, iblock, xblock)
Jets.getblock(x::DArray, R::JetDSpace, iblock::Integer) = remotecall_fetch(getblocklocal, blockproc(R, iblock), x, R, iblock)

function Jets.setblock!(x::DArray, R::JetDSpace, iblock::Integer, xblock::AbstractArray)
    function _block!(x, R, iblock, xblock)
        j0 = localindices(x)[1][1]
        _x = localpart(x)
        for (i,j) in enumerate(R.blkindices[iblock])
            _x[j-j0+1] = xblock[i]
        end
        nothing
    end
    remotecall_fetch(_block!, blockproc(R, iblock), x, R, iblock, xblock)
end

for f in (:ones, :rand, :zeros)
    @eval (Base.$f)(R::JetDSpace{T}) where {T} = DArray(I->($f)(T, length(I[1])), procs(R), indices(R))
end
Base.Array(R::JetDSpace{T}) where {T} = DArray(I->Array{T,1}(undef, length(I[1])), procs(R), indices(R))

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

    blkranges(jets) = DArray(I->[range(jets[i]) for i in I[1]], pids, indices(jets, 1))
    blkspaces = blkranges(jets)

    counts = zeros(Int, length(pids))
    _count(spaces) = mapreduce(length, +, localpart(spaces))
    @sync for (i,pid) in enumerate(pids)
        @async begin
            counts[i] = remotecall_fetch(_count, pid, blkspaces)
        end
    end
    idxs = Vector{UnitRange{Int}}(undef, length(pids))
    lst = 0
    for i = 1:length(idxs)
        frst = lst + 1
        lst = frst + counts[i] - 1
        idxs[i] = frst:lst
    end

    blkidxs = DArray(I->[0:0 for i in I[1]], pids, indices(jets, 1))
    function _blkidxs!(_blkidxs, _blkspaces, firstindex)
        lst = firstindex - 1
        for (i,idx) in enumerate(localindices(_blkidxs)[1])
            frst = lst + 1
            lst = frst + length(_blkspaces[idx]) - 1
            localpart(_blkidxs)[i] = frst:lst
        end
        nothing
    end
    @sync for (ipid,pid) in enumerate(pids)
        @async remotecall_fetch(_blkidxs!, pid, blkidxs, blkspaces, idxs[ipid][1])
    end

    rng = JetDSpace(blkspaces, blkidxs, idxs)

    function build(I, jets)
        irng = indices(jets, 1)[I[1][1]]
        jrng = indices(jets, 2)[I[2][1]]
        _jets = [jets[i,j] for i in irng, j in jrng]
        [JetBlock(_jets) for k=1:1, l=1:1]
    end
    n1,n2 = length(indices(jets, 1)),length(indices(jets, 2))
    _jets = DArray(I->build(I, jets), (n1,n2), procs(jets), [n1,n2])

    Jet(dom = dom, rng = rng, f! = JetDBlock_f!, df! = JetDBlock_df!, df′! = JetDBlock_df′!, s = (jets=_jets, blockmap=indices(jets, 1)))
end

function addmasterpid(pids)
    if myid() ∉ pids
        return [myid();pids]
    end
    pids
end

function JetDBlock_f!(d::DArray, m::AbstractArray; jets, kwargs...)
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

function JetDBlock_df!(d::DArray, m::AbstractArray; jets, kwargs...)
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

function JetDBlock_df′!(m::AbstractArray, d::DArray; jets, kwargs...)
    pids = procs(jets)
    _m = ArrayFutures(m, addmasterpid(pids))
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
    mₒ
end

Distributed.procs(A::Jop{Jet{D,R,typeof(JetDBlock_f!)}}) where {D<:JetAbstractSpace, R<:JetAbstractSpace} = procs(state(A).jets)

Jets.nblocks(jet::Jet{D,R,typeof(JetDBlock_f!)}) where {D,R}= size(state(jet).jets)
Jets.nblocks(jet::Jet{D,R,typeof(JetDBlock_f!)}, i::Integer) where {D,R} = size(state(jet).jets, i)
Jets.nblocks(A::Jop{T}) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetDBlock_f!)}} = nblocks(jet(A))
Jets.nblocks(A::Jop{T}, i::Integer) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetDBlock_f!)}} = nblocks(jet(A), i)

function Jets.getblock(jet::Jet{D,R,typeof(JetDBlock_f!)}, i::Integer, j::Integer) where {D,R}
    jets = state(jet).jets
    blockmap = state(jet).blockmap
    ipid = findfirst(rng->i∈rng, blockmap)
    pid = procs(jets)[ipid]

    function _getblock(jets, i, j, ipid, blockmap)
        state(localpart(jets)[1]).jets[i - blockmap[ipid][1] + 1,j]
    end
    remotecall_fetch(_getblock, pid, jets, i, j, ipid, blockmap)
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
