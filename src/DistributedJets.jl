#=
type stability / efficiency of JopBlock constructor ??
=#
module DistributedJets

using Distributed, DistributedArrays, Jets, LinearAlgebra, ParallelOperations
import Jets:JetAbstractSpace, JopAdjoint, JopBlock

#
# DArray extensions
#
function DistributedArrays.DArray(init::Function, pids::AbstractVector{Int}, rngs::Vector{UnitRange{Int}}...)
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
Jets.getblock(x::DArray, R::JetDSpace, iblock::Integer) = remotecall_fetch(getblocklocal, blockprop(R, iblock), x, R, iblock)

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

struct JopDBlock{T<:JopBlock,D<:JetAbstractSpace,R<:JetAbstractSpace} <: Jop
    dom::D
    rng::R
    ops::DArray{T, 2, Array{T,2}}
end

function Jets.JopBlock(ops::DArray{T,2}) where {T<:Jop}
    length(ops.cuts[2]) < 3 || error("Distributed support is across rows")

    function _domain(ops)
        _ops = localpart(ops)
        N = size(_ops,2)
        N == 1 && (return domain(_ops[1,1]))
        JetBSpace([domain(_ops[1,i]) for i=1:N])
    end
    dom = remotecall_fetch(_domain, procs(ops)[1], ops)

    pids = procs(ops)[:]

    blkranges(ops) = DArray(I->[range(ops[i]) for i in I[1]], pids, indices(ops, 1))
    blkspaces = blkranges(ops)

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

    blkidxs = DArray(I->[0:0 for i in I[1]], pids, indices(ops, 1))
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

    _ops = DArray(I->[JopBlock([ops[i,j] for i in I[1], j in I[2]]) for k=1:1, l=1:1], pids, indices(ops, 1), indices(ops, 2))

    JopDBlock(dom, rng, _ops)
end

Base.adjoint(A::JopDBlock{D,R,T}) where {D,R,T} = JopAdjoint(A)

Jets.domain(A::JopDBlock) = A.dom
Base.range(A::JopDBlock) = A.rng
Distributed.procs(A::JopDBlock) = procs(A.ops)
Distributed.procs(A::JopAdjoint{T}) where {T<:JopDBlock} = procs(A.op.ops)

Base.getindex(A::JopDBlock, i, j) = A.ops[i,j]
Base.getindex(A::JopAdjoint{T}, i, j) where {T<:JopDBlock} = A.op.ops[j,i]

Jets.indices(A::JopDBlock, i::Integer) = indices(A.ops, i)
Jets.indices(A::JopAdjoint{T}, i::Integer) where {T<:JopDBlock} = indices(A.op.ops, i)

function addmasterpid(pids)
    if myid() ∉ pids
        return [myid();pids]
    end
    pids
end

function LinearAlgebra.mul!(d::DArray, A::JopDBlock, m::AbstractArray)
    pids = procs(A)
    _m = bcast(m, addmasterpid(pids))
    function _mul!(d, A, _m)
        mul!(localpart(d), localpart(A.ops)[1], localpart(_m))
        nothing
    end
    @sync for pid in pids
        @async remotecall_fetch(_mul!, pid, d, A, _m)
    end
    d
end

function LinearAlgebra.mul!(m::AbstractArray, A::JopAdjoint{T}, d::DArray) where {T<:JopDBlock}
    pids = procs(A)
    _m = ArrayFutures(m, addmasterpid(pids))
    function _mul!(m, A, d)
        mul!(localpart(_m), JopAdjoint(localpart(A.op.ops)[1,1]), localpart(d))
        nothing
    end
    @sync for pid in pids
        @async remotecall_fetch(_mul!, pid, _m, A, d)
    end
    reduce!(_m)
    m
end

function Jets.jacobian(A::JopDBlock, m::AbstractArray)
    pids = procs(A)[:]
    _m = bcast(m, addmasterpid(pids))
    ops = DArray(I->[JopBlock([jacobian(A[i,j], localpart(_m)) for i in I[1], j in I[2]]) for k=1:1, l=1:1], pids, indices(A, 1), indices(A, 2))
    JopBlock(ops)
end

export blockproc

end
