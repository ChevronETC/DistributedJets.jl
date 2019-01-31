module DistributedJets

using Distributed, DistributedArrays, Jets, LinearAlgebra, ParallelOperations
import Jets:JetAbstractSpace,JopAdjoint,JopBlock

#
# DArray extensions
#
function DistributedArrays.DArray(init::Function, rngs::Vector{UnitRange{Int}}...; pids=workers())
    @assert length(pids) == length(rngs[1])
    idxs, cuts = DistributedArrays.chunk_idxs(rngs...)
    @show idxs, cuts
    dims = ntuple(i->idxs[end][i][end], length(rngs))
    DistributedArrays.DArray(DistributedArrays.next_did(), init, dims, pids[:], idxs, cuts)
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

indices(cuts::Vector{Int}) = [cuts[i]:(cuts[i+1]-1) for i=1:(length(cuts)-1)]

#
# Distributed block operators
#
struct JetDSpace{T,S<:JetAbstractSpace{T,1}} <: JetAbstractSpace{T,1}
    spaces::DArray{S,1,Array{S,1}}
    indices::Vector{UnitRange{Int}}
    pids::Vector{Int}
end

indices(R::JetDSpace) = R.indices

for f in (:ones, :rand, :zeros)
    @eval (Base.$f)(R::JetDSpace{T}) where {T} = DArray(I->($f)(T, length(I[1])), indices(R))
end

struct JopDBlock{T<:JopBlock,D<:JetAbstractSpace,R<:JetAbstractSpace} <: Jop
    dom::D
    rng::R
    ops::DArray{T, 2, Array{T,2}}
end

function Jets.JopBlock(ops::DArray{T,2}) where {T<:Jop}
    length(ops.cuts[2]) < 3 || error("Distributed support is in the column-sense")

    function _domain(ops)
        _ops = localpart(ops)
        N = size(_ops,2)
        N == 1 && (return domain(_ops[1,1]))
        JetBSpace([domain(_ops[1,i]) for i=1:N])
    end
    dom = remotecall_fetch(_domain, procs(ops)[1], ops)

    ranges(ops) = DArray(I->[range(ops[i]) for i in I[1]], indices(ops.cuts[1]); pids=procs(ops))
    spaces = ranges(ops)

    pids = procs(ops)[:]

    counts = zeros(Int, length(pids))
    _count(spaces) = mapreduce(length, +, localpart(spaces))
    @sync for (i,pid) in enumerate(pids)
        @async begin
            counts[i] = remotecall_fetch(_count, pid, spaces)
        end
    end
    idxs = Vector{UnitRange{Int}}(undef, size(ops,1))
    last = 0
    for i = 1:length(idxs)
        frst = last + 1
        last = frst + counts[i] - 1
        idxs[i] = frst:last
    end
    rng = JetDSpace(spaces, idxs, pids)

    _ops = DArray(I->[JopBlock(localpart(ops))], (length(pids),), pids, [length(pids),1])

    JopDBlock(dom, rng, _ops)
end
Jet.JopBlock(ops::DArray{T,1}) where {T<:Jop} = JopBlock(reshape(ops, length(ops), 1))

Base.adjoint(A::JopDBlock{D,R,T}) where {D,R,T} = JopAdjoint(A)

Jets.domain(A::JopDBlock) = A.dom
Base.range(A::JopDBlock) = A.rng
Base.procs(A::JopDBlock) = procs(A.ops)

Base.getindex(A::JopDBlock, i, j) = A.ops[i,j]
Base.getindex(A::JopAdjoint{T}, i, j) where {T<:JopDBlock} = A.op.ops[j,i]

function addmasterpid(pids)
    if myid() ∉ pids
        return [myid();pids]
    end
    pids
end

function LinearAlgebra.mul!(d::DArray, A::JopDBlock, m::AbstractArray)
    pids = procs(A)
    _m = bcast(m, pids=addmasterpid(pids))
    function _mul(d, A, _m)
        mul!(localpart(d), localpart(A.ops), localpart(_m))
        nothing
    end
    @sync for pid in pids
        @async remotecall_fetch(_mul!, pid, d, A, _m)
    end
    d
end

function LinearAlgebra.mul!(m::AbstractArray, A::JopAdjoint{T}, d::DArray) where {T<:JopDBlock}
    pids = = procs(A)
    _m = ArrayFutures(m, addmasterpid(pids))
    function _mul!(m, A, d)
        mul!(localpart(_m), JopAdjoint(localpart(A.op.ops)), localpart(d))
        nothing
    end
    @sync for pid in pids
        @async remotecall_fetch(_mul!, pid, _m, A, d)
    end
    reduce!(_m)
    m
end

function Jets.jacobian(A::JopDBlock, m::AbstractArray)
    pids = procs(A)
    _m = bcast(m, pids=addmasterpid(pids))
    ops = DArray(I->[jacobian(A[i,j], localpart(_m)) for i in I[1], j in I[2]], indices(A.ops), pids)
    JopDBlock(ops)
end

end
