module DistributedJets

using Distributed, DistributedArrays, Jets, LinearAlgebra, ParallelOperations

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
struct JetDSpace{T,S<:Jets.JetAbstractSpace{T,1}} <: Jets.JetAbstractSpace{T,1}
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

struct DBArray{T,A<:Jets.BlockArray{T},B} <: AbstractArray{T,1}
    darray::DArray{A,1,B}
    indices::Vector{UnitRange{Int}}
    blkindices::Vector{UnitRange{Int}}
end

# DBArray array interface implementation <--
Base.IndexStyle(::Type{T}) where {T<:DBArray} = IndexLinear()
Base.size(x::DBArray) = (x.indices[end][end],)
Jets.indices(x::DBArray, i::Integer) = indices(x.darray, i)

DBArray_local_getindex(x, j) = getindex(localpart(x), j)

function Base.getindex(x::DBArray, i::Int)
    ipid = findfirst(rng->i∈rng, x.indices)::Int
    pid = procs(x)[ipid]
    iₒ = x.indices[ipid][1]
    remotecall_fetch(DBArray_local_getindex, pid, x, i-iₒ+1)
end

function Base.similar(x::DBArray, ::Type{T}) where {T}
    darray = DArray(I->[similar(localpart(x), T)], procs(x)[:], indices(x, 1))
    DBArray(darray, x.indices, x.blkindices)
end
Base.similar(x::DBArray{T}) where {T} = similar(x, T)

DBArray_local_norm(x::DBArray, p) = norm(localpart(x), p)

function LinearAlgebra.norm(x::DBArray{T}, p::Real=2) where {T}
    _T = real(T)
    pids = procs(x)
    z = zeros(_T, length(pids))
    @sync for (ipid,pid) in enumerate(pids)
        @async begin
            z[ipid] = remotecall_fetch(DBArray_local_norm, pid, x, p)::_T
        end
    end
    if p == Inf
        maximum(z)
    elseif p == 0
        sum(z)
    else
        _p = _T(p)
        mapreduce(_z->_z^_p, +, z)^(one(_T)/_p)
    end
end

DBArray_local_dot(x::DBArray, y::DBArray) = dot(localpart(x), localpart(y))

function LinearAlgebra.dot(x::DBArray{T}, y::DBArray{T}) where {T}
    pids = procs(x)
    z = zeros(T, length(pids))
    @sync for (ipid,pid) in enumerate(pids)
        @async begin
            z[ipid] = remotecall_fetch(DBArray_local_dot, pid, x, y)::T
        end
    end
    sum(z)
end

_localpart(x::DBArray) = localpart(x.darray)
DistributedArrays.localpart(x::DBArray) = _localpart(x)[1]::Jets.BlockArray
Distributed.procs(x::DBArray) = procs(x.darray)
Jets.nblocks(x::DBArray) = x.blkindices[end][end]

function Base.collect(x::DBArray{T,A}) where {T,A}
    _x = A[]
    _indices = UnitRange{Int}[]
    n = 0
    for pid in procs(x)
        y = remotecall_fetch(localpart, pid, x)
        _x = [_x; y.arrays]
        for i = 1:length(y.indices)
            push!(_indices, ((n+y.indices[i][1]):(n+y.indices[i][end])))
        end
        n = _indices[end][end]
    end
    Jets.BlockArray(_x, _indices)
end

Base.convert(::Jets.BlockArray, x::DBArray) = collect(x)
Base.convert(::Array, x::DBArray) = convert(Array, collect(x))
# -->

# DBArray broadcasting implementation --<
struct DBArrayStyle <: Broadcast.AbstractArrayStyle{1} end
Base.BroadcastStyle(::Type{<:DBArray}) = DBArrayStyle()
DBArrayStyle(::Val{1}) = DBArrayStyle()

function Base.similar(bc::Broadcast.Broadcasted{DBArrayStyle}, ::Type{T}) where {S,T}
    A = find_dbarray(bc)
    similar(A)
end
find_dbarray(bc::Broadcast.Broadcasted) = find_dbarray(bc.args)
find_dbarray(args::Tuple) = find_dbarray(find_dbarray(args[1]), Base.tail(args))
find_dbarray(x) = x
find_dbarray(a::DBArray, rest) = a
find_dbarray(::Any, rest) = find_dbarray(rest)

DistributedArrays.localpart(bc::Broadcast.Broadcasted{DBArrayStyle}) = Broadcast.Broadcasted{Jets.BlockArrayStyle}(bc.f, map(arg->localpart(arg), bc.args))

function DBArray_local_copyto!(dest, bc)
    copyto!(localpart(dest), localpart(bc))
    nothing
end

function Base.copyto!(dest::DBArray, bc::Broadcast.Broadcasted{DBArrayStyle})
    @sync for pid in procs(dest)
        @async remotecall_fetch(DBArray_local_copyto!, pid, dest, bc)
    end
    dest
end
# -->

DistributedArrays.empty_localpart(T,N,::Type{A}) where {B<:Jets.BlockArray,A<:Vector{B}} = Jets.BlockArray([Array{T}(undef, ntuple(zero, N))], [0:0])
for f in (:Array, :ones, :rand, :zeros)
    @eval (Base.$f)(R::JetDSpace) = DBArray(DArray(i->[($f)(localpart(R.blkspaces)[1])], procs(R), indices(R)), R.indices, R.blkindices)
end

getblocklocal(x::DBArray, δblock) = getblock(localpart(x), δblock)
function Jets.getblock(x::DBArray{T,A}, iblock::Integer) where {T,B,A<:Jets.BlockArray{T,B}}
    ipid = findfirst(rng->iblock∈rng, x.blkindices)
    remotecall_fetch(getblocklocal, procs(x)[ipid], x, iblock - x.blkindices[ipid][1] + 1)::B
end
function Jets.getblock!(x::DBArray{T,A}, iblock::Integer, xblock::AbstractArray) where {T,B,A<:Jets.BlockArray{T,B}}
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

Jets.JopBlock(A::DArray{T,2}) where {T<:Jop} = JopNl(Jets.JetBlock(A))
Jets.JopBlock(A::DArray{T,2}) where {T<:JopLn} = JopLn(Jets.JetBlock(A))

function JetDBlock_domain(ops)
    _ops = localpart(ops)
    N = size(_ops, 2)
    N == 1 && (return domain(_ops[1,1]))
    Jets.JetBSpace([domain(_ops[1,i]) for i=1:N])
end

function JetDBlock_build(I, ops)
    irng = indices(ops, 1)[I[1][1]]
    jrng = indices(ops, 2)[I[2][1]]
    _ops = [ops[i,j] for i in irng, j in jrng]
    [Jets.JopBlock(_ops) for k=1:1, l=1:1]
end

JetDBlock_countidxs(spaces) = length(localpart(spaces)[1])
JetDBlock_countblks(spaces) = nblocks(localpart(spaces)[1])

function Jets.JetBlock(ops::DArray{T,2}) where {T<:Jop}
    length(ops.cuts[2]) < 3 || error("Distributed support is across rows")

    dom = remotecall_fetch(JetDBlock_domain, procs(ops)[1], ops)

    pids = procs(ops)[:]

    n1,n2 = length(indices(ops, 1)),length(indices(ops, 2))
    _ops = DArray(I->JetDBlock_build(I, ops), (n1,n2), pids, [n1,n2])

    blkspaces = DArray(I->[range(localpart(_ops)[1]) for i=1:1], (n1,), pids, [n1])

    countidxs = zeros(Int, n1)
    countblks = zeros(Int, n1)
    @sync for (i,pid) in enumerate(pids)
        @async begin
            countidxs[i] = remotecall_fetch(JetDBlock_countidxs, pid, blkspaces)
            countblks[i] = remotecall_fetch(JetDBlock_countblks, pid, blkspaces)
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

    Jet(dom = dom, rng = rng, f! = JetDBlock_f!, df! = JetDBlock_df!, df′! = JetDBlock_df′!, s = (ops=_ops, dom=dom, blockmap=indices(ops, 1)))
end

function addmasterpid(pids)
    if myid() ∉ pids
        return [myid();pids]
    end
    pids
end

function JetDBlock_local_f!(d, _m, ops)
    op = localpart(ops)[1]
    mul!(localpart(d), op, localpart(_m))
    nothing
end

function JetDBlock_f!(d::DBArray, m::AbstractArray; ops, kwargs...)
    pids = procs(ops)
    _m = bcast(m, addmasterpid(pids))
    @sync for pid in pids
        @async remotecall_fetch(JetDBlock_local_f!, pid, d, _m, ops)
    end
    d
end

function JetDBlock_local_df!(d, _m, ops)
    op = localpart(ops)[1]
    mul!(localpart(d), JopLn(op), localpart(_m))
    nothing
end

function JetDBlock_df!(d::DBArray, m::AbstractArray; ops, kwargs...)
    pids = procs(ops)
    _m = bcast(m, addmasterpid(pids))
    @sync for pid in pids
        @async remotecall_fetch(JetDBlock_local_df!, pid, d, _m, ops)
    end
    d
end

function JetDBlock_local_df′!(_m, d, ops)
    op = localpart(ops)[1]
    mul!(localpart(_m), (JopLn(op))', localpart(d))
    nothing
end

function JetDBlock_df′!(m::AbstractArray, d::DBArray; ops, dom, kwargs...)
    m .= 0
    pids = procs(ops)
    _m = TypeFutures(m, zeros, addmasterpid(pids), dom)
    @sync for pid in pids
        @async remotecall_fetch(JetDBlock_local_df′!, pid, _m, d, ops)
    end
    reduce!(_m)
    m
end

function JetDBlock_local_point!(ops, _mₒ)
    op = localpart(ops)[1]
    Jets.point!(jet(op), localpart(_mₒ))
    nothing
end

function Jets.point!(j::Jet{D,R,typeof(JetDBlock_f!)}, mₒ::AbstractArray) where {D<:Jets.JetAbstractSpace,R<:Jets.JetAbstractSpace}
    ops = state(j).ops
    pids = procs(ops)
    _mₒ = bcast(mₒ, addmasterpid(pids))
    @sync for pid in pids
        @async remotecall_fetch(JetDBlock_local_point!, pid, ops, _mₒ)
    end
    j
end

Distributed.procs(A::T) where {D,R,J<:Jet{D,R,typeof(JetDBlock_f!)},T<:Jop{J}} = procs(state(A).ops)

DistributedArrays.localpart(j::Jet{D,R,typeof(JetDBlock_f!)}) where {D,R} = jet(localpart(state(j).ops)[1])
DistributedArrays.localpart(A::T) where {D,R,J<:Jet{D,R,typeof(JetDBlock_f!)},T<:Jop{J}} = localpart(state(A).ops)[1]

JetDBlock_local_getblock(ops, δi, j) = state(localpart(ops)[1]).ops[δi,j]

function Jets.getblock(op::Jop{T}, i::Integer, j::Integer) where {D,R,T<:Jet{D,R,typeof(JetDBlock_f!)}}
    ops = state(jet(op)).ops
    blockmap = state(jet(op)).blockmap
    ipid = findfirst(rng->i∈rng, blockmap)
    pid = procs(ops)[ipid]
    remotecall_fetch(JetDBlock_local_getblock, pid, ops, i - blockmap[ipid][1] + 1, j)
end

Jets.getblock(::Type{JopLn}, A::Jop{T}, i::Integer, j::Integer) where {T<:Jet{<:Jets.JetAbstractSpace,<:Jets.JetAbstractSpace,typeof(JetDBlock_f!)}} = getblock(A, i, j)::JopLn
Jets.getblock(::Type{JopNl}, F::Jop{T}, i::Integer, j::Integer) where {T<:Jet{<:Jets.JetAbstractSpace,<:Jets.JetAbstractSpace,typeof(JetDBlock_f!)}} = getblock(F, i, j)::JopNl
Jets.getblock(A::Jets.JopAdjoint{T}, i::Integer, j::Integer) where {T<:Jet{<:Jets.JetAbstractSpace,<:Jets.JetAbstractSpace,typeof(JetDBlock_df!)}} = Jets.JopAdjoint(getblock(A.op, j, i))

Jets.indices(jet::Jet{D,R,typeof(JetDBlock_f!)}, i::Integer) where {D,R} = indices(state(A).ops, i)
Jets.indices(A::Jop{T}, i::Integer) where {T<:Jet{<:Jets.JetAbstractSpace,<:Jets.JetAbstractSpace,typeof(JetDBlock_f!)}} = indices(jet(A), i)
Jets.indices(A::Jets.JopAdjoint{Jet{D,R,typeof(JetDBlock_f!)}}, i::Integer) where {D,R} = indices(A.op, i == 1 ? 2 : 1)

export blockproc

end
