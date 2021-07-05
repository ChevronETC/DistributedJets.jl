module DistributedJets

using Distributed, DistributedArrays, DistributedOperations, JSON, Jets, LinearAlgebra, Statistics

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
Jets.indices(A::DArray{T}) where {T} = A.indices
Jets.indices(A::DArray{T,1}) where {T} = indices(A.cuts[1])
Jets.indices(A::DArray{T}, i::Integer) where {T} = indices(A.cuts[i])

#
# Distributed block operators
#
struct JetDSpace{T,S<:Jets.JetAbstractSpace{T,1}} <: Jets.JetAbstractSpace{T,1}
    blkspaces::DArray{S,1,Array{S,1}}
    blkindices::Vector{UnitRange{Int}}
    indices::Vector{UnitRange{Int}}
end

function JetDSpace(blkspaces::DArray{S,1}) where {S<:JetBSpace}
    blkₒ,iₒ = 1,1
    blkindices = Vector{UnitRange{Int}}(undef, length(blkspaces))
    indices = Vector{UnitRange{Int}}(undef, length(blkspaces))
    for i = 1:length(blkspaces)
        blkspacesᵢ = blkspaces[i]
        blkindices[i] = blkₒ:(blkₒ+nblocks(blkspacesᵢ)-1)
        indices[i] = iₒ:(iₒ+length(blkspacesᵢ)-1)
        blkₒ = blkindices[i][end] + 1
        iₒ = indices[i][end] + 1
    end

    JetDSpace(blkspaces, blkindices, indices)
end

Base.:(==)(R::JetDSpace, S::JetDSpace) = R.blkspaces == S.blkspaces && R.blkindices == S.blkindices && R.indices == S.indices

"""
    size(R)
returns the size of R
"""
Base.size(R::JetDSpace) = (R.indices[end][end],)
"""
    eltype(R)
returns the element type of R 
"""
Base.eltype(R::Type{JetDSpace{T,S}}) where {T,S} = T
Base.eltype(R::Type{JetDSpace{T}}) where {T} = T
Base.vec(R::JetDSpace) = R

Jets.indices(R::JetDSpace) = R.indices
Jets.space(R::JetDSpace, iblock::Integer) where {T,S} = R.blkspaces[iblock]
Jets.nblocks(R::JetDSpace) = R.blkindices[end][end]

DistributedArrays.localindices(R::JetDSpace) = R.indices[findfirst(pid->pid==myid(), procs(R))]
DistributedArrays.localpart(R::JetDSpace) = localpart(R.blkspaces)[1]
"""
    localblockindices(R)
returns the block indicies that are local to the calling process
"""
localblockindices(R::JetDSpace) = R.blkindices[findfirst(pid->pid==myid(), procs(R))]
localblockindices(R::Jets.JetBSpace) = 1:length(R.indices)
localblockindices(R::Jets.JetAbstractSpace) = 1:1
"""
    blockmap(A)
returns map between process id and block indicies
"""
blockmap(R::JetDSpace) = R.blkindices

Distributed.procs(R::JetDSpace) = procs(R.blkspaces)
Distributed.nprocs(R::JetDSpace) = length(procs(R.blkspaces))

struct DBArray{T,A<:Jets.BlockArray{T},B} <: AbstractArray{T,1}
    darray::DArray{A,1,B}
    indices::Vector{UnitRange{Int}}
    blkindices::Vector{UnitRange{Int}}
end

function DBArray_localpart(f, blkidxs)
    arrays = [f(blkidx) for blkidx in blkidxs]
    indices = Vector{UnitRange{Int}}(undef, length(blkidxs))

    ibeg = 1
    for i = 1:length(arrays)
        iend = ibeg + length(arrays[i]) - 1
        indices[i] = ibeg:iend
        ibeg = iend + 1
    end

    Jets.BlockArray(arrays, indices)
end

function DBArray_local_length(future)
    x = fetch(future)
    isa(x, RemoteException) && throw(x)
    length(x)
end

function DBArray(f::Function, blkidxs::Vector{UnitRange{Int}}, pids=workers())
    futures = Dict{Int,Future}()
    for (ipid,pid) in enumerate(pids)
        futures[pid] = remotecall(DBArray_localpart, pid, f, blkidxs[ipid])
    end

    idxs = Vector{UnitRange{Int}}(undef, length(pids))
    ibeg = 1
    for (ipid,pid) in enumerate(pids)
        iend = ibeg + remotecall_fetch(DBArray_local_length, pid, futures[pid]) - 1
        idxs[ipid] = ibeg:iend
        ibeg = iend + 1
    end

    darray = DArray(i->[fetch(futures[myid()])], pids, idxs)
    DBArray(darray, idxs, blkidxs)
end

function DBArray(f::Function, nblks::Tuple, pids::AbstractArray, dist::AbstractArray)
    blkidxs, cuts = DistributedArrays.chunk_idxs([nblks...], dist) # cuts is not used
    DBArray(f, [blkidx[1] for blkidx in blkidxs], pids)
end

DBArray(f::Function, nblks::Tuple, pids::AbstractArray) = DBArray(f, nblks, pids, DistributedArrays.defaultdist(nblks, pids))
DBArray(f::Function, nblks::Tuple) = DBArray(f, nblks, workers()[1:min(nworkers(),nblks[1])], DistributedArrays.defaultdist(nblks, workers()[1:min(nworkers(),nblks[1])]))

function Jets.space(x::DBArray)
    blkspaces = DArray(I->[space(localpart(x)) for i in I[1]], (nprocs(x),), procs(x))
    JetDSpace(blkspaces)
end

# DBArray array interface implementation <--
Base.IndexStyle(::Type{T}) where {T<:DBArray} = IndexLinear()
Base.size(x::DBArray) = (x.indices[end][end],)
Jets.indices(x::DBArray) = indices(x.darray)
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
Base.similar(x::DBArray, ::Type{T}, n::Integer) where {T} = length(x) == n ? similar(x, T) : Array{T}(undef, n)
Base.similar(x::DBArray, ::Type{T}, dims::Tuple{Int}) where {T} = similar(x, T, dims[1])

DBArray_local_norm(x::DBArray, p) = norm(localpart(x), p)

function LinearAlgebra.norm(x::DBArray{T}, p::Real=2) where {T}
    _T = float(real(T))
    pids = procs(x)
    z = zeros(_T, length(pids))
    @sync for (ipid,pid) in enumerate(pids)
        @async begin
            z[ipid] = remotecall_fetch(DBArray_local_norm, pid, x, p)::_T
        end
    end
    if p == Inf
        maximum(z)
    elseif p == -Inf
        minimum(z)
    elseif p == 0 || p == 1
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

DBArray_local_extrema(x::DBArray) = extrema(localpart(x))

function Base.extrema(x::DBArray{T}) where {T}
    pids = procs(x)
    mn = zeros(T, length(pids))
    mx = zeros(T, length(pids))
    @sync for (ipid,pid) in enumerate(pids)
        @async begin
            mn[ipid],mx[ipid] = remotecall_fetch(DBArray_local_extrema, pid, x)::Tuple{T,T}
        end
    end
    minimum(mn),maximum(mx)
end

DBArray_local_mapreduce(f, op, x; kw...) = mapreduce(f, op, localpart(x); kw...)

function Base.mapreduce(f, op, x::DBArray{T}; kw...) where {T}
    pids = procs(x)
    mx = zeros(T, length(pids))
    @sync for (ipid,pid) in enumerate(pids)
        @async begin
            mx[ipid] = remotecall_fetch(DBArray_local_mapreduce, pid, f, op, x; kw...)::T
        end
    end
    mapreduce(f, op, mx)
end

DBArray_local_mean(x::DBArray) = sum(localpart(x))

function Statistics.mean(x::DBArray{T}) where {T}
    pids = procs(x)
    y = zeros(T, length(pids))
    @sync for (ipid,pid) in enumerate(pids)
        @async begin
            y[ipid] = remotecall_fetch(DBArray_local_mean, pid, x)
        end
    end
    sum(y) / length(x)
end

DBArray_local_var(x::DBArray, μ) = sum((localpart(x) .- μ).^2)

function Statistics.var(x::DBArray{T}; corrected=true, mean=nothing) where {T}
    pids = procs(x)
    μ = mean === nothing ? Statistics.mean(x) : mean
    y = zeros(T, length(pids))
    @sync for (ipid,pid) in enumerate(pids)
        @async begin
            y[ipid] = remotecall_fetch(DBArray_local_var, pid, x, μ)
        end
    end
    σ² = sum(y)
    if corrected
        σ² /= (length(x) - 1)
    else
        σ² /= length(x)
    end
    σ²
end

_localpart(x::DBArray) = localpart(x.darray)
DistributedArrays.localpart(x::DBArray) = _localpart(x)[1]::Jets.BlockArray
"""
    procs(R)
returns Julia processes associated with R
"""
Distributed.procs(x::DBArray) = procs(x.darray)
"""
    nprocs(R)
returns the number of Julia processes associated with R
"""
Distributed.nprocs(x::DBArray) = length(procs(x))
Jets.nblocks(x::DBArray) = x.blkindices[end][end]
"""
    localindices(R)
returns the indicies of R that are local to the calling process
"""
DistributedArrays.localindices(x::DBArray) = x.indices[findfirst(ipid->ipid==myid(), procs(x))]
localblockindices(x::DBArray) = x.blkindices[findfirst(ipid->ipid==myid(), procs(x))]
blockmap(x::DBArray) = x.blkindices

"""
    collect(d)
collect the block array `d` to the calling process, returning a block array
"""
function Base.collect(x::DBArray{T,A}) where {T,B,A<:Jets.BlockArray{T,B}}
    _x = B[]
    _indices = UnitRange{Int}[]
    n = 0
    for pid in procs(x)
        y = remotecall_fetch(localpart, pid, x)::Jets.BlockArray{T,B}
        _x = [_x; y.arrays]
        for i = 1:length(y.indices)
            push!(_indices, ((n+y.indices[i][1]):(n+y.indices[i][end])))
        end
        n = _indices[end][end]
    end
    Jets.BlockArray(_x, _indices)
end

"""
    convert(Array,d)
collect the block array `d` and convert it to a Julia array
"""
Base.convert(::Jets.BlockArray, x::DBArray) = collect(x)
Base.convert(::Array, x::DBArray) = convert(Array, collect(x))

DBArray_local_fill!(x::DBArray, a) = fill!(localpart(x), a)

function Base.fill!(x::DBArray, a)
    @sync for pid in procs(x)
        @async remotecall_fetch(DBArray_local_fill!, pid, x, a)
    end
    x
end

# DBArrays are vectors
function Base.reshape(x::DBArray, R::JetDSpace)
    length(x) == length(R) || error("dimension mismatch, unable to reshap distributed block arrays")
    x
end
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
"""
    getblock(d,i)
retrive block i from a distributed block operator
"""
function Jets.getblock(x::DBArray{T,A}, iblock::Integer) where {T,B,A<:Jets.BlockArray{T,B}}
    ipid = findfirst(rng->iblock∈rng, x.blkindices)
    remotecall_fetch(getblocklocal, procs(x)[ipid], x, iblock - x.blkindices[ipid][1] + 1)::B
end
"""
    getblock!(d, i, x)
get block `i::Int` and put it into `x`
"""
function Jets.getblock!(x::DBArray{T,A}, iblock::Integer, xblock::AbstractArray) where {T,B,A<:Jets.BlockArray{T,B}}
    ipid = findfirst(rng->iblock∈rng, x.blkindices)::Int
    xblock .= remotecall_fetch(getblocklocal, procs(x)[ipid], x, iblock - x.blkindices[ipid][1] + 1)::B
end

function setblocklocal!(x::DBArray, δblock::Integer, xblock)
    setblock!(localpart(x), δblock, xblock)
    nothing
end
"""
    setblock!(d,i,x)
put `x` into block `i::Int`
"""
function Jets.setblock!(x::DBArray, iblock::Integer, xblock)
    ipid = findfirst(rng->iblock∈rng, x.blkindices)::Int
    remotecall_fetch(setblocklocal!, procs(x)[ipid], x, iblock - x.blkindices[ipid][1] + 1, xblock)
end

Jets.JopBlock(A::DArray{T,2}; kwargs...) where {T<:Jop} = JopNl(Jets.JetBlock(A; kwargs...))
Jets.JopBlock(A::DArray{T,2}; kwargs...) where {T<:JopLn} = JopLn(Jets.JetBlock(A; kwargs...))

function JetDBlock_build(I, ops::DArray{T,2}, dadom) where {T<:Jop}
    irng = indices(ops, 1)[I[1][1]]
    jrng = indices(ops, 2)[I[2][1]]
    _ops = T[ops[i,j] for i in irng, j in jrng]
    [Jets.JopBlock(_ops; dadom=dadom) for k=1:1, l=1:1]
end

JetDBlock_countidxs(spaces) = length(localpart(spaces)[1])
JetDBlock_countblks(spaces) = nblocks(localpart(spaces)[1])

function JetDBlock_count_idxs_blks(blkspaces)
    pids = procs(blkspaces)
    n = length(blkspaces)
    countidxs = zeros(Int, n)
    countblks = zeros(Int, n)
    @sync for (i,pid) in enumerate(pids)
        @async begin
            countidxs[i] = remotecall_fetch(JetDBlock_countidxs, pid, blkspaces)::Int
            countblks[i] = remotecall_fetch(JetDBlock_countblks, pid, blkspaces)::Int
        end
    end
    countidxs,countblks
end

function JetDBlock_collect_idxs(countidxs, countblks, pids)
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
    idxs,blkidxs
end

function JetDBlock_space(blkspaces, blkidxs, idxs)
    length(blkidxs) == 1 && (return blkspaces[1])
    JetDSpace(blkspaces, blkidxs, idxs)
end

function localpart_diag_darray_spaces(_ops)
    ops = state(localpart(_ops)[1]).ops
    for j = 1:size(ops,1)
        opsj = @view ops[j,:]
        s = mapreduce(!iszero, +, opsj)
        s == 1 || error("Expecting one non-zero column block per row in block diagonal construction, got $s.")
    end
    ops = state(localpart(_ops)[1]).ops[:,1:1]
    A = @blockop adjoint.(JopLn.(jet.(ops))) dadom=true
    range(A)
end

function Jets.JetBlock(ops::DArray{T,2}; perfstatfile="", isdiag=false) where {T<:Jop}
    pids = procs(ops)

    n1,n2 = length(indices(ops, 1)),length(indices(ops, 2))

    (isdiag && n2 != 1) && error("block diagonal jets must define distribution along rows.")

    _ops = DArray(I->JetDBlock_build(I, ops, n2 > 1), (n1,n2), pids[:], [n1,n2])

    blkspaces_rng = DArray(I->[range(localpart(_ops)[1]) for i in I[1]], (n1,), pids[:,1], [n1])

    local blkspaces_dom
    if isdiag
        # TODO: fragile
        # assumption that there is only one column block per block-diagnol block
        blkspaces_dom = DArray(I->[localpart_diag_darray_spaces(_ops) for i in I[1]], (n1,), pids[:,1], [n1])
    else
        blkspaces_dom = DArray(I->[domain(localpart(_ops)[1]) for i in I[1]], (n2,), pids[1,:], [n2])
    end

    countidxs_dom,countblks_dom = JetDBlock_count_idxs_blks(blkspaces_dom)
    countidxs_rng,countblks_rng = JetDBlock_count_idxs_blks(blkspaces_rng)

    idxs_rng,blkidxs_rng = JetDBlock_collect_idxs(countidxs_rng, countblks_rng, pids[:,1])
    idxs_dom,blkidxs_dom = JetDBlock_collect_idxs(countidxs_dom, countblks_dom, isdiag ? pids[:,1] : pids[1,:])

    dom = JetDBlock_space(blkspaces_dom, blkidxs_dom, idxs_dom)
    rng = JetDBlock_space(blkspaces_rng, blkidxs_rng, idxs_rng)

    perfstatfile == "" || rm(perfstatfile, force=true)

    Jet(dom = dom, rng = rng, f! = JetDBlock_f!, df! = JetDBlock_df!, df′! = JetDBlock_df′!, s = (ops=_ops, dom=dom, rng=rng, blockmap=indices(ops), isdiag=isdiag, perfstatfile=perfstatfile))
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

function _JetDBlock_f!(d::DBArray, m::AbstractArray, ops)
    pids = procs(ops)
    _m = bcast(m, addmasterpid(pids))
    @sync for pid in pids
        @async remotecall_fetch(JetDBlock_local_f!, pid, d, _m, ops)
    end
    nothing
end

function JetDBlock_f!(d::DBArray, m::AbstractArray; ops, perfstatfile, kwargs...)
    _JetDBlock_f!(d, m, ops)
    perfstat(ops, "f", perfstatfile)
    d
end

function JetDBlock_local_df!(d, _m, ops)
    op = localpart(ops)[1]
    mul!(localpart(d), JopLn(op), localpart(_m))
    nothing
end

function _JetDBlock_df!(d::DBArray, m::AbstractArray, ops)
    pids = procs(ops)
    _m = bcast(m, addmasterpid(pids))
    @sync for pid in pids
        @async remotecall_fetch(JetDBlock_local_df!, pid, d, _m, ops)
    end
    nothing
end

function JetDBlock_df!(d::DBArray, m::AbstractArray; ops, perfstatfile, kwargs...)
    _JetDBlock_df!(d, m, ops)
    perfstat(ops, "df", perfstatfile)
    d
end

function JetDBlock_local_df′!(_m, d, ops)
    op = localpart(ops)[1]
    mul!(localpart(_m), (JopLn(op))', localpart(d))
    nothing
end

function _JetDBlock_df′!(m::AbstractArray, d::DBArray, ops, dom)
    m .= 0
    pids = procs(ops)
    _m = TypeFutures(m, zeros, addmasterpid(pids), dom)
    @sync for pid in pids
        @async remotecall_fetch(JetDBlock_local_df′!, pid, _m, d, ops)
    end
    reduce!(_m)
    nothing
end

function JetDBlock_df′!(m::AbstractArray, d::DBArray; ops, dom, perfstatfile, kwargs...)
    _JetDBlock_df′!(m, d, ops, dom)
    perfstat(ops, "df′", perfstatfile)
    m
end

function JetDBlock_local_row_cols_f!(_d, m, ops, blockmap, irow_pid, icol_pid, irow)
    __d = localpart(_d)
    dtmp = similar(__d)
    for icol in blockmap[irow_pid,icol_pid][2]
        op = getblock(ops, blockmap, irow, icol)
        if !iszero(op)
            __d .+= mul!(dtmp, op, getblock(m, icol))
        end
    end
    nothing
end

function JetDBlock_local_rows_f!(d, m, ops, irow_pid, blockmap)
    pids = procs(ops)[irow_pid,:]
    _pids = addmasterpid(pids)
    for irow in blockmap[irow_pid,1][1]
        dblk = getblock(d, irow)
        dblk .= 0
        _d = ArrayFutures(dblk, _pids)
        @sync for (icol_pid,col_pid) in enumerate(pids)
            @async remotecall_fetch(JetDBlock_local_row_cols_f!, col_pid, _d, m, ops, blockmap, irow_pid, icol_pid, irow)
        end
        reduce!(_d)
    end
    nothing
end

function JetDBlock_local_rows_diag_f!(d, m, ops, irow_pid, blockmap)
    for irow in blockmap[irow_pid,1][1]
        mul!(getblock(d, irow), getblock(ops, blockmap, irow, irow), getblock(m, irow))
    end
    nothing
end

function JetDBlock_f!(d::DBArray, m::DBArray; ops, blockmap, isdiag, perfstatfile, kwargs...)
    pids = procs(ops)
    @sync for (irow_pid,row_pid) in enumerate(pids[:,1])
        if isdiag
            @async remotecall_fetch(JetDBlock_local_rows_diag_f!, row_pid, d, m, ops, irow_pid, blockmap)
        else
            @async remotecall_fetch(JetDBlock_local_rows_f!, row_pid, d, m, ops, irow_pid, blockmap)
        end
    end
    perfstat(ops, "f", perfstatfile)
    d
end

function JetDBlock_local_row_cols_df!(_d, m, ops, blockmap, irow_pid, icol_pid, irow)
    __d = localpart(_d)
    dtmp = similar(__d)
    for icol in blockmap[irow_pid,icol_pid][2]
        op = getblock(ops, blockmap, irow, icol)
        if !iszero(op)
            __d .+= mul!(dtmp, JopLn(op), getblock(m, icol))
        end
    end
    nothing
end

function JetDBlock_local_rows_df!(d, m, ops, irow_pid, blockmap)
    pids = procs(ops)[irow_pid,:]
    _pids = addmasterpid(pids)
    for irow in blockmap[irow_pid,1][1]
        dblk = getblock(d, irow)
        dblk .= 0
        _d = ArrayFutures(dblk, _pids)
        @sync for (icol_pid,col_pid) in enumerate(pids)
            @async remotecall_fetch(JetDBlock_local_row_cols_df!, col_pid, _d, m, ops, blockmap, irow_pid, icol_pid, irow)
        end
        reduce!(_d)
    end
    nothing
end

function JetDBlock_local_rows_diag_df!(d, m, ops, irow_pid, blockmap)
    for irow in blockmap[irow_pid,1][1]
        mul!(getblock(d, irow), JopLn(getblock(ops, blockmap, irow, irow)), getblock(m, irow))
    end
    nothing
end

function JetDBlock_df!(d::DBArray, m::DBArray; ops, blockmap, isdiag, perfstatfile, kwargs...)
    pids = procs(ops)
    @sync for (irow_pid,row_pid) in enumerate(pids[:,1])
        if isdiag
            @async remotecall_fetch(JetDBlock_local_rows_diag_df!, row_pid, d, m, ops, irow_pid, blockmap)
        else
            @async remotecall_fetch(JetDBlock_local_rows_df!, row_pid, d, m, ops, irow_pid, blockmap)
        end
    end
    perfstat(ops, "df", perfstatfile)
    d
end

function JetDBlock_local_col_rows_df′!(_m, d, ops, blockmap, icol_pid, irow_pid, icol)
    __m = localpart(_m)
    mtmp = similar(__m)
    for irow in blockmap[irow_pid,icol_pid][1]
        op = getblock(ops, blockmap, irow, icol)
        if !iszero(op)
            __m .+= mul!(mtmp, (JopLn(op))', getblock(d, irow))
        end
    end
    nothing
end

function JetDBlock_local_cols_df′!(m, d, ops, icol_pid, blockmap)
    pids = procs(ops)[:,icol_pid]
    _pids = addmasterpid(pids)
    for icol in blockmap[1,icol_pid][2]
        mblk = getblock(m, icol)
        mblk .= 0
        _m = ArrayFutures(mblk, _pids)
        @sync for (irow_pid,row_pid) in enumerate(pids)
            @async remotecall_fetch(JetDBlock_local_col_rows_df′!, row_pid, _m, d, ops, blockmap, icol_pid, irow_pid, icol)
        end
        reduce!(_m)
    end
    nothing
end

function JetDBlock_local_cols_diag_df′!(m, d, ops, icol_pid, blockmap)
    for icol in blockmap[icol_pid,1][1]
        mul!(getblock(m, icol), (JopLn(getblock(ops, blockmap, icol, icol)))', getblock(d, icol))
    end
    nothing
end

function JetDBlock_df′!(m::DBArray, d::DBArray; ops, blockmap, isdiag, perfstatfile, kwargs...)
    pids = isdiag ? (procs(ops))' : procs(ops)
    @sync for (icol_pid,col_pid) in enumerate(pids[1,:])
        if isdiag
            @async remotecall_fetch(JetDBlock_local_cols_diag_df′!, col_pid, m, d, ops, icol_pid, blockmap)
        else
            @async remotecall_fetch(JetDBlock_local_cols_df′!, col_pid, m, d, ops, icol_pid, blockmap)
        end
    end
    perfstat(ops, "df′", perfstatfile)
    m
end

function _perfstat_local(ops, operation)
    _ops = state(localpart(ops)[1]).ops
    _s = [perfstat(_ops[iblkrow,iblkcol]) for iblkrow=1:size(_ops,1), iblkcol=1:size(_ops,2)][:]
    Dict("id"=>myid(), "hostname"=>gethostname(), "operation"=>operation, "localblock"=>_s)
end

function Jets.perfstat(ops, operation, perfstatfile)
    if perfstatfile != ""
        pids = procs(ops)
        s = Vector{Dict{String,Any}}(undef, length(pids))
        @sync for (i, pid) in enumerate(procs(ops))
            @async s[i] = remotecall_fetch(_perfstat_local, pid, ops, operation)
        end
        if isfile(perfstatfile)
            _s = JSON.parse(read(perfstatfile, String))
            push!(_s["step"], Dict("pid"=>s))
            write(perfstatfile, json(_s, 1))
        else
            write(perfstatfile, json(Dict("step"=>[Dict("pid"=>s)]), 1))
        end
    end
    nothing
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

function JetDBlock_local_row_cols_point!(mₒ, ops, blockmap, irow_pid, icol_pid, irow)
    icols = blockmap[irow_pid,icol_pid][2]
    for icol in icols
        op = getblock(ops, blockmap, irow, icol)
        if !iszero(op)
            Jets.point!(jet(op), getblock(mₒ, icol))
        end
    end
    nothing
end

function JetDBlock_local_rows_point!(mₒ, ops, irow_pid, blockmap)
    pids = procs(ops)[irow_pid,:]
    _pids = addmasterpid(pids)
    for irow in blockmap[irow_pid,1][1]
        @sync for (icol_pid,col_pid) in enumerate(pids)
            @async remotecall_fetch(JetDBlock_local_row_cols_point!, col_pid, mₒ, ops, blockmap, irow_pid, icol_pid, irow)
        end
    end
    nothing
end

function JetDBlock_local_rows_diag_point!(mₒ, ops, irow_pid, blockmap)
    for irow in blockmap[irow_pid,1][1]
        op = getblock(ops, blockmap, irow, irow)
        Jets.point!(jet(op), getblock(mₒ, irow))
    end
    nothing
end

function Jets.point!(j::Jet{D,R,typeof(JetDBlock_f!)}, mₒ::DBArray) where {D<:Jets.JetAbstractSpace,R<:Jets.JetAbstractSpace}
    pids = procs(j)
    @sync for (irow_pid,row_pid) in enumerate(pids[:,1])
        if state(j).isdiag
            @async remotecall_fetch(JetDBlock_local_rows_diag_point!, row_pid, mₒ, state(j).ops, irow_pid, state(j).blockmap)
        else
            @async remotecall_fetch(JetDBlock_local_rows_point!, row_pid, mₒ, state(j).ops, irow_pid, state(j).blockmap)
        end
    end
    j
end

Distributed.procs(jet::J) where {D,R,J<:Jet{D,R,typeof(JetDBlock_f!)}} = procs(state(jet).ops)
Distributed.procs(A::T) where {D,R,J<:Jet{D,R,typeof(JetDBlock_f!)},T<:Jop{J}} = procs(jet(A))
Distributed.nprocs(A::T) where {D,R,J<:Jet{D,R,typeof(JetDBlock_f!)},T<:Jop{J}} = length(procs(A))

DistributedArrays.localpart(j::Jet{D,R,typeof(JetDBlock_f!)}) where {D,R} = jet(localpart(state(j).ops)[1])
DistributedArrays.localpart(A::T) where {D,R,J<:Jet{D,R,typeof(JetDBlock_f!)},T<:JopLn{J}} = JopLn(localpart(state(A).ops)[1])
DistributedArrays.localpart(A::T) where {D,R,J<:Jet{D,R,typeof(JetDBlock_f!)},T<:JopAdjoint{J}} = JopAdjoint(localpart(state(A).ops)[1])
DistributedArrays.localpart(A::T) where {D,R,J<:Jet{D,R,typeof(JetDBlock_f!)},T<:Jop{J}} = localpart(state(A).ops)[1]

function localblockindices(j::Jet{D,R,typeof(JetDBlock_f!)}) where {D,R}
    blkmap = blockmap(j)
    pids = procs(j)
    mypid = myid()
    for i2 = 1:size(pids,2), i1 = 1:size(pids,1)
        if mypid == pids[i1,i2]
            return blkmap[i1,i2]
        end
    end
    (0:0,0:0)
end
localblockindices(j::Jet{D,R,typeof(JetDBlock_f!)}, i) where {D,R} = localblockindices(j)[i]
localblockindices(A::T) where {D,R,J<:Jet{D,R,typeof(JetDBlock_f!)},T<:Jop{J}} = localblockindices(jet(A))
localblockindices(A::T, i) where {D,R,J<:Jet{D,R,typeof(JetDBlock_f!)},T<:Jop{J}} = localblockindices(jet(A), i)

localblockindices(j::Jet{D,R,typeof(Jets.JetComposite_f!)}) where {D,R} = error("localblockindices does not support composite operators; consider using `localblockindices(range(F))` or `localblockindices(domain(F))`")
localblockindices(A::T) where {D,R,J<:Jet{D,R,typeof(Jets.JetComposite_f!)},T<:Jop{J}} = localblockindices(jet(A))
localblockindices(A::T, i) where {D,R,J<:Jet{D,R,typeof(Jets.JetComposite_f!)},T<:Jop{J}} = localblockindices(jet(A))

blockmap(jet::T) where {D,R,T<:Jet{D,R,typeof(JetDBlock_f!)}} = state(jet).blockmap
blockmap(op::Jop) = blockmap(jet(op))

JetDBlock_local_getblock(ops, δi, δj) = getblock(localpart(ops)[1], δi, δj)

function Jets.getblock(ops::DArray, blkmap, i::Integer, j::Integer)
    local ipid, jpid
    for _ipid in 1:size(blkmap,1), _jpid in 1:size(blkmap,2)
        if i ∈ blkmap[_ipid,_jpid][1] && j ∈ blkmap[_ipid,_jpid][2]
            ipid,jpid = _ipid,_jpid
            break
        end
    end
    pid = procs(ops)[ipid,jpid]
    remotecall_fetch(JetDBlock_local_getblock, pid, ops, i - blkmap[ipid,jpid][1][1] + 1, j - blkmap[ipid,jpid][2][1] + 1)
end

Jets.isblockop(A::Jop{<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetDBlock_f!)}}) = true

Jets.getblock(jet::Jet{D,R,typeof(JetDBlock_f!)}, i, j) where {D,R} = getblock(state(jet).ops, blockmap(jet), i, j)

close_localpart(ops::DArray) = close(localpart(ops)[1])

function Base.close(j::Jet{D,R,typeof(JetDBlock_f!)}) where {D,R}
    ops = state(j).ops
    @sync for pid in procs(ops)
        @async remotecall_fetch(close_localpart, pid, ops)
    end
end

export DBArray, JetDSpace, blockmap, localblockindices

end
