# DistributedJets.jl

This package contains distributed block operators and vectors for Jets.jl.  It
builds on top of the block operators in
https://dev.azure.com/chevron/ETC-ESD-Jets.jl, providing a parallel distributed
version of block operators and block vectors that are used to orchestrate
distributed (in-memory) storage and compute.

# Distributed block operators
Similar to Jets.jl, we create a block operator using the `@blockop` macro,
except that instead of using an array comprehension, we use a DArray
constructor.  For example,
```julia
using Pkg
Pkg.add("Distributed","DistributedArrays", "DistributedJets","JetPack","Jets")
using Distributed
addprocs(6)
@everywhere using DistributedArrays, DistributedJets, JetPack, Jets
A = @blockop DArray(I->[JopDiagonal(rand(10)) for i in I[1], j in I[2]], (3,2), workers(), [3,2])
```
`A` is a distributed block operator with 3 column blocks and 2 row blocks.  Each
block resides on a separate Julia process.  Vectors in the domain of `A` are
distributed using processes corresponding the row blocks of `A`.  Likewise,
vectors in the range of `A` are distributed using processes corresponding to the
column blocks of `A`.
```julia
R = domain(A) # JetBSpace consistent with the range of A
m = rand(R) # distributed block array in the domain of A
d = A*m # d is a distributed block array in the range of A
```

## Tall and skinny distributed block operators
We support block operators that are tall and skinny with N row blocks, and
one column block.  In this case, the model space is not distributed, instead
residing entirely on the master process.  For example,
```julia
A = @blockop DArray(I->[JopDiagonal(rand(10)) for i in I[1], j in I[2]], (6,1))
R = domain(A) # JeSpace consistent with the domain of A
m = rand(domain(A)) # m is a Julia array on the master process
d = A*m # d is a distributed block array in the range of A
```

## Distributed block diagonal operators
we support block diagonal operators.  In this case the domain and range
share the same distributed space.  Please note that in this case the
paralleldistribution of the operator must be along the row direction of
the block operator. In addition, one must pass the extra argument `isdiag=true` to the
`@blockop` macro.  For example,
```julia
A = @blockop DArray(I->[i==j ? JopDiagonal(rand(10)) : JopZeroBlock(JetSpace(Float64,10),JetSpace(Float64,10)) for i in I[1], j in I[2]], (6,6), workers(), [6,1]) isdiag=true
m = rand(domain(A)) # m is a distributed block array in the domain of A
d = A * m # d is a distribution block array in the range of A
```
We note that other patterns of sparse distributed operators are not currently
supported.  More generic support for sparse structures in distributed block
operators would likely be best achieved by first created a sparse version of the
DistributedArrays.jl package.

## Methods for distributed block spaces
The following methods are defined for distributed block spaces,
```julia
zeros(R) # distributed block array in R, initialized to zeros
ones(R) # distributd block array in R, initialized to ones
rand(R) # distributed block array in R, initialized to random numbers
Array(R) # distributed block array in R with un-initialized values
size(R) # the size of a distributed block space
length(R) # the length (number of dimensions) of a distributed block space
eltype(R) # the element type of the space
ndims(R) # the number of dimensions of the space
localindices(R) # the indices that are local to the calling process
localblockindices(R) # the block indices that are local to the calling process
nprocs(R) # number of Julia process associated with R
procs(R) # Julia processes associated with R
```

## Methods for distributed block arrays
The following methods are defined for distributed block arrays,
```julia
size(d)
length(d)
getblock(d, i) # retrieve block i from a block vector
getblock!(d, i, x) # retrieve block i into x from a block vector
setblock!(d, i, x) # put x into block i
localblockindices(d) # get a list of block indices that are local to the calling process
nprocs(d) # get the number of processes associated with the block vector
procs(d) # get a list of processes associated with the block vector
collect(d) # collect a block array to the calling process, returning a block array
convert(Array,d) # collect a block array to the calling process, returning a Julia array
```

## Methods for block operators
```julia
getblock(A,i,j) # retrive block (i,j) from a distributed block operator
localblockindices(A) # get a list of local block indices associated with the calling process
localblockindices(A,i) # get a list of local block indices associated with the calling process along dimension i
nprocs(A) # number of processes associated with A
procs(A) # list of processes associated with A
blockmap(A) # map between process id and block indices
domain(A)
range(A)
size(A)
size(A,i)
shape(A)
shape(A,i)
close(A)
state(A)
state!(A)
```

## Notes
When using the `getblock` and `setblock` methods, one must take care to understand
how data is moved between machines in the cluster.  If `getblock(A,i,j)` is called
from the process that contains block `i,j` then this operation is copy-free. On
the other hand if `getblock(A,i,j)` is run from a process that does not contain
block `i,j`, then the call results in a copy of the block from the process
that it resides on to the calling process.  The same is true for the `setblock!`
method, and for the same methods when applied to distributed block arrays.

## Useful patterns

### Computing cost over a set of shots from a distributed block array.
```julia
@everywhere costperblock(dmod,dobs) = 0.5*norm(dobs .- dmod)^2

@everywhere costperpid(fmod, fobs)
    _fmod = localpart(fmod)
    _fobs = localpart(fobs)
    obj = 0.0
    for iblock = 1:nblocks(_fmod,1)
        obj += costperblock(getblock(_fmod,iblock), getblock(_fobs,iblock))
    end
    obj
end

function cost(m, F, dobs)
    dmod = F*m #F is a block operators
    phi = zeros(nprocs(F))
    @sync for (ipid,pid) in enumerate(procs(F))
        @async begin
            phi[ipid] = remotecall_fetch(costperpid, pid, dmod, dobs)
        end
    end
    sum(phi)
end

```
Note that the above can be done in a single line. Above lines are meant to illustrate how to use the block structure.
```julia
    cost(m,F,d) = 0.5*norm(F*m .-  d)^2
```

## Create a block wavefield modeling operator from the geometry in a JavaSeis file
```julia
Pkg.add("Distributed","DistributedArrays","DistributedJets","Jets","JetPackWave","TeaSeis","ParallelOperations")
using Distributed
addprocs(2)
@everywhere using DistributedArrays,DistributedJets,Jets,JetPackWave,TeaSeis,ParallelOperations

function buildblock(ishot,ρ,io)
    h = readframehdrs(io,ishot)
    JopNlProp3DAcoIsoDenQ_DEO2_FDTD(
        sz = -get(prop(io,"SOU_ELEV"), h, 1),
        sy = get(prop(io,"SOU_Y"), h, 1),
        sx = get(prop(io,"SOU_X"), h, 1),
        rz = [-get(prop(io,"REC_ELEV"), h, i) for i = 1:fold(io,h)],
        ry = [-get(prop(io,"REC_Y"), h, i) for i = 1:fold(io,h)],
        rx = [-get(prop(io,"REC_X"), h, i) for i = 1:fold(io,h)],
        ntrec = size(io,1),
        dtrec = pincs(io,1),
        dtmod = 0.0001,
        b = 1 ./ ρ,
        dz = 20.0,
        dy = 20.0,
        dx = 20.0)
end

function buildblocks(I,ρ_futures)
    io = jsopen("data.js")
    ρ = localpart(ρ_futures)
    F = [buildblock(ishot,ρ,io) for ishot in I[1], j in 1:1]
    close(io)
    F
end

io = jsopen("data.js")
nshots = size(io,3) # assume one shot per frame
close(io)

nz,ny,nx=512,512,512
ρ = 1.0*ones(nz,ny,nx)
ρ_futures = bcast(ρ)

F = @blockop DArray(I->buildblocks(I, ρ_futures), (nshots,1))
```

## Populate a distributed array from a JavaSeis file
Given `F` built in the previous example, we can populate an array the range of
`F` from a JavaSeis file.
```julia
@everywhere function readblocks!(d)
    io = jsopen("data.js")
    for ishot in localblockindices(d)
        setblock!(d, ishot, readframetrcs(io, ishot))
    end
    close(io)
end

d = zeros(range(F))
@sync for pid in procs(d)
    @async remotecall_fetch(readblocks!, pid, d)
end
```
