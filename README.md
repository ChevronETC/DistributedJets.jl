| **Documentation**                                                               |
|:-------------------------------------------------------------------------------:|
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://chevronetc.github.io/DistributedJets.jl/dev/) |
# DistributedJets.jl

This package contains distributed block operators and vectors for Jets.jl.  It
builds on top of the block operators in Jets.jl, providing a parallel distributed
version of block operators and block vectors that are used to orchestrate
distributed (in-memory) storage and compute.

# Distributed block operators
Similar to Jets.jl, we create a block operator using the `@blockop` macro,
except that instead of using an array comprehension, we use a DArray
constructor.  For example,
```julia
using Pkg
Pkg.add(["Distributed","DistributedArrays", "DistributedJets","JetPack","Jets"])
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
