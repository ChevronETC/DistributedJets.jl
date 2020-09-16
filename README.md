# DistributedJets.jl
  
| **Documentation** | **Action Statuses** |
|:---:|:---:|
| [![][docs-dev-img]][docs-dev-url] [![][docs-stable-img]][docs-stable-url] | [![][doc-build-status-img]][doc-build-status-url] [![][build-status-img]][build-status-url] [![][code-coverage-img]][code-coverage-results] |

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

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://chevronetc.github.io/DistributedJets.jl/dev/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://ChevronETC.github.io/DistributedJets.jl/stable

[doc-build-status-img]: https://github.com/ChevronETC/DistributedJets.jl/workflows/Documentation/badge.svg
[doc-build-status-url]: https://github.com/ChevronETC/DistributedJets.jl/actions?query=workflow%3ADocumentation

[build-status-img]: https://github.com/ChevronETC/DistributedJets.jl/workflows/Tests/badge.svg
[build-status-url]: https://github.com/ChevronETC/DistributedJets.jl/actions?query=workflow%3A"Tests"

[code-coverage-img]: https://codecov.io/gh/ChevronETC/DistributedJets.jl/branch/master/graph/badge.svg
[code-coverage-results]: https://codecov.io/gh/ChevronETC/DistributedJets.jl
