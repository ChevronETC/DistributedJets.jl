# DistributedJets.jl
  
| **Documentation** | **Action Statuses** |
|:---:|:---:|
| [![][docs-dev-img]][docs-dev-url] [![][docs-stable-img]][docs-stable-url] | [![][doc-build-status-img]][doc-build-status-url] [![][build-status-img]][build-status-url] [![][code-coverage-img]][code-coverage-results] |

This package contains distributed block operators and vectors for
[Jets.jl](https://github.com/ChevronETC/Jets.jl).  It builds on top
of the block operators in Jets.jl, providing a parallel distributed
version of block operators and block vectors that are used to orchestrate
distributed (in-memory) storage and compute.

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
