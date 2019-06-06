# DistributedJets.jl

Distributed block operators for Jets.jl


Pattern for computing cost over a set of shots from a distributed block array. 
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
