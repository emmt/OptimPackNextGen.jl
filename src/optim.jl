#
# optim.jl -
#
# Common methods and constants for optimization algorithms.
#
#------------------------------------------------------------------------------
#
# This file is part of TiPi.jl which is licensed under the MIT "Expat" License:
#
# Copyright (C) 2015, Éric Thiébaut.
#
#------------------------------------------------------------------------------

module Optimization

using TiPi.Algebra

export LINE_SEARCH, NEW_ITERATE, CONVERGENCE
export TOO_MANY_ITERATIONS, TOO_MANY_EVALUATIONS
export initial_step

const LINE_SEARCH          = 0 # line search in progress
const NEW_ITERATE          = 1 # a new iterate is available for inspection
const CONVERGENCE          = 2 # algorithm has converged
const TOO_MANY_ITERATIONS  = 3 # too many iterations
const TOO_MANY_EVALUATIONS = 4 # too many evaluations

reason = Dict{Int,ASCIIString}(LINE_SEARCH => "line search in progress",
                               NEW_ITERATE => "a new iterate is available",
                               CONVERGENCE => "algorithm has converged",
                               TOO_MANY_ITERATIONS => "too many iterations",
                               TOO_MANY_EVALUATIONS => "too many evaluations")

"""
### Estimate initial step length

The call:
```
    initial_step(x, d, slen) --
```

yields the initial step length for the first iteration or after a restart.
`x` are the current variables, `d` is the search direction, `slen=(alen,rlen)`
where `alen` and `rlen` are an absolute and relative step length (`alen` > 0
and `rlen` >= 0).

The result is: `a/||d|`| where `||d||` is the Euclidean norm of `d` and:
```
    a = rlen*||x||   if rlen*||x|| > 0
      = alen         otherwise
```
"""
function initial_step{T,N}(x::Array{T,N}, d::Array{T,N}, slen::NTuple{2})
    @assert(size(x) == size(d))
    dnorm = norm2(d)
    len1::Cdouble = slen[1]
    len2::Cdouble = slen[2]
    if len2 > 0
        len2 *= norm2(x)
    end
    (len2 > 0 ? len2 : len1)/dnorm
end

end # module
