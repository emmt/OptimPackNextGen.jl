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
export iterate!, start!, LineSearch, ArmijoLineSearch, BacktrackLineSearch

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

abstract LineSearch

function start!(ls::LineSearch, f0::Real, df0::Real, stp::Real)
    start!(ls, convert(Cdouble, f0), convert(Cdouble, df0), convert(Cdouble, stp))
end

function iterate!(ls::LineSearch, f1::Real, df1::Real, stp::Real)
    iterate!(ls, convert(Cdouble, f1), convert(Cdouble, df1), convert(Cdouble, stp))
end

type ArmijoLineSearch <: LineSearch
    ftol::Cdouble
    finit::Cdouble
    ginit::Cdouble
    function ArmijoLineSearch(ftol::Real)
        @assert(0 < ftol < 1)
        new(ftol, 0, 0)
    end
end
ArmijoLineSearch() = ArmijoLineSearch(1e-4)

function start!(ls::ArmijoLineSearch, f0::Cdouble, df0::Cdouble, stp::Cdouble)
    ls.finit = f0
    ls.ginit = df0
    return (LINE_SEARCH, stp)
end

function iterate!(ls::ArmijoLineSearch, f1::Cdouble, df1::Cdouble, stp::Cdouble)
    if f1 ≤ ls.finit + ls.ftol*stp*ls.ginit
        return (NEW_ITERATE, stp)
    else
        return (LINE_SEARCH, stp/2)
    end
end

type BacktrackLineSearch <: LineSearch
    t1::Cdouble
    t2::Cdouble
    ftol::Cdouble
    finit::Cdouble
    ginit::Cdouble
    function BacktrackLineSearch(ftol::Real, t1::Real, t2::Real)
        @assert(0 < ftol < 1)
        @assert(0 < t1 < t2 < 1)
        new(t1, t2, ftol, 0, 0)
    end
end
BacktrackLineSearch() = BacktrackLineSearch(1e-4, 0.1, 0.9)

function start!(ls::BacktrackLineSearch, f0::Cdouble, df0::Cdouble, stp::Cdouble)
    ls.finit = f0
    ls.ginit = df0
    return (LINE_SEARCH, stp)
end

function iterate!(ls::BacktrackLineSearch, f1::Cdouble, df1::Cdouble, stp::Cdouble)
    if f1 ≤ ls.finit + ls.ftol*stp*ls.ginit
        return (NEW_ITERATE, stp)
    else
        q = -ls.ginit*stp*stp
        r = 2*(f1 - ls.finit - ls.ginit*stp)
        if r > 0 && ls.t1*r ≤ q ≤ stp*ls.t2*r
            # quadratic interpolation
            return (LINE_SEARCH, q/r)
        else
            # bissection
            return (LINE_SEARCH, stp/2)
        end
    end
end

end # module
