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
using TiPi.ConvexSets

export blmvm!, vmlmb!
export LINE_SEARCH, NEW_ITERATE, CONVERGENCE
export TOO_MANY_ITERATIONS, TOO_MANY_EVALUATIONS
export NO_FUNCTION_CHANGE, NO_GRADIENT_CHANGE
export initial_step
export iterate!, start!, usederivative
export LineSearch, ArmijoLineSearch, BacktrackLineSearch

const LINE_SEARCH          = 0 # line search in progress
const NEW_ITERATE          = 1 # a new iterate is available for inspection
const CONVERGENCE          = 2 # algorithm has converged
const TOO_MANY_ITERATIONS  = 3 # too many iterations
const TOO_MANY_EVALUATIONS = 4 # too many evaluations
const NO_FUNCTION_CHANGE   = 5 # no function change between iterations
const NO_GRADIENT_CHANGE   = 6 # no gradient change between iterations
const WOULD_BLOCK          = 7 # search direction infeasible

reason = Dict{Int,ASCIIString}(LINE_SEARCH => "line search in progress",
                               NEW_ITERATE => "a new iterate is available",
                               CONVERGENCE => "algorithm has converged",
                               TOO_MANY_ITERATIONS => "too many iterations",
                               TOO_MANY_EVALUATIONS => "too many evaluations",
                               NO_FUNCTION_CHANGE => "no function change between iterations",
                               NO_GRADIENT_CHANGE => "no gradient change between iterations",
                               WOULD_BLOCK => "search direction infeasible")

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

#------------------------------------------------------------------------------
# LINE SEARCH METHODS

"""
### Line Search Methods

The objective of line search is to (approximately) minimize:
```
     ϕ(α) = f(x0 + α d)
```
for `α > 0` and with `x0` the variables at the start of the line search, `d`
the search direction and `α` the step lenght.

Implemented line search methods are sub-types of the `LineSearch` abstract
type.  To use a given line search method, say `SomeLineSearch`, first create
the line search object:
```
    lnsrch = SomeLineSearch(...)
```
At every iterate `x0` of the optimization algorithm, the line search is
initiated with:
```
    (status, nextstep) = start!(lnsrch, f0, df0, nextstep)
```
where `f0 = f(x0)` and `df0 = ⟨ d, ∇f(x0) ⟩` are the function value and the
directional derivative at the position `x0`, and `nextstep` is the lenght of
the first step to take.  For subsequent steps, checking the convergence
of the line search and computating the next step length are performed by:
```
    (status, nextstep) = start!(lnsrch, α, f1, df1)
```
where `f1 = f(x + α d)` and `df1 = ⟨ d, ∇f(x + α d) ⟩` are the function value
and the directional derivative at the position `x0 + α d`, that is for a step
length `α`.

The result of the `start!` and `iterate!` methods is a tuple with the line
search status and the length of the next step to take.  The status is either
`LINE_SEARCH` or `NEW_ITERATE` to indicate whether the line search is in
progress or has converged.  If the returned status is `LINE_SEARCH`, the
caller should take the proposed step, compute the function value and, possibly
the directional derivative, and then call `iterate!` again.  If the returned
status is `NEW_ITERATE`, the line search has converged and a new iterate is
available for inspection (the next step length is equal to the current one).

Although the directional derivative `df0` at the start of the line search is
required, not all line search methods need the directional derivative for
refining the step lenght (i.e. when calling `iterate!`).  To check that, call:
```
    usederivative(lnsrch)
```
which return a boolean result.

For unconstrained optimization, the directional derivative is:
```
    ϕ'(α) = ⟨ d, g(x0 + α d) ⟩
```
with `g(x) = ∇f(x)`.  However, if constraints are implemented, the function
`ϕ(α)` becomes:
```
    ϕ(α) = f(x(α))
```
where `x(α) = P(x0 + α d)` and `P(...)` is the projection onto the feasible
(convex) set.  Then:
```
    ϕ'(α) ≈ ⟨ (x(α) - x(0))/α, g(x(α)) ⟩
```
"""
abstract LineSearch

function start!(ls::LineSearch, f0::Real, df0::Real, stp::Real)
    start!(ls, convert(Cdouble, f0), convert(Cdouble, df0), convert(Cdouble, stp))
end

function iterate!(ls::LineSearch, stp::Real, f1::Real, df1::Real)
    iterate!(ls, convert(Cdouble, stp), convert(Cdouble, f1), convert(Cdouble, df1))
end

usederivative(ls::LineSearch) = true

@doc (@doc LineSearch) start!
@doc (@doc LineSearch) iterate!
@doc (@doc LineSearch) usederivative

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

# Armijo's line search does not use the directional derivative to refine the
# step.
usederivative(ls::ArmijoLineSearch) = false

function start!(ls::ArmijoLineSearch, f0::Cdouble, df0::Cdouble, stp::Cdouble)
    ls.finit = f0
    ls.ginit = df0
    return (LINE_SEARCH, stp)
end

function iterate!(ls::ArmijoLineSearch, stp::Cdouble, f1::Cdouble, df1::Cdouble)
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

# Backtracking line search does not use the directional derivative to refine
# the step.
usederivative(ls::BacktrackLineSearch) = false

function start!(ls::BacktrackLineSearch, f0::Cdouble, df0::Cdouble, stp::Cdouble)
    ls.finit = f0
    ls.ginit = df0
    return (LINE_SEARCH, stp)
end

function iterate!(ls::BacktrackLineSearch, stp::Cdouble, f1::Cdouble, df1::Cdouble)
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

#-------------------------------------------------------------------------------
# OPTIMIZATION ALGORITHMS

include("blmvm.jl")
include("vmlmb.jl")

end # module
