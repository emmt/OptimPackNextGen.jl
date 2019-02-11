#
# linesearches.jl --
#
# Line search methods for OptimPackNextGen.
#
# -----------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2015-2019, Éric Thiébaut.
# <https://github.com/emmt/OptimPackNextGen.jl>.
#

module LineSearches

import ...OptimPackNextGen
import OptimPackNextGen.Float

export
    start!,
    iterate!,
    getreason,
    getstatus,
    getstep,
    gettask,
    usederivatives,
    LineSearch,
    ArmijoLineSearch,
    MoreToraldoLineSearch,
    MoreThuenteLineSearch

const REASONS = Dict{Symbol,String}(
    :NOT_STARTED => "Algorithm not yet started",
    :F_LE_FMIN => "F(X) ≤ FMIN",
    :COMPUTE_FG => "Caller must compute f(x) and g(x)",
    :NEW_X => "A new iterate is available for examination",
    :FINAL_X => "A solution has been found within tolerances",
    :FATOL_TEST_SATISFIED => "FATOL test satisfied",
    :FRTOL_TEST_SATISFIED => "FRTOL test satisfied",
    :GATOL_TEST_SATISFIED => "GATOL test satisfied",
    :GRTOL_TEST_SATISFIED => "GRTOL test satisfied",
    :FIRST_WOLFE_SATISFIED =>"First Wolfe condition satisfied",
    :STRONG_WOLFE_SATISFIED => "Strong Wolfe conditions both satisfied",
    :STP_EQ_STPMIN => "Step at lower bound",
    :STP_EQ_STPMAX => "Step at upper bound",
    :XTOL_TEST_SATISFIED => "XTOL test satisfied",
    :ROUNDING_ERRORS => "Rounding errors prevent progress",
    :NOT_DESCENT => "Search direction is not a descent direction",
    :INIT_STP_LE_ZERO => "Initial step ≤ 0"
)

"""
## Line search methods

Line search methods are instances of types derived from the abstract type
`LineSearch{T}` which is parameterized by the floating point type `T` for the
computations.  Assuming `SomeLineSearch` is a concrete line search type, a
typical line search is performed as follows:

    # Create an instance of the line search method:
    ls = SomeLineSearch(T)

    # Start the line search and loop until a step satisfying
    # some conditions is found:
    x0 = ...            # initial variables
    f0 = func(x)        # function value at x0
    g0 = grad(x)        # gradient at x0
    d = ...             # search direction
    dtg0 = vdot(d, g0)  # directional derivative at x0
    stp = ...           # initial step
    stpmin = ...        # lower bound for the step (usually zero)
    stpmax = ...        # upper bound for the step (usually a large number)
    task = start!(ls, f0, dtg0, stp; stpmin = stpmin, stpmax = stpmax)
    while task == :SEARCH
        stp = getstep(ls) # get step lenght to try
        x = x0 + stp*d    # compute trial point
        f = func(x)       # function value at x
        g = grad(x)       # gradient at x
        dtg = vdot(d, g)  # directional derivative at x
        task = iterate!(ls, stp, f, dtg)
    end
    task = gettask(ls)
    if task != :CONVERGENCE
        if task == :WARNING
            @warn getreason(ls)
        else
            error(getreason(ls))
        end
    end

Note that the same line search instance may be re-used for subsequent line
searches (with the same settings).

"""
abstract type LineSearch{T<:AbstractFloat} end

"""
    getstep(ls)

yields the length of the next step to try in the line search implemented by
`ls`.
"""
getstep(ls::LineSearch) = ls.stp

"""
    gettask(ls)

yields the current pending task in the line search implemented by `ls`.
"""
gettask(ls::LineSearch) = ls.task

"""
    getstatus(ls)

yields the current status of the line search implemented by `ls`.
"""
getstatus(ls::LineSearch) = ls.status

# Helper function.
function report!(ls::LineSearch, task::Symbol, status::Symbol)
    ls.status = status
    ls.task = task
    return task
end

"""
    getreason(ls)

yields a textual explanation for the current state of the line search
implemented by `ls`.
"""
getreason(ls::LineSearch) = REASONS[ls.status]

"""
    usederivative(ls)

indicates whether the line search instance `ls` requires the derivative of the
objective function when calling `iterate!`.  Alternatively `ls` can also be the
line search type.  Note that the derivative is always needed by the `start!`
method.
"""
usederivatives(::T) where {T<:LineSearch} = usederivatives(T)

"""
    start!(ls, f0, g0, stp; stpmin = 0, stpmax = Inf) -> task

starts a new line search with the method implemented by the line search
instance `ls`, the returned value is the next task to perform (see `iterate!`
for the interpretation of this value).  Arguments `f0` and `g0` are the value
and the directional derivative of the objective function at the start of the
search (that is for a step length equal to zero).  Argument `stp` is the length
of the next step to try and must be nonnegative.  Keywords `stpmin` and
`stpmax` can be used to specify bounds for the length of the step.
"""
function start!(ls::LineSearch{T},
                f0::Real, g0::Real, stp::Real;
                stpmin::Real = zero(T),
                stpmax::Real = typemax(T)) where {T<:AbstractFloat}
    return start!(ls, convert(T, f0), convert(T, g0), convert(T, stp),
                  convert(T, stpmin), convert(T, stpmax))
end

"""
    iterate!(ls, stp, f, g) -> task

performs one iteration of the line search implemented by `ls` with `f` and `g`
the value and directional derivative of the objective function for the step
`stp` (which must be equal to `getstep(ls)`).

The returned `task` is one of the following symbolic values:

* `:SEARCH` to indicate that the caller should take the new trial step, given
  by `getstep(ls)`, and call `iterate!` with the updated values of the
  objective function and of its directional derivative.

* `:CONVERGENCE` to indicate that the line search criterion holds.

* `:WARNING` to indicate that no further progresses are possible.

""" iterate!

#-------------------------------------------------------------------------------

mutable struct ArmijoLineSearch{T<:AbstractFloat} <: LineSearch{T}
    task::Symbol
    status::Symbol
    ftol::T
    finit::T
    ginit::T
    stp::T
    stpmin::T
end

"""

    ArmijoLineSearch(T; ftol = 1e-4) -> ls

yields a line search instance for floating-point type `T` which implements
Armijo's method.  Keyword `ftol` can be used to specify a nonnegative tolerance
for the sufficient decrease condition.


## Description

The objective of Armijo's method is to find a step `stp` that satisfies the
sufficient decrease condition (1st Wolfe condition):

    f(stp) ≤ f(0) + ftol*stp*f'(0),

where `stp` is smaller or equal the initial step (backtracking).  The method
consists in reducing the step (by a factor 2) until the criterion holds.


## References

* L. Armijo, "*Minimization of functions having Lipschitz continuous first
  partial derivatives*" in Pacific Journal of Mathematics, vol. 16, pp. 1–3
  (1966).

"""
function ArmijoLineSearch(::Type{T} = Float;
                          ftol::Real = 1e-4) where {T<:AbstractFloat}
    @assert 0 < ftol ≤ 1/2
    ArmijoLineSearch{T}(:START, :NOT_STARTED, ftol, 0, 0, 0, 0)
end

# Armijo's line search does not use the directional derivative to refine the
# step.
usederivatives(::Type{<:ArmijoLineSearch}) = false

# start! method for Armijo's line search has the same implementation as the
# Moré & Toraldo line search.

function iterate!(ls::ArmijoLineSearch, stp::Real, f::Real, g::Real)
    @assert ls.task == :SEARCH
    @assert stp == ls.stp
    if f ≤ ls.finit + ls.ftol*stp*ls.ginit
        return report!(ls, :CONVERGENCE, :FIRST_WOLFE_CONDITION_SATISFIED)
    elseif stp > ls.stpmin
        ls.stp = max(stp/2, ls.stpmin)
        return report!(ls, :SEARCH, :COMPUTE_FG)
    else
        ls.stp = ls.stpmin
        return report!(ls, :WARNING, :STP_EQ_STPMIN)
    end
end

#-------------------------------------------------------------------------------

mutable struct MoreToraldoLineSearch{T<:AbstractFloat} <: LineSearch{T}
    task::Symbol
    status::Symbol
    gamma1::T
    gamma2::T
    ftol::T
    finit::T
    ginit::T
    stp::T
    stpmin::T
    strict::Bool
end

"""

    MoreToraldoLineSearch(T; ...) -> ls

yields a backtracking line search proposed by Moré & Toraldo (1991) and which
is based on a safeguarded quadratic interpolation.  Argument `T` is the
floating-point type for the computations.

Keyword `ftol` can be used to specify a nonnegative tolerance for the
sufficient decrease condition.

Keyword `gamma` can be used to specify a tuple of two values to safeguard the
quadratic interpolation of the step and such that `0 < gamma[1] < gamma[2] <
1`).  In Moré & Toraldo (1991, GPCG algorithm) `gamma = (0.01,0.5)` while in
Birgin et al. (2000, SPG2 algorithm) `gamma = (0.1,0.9)`.  The default settings
are `gamma = (0.1, 0.5)`.

If keyword `strict` is set to `false`, a variant of the method is used which
takes a bissection step whenever the curvature of the model is not striclty
positive.  By default, `strict` is `true`.


## Description

The objective of Moré & Toraldo method is to find a step `stp` that satisfies
the sufficient decrease condition (1st Wolfe condition):

    f(stp) ≤ f(0) + ftol*stp*f'(0),

where `stp` is smaller or equal the initial step (backtracking).  The principle
of the method is to reduce the step by a safeguarded quadratic interpolation
(or by a bissection if the minimum of the quadratic interpolation falls outside
the current search interval) until the criterion is met.

The new step is computed as follows:

    newstp = clamp(fact, gamma[1], gamma[2])*stp

where `fact` is the factor by which the current step should be multiplied to
reach the minimum of the quadratic interpolation of the objective function.


## References

* J.J. Moré & G. Toraldo, "*On the Solution of Large Quadratic Programming
  Problems with Bound Constraints*", SIAM J. Optim., vol. 1, pp. 93–113 (1991).

* E.G. Birgin, J.M. Martínez & M. Raydan, "*Nonmonotone Spectral Projected
  Gradient Methods on Convex Sets*", SIAM J. Optim., vol. 10, pp. 1196–1211
  (2000).

"""
function MoreToraldoLineSearch(::Type{T} = Float;
                               ftol::Real = 1e-4,
                               strict::Bool = false,
                               gamma::NTuple{2,Real} = (0.1,0.5)
                               ) where {T<:AbstractFloat}
    @assert 0 < ftol ≤ 1/2
    @assert 0 < gamma[1] < gamma[2] < 1
    MoreToraldoLineSearch{T}(:START, :NOT_STARTED, gamma[1], gamma[2], ftol,
                             0, 0, 0, 0, strict)
end

# Backtracking line search does not use the directional derivative to refine
# the step but it does use the initial derivative.
usederivatives(::Type{<:MoreToraldoLineSearch}) = false

function start!(ls::Union{MoreToraldoLineSearch,ArmijoLineSearch},
                f0::Real, g0::Real, stp::Real, stpmin::Real, stpmax::Real)
    # Check the input arguments for errors.
    if stpmax < stpmin
        throw(ArgumentError("STPMAX < STPMIN"))
    end
    if stpmin < 0
        throw(ArgumentError("STPMIN < 0"))
    end
    if stp > stpmax
         throw(ArgumentError("STP > STPMAX"))
    end
    if stp < stpmin
         throw(ArgumentError("STP < STPMIN"))
    end
    if g0 ≥ 0
        throw(ArgumentError("Not a descent direction"))
        #return report!(ls, :ERROR, :NOT_DESCENT)
    end

    # Store objective function and directional derivative at start of line
    # search.
    ls.finit = f0
    ls.ginit = g0
    ls.stp = stp
    ls.stpmin = stpmin
    return report!(ls, :SEARCH, :COMPUTE_FG)
end

function iterate!(ls::MoreToraldoLineSearch, stp::Real, f::Real, g::Real)
    @assert ls.task == :SEARCH
    @assert stp == ls.stp
    if f ≤ ls.finit + ls.ftol*stp*ls.ginit
        return report!(ls, :CONVERGENCE, :FIRST_WOLFE_CONDITION_SATISFIED)
    elseif stp > ls.stpmin
        q = -ls.ginit*stp
        r = f - (ls.finit - q)
        if (ls.strict ? r > 0 && q > 0 : r != 0)
            # Safeguarded quadratic interpolation.
            stp *= clamp(q/(r + r), ls.gamma1, ls.gamma2)
        else
            # Bissection.
            stp /= 2
        end
        ls.stp = max(stp, ls.stpmin)
        return report!(ls, :SEARCH, :COMPUTE_FG)
    else
        ls.stp = ls.stpmin
        return report!(ls, :WARNING, :STP_EQ_STPMIN)
    end
end

#-------------------------------------------------------------------------------

"""

`MoreThuenteLineSearch{T}` defines the work-space for Moré & Thuente line
search.

Line search methods which need to compute a cubic step via `cstep!` need to
store some parameters in an instance of `LineSearchInterval` which has the
following members:

* `lower` and `upper` contain respectively a lower and an upper bound for the
  step;

* `stx`, `fx`, `dx` contain the values of the step, function, and derivative at
  the best step so far;

* `sty`, `fy`, `dy` contain the value of the step, function, and derivative at
  the other endpoint`sty`;

* `brackt` indicates whether a minimum is bracketed in the interval
  `(stx,sty)`.

At the start of the line search, `brackt = false`, `stx = sty = 0` while `fx =
fy = f0` and `dx = dy = g0` the value of the function and its derivative for
the step `stp = 0`.  Then `cstep!` can be called (typically in the `iterate!`
method) to compute a new step (based on cubic or quadratic interpolation) and
to maintain the interval of search.

"""
mutable struct MoreThuenteLineSearch{T<:AbstractFloat} <: LineSearch{T}
    task::Symbol
    status::Symbol
    ftol::T
    gtol::T
    xtol::T
    stpmin::T
    stpmax::T
    finit::T
    ginit::T
    stp::T
    stx::T
    fx::T
    gx::T
    sty::T
    fy::T
    gy::T
    lower::T     # current lower bound for the step
    upper::T     # current upper bound for the step
    width::T     # current width of the interval
    oldwidth::T  # previous width of the interval
    stage::Int
    brackt::Bool # minimum has been bracketed?
end

function MoreThuenteLineSearch(::Type{T} = Float;
                               ftol::Real = 0.001,
                               gtol::Real = 0.9,
                               xtol::Real = 0.1) where {T<:AbstractFloat}
    @assert 0 < ftol < 1
    @assert 0 < gtol < 1
    @assert 0 < xtol < 1
    MoreThuenteLineSearch{T}(:START, :NOT_STARTED,
                             ftol, gtol, xtol, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0, false)
end

# Moré & Thuente line search does use the directional derivative to refine the
# step.
usederivatives(::Type{<:MoreThuenteLineSearch}) = true

# A few constants.
const XTRAPL = 1.1
const XTRAPU = 4.0
const STPMAX = 1e20

function start!(ls::MoreThuenteLineSearch{T},
                f0::Real, g0::Real, stp::Real;
                ftol::Real = ls.ftol,
                gtol::Real = ls.gtol,
                xtol::Real = ls.xtol,
                stpmin::Real = zero(T),
                stpmax::Real = stp*STPMAX) where {T<:AbstractFloat}
    start!(ls, f0, g0, stp, ftol, gtol, xtol, stpmin, stpmax)
end

function start!(ls::MoreThuenteLineSearch,
                f0::Real, g0::Real, stp::Real,
                ftol::Real, gtol::Real, xtol::Real,
                stpmin::Real, stpmax::Real)

    # Check the input arguments for errors.
    if stpmax < stpmin
        throw(ArgumentError("STPMAX < STPMIN"))
    end
    if stpmin < 0
        throw(ArgumentError("STPMIN < 0"))
    end
    if xtol < 0
         throw(ArgumentError("XTOL < 0"))
    end
    if ftol ≤ 0
         throw(ArgumentError("FTOL ≤ 0"))
    end
    if gtol ≤ 0
         throw(ArgumentError("GTOL ≤ 0"))
    end
    if g0 ≥ 0
         throw(ArgumentError("Not a descent direction"))
    end
    if stp > stpmax
         throw(ArgumentError("STP > STPMAX"))
    end
    if stp < stpmin
         throw(ArgumentError("STP < STPMIN"))
    end

    # Initialize local variables.
    # The variables STX, FX, GX contain the values of the step,
    # function, and derivative at the best step.
    # The variables STY, FY, GY contain the value of the step,
    # function, and derivative at STY.
    # The variables STP, F, G contain the values of the step,
    # function, and derivative at STP.
    ls.ftol      = ftol
    ls.gtol      = gtol
    ls.xtol      = xtol
    ls.stpmin    = stpmin
    ls.stpmax    = stpmax
    ls.finit     = f0
    ls.ginit     = g0
    ls.stp       = stp
    ls.stx       = 0
    ls.fx        = f0
    ls.gx        = g0
    ls.sty       = 0
    ls.fy        = f0
    ls.gy        = g0
    ls.lower     = 0
    ls.upper     = stp + stp*XTRAPU
    ls.width     = stpmax - stpmin
    ls.oldwidth  = 2*(stpmax - stpmin)
    ls.stage     = 1
    ls.brackt    = false
    report!(ls, :SEARCH, :COMPUTE_FG)
end

function iterate!(ls::MoreThuenteLineSearch{T},
                  stp::Real, f::Real, g::Real) where {T<:AbstractFloat}
    return iterate!(ls, convert(T, stp), convert(T, f), convert(T, g))
end

function iterate!(ls::MoreThuenteLineSearch{T},
                  stp::T, f::T, g::T) where {T<:AbstractFloat}

    @assert ls.task == :SEARCH
    @assert stp == ls.stp

    P66 = T(0.66)
    HALF = T(0.5)
    ZERO = zero(T)

    # If psi(stp) ≤ 0 and f'(stp) ≥ 0 for some step, then the algorithm enters
    # the second stage.
    gtest = ls.ftol*ls.ginit
    ftest = ls.finit + stp*gtest
    if ls.stage == 1 && f ≤ ftest && g ≥ ZERO
        ls.stage = 2
    end

    # Test for termination: convergence or warnings.
    if f ≤ ftest && abs(g) ≤ -ls.gtol*ls.ginit
        return report!(ls, :CONVERGENCE, :STRONG_WOLFE_SATISFIED)
    end
    if stp == ls.stpmin && (f > ftest || g ≥ gtest)
        return report!(ls, :WARNING, :STP_EQ_STPMIN)
    end
    if stp == ls.stpmax && f ≤ ftest && g ≤ gtest
        return report!(ls, :WARNING, :STP_EQ_STPMAX)
    end
    if ls.brackt
        if ls.upper - ls.lower ≤ ls.xtol*ls.upper
            return report!(ls, :WARNING, :XTOL_TEST_SATISFIED)
        end
        if stp ≤ ls.lower || stp ≥ ls.upper
            return report!(ls, :WARNING, :ROUNDING_ERRORS)
        end
    end

    # A modified function is used to predict the step during the first stage if
    # a lower function value has been obtained but the decrease is not
    # sufficient.

    if ls.stage == 1 && f ≤ ls.fx && f > ftest

        # Call CSTEP to update STX, STY, and to compute the new step for the
        # modified function and its derivatives.
        info, ls.brackt,
        ls.stx, fxm, gxm,
        ls.sty, fym, gym,
        stp = cstep(ls.brackt, ls.lower, ls.upper,
                    ls.stx, ls.fx - ls.stx*gtest, ls.gx - gtest,
                    ls.sty, ls.fy - ls.sty*gtest, ls.gy - gtest,
                    stp, f - stp*gtest, g - gtest)

        # Reset the function and derivative values for F.
        ls.fx = fxm + ls.stx*gtest
        ls.fy = fym + ls.sty*gtest
        ls.gx = gxm + gtest
        ls.gy = gym + gtest

    else

        # Call CSTEP to update STX, STY, and to compute the new step.
        info, ls.brackt,
        ls.stx, ls.fx, ls.gx,
        ls.sty, ls.fy, ls.gy,
        stp = cstep(ls.brackt, ls.lower, ls.upper,
                    ls.stx, ls.fx, ls.gx,
                    ls.sty, ls.fy, ls.gy,
                    stp, f, g)

    end

    # Decide if a bisection step is needed.
    if ls.brackt
        wcur = abs(ls.sty - ls.stx)
        if wcur ≥ P66*ls.oldwidth
            stp = ls.stx + HALF*(ls.sty - ls.stx)
        end
        ls.oldwidth = ls.width
        ls.width = wcur
    end

    # Set the minimum and maximum steps allowed for STP.
    if ls.brackt
        if ls.stx ≤ ls.sty
            ls.lower = ls.stx
            ls.upper = ls.sty
        else
            ls.lower = ls.sty
            ls.upper = ls.stx
        end
    else
        ls.lower = stp + XTRAPL*(stp - ls.stx)
        ls.upper = stp + XTRAPU*(stp - ls.stx)
    end

    # Force the step to be within the bounds STPMAX and STPMIN.
    stp = clamp(stp, ls.stpmin, ls.stpmax)

    # If further progress is not possible, let STP be the best point obtained
    # during the search.
    if (ls.brackt && (stp ≤ ls.lower || stp ≥ ls.upper ||
                      ls.upper - ls.lower ≤ ls.xtol*ls.upper))
        stp = ls.stx
    end

    # Save next step to try.
    ls.stp = stp

    # Obtain another function and derivative.
    return report!(ls, :SEARCH, :COMPUTE_FG)
end

"""
# Compute a safeguarded cubic step.

    cstep(brackt, stpmin, stpmax,
          stx, fx, dx,
          sty, fy, dy,
          stp, fp, dp) ->  info, brackt, stx,fx,dx, sty,fy,dy, stp

The function `cstep` computes a safeguarded step for a search procedure and
updates an interval that contains a step that satisfies a sufficient decrease
and a curvature condition.  The algorithm is described in:

* J.J. Moré and D.J. Thuente, "*Line search algorithms with guaranteed
  sufficient decrease*" in ACM Transactions on Mathematical Software, vol. 20,
  pp. 286–307 (1994).

The parameter `stx` contains the step with the least function value.  If
`brackt` is set to true (*i.e.,* non-zero) then a minimizer has been bracketed
in an interval with endpoints `stx` and `sty`.  The parameter `stp` contains
the current step.  The subroutine assumes that if `brackt` is true then:

     min(stx,sty) < stp < max(stx,sty),

and that the derivative at `stx` is negative in the direction of the step.

On output, the updated parameters are returned.

Parameter `brackt` specifies if a minimizer has been bracketed.  Initially
`brackt` must be set to false.  The returned value of `brackt` indicates if a
minimizer has been bracketed.

Parameters `stpmin` and `stpmax` specify the lower and the upper bounds for the
step.

Parameters `stx`, `fx` and `dx` specify the step, the function and the
derivative at the best step obtained so far.  The derivative must be negative
in the direction of the step, that is, `dx` and `stp - stx` must have opposite
signs.  On return, these parameters are updated appropriately.

Parameters `sty`, `fy` and `dy` specify the step, the function and the
derivative at the other endpoint of the interval of uncertainty.  On return,
these parameters are updated appropriately.

Parameters `stp`, `fp` and `dp` specify the step, the function and the
derivative at the current step.  If `brackt` is true, then `stp` must be
between `stx` and `sty`.  On return, these parameters are updated
appropriately.  The returned value of `stp` is the next trial step.

The returned value `info` indicates which case occured for computing the new
step.

"""
function cstep(brackt::Bool, stpmin::Real, stpmax::Real,
               stx::Real, fx::Real, dx::Real,
               sty::Real, fy::Real, dy::Real,
               stp::Real, fp::Real, dp::Real)
    cstep(brackt,
          promote(stpmin, stpmax, stx, fx, dx, sty, fy, dy, stp, fp, dp)...)
end

function cstep(brackt::Bool, stpmin::T, stpmax::T,
               stx::T, fx::T, dx::T,
               sty::T, fy::T, dy::T,
               stp::T, fp::T, dp::T) where {T<:AbstractFloat}

    ZERO = zero(T)
    ONE = one(T)
    TWO = convert(T, 2)
    THREE = convert(T, 3)
    P66 = convert(T, 0.66)
    local theta::T, gamma::T, p::T, q::T, r::T, s::T, t::T

    info = 0

    # Check the input parameters for errors.
    if brackt && (stp ≤ min(stx, sty) || stp ≥ max(stx, sty))
        throw(ArgumentError("STP outside bracket (STX,STY)"))
    end
    if dx*(stp - stx) ≥ ZERO
        throw(ArgumentError("descent condition violated"))
    end
    if stpmax < stpmin
        throw(ArgumentError("STPMAX < STPMIN"))
    end

    # Determine if the derivatives have opposite sign.
    opposite = ((dx < ZERO && dp > ZERO) || (dx > ZERO && dp < ZERO))

    if fp > fx
        # First case.  A higher function value.  The minimum is bracketed.  If
        # the cubic step is closer to STX than the quadratic step, the cubic
        # step is taken, otherwise the average of the cubic and quadratic steps
        # is taken.
        info = 1
        theta = THREE*(fx - fp)/(stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s*sqrt((theta/s)^2 - (dx/s)*(dp/s))
        if stp < stx
            gamma = -gamma
        end
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p/q
        stpc = stx + r*(stp - stx)
        stpq = stx + ((dx/((fx - fp)/(stp - stx) + dx))/TWO)*(stp - stx)
        if abs(stpc - stx) < abs(stpq - stx)
            stpf = stpc
        else
            stpf = stpc + (stpq - stpc)/TWO
        end
        brackt = true
    elseif opposite
        # Second case.  A lower function value and derivatives of opposite
        # sign.  The minimum is bracketed.  If the cubic step is farther from
        # STP than the secant (quadratic) step, the cubic step is taken,
        # otherwise the secant step is taken.
        info = 2
        theta = THREE*(fx - fp)/(stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s*sqrt((theta/s)^2 - (dx/s)*(dp/s))
        if stp > stx
            gamma = -gamma
        end
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p/q
        stpc = stp + r*(stx - stp)
        stpq = stp + (dp/(dp - dx))*(stx - stp)
        if abs(stpc - stp) > abs(stpq - stp)
            stpf = stpc
        else
            stpf = stpq
        end
        brackt = true
    elseif abs(dp) < abs(dx)
        # Third case.  A lower function value, derivatives of the same sign,
        # and the magnitude of the derivative decreases.  The cubic step is
        # computed only if the cubic tends to infinity in the direction of the
        # step or if the minimum of the cubic is beyond STP.  Otherwise the
        # cubic step is defined to be the secant step.
        info = 3
        theta = THREE*(fx - fp)/(stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        # The case GAMMA = 0 only arises if the cubic does not tend to infinity
        # in the direction of the step.
        t = (theta/s)^2 - (dx/s)*(dp/s)
        gamma = (t > ZERO ? s*sqrt(t) : ZERO)
        if stp > stx
            gamma = -gamma
        end
        p = (gamma - dp) + theta
        #q = ((gamma - dp) + gamma) + dx
        q = (gamma + (dx - dp)) + gamma
        r = p/q
        if r < ZERO && gamma != ZERO
            stpc = stp + r*(stx - stp)
        elseif stp > stx
            stpc = stpmax
        else
            stpc = stpmin
        end
        stpq = stp + (dp/(dp - dx))*(stx - stp)
        if brackt
            # A minimizer has been bracketed.  If the cubic step is closer to
            # STP than the secant step, the cubic step is taken, otherwise the
            # secant step is taken.
            stpf = abs(stpc - stp) < abs(stpq - stp) ? stpc : stpq
            t = stp + P66*(sty - stp)
            if stp > stx ? stpf > t : stpf < t
                stpf = t
            end
        else
            # A minimizer has not been bracketed. If the cubic step is farther
            # from stp than the secant step, the cubic step is taken, otherwise
            # the secant step is taken.
            stpf = abs(stpc - stp) > abs(stpq - stp) ? stpc : stpq
            stpf = max(stpmin, min(stpf, stpmax))
        end
    else
        # Fourth case.  A lower function value, derivatives of the same sign,
        # and the magnitude of the derivative does not decrease.  If the
        # minimum is not bracketed, the step is either STPMIN or STPMAX,
        # otherwise the cubic step is taken.
        info = 4
        if brackt
            theta = THREE*(fp - fy)/(sty - stp) + dy + dp
            s = max(abs(theta), abs(dy), abs(dp))
            t = theta/s
            gamma = s*sqrt((theta/s)^2 - (dy/s)*(dp/s))
            if stp > sty
                gamma = -gamma
            end
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p/q
            stpc = stp + r*(sty - stp)
            stpf = stpc
        elseif stp > stx
            stpf = stpmax
        else
            stpf = stpmin
        end
    end

    # Update the interval which contains a minimizer and guess for next step.
    if fp > fx
        sty = stp
        fy = fp
        dy = dp
    else
        if opposite
            sty = stx
            fy = fx
            dy = dx
        end
        stx = stp
        fx = fp
        dx = dp
    end
    stp = stpf
    return (info, brackt, stx,fx,dx, sty,fy,dy, stp)
end

end # module
