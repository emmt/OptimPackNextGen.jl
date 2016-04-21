#
# lnsrch.jl --
#
# Line search methods for TiPi.
#
#------------------------------------------------------------------------------
#
# This file is part of TiPi.jl licensed under the MIT "Expat" License.
#
# Copyright (C) 2015-2016, Éric Thiébaut.
#
#------------------------------------------------------------------------------

module LineSearch

export start!, iterate!, get_task, get_reason, get_step,
       AbstractLineSearch, MoreThuenteLineSearch

# Use the same floating point type for scalars as in TiPi.
import ..Float

"""
## Line search methods

Line search methods are instances of types derived from the abstract type
`AbstractLineSearch`.  Assuming `SomeLineSearch` is a concrete line search
type, a typical line search is perfomred as follows:

    # Create an instance of the line search method:
    ls = SomeLineSearch(...)

    # Start the line search and loop until a step satisfying
    # some conditions is found:
    x0 = ...            # initial variables
    f0 = func(x)        # function value at x0
    g0 = grad(x)        # gradient at x0
    d = ...             # search direction
    dtg0 = inner(d, g0) # directional derivative at x0
    stp = ...           # initial step
    stpmin = ...        # lower bound for the step
    stpmax = ...        # upper bound for the step
    searching = start!(ls, stp, f0, dtg0, stpmin, stpmax)
    while searching
        x = x0 + stp*d   # compute trial point
        f = func(x)      # function value at x
        g = grad(x)      # gradient at x
        dtg = inner(d,g) # directional derivative at x
        (stp, searching) = iterate!(ls, stp, f, dtg)
    end
    task = get_task(ls)
    if task != :CONVERGENCE
        if task == :ERROR
            error(get_reason(ls)
        else
            println("warning: ", get_reason(ls))
        end
    end

"""
abstract AbstractLineSearch

"""
`get_task(ls)` yields the current pending task for the line search instance
`ls`.  The result is one of the following symbols:

* `:START` if line search `ls` has not yet been started with `start!(ls, ...)`.

* `:SEARCH` if line search `ls` is in progress; the caller shall compute the
  value of the function and its derivative at the current trial step and call
  `iterate!(ls, ...)`.

* `:CONVERGENCE` if line search `ls` has converged.

* `:WARNING` if line search `ls` has been stopped with a warning; use
  `get_reason(ls)` to retrieve a textual explanation.

* `:ERROR` if there are any errors; use `get_reason(ls)` to retrieve a textual
  error message.
"""
get_task(ls::AbstractLineSearch) = ls.task

"""
`get_step(ls)` yields the current trial step for the line search instance `ls`.
"""
get_step(ls::AbstractLineSearch) = ls.step


"""
`get_reason(ls)` yields the error or warning message for the line search
instance `ls`.
"""
get_reason(ls::AbstractLineSearch) = ls.reason

"""
`requires_derivative(ls)` indicates whether the line search instance `ls`
requires the derivative of the function.  Alternatively `ls` can also be
the line search type.
"""
requires_derivative{T<:AbstractLineSearch}(::T) = requires_derivative(T)

"""
A line search is started by:

    search = start!(ls, stp, f0, g0, stpmin, stpmax)

which returns a boolean `search` indicating whether to try `stp` as the first
step.  If `search` is true then the caller shall compute the value of the
function and its derivative at the setp `stp` and call `iterate!`

Arguments are:

* `stp` is a positive initial estimate of the next step to take.

* `f0` is the value of the function at 0.

* `g0` is the derivative of the function at 0.

* `stpmin` is a nonnegative lower bound for the step.

* `stpmax` is a nonnegative upper bound for the step.

* `search` is a boolean indicating whether to try `stp` as the first step.

"""
function start!(ls::AbstractLineSearch, stp::Real, f0::Real, g0::Real,
                stpmin::Real, stpmax::Real)
    start!(ls, Float(stp), Float(f0), Float(g0), Float(stpmin), Float(stpmax))
    #error("`start!` method not implemented for this line search method")
end

"""
The call:

    (stp, search) = iterate!(ls, stp, f, g)

performs next line search iteration and return the next trial step `stp` and a
boolean `search` indicating whether to continue searching.  If `search` is true
then the caller shall compute the value of the function and its derivative for
the output value `stp` of the step and call `iterate!` again.  Otherwise, the
line search is terminated (the value of `stp` is left unchanged) either because
of convergence, failure or warnings.

Arguments are:

* `stp` is the current estimate of a satisfactory step.

* `f` is the value of the function at 0.

* `g` is the derivative of the function at 0.

"""
function iterate!(ls::AbstractLineSearch, stp::Real, f::Real, g::Real)
    iterate!(ls, Float(stp), Float(f), Float(g))
    #error("`iterate!` method not implemented for this line search method")
end

# Private method to initialize common members of line search instances.
function initialize!(ls::AbstractLineSearch)
    ls.step = 0
    ls.task = :START
    ls.reason = ""
    return ls
end

# Private method for the starting.
function starting!(ls::AbstractLineSearch, stp::Float)
    ls.step = stp
    ls.task = :SEARCH
    ls.reason = ""
    return true
end

# Private method to provide the next trial step.
function searching!(ls::AbstractLineSearch, stp::Float)
    ls.step = stp
    ls.task = :SEARCH
    ls.reason = ""
    return (ls.step, true)
end

# Private method to indicate that the line search has converged.
function convergence!(ls::AbstractLineSearch, reason::AbstractString)
    ls.task = :CONVERGENCE
    ls.reason = reason
    return (ls.step, false)
end

# Private method to indicate that the line search is terminated with a warning.
function warning!(ls::AbstractLineSearch, reason::AbstractString)
    ls.task = :WARNING
    ls.reason = reason
    return (ls.step, false)
end

# Private method to indicate that the line search has failed for some reason.
function failure!(ls::AbstractLineSearch, reason::AbstractString)
    ls.task = :ERROR
    ls.reason = reason
    return (ls.step, false)
end

#------------------------------------------------------------------------------
"""
## Moré & Thuente line search method

Moré & Thuente line search method finds a step that satisfies a sufficient
decrease condition and a curvature condition.

The algorithm is designed to find a step `stp` that satisfies the sufficient
decrease condition:

      f(stp) <= f(0) + ftol*stp*f'(0),

and the curvature condition:

      abs(f'(stp)) <= gtol*abs(f'(0)).

If `ftol` is less than `gtol` and if, for example, the function is bounded
below, then there is always a step which satisfies both conditions.

Each call to `iterate!` updates an interval with endpoints `stx` and `sty`.
The interval is initially chosen so that it contains a minimizer of the
modified function:

      psi(stp) = f(stp) - f(0) - ftol*stp*f'(0).

If `psi(stp) <= 0` and `f'(stp) >= 0` for some step `stp`, then the interval is
chosen so that it contains a minimizer of `f`.

If no step can be found that satisfies both conditions, then the algorithm
stops with a warning.  In this case `stp` only satisfies the sufficient
decrease condition.

The line search instance is created by:

    ls = MoreThuenteLineSearch(ftol=..., gtol=..., xtol=...)

where:

* `ftol` specifies a nonnegative tolerance for the sufficient decrease
         condition.

* `gtol` specifies a nonnegative tolerance for the curvature condition.

* `xtol` specifies a nonnegative relative tolerance for an acceptable step. The
         subroutine exits with a warning if the relative difference between
         `sty` and `stx` is less than `xtol`.

Default values for these parameters are `ftol=1e-3`, `gtol=0.9` and xtol=0.1
which are suitable for quasi Newton line search.


### References

* Moré, J. J. & Thuente, D. J. "Line search algorithms with guaranteed
  sufficient decrease", ACM Trans. Math. Softw., ACM Press, vol. 20,
  pp. 286-307 (1994).


### History

* MINPACK-1 Project. June 1983.  Argonne National Laboratory.  Jorge J. Moré
   and David J. Thuente.

* MINPACK-2 Project. November 1993.  Argonne National Laboratory and University
  of Minnesota.  Brett M. Averick, Richard G. Carter, and Jorge J. Moré.

* TiPi.jl Project.  April 2016.  Centre de Recherche Astrophysique de Lyon.
  Conversion to Julia by Éric Thiébaut.

"""
type MoreThuenteLineSearch <: AbstractLineSearch

    # Common to all line search instances.
    reason::AbstractString
    task::Symbol
    step::Float

    # More & Thuente line search parameters.
    ftol::Float
    gtol::Float
    xtol::Float
    stpmin::Float
    stpmax::Float
    finit::Float
    ginit::Float
    gtest::Float
    width::Float
    width1::Float
    stage::Int

    # Parameters shared with `cstep!`.
    #
    # The members `stx`, `fx`, `dx` contain the values of the step,
    # function, and derivative at the best step.
    #
    # The members `sty`, `fy`, `dy` contain the value of the step,
    # function, and derivative at `sty`.
    #
    smin::Float # minimum step in cstep!
    smax::Float # maximum step in cstep!
    stx::Float
    fx::Float
    dx::Float
    sty::Float
    fy::Float
    dy::Float
    brackt::Bool

    function MoreThuenteLineSearch(;
                                  ftol::Float=1e-3,
                                  gtol::Float=0.9,
                                  xtol::Float=0.1)
        @assert ftol ≥ 0
        @assert gtol ≥ 0
        @assert xtol ≥ 0

        ws = new()

        ws.ftol = ftol
        ws.gtol = gtol
        ws.xtol = xtol
        ws.smin = 0
        ws.smax = 0
        ws.stx = 0
        ws.fx = 0
        ws.dx = 0
        ws.sty = 0
        ws.fy = 0
        ws.dy = 0

        ws.stpmin = 0
        ws.stpmax = 0
        ws.finit = 0
        ws.ginit = 0
        ws.gtest = 0
        ws.width = 0
        ws.width1 = 0
        ws.brackt = false
        ws.stage = 0

        return initialize!(ws)
    end
end

requires_derivative(::Type{MoreThuenteLineSearch}) = true

const xtrapl = Float(1.1)
const xtrapu = Float(4.0)

# The arguments `stp`, `f`, `g` contain the values of the step,
# function, and directional derivative at `stp`.
function start!(ws::MoreThuenteLineSearch, stp::Float, f::Float, g::Float,
                stpmin::Float, stpmax::Float)

    @assert 0 ≤ stpmin ≤ stpmax
    @assert stpmin ≤ stp ≤ stpmax
    @assert g < 0 "not a descent direction"

    ws.stpmin = stpmin
    ws.stpmax = stpmax
    ws.brackt = false
    ws.stage = 1
    ws.finit = f
    ws.ginit = g
    ws.gtest = ws.ftol*ws.ginit
    ws.width = ws.stpmax - ws.stpmin
    ws.width1 = 2*ws.width

    ws.stx = 0
    ws.fx = ws.finit
    ws.dx = ws.ginit
    ws.sty = 0
    ws.fy = ws.finit
    ws.dy = ws.ginit
    ws.smin = 0
    ws.smax = stp + xtrapu*stp

    return starting!(ws, stp)

end

function iterate!(ws::MoreThuenteLineSearch, stp::Float, f::Float, g::Float)
    # If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the algorithm
    # enters the second stage.
    ftest::Float = ws.finit + stp*ws.gtest
    if ws.stage == 1 && f <= ftest && g >= 0
        ws.stage = 2
    end

    # Test for termination (convergence or warnings).
    if f <= ftest && abs(g) <= -ws.gtol*ws.ginit
        return convergence!(ws, "strong Wolfe conditions hold")
    elseif stp == ws.stpmin && (f > ftest || g >= ws.gtest)
        return warning!(ws, "stp = stpmin")
    elseif stp == ws.stpmax && f <= ftest && g <= ws.gtest
        return warning!(ws, "stp = stpmax")
    elseif ws.brackt && ws.smax - ws.smin <= ws.xtol*ws.smax
        return warning!(ws, "xtol test satisfied")
    elseif ws.brackt && (stp <= ws.smin || stp >= ws.smax)
        return warning!(ws, "rounding errors prevent progress")
    end

    # A modified function is used to predict the step during the first stage
    # if a lower function value has been obtained but the decrease is not
    # sufficient.


    if ws.stage == 1 && f <= ws.fx && f > ftest

        # Define the modified function and derivative values.
        ws.fx -= ws.stx*ws.gtest
        ws.fy -= ws.sty*ws.gtest
        ws.dx -= ws.gtest
        ws.dy -= ws.gtest

        # Call `cstep!` to update `stx`, `sty`, and to compute the new step.
        stp = cstep!(ws, stp, f - stp*ws.gtest, g - ws.gtest)

        # Reset the function and derivative values for f.
        ws.fx += ws.stx*ws.gtest
        ws.fy += ws.sty*ws.gtest
        ws.dx += ws.gtest
        ws.dy += ws.gtest

    else

        # Call `cstep!` to update `stx`, `sty`, and to compute the new step.
        stp = cstep!(ws, stp, f, g)

    end

    # Decide if a bisection step is needed.
    if ws.brackt
        if (abs(ws.sty - ws.stx) >= 0.66*ws.width1)
            stp = ws.stx + 0.5*(ws.sty - ws.stx)
        end
        ws.width1 = ws.width
        ws.width = abs(ws.sty - ws.stx)
    end

    # Set the minimum and maximum steps allowed for `stp`.
    if ws.brackt
        ws.smin = min(ws.stx, ws.sty)
        ws.smax = max(ws.stx, ws.sty)
    else
        ws.smin = stp + xtrapl*(stp - ws.stx)
        ws.smax = stp + xtrapu*(stp - ws.stx)
    end

    # Force the step to be within the bounds `stpmax` and `stpmin`.
    stp = max(stp, ws.stpmin)
    stp = min(stp, ws.stpmax)

    # If further progress is not possible, let `stp` be the best point
    # obtained during the search.

    if (ws.brackt && (stp <= ws.smin || stp >= ws.smax) ||
        (ws.brackt && ws.smax - ws.smin <= ws.xtol*ws.smax))
        stp = ws.stx
    end

    # Obtain another function and derivative.
    return searching!(ws, stp)

end

"""
## Compute a safeguarded cubic step

The call:

     nextstep = cstep!(ws, stp, fp, dp)

computes a safeguarded step for a search procedure and updates an interval that
contains a step that satisfies a sufficient decrease and a curvature condition.

The argument `stp` is the current step, the argument `fp` and `dp` respectively
give the function value and derivative at `stp`.  The returned value `nextstep`
is the new trial step.

The parameter `ws.stx` contains the step with the least function value.  If
`ws.brackt` is set to true then a minimizer has been bracketed in an interval
with endpoints `ws.stx` and `ws.sty`.  The subroutine assumes that if
`ws.brackt` is set to true then:

    min(ws.stx, ws.sty) < stp < max(ws.stx, ws.sty),

and that the derivative at `stx` is negative in the direction of the step.

Workspace `ws` is used as follows:

* `ws.stx` is the best step obtained so far and is an endpoint of the interval
  that contains the minimizer.  On exit, `ws.stx` is the updated best step.

* `ws.fx` is the function value at `ws.stx`.  On exit, `ws.fx` is the updated
  function value at `ws.stx`.

* `ws.dx` is derivative of the function at `ws.stx`.  The derivative must be
  negative in the direction of the step, that is, `ws.dx` and `stp - ws.stx`
  must have opposite signs.  On exit, `ws.dx` is the updated derivative of the
  function at `ws.stx`.

* `ws.sty` is the second endpoint of the interval that contains the minimizer.
  On exit, `ws.sty` is the updated endpoint of the interval that contains the
  minimizer.

* `ws.fy` is the function value at `ws.sty`.  On exit, `ws.fy` is the updated
  function value at `ws.sty`.

* `ws.dy` is derivative of the function at `ws.sty`.  On exit, `ws.dy` is the
  updated derivative of the function at `ws.sty`.

* `ws.brackt` is a boolean variable which specifies if a minimizer has been
  bracketed.  Initially `ws.brackt` must be set to `false`.  On exit,
  `ws.brackt` specifies if a minimizer has been bracketed.

* `ws.smin` is a lower bound for the step.  Its value is left unchanged.

* `ws.smax` is an upper bound for the step.  Its value is left unchanged.


### History

* MINPACK-1 Project.  June 1983.  Argonne National Laboratory.
  Jorge J. Moré and David J. Thuente.

* MINPACK-2 Project.  November 1993.  Argonne National Laboratory and
  University of Minnesota.  Brett M. Averick and Jorge J. Moré.

* TiPi.jl Project.  April 2016.  Centre de Recherche Astrophysique de Lyon.
  Conversion to Julia by Éric Thiébaut.

"""
function cstep!(ws::MoreThuenteLineSearch, stp::Float, fp::Float, dp::Float)

    const ZERO::Float = 0
    const TWO::Float = 2
    const THREE::Float = 3
    const P66::Float = 0.66

    stx::Float = ws.stx
    fx::Float  = ws.fx
    dx::Float  = ws.dx
    sty::Float = ws.sty
    fy::Float  = ws.fy
    dy::Float  = ws.dy
    stpmin::Float = ws.smin
    stpmax::Float = ws.smax

    opposite = (dx < ZERO < dp) || (dp < ZERO < dx)

    if fp > fx

        # First case: A higher function value.  The minimum is bracketed.  If
        # the cubic step is closer to `stx` than the quadratic step, the cubic
        # step is taken, otherwise the average of the cubic and quadratic steps
        # is taken.

        theta = THREE*(fx-fp)/(stp-stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s*sqrt((theta/s)^2-(dx/s)*(dp/s))
        if stp < stx; gamma = -gamma; end
        p = (gamma-dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p/q
        stpc = stx + r*(stp - stx)
        stpq = stx + ((dx/((fx - fp)/(stp - stx) + dx))/TWO)*(stp - stx)
        if abs(stpc - stx) < abs(stpq - stx)
            stpf = stpc
        else
            stpf = stpc + (stpq - stpc)/TWO
        end
        ws.brackt = true

    elseif opposite

        # Second case: A lower function value and derivatives of opposite sign.
        # The minimum is bracketed.  If the cubic step is farther from `stp`
        # than the secant step, the cubic step is taken, otherwise the secant
        # step is taken.

        theta = THREE*(fx - fp)/(stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s*sqrt((theta/s)^2 - (dx/s)*(dp/s))
        if stp > stx; gamma = -gamma; end
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
        ws.brackt = true

      elseif abs(dp) < abs(dx)

        # Third case: A lower function value, derivatives of the same sign, and
        # the magnitude of the derivative decreases.
        #
        # The cubic step is computed only if the cubic tends to infinity in the
        # direction of the step or if the minimum of the cubic is beyond
        # stp. Otherwise the cubic step is defined to be the secant step.

        theta = THREE*(fx - fp)/(stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))

        # The case `gamma = 0` only arises if the cubic does not tend to
        # infinity in the direction of the step.

        gamma = s*sqrt(max(ZERO, (theta/s)^2 - (dx/s)*(dp/s)))
        if stp > stx; gamma = -gamma; end
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p/q
        if r < ZERO && gamma != ZERO
            stpc = stp + r*(stx - stp)
        elseif stp > stx
            stpc = ws.stpmax
        else
            stpc = ws.stpmin
        end
        stpq = stp + (dp/(dp - dx))*(stx - stp)

        if ws.brackt

            # A minimizer has been bracketed.  If the cubic step is closer to
            # `stp` than the secant step, the cubic step is taken, otherwise
            # the secant step is taken.

            if abs(stpc - stp) < abs(stpq - stp)
               stpf = stpc
            else
               stpf = stpq
            end
            if stp > stx
               stpf = min(stp + P66*(sty - stp), stpf)
            else
               stpf = max(stp + P66*(sty - stp), stpf)
            end

         else

            # A minimizer has not been bracketed.  If the cubic step is farther
            # from `stp` than the secant step, the cubic step is taken,
            # otherwise the secant step is taken.

            if abs(stpc - stp) > abs(stpq - stp)
               stpf = stpc
            else
               stpf = stpq
            end
            stpf = min(ws.stpmax, stpf)
            stpf = max(ws.stpmin, stpf)

         end

    else

        # Fourth case: A lower function value, derivatives of the same sign,
        # and the magnitude of the derivative does not decrease.  If the
        # minimum is not bracketed, the step is either `stpmin` or `stpmax`,
        # otherwise the cubic step is taken.

        if ws.brackt
            theta = THREE*(fp - fy)/(sty - stp) + dy + dp
            s = max(abs(theta), abs(dy), abs(dp))
            gamma = s*sqrt((theta/s)^2-(dy/s)*(dp/s))
            if stp > sty; gamma = -gamma; end
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p/q
            stpc = stp + r*(sty - stp)
            stpf = stpc
        elseif stp > stx
            stpf = ws.stpmax
        else
            stpf = ws.stpmin
        end
    end

    # Update the interval which contains a minimizer.
    if fp > fx
        ws.sty = stp
        ws.fy = fp
        ws.dy = dp
    else
        if opposite
            ws.sty = stx
            ws.fy = fx
            ws.dy = dx
        end
        ws.stx = stp
        ws.fx = fp
        ws.dx = dp
    end

    # Return the new step.
    return stpf

end

end # module
