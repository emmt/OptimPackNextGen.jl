#
# spg.jl --
#
# Implements Spectral Projected Gradient Method (Version 2: "continuous
# projected gradient direction") to find the local minimizers of a given
# function with convex constraints, described in:
#
# [1] E. G. Birgin, J. M. Martinez, and M. Raydan, "Nonmonotone spectral
#     projected gradient methods on convex sets", SIAM Journal on Optimization
#     10, pp. 1196-1211 (2000).
#
# [2] E. G. Birgin, J. M. Martinez, and M. Raydan, "SPG: software for
#     convex-constrained optimization", ACM Transactions on Mathematical
#     Software (TOMS) 27, pp. 340-349 (2001).
#
# ----------------------------------------------------------------------------
#
# This file is part of TiPi.
# Copyright (C) 2014, Éric Thiébaut <eric.thiebaut@univ-lyon1.fr>.
# All rights reserved.
#

module SPG

using TiPi
import TiPi.Float
using TiPi.Algebra

export spg2, spg2!, spg2_reason

const WORK_IN_PROGRESS     =  0
const INFNORM_CONVERGENCE  =  1
const TWONORM_CONVERGENCE  =  2
const TOO_MANY_ITERATIONS  = -1
const TOO_MANY_EVALUATIONS = -2

type SPG2Info
    f::Float      # The final/current function value.
    fbest::Float  # The best function value so far.
    pginfn::Float # ||projected grad||_inf at the final/current iteration.
    pgtwon::Float # ||projected grad||_2 at the final/current iteration.
    iter::Int     # The number of iterations.
    fcnt::Int     # The number of function (and gradient) evaluations.
    pcnt::Int     # The number of projections.
    status::Int   # Termination parameter.
    SPG2Info() = new(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)
end

doc"""
# Spectral Projected Gradient Method

    x = spg2(fg!, prj!, x0, m)

SPG2 implements the Spectral Projected Gradient Method (Version 2: "continuous
projected gradient direction") to find the local minimizers of a given function
with convex constraints, described in the references below.

The user must supply the functions `fg!` and `prj!` to evaluate the objective
function and its gradient and to project an arbitrary point onto the feasible
region.  These functions must be defined as:

    function fg!{T}(x::T, g::T)
        g[:] = gradient_at(x)
        return function_value_at(x)
     end

     function prj!{T}(dst::T, src::T)
         dst[:] = projection_of(src)
         return dst
     end

Argument `x0` is the initial solution and argument `m` is the number of previous
function values to be considered in the nonmonotone line search.  If `m ≤ 1`,
then a monotone line search with Armijo-like stopping criterion will be used.

The following keywords are available:

* `eps1` specifies the stopping criterion `‖pg‖_∞ ≤ eps1` with `pg` the
  projected gradient.  By default, `eps1 = 1e-6`.

* `eps2` specifies the stopping criterion `‖pg‖_2 ≤ eps2` with `pg` the
  projected gradient.  By default, `eps2 = 1e-6`.

* `eta` specifies a scaling parameter for the gradient.  The projected gradient
  is computed as `(x - prj(x - eta*g))/eta` (with `g` the gradient at `x`)
  instead of `x - prj(x - g)` which corresponds to the default behavior (same
  as if `eta=1`) and is usually used in methodological publications although it
  does not scale correctly (for instance, if you make a change of variables or
  simply multiply the function by some factor).

* `maxit` specifies the maximum number of iterations.

* `maxfc` specifies the maximum number of function evaluations.

* `ws` is an instance of `SPG2Info` to store information about the final
  iterate.

* `verb` indicates whether to print some information at each iteration.

* `printer` specifies a subroutine to print some information at each iteration.
  This subroutine will be called as `printer(io, ws)` with `io` the output
  stream and `ws` an instance of `SPG2Info` with information about the current
  iterate.

* `io` specifes the output stream for iteration information.  It is `STDOUT` by
  default.

The `SPG2Info` type has the following members:

* `f` is the function value.
* `fbest` is the best function value so far.
* `pginfn` is the infinite norm of the projected gradient.
* `pgtwon` is the Eucliddean norm of the projected gradient.
* `iter` is the number of iterations.
* `fcnt` is the number of function (and gradient) evaluations.
* `pcnt` is the number of projections.
* `status` indicates the type of termination:

    Status                        Reason
    ----------------------------------------------------------------------------
    SPG.WORK_IN_PROGRESS (0)      work in progress
    SPG.INFNORM_CONVERGENCE (1)   convergence with projected gradient infinite-norm
    SPG.TWONORM_CONVERGENCE (2)   convergence with projected gradient 2-norm
    SPG.TOO_MANY_ITERATIONS (-1)  too many iterations
    SPG.TOO_MANY_EVALUATIONS (-2) too many function evaluations


## References

* E. G. Birgin, J. M. Martinez, and M. Raydan, "Nonmonotone spectral projected
  gradient methods on convex sets", SIAM Journal on Optimization 10,
  pp. 1196-1211 (2000).

* E. G. Birgin, J. M. Martinez, and M. Raydan, "SPG: software for
  convex-constrained optimization", ACM Transactions on Mathematical Software
  (TOMS) 27, pp. 340-349 (2001).
"""
function spg2{T}(fg!, prj!, x0::T, m::Integer; kws...)
    spg2!(fg!, prj!, vcopy(x0), m; kws...)
end

REASON = Dict{Int,ASCIIString}(WORK_IN_PROGRESS => "work in progress",
                               INFNORM_CONVERGENCE => "convergence with projected gradient infinite-norm",
                               TWONORM_CONVERGENCE => "convergence with projected gradient 2-norm",
                               TOO_MANY_ITERATIONS => "too many iterations",
                               TOO_MANY_EVALUATIONS => "too many function evaluations")

spg2_reason(ws::SPG2Info) = spg2_reason(ws.status)

spg2_reason(status::Int) = get(REASON, Int(status), "unknown status")

function spg2!{T}(fg!, prj!, x::T, m::Integer;
                  ws::SPG2Info=SPG2Info(),
                  maxit::Integer=typemax(Int),
                  maxfc::Integer=typemax(Int),
                  eps1::Real=1e-6,
                  eps2::Real=1e-6,
                  eta::Real=1.0,
                  printer::Function=default_printer,
                  verb::Bool=false,
                  io::IO=STDOUT)

    # Initialization.
    m::Int = m
    maxit::Int = maxit
    maxfc::Int = maxfc
    eps1::Float = eps1
    eps2::Float = eps2
    eta::Float = eta
    @assert eps1 ≥ 0.0
    @assert eps2 ≥ 0.0
    @assert eta > 0.0
    lmin::Float = 1e-30
    lmax::Float = 1e+30
    ftol::Float = 1e-4
    amin::Float = 0.1
    amax::Float = 0.9
    iter::Int = 0
    fcnt::Int = 0
    pcnt::Int = 0
    status::Int = WORK_IN_PROGRESS
    if m > 1
        lastfv = Array(Float, max(m,0))
        fill!(lastfv, -Inf)
    end
    x0::T = vcopy(x)
    g::T = vcreate(x)
    d::T = vcreate(x)
    s::T = vcreate(x)
    y::T = vcreate(x)
    g0::T = vcreate(x)
    pg::T = vcreate(x)
    xbest::T = vcreate(x)
    f::Float = 0.0
    f0::Float = 0.0
    fbest::Float = 0.0
    fmax::Float = 0.0
    pgtwon::Float = 0.0
    pginfn::Float = 0.0
    sty::Float = 0.0
    sts::Float = 0.0
    lambda::Float = 0.0
    delta::Float = 0.0
    stp::Float = 0.0
    q::Float = 0.0
    r::Float = 0.0

    # Project initial guess.
    prj!(x, x)
    pcnt += 1

    # Evaluate function and gradient.
    f = fg!(x, g)
    fcnt += 1

    # Initialize best solution and best function value.
    fbest = f
    vcopy!(xbest, x)

    # Main loop.
    while true

        # Compute continuous projected gradient (and its norms).
        # as: `pg = (x - prj(x - eta*g))/eta`
        vcombine!(pg, 1/eta, x, -1/eta, prj!(pg, vcombine!(pg, 1, x, -eta, g)))
        pcnt += 1
        pgtwon = vnorm2(pg)
        pginfn = vnorminf(pg)

        # Print iteration information
        if verb
            ws.f = f
            ws.fbest = fbest
            ws.pgtwon = pgtwon
            ws.pginfn = pginfn
            ws.iter = iter
            ws.fcnt = fcnt
            ws.pcnt = pcnt
            ws.status = status
            printer(ws)
        end

        # Test stopping criteria.
        if pginfn ≤ eps1
            # Gradient infinite-norm stopping criterion satisfied, stop.
            status = INFNORM_CONVERGENCE
            break
        end
        if pgtwon ≤ eps2
            # Gradient 2-norm stopping criterion satisfied, stop.
            status = TWONORM_CONVERGENCE
            break
        end
        if iter ≥ maxit
            # Maximum number of iterations exceeded, stop.
            status = TOO_MANY_ITERATIONS
            break
        end
        if fcnt ≥ maxfc
            # Maximum number of function evaluations exceeded, stop.
            status = TOO_MANY_EVALUATIONS
            break
        end

        # Store function value for the nonmonotone line search and find maximum
        # function value since m last calls.
        if m > 1
            lastfv[(iter%m) + 1] = f
            fmax = maximum(lastfv)
        else
            fmax = f
        end

        # Compute spectral steplength.
        if iter == 0
            # Initial steplength.
            lambda = min(lmax, max(lmin, 1.0/pginfn))
        else
            vcombine!(s, 1, x, -1, x0) # s = x - x0
            vcombine!(y, 1, g, -1, g0) # y = g - g0
            sty = vdot(s, y)
            if sty > 0.0
                # Safeguarded Barzilai & Borwein spectral steplength.
                sts = vdot(s, s)
                lambda = min(lmax, max(lmin, sts/sty))
            else
                lambda = lmax
            end
        end

        # Save current point.
        vcopy!(x0, x)
        vcopy!(g0, g)
        f0 = f

        # Compute the spectral projected gradient direction and delta = ⟨g,d⟩
        prj!(x, vcombine!(x, 1, x0, -lambda, g0)) # x = prj(x0 - lambda*g0)
        pcnt += 1
        vcombine!(d, 1, x, -1, x0) # d = x - x0
        delta = vdot(g0, d)

        # Nonmonotone line search.
        stp = 1.0 # Step length for first trial.
        while true
            # Evaluate function and gradient at trial point.
            f = fg!(x, g)
            fcnt += 1

            # Compare the new function value against the best function value
            # and, if smaller, update the best function value and the
            # corresponding best point.
            if f < fbest
                fbest = f
                vcopy!(xbest, x)
            end

            # Test stopping criteria.
            if f ≤ fmax + stp*ftol*delta
                # Nonmonotone Armijo-like stopping criterion satisfied, stop.
                break
            end
            if fcnt ≥ maxfc
                # Maximum number of function evaluations exceeded, stop.
                status = TOO_MANY_EVALUATIONS
                break
            end

            # Safeguarded quadratic interpolation.
            q = -delta*(stp*stp)
            r = (f - f0 - stp*delta)*2.0
            if r > 0.0 && amin*r ≤ q ≤ amax*stp*r
                stp = q/r
            else
                stp *= 0.5
            end

            # Compute trial point.
            vcombine!(x, 1, x0, stp, d) # x = x0 + stp*d
        end

        if status != WORK_IN_PROGRESS
            # The number of function evaluations was exceeded inside the line
            # search.
            break
        end

        # Proceed with next iteration.
        iter += 1

    end
    ws.f = f
    ws.fbest = fbest
    ws.pgtwon = pgtwon
    ws.pginfn = pginfn
    ws.iter = iter
    ws.fcnt = fcnt
    ws.pcnt = pcnt
    ws.status = status
    if verb
        io::IO = STDERR
        reason = spg2_reason(status)
        if status < 0
            print_with_color(:red, io, "# WARNING: ", reason)
        else
            print_with_color(:green, io, "# SUCCESS: ", reason)
        end
        println(io)
    end
    return xbest
end

function default_printer(ws::SPG2Info)
    if ws.iter == 0
        println("#  ITER   EVAL   PROJ             F(x)              ‖PG(X)‖_2 ‖PG(X)‖_∞")
        println("# ------ ------ ------ ---------------------------- --------- ---------")
    end
    @printf(" %6d %6d %6d %3s %24.17e %9.2e %9.2e\n",
            ws.iter, ws.fcnt, ws.pcnt, (ws.f ≤ ws.fbest ? "(*)" : "   "),
            ws.f, ws.pgtwon, ws.pginfn)
end

end # module
