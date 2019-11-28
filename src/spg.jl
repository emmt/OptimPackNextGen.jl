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
# -----------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2015-2019, Éric Thiébaut.
# <https://github.com/emmt/OptimPackNextGen.jl>.
#

module SPG

export
    spg,
    spg!

using Compat
using Compat.Printf

using LazyAlgebra
using ...OptimPackNextGen

import OptimPackNextGen: getreason

const SEARCHING            =  0
const INFNORM_CONVERGENCE  =  1
const TWONORM_CONVERGENCE  =  2
const TOO_MANY_ITERATIONS  = -1
const TOO_MANY_EVALUATIONS = -2

mutable struct Info
    f::Float64      # The final/current function value.
    fbest::Float64  # The best function value so far.
    pginfn::Float64 # ||projected grad||_inf at the final/current iteration.
    pgtwon::Float64 # ||projected grad||_2 at the final/current iteration.
    iter::Int       # The number of iterations.
    fcnt::Int       # The number of function (and gradient) evaluations.
    pcnt::Int       # The number of projections.
    status::Int     # Termination parameter.
end

Info() = Info(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)

"""
# Spectral Projected Gradient Method

The `spg` method implements the Spectral Projected Gradient Method (Version 2:
"continuous projected gradient direction") to find the local minimizers of a
given function with convex constraints, described in the references below.  A
typical use is:

```julia
spg(fg!, prj!, x0, m) -> x
```

The user must supply the functions `fg!` and `prj!` to evaluate the objective
function and its gradient and to project an arbitrary point onto the feasible
region.  These functions must be defined as:

```julia
function fg!(x::T, g::T) where {T}
   g[:] = gradient_at(x)
   return function_value_at(x)
end

function prj!(dst::T, src::T) where {T}
    dst[:] = projection_of(src)
    return dst
end
```

Argument `x0` is the initial solution and argument `m` is the number of
previous function values to be considered in the nonmonotone line search.  If
`m ≤ 1`, then a monotone line search with Armijo-like stopping criterion will
be used.

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

* `ws` is an instance of `SPG.Info` to store information about the final
  iterate.

* `verb` indicates whether to print some information at each iteration.

* `printer` specifies a subroutine to print some information at each iteration.
  This subroutine will be called as `printer(io, ws)` with `io` the output
  stream and `ws` an instance of `SPG.Info` with information about the current
  iterate.

* `io` specifes the output stream for iteration information.  It is `stdout` by
  default.

The `SPG.Info` type has the following members:

* `f` is the function value.
* `fbest` is the best function value so far.
* `pginfn` is the infinite norm of the projected gradient.
* `pgtwon` is the Eucliddean norm of the projected gradient.
* `iter` is the number of iterations.
* `fcnt` is the number of function (and gradient) evaluations.
* `pcnt` is the number of projections.
* `status` indicates the type of termination:

    Status                        Reason
    -------------------------------------------------------------------------
    SPG.SEARCHING (0)             Work in progress
    SPG.INFNORM_CONVERGENCE (1)   Convergence with projected gradient
                                  infinite-norm
    SPG.TWONORM_CONVERGENCE (2)   Convergence with projected gradient 2-norm
    SPG.TOO_MANY_ITERATIONS (-1)  Too many iterations
    SPG.TOO_MANY_EVALUATIONS (-2) Too many function evaluations


## References

* E. G. Birgin, J. M. Martinez, and M. Raydan, "Nonmonotone spectral projected
  gradient methods on convex sets", SIAM Journal on Optimization 10,
  pp. 1196-1211 (2000).

* E. G. Birgin, J. M. Martinez, and M. Raydan, "SPG: software for
  convex-constrained optimization", ACM Transactions on Mathematical Software
  (TOMS) 27, pp. 340-349 (2001).
"""
spg(fg!, prj!, x0, m::Integer; kwds...) =
    spg!(fg!, prj!, vcopy(x0), m; kwds...)

REASON = Dict{Int,String}(
    SEARCHING => "Work in progress",
    INFNORM_CONVERGENCE => "Convergence with projected gradient infinite-norm",
    TWONORM_CONVERGENCE => "Convergence with projected gradient 2-norm",
    TOO_MANY_ITERATIONS => "Too many iterations",
    TOO_MANY_EVALUATIONS => "Too many function evaluations")

getreason(ws::Info) = get(REASON, ws.status, "unknown status")

function spg!(fg!, prj!, x, m::Integer;
              ws::Info = Info(),
              maxit::Integer = typemax(Int),
              maxfc::Integer = typemax(Int),
              eps1::Real = 1e-6,
              eps2::Real = 1e-6,
              eta::Real = 1.0,
              printer::Function = default_printer,
              verb::Bool = false,
              io::IO = stdout)
    _spg!(fg!, prj!, x, Int(m), ws, Int(maxit), Int(maxfc),
          Float64(eps1), Float64(eps2), Float64(eta),
          printer, verb, io)
end

function _spg!(fg!, prj!, x::T, m::Int, ws::Info,
               maxit::Int, maxfc::Int,
               eps1::Float64, eps2::Float64, eta::Float64,
               printer::Function, verb::Bool, io::IO) where {T}
    # Initialization.
    @assert m ≥ 1
    @assert eps1 ≥ 0
    @assert eps2 ≥ 0
    @assert eta > 0
    lmin = Float64(1e-30)
    lmax = Float64(1e+30)
    ftol = Float64(1e-4)
    amin = Float64(0.1)
    amax = Float64(0.9)
    iter = 0
    fcnt = 0
    pcnt = 0
    status = SEARCHING
    pgtwon = Float64(Inf)
    pginfn = Float64(Inf)

    # Allocate workspaces making a few aliases to save memory.
    #
    # 1. `x0` and `g0` are updated right after computing `s = x - x0` and
    #    `y = g - g0` the variable and gradient changes, we can use `x0`
    #    and `g0` as scratch workspaces to temporarily store `s` and `y`.
    #
    # 2. The projected gradient `pg` and the search direction can share the
    #    same workspace.
    #
    lastfv = fill!(Array{Float64}(undef, m), -Inf)
    g = vcreate(x)
    d = pg = vcreate(x)
    s = x0 = vcreate(x)
    y = g0 = vcreate(x)
    xbest = vcreate(x)

    # Project initial guess.
    prj!(x, x)
    vcopy!(x0, x) # FIXME: not necessary (idem for g0)
    pcnt += 1

    # Evaluate function and gradient.
    f = Float64(fg!(x, g))
    fcnt += 1

    # Initialize best solution and best function value.
    fbest = f
    vcopy!(xbest, x)

    # Main loop.
    while true

        # Compute continuous projected gradient (and its norms)
        # as: `pg = (x - prj(x - eta*g))/eta`.
        vcombine!(pg, 1/eta, x, -1/eta, prj!(pg, vcombine!(pg, 1, x, -eta, g)))
        pcnt += 1
        pgtwon = vnorm2(pg)
        pginfn = vnorminf(pg)

        # Print iteration information.
        if verb
            ws.f = f
            ws.fbest = fbest
            ws.pgtwon = pgtwon
            ws.pginfn = pginfn
            ws.iter = iter
            ws.fcnt = fcnt
            ws.pcnt = pcnt
            ws.status = status
            printer(io, ws)
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
            # Initial steplength. (FIXME: check type stability)
            lambda = min(lmax, max(lmin, 1/pginfn))
        else
            vcombine!(s, 1, x, -1, x0)
            vcombine!(y, 1, g, -1, g0)
            sty = vdot(s, y)
            if sty > 0
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
            f = Float64(fg!(x, g))
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
            r = (f - f0 - stp*delta)*2
            if r > 0 && amin*r ≤ q ≤ amax*stp*r
                stp = q/r
            else
                stp /= 2
            end

            # Compute trial point.
            vcombine!(x, 1, x0, stp, d) # x = x0 + stp*d
        end

        if status != SEARCHING
            # The number of function evaluations was exceeded inside the line
            # search.
            break
        end

        # Proceed with next iteration.
        iter += 1

    end

    # Store information and report final status.
    ws.f = fbest
    ws.fbest = fbest
    ws.pgtwon = pgtwon
    ws.pginfn = pginfn
    ws.iter = iter
    ws.fcnt = fcnt
    ws.pcnt = pcnt
    ws.status = status
    if verb
        reason = getreason(ws)
        if status < 0
            printstyled(stderr, "# WARNING: ", reason, "\n"; color=:red)
        else
            printstyled(io, "# SUCCESS: ", reason, "\n"; color=:green)
        end
    end

    # Make sure to return the best solution so far.
    fbest < f && vcopy!(x, xbest)
    return x
end

function default_printer(io::IO, nfo::Info)
    if nfo.iter == 0
        @printf(io, "# %s\n# %s\n",
                " ITER   EVAL   PROJ             F(x)              ‖PG(X)‖_2 ‖PG(X)‖_∞",
                "---------------------------------------------------------------------")
    end
    @printf(io, " %6d %6d %6d %3s %24.17e %9.2e %9.2e\n",
            nfo.iter, nfo.fcnt, nfo.pcnt, (nfo.f ≤ nfo.fbest ? "(*)" : "   "),
            nfo.f, nfo.pgtwon, nfo.pginfn)
end

end # module
