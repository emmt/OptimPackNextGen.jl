"""

Module `OptimPackNextGen.SPG` implements the Spectral Projected Gradient Method
(Version 2: "continuous projected gradient direction") to find the local
minimizers of a given function with convex constraints, described in:

- E. G. Birgin, J. M. Martinez, and M. Raydan, "Nonmonotone spectral projected
  gradient methods on convex sets", SIAM Journal on Optimization 10,
  pp. 1196-1211 (2000).

- E. G. Birgin, J. M. Martinez, and M. Raydan, "SPG: software for
  convex-constrained optimization", ACM Transactions on Mathematical Software
  (TOMS) 27, pp. 340-349 (2001).

"""
module SPG

export
    issuccess,
    spg,
    spg!,
    spg_CUTEst

# Imports from other packages.
using LinearAlgebra
using NumOptBase
using NumOptBase: copy!
using Printf
using TypeUtils

# Imports from parent module.
using  ..OptimPackNextGen
using  ..OptimPackNextGen: auto_differentiate!, copy_variables
import ..OptimPackNextGen: get_reason

"""
    SPG.Status

is the enumeration for the algorithm status in SPG method.

"""
@enum Status begin
    TOO_MANY_EVALUATIONS = -2
    TOO_MANY_ITERATIONS  = -1
    SEARCHING            =  0
    INFNORM_CONVERGENCE  =  1
    TWONORM_CONVERGENCE  =  2
    FUNCTION_CONVERGENCE =  3
end

"""
    SPG.Stats(; fx::Real, pgtwon::Real, pginfn::Real, seconds::Real,
                iter::Integer, fcnt::Integer, pcnt::Integer, status::Status)

yields an immutable object collecting information returned by the SPG method.
All members are mandatory and are specified by keyword:

- `fx` is the objective function value.
- `pgtwon` is the Euclidean norm of projected gradient.
- `pginfn` is the infinite norm of projected gradient.
- `seconds` is the execution time in seconds.
- `iter` is the number of iterations.
- `fcnt` is the number of function (and gradient) evaluations.
- `pcnt` is the number of projections.
- `status` is algorithm status.

"""
struct Stats
    fx::Float64      # Objective function value.
    pgtwon::Float64  # Euclidean norm of projected grad.
    pginfn::Float64  # Infinite norm of projected grad.
    seconds::Float64 # Execution time in seconds.
    iter::Int        # Number of iterations.
    fcnt::Int        # Number of function (and gradient) evaluations.
    pcnt::Int        # Number of projections.
    status::Status   # Algorithm status.
    function Stats(; fx::Real, pgtwon::Real, pginfn::Real, seconds::Real,
                   iter::Integer, fcnt::Integer, pcnt::Integer, status::Status)
        return new(fx, pgtwon, pginfn, seconds, iter, fcnt, pcnt, status)
    end
end

LinearAlgebra.issuccess(stats::Stats) = issuccess(stats.status)
LinearAlgebra.issuccess(status::Status) = Integer(status) > 0

get_reason(stats::Stats) = get_reason(stats.status)
get_reason(status::Status) =
    status == TOO_MANY_EVALUATIONS ? "Too many function evaluations" :
    status == TOO_MANY_ITERATIONS  ? "Too many iterations" :
    status == SEARCHING            ? "Search in progress" :
    status == INFNORM_CONVERGENCE  ? "Convergence with projected gradient infinite-norm" :
    status == TWONORM_CONVERGENCE  ? "Convergence with projected gradient 2-norm" :
    status == FUNCTION_CONVERGENCE ? "Function does not change in the last `mem` iterations" :
    "Unknown status"

# Default settings.
const default_mem  = 10
const default_eps1 = 1.0e-6
const default_eps2 = 1.0e-6
const default_eta  = 1.0
const default_lmin = 1.0e-30
const default_lmax = 1.0e+30
const default_ftol = 1.0e-4
const default_amin = 0.1
const default_amax = 0.9

"""
    spg(fg!, prj!, x0; kwds...) -> x, stats

attempts to solve the constrained problem:

    min f(x)   subject to   x ∈ Ω ⊆ ℝⁿ

by the Spectral Projected Gradient (SPG) method (Version 2: "continuous
projected gradient direction" described in the references below). Arguments
`fg!` and `prj!` implement the objective function `f(x)` and its gradient
`∇f(x)`. Argument `x0 ∈ ℝⁿ` is the initial solution. The result is a 2-tuple
`(x, stats)` with `x ∈ Ω` and `stats` a structure with information about the
algorithm computations (see [`OptimPAckNextGen.SPG.Stats`](@ref). Provided
`issuccess(stats)` is true, `x` is an approximate local minimizer of the
objective function on the feasible set `Ω`.

The user must supply the functions `fg!` and `prj!` to evaluate the objective
function and its gradient and to project an arbitrary point onto the feasible
region. These functions must be defined as:

    function fg!(x::T, g::T) where {T}
       g[:] = gradient_at(x)
       return function_value_at(x)
    end

    function prj!(dst::T, src::T) where {T}
        dst[:] = projection_of(src)
        return dst
    end

If the feasible set consists in simple separable bounds on the variables,
another possibility is to call:

    spg(fg!, Ω, x0) -> x, stats

with `Ω` a bounded set (of type `BoundedSet` defined in package `NumOptBase`)
to specify the feasible subset for the variables `x`.

The following keywords are available:

* `mem` is the number of previous function values to be considered in the
  nonmonotone line search. If `mem ≤ 1`, then a monotone line search with
  Armijo-like stopping criterion will be used. By default, `mem =
  $default_mem`.

* `autodiff` is a boolean specifying whether to rely on automatic
  differentiation by calling [`OptimPackNextGen.auto_differentiate!](@ref). If
  not specified, this keyword is assumed to be `false`.

* `eps1` specifies the stopping criterion `‖pg‖_∞ ≤ eps1` with `pg` the
  projected gradient. By default, `eps1 = $(default_eps1)`.

* `eps2` specifies the stopping criterion `‖pg‖₂ ≤ eps2` with `pg` the
  projected gradient. By default, `eps2 = $(default_eps2)`.

* `eta` specifies a scaling parameter for the gradient. The projected gradient
  is computed as `(x - prj(x - eta*g))/eta` (with `g` the gradient at `x`)
  instead of `x - prj(x - g)` which corresponds to the default behavior (same
  as if `eta=1`) and is usually used in methodological publications although it
  does not scale correctly (for instance, if you make a change of variables or
  simply multiply the function by some factor).

* `lmin` and `lmax` specify the limits of the spectral steplength. By default,
  `lmin = $(default_lmin)` and `lmax = $(default_lmax)`.

* `amin` and `amax` specify the parameters for safeguarding the quadratic
  interpolation step. By default, `amin = $(default_amin)` and `amax =
  $(default_amax)`.

* `ftol` specifies the relative function tolerance for the non-monotone
  Armijo-like stopping criterion. By default, `ftol = $(default_ftol)`.

* `maxit` specifies the maximum number of iterations.

* `maxfc` specifies the maximum number of function evaluations.

* `verb` specifies the verbosity level. It can be a boolean to specify whether
  to call the observer at every iteration or an integer to call the observer
  every `verb` iteration(s). The observer is never called if `verb` is less or
  equal zero. The default is `verb = false`.

* `observer` specifies a callable to inspect the solution and/or print some
  information at each iteration. This subroutine will be called as
  `observer(output, stats, x, best)` where `output` is the output stream,
  `stats` collects information about the current iteration (see below), `x` is
  the current iterate, and `best` is a boolean indicating whether `x` is the
  best solution found so far.

* `output` (`stdout` by default) specifes the output stream for iteration
  information.

The `stats` object has the following properties:

* `stats.fx` is the objective function value `f(x)`.

* `stats.pgtwon` is the Euclidean norm of the projected gradient of the last
  iterate.

* `stats.pginfn` is the infinite norm of the projected gradient of the last
  iterate.

* `stats.seconds` is the execution time in seconds.

* `stats.iter` is the number of iterations, `0` for the starting point.

* `stats.fcnt` is the number of function (and gradient) evaluations.

* `stats.pcnt` is the number of projections.

* `stats.status` indicates the final status of the algorithm (see below).

* `stats.info` provides details about the state of the algorithm. Method
  `get_reason` can be used to retrieve a descriptive message.

Possible `status` values are:

| Status                     | Reason                                                |
|:---------------------------|:------------------------------------------------------|
| `SPG.SEARCHING`            | Work in progress                                      |
| `SPG.INFNORM_CONVERGENCE`  | Convergence with projected gradient infinite-norm     |
| `SPG.TWONORM_CONVERGENCE`  | Convergence with projected gradient 2-norm            |
| `SPG.FUNCTION_CONVERGENCE` | Function does not change in the last `mem` iterations |
| `SPG.TOO_MANY_ITERATIONS`  | Too many iterations                                   |
| `SPG.TOO_MANY_EVALUATIONS` | Too many function evaluations                         |

Method `issuccess(stats)` yields whether the algorithm converged according to
one of the convergence criteria.

## References

* E. G. Birgin, J. M. Martinez, and M. Raydan, "Nonmonotone spectral projected
  gradient methods on convex sets", SIAM Journal on Optimization 10,
  pp. 1196-1211 (2000).

* E. G. Birgin, J. M. Martinez, and M. Raydan, "SPG: software for
  convex-constrained optimization", ACM Transactions on Mathematical Software
  (TOMS) 27, pp. 340-349 (2001).

"""
function spg(fg!, prj!, x0::AbstractArray; kwds...)
    x = copy_variables(x0)
    stats = spg!(fg!, prj!, x; kwds...)
    return x, stats
end

spg!(fg!, Ω::BoundedSet{T,N}, x::AbstractArray{T,N}; kwds...) where {T,N} =
    spg!(fg!, Projector(Ω), x; kwds...)

spg!(fg!, Ω::BoundedSet, x::AbstractArray{T,N}; kwds...) where {T,N} =
    spg!(fg!, BoundedSet{T,N}(Ω), x; kwds...)

function spg!(fg!, prj!, x::AbstractArray;
              mem::Integer    = default_mem,
              autodiff::Bool  = false,
              maxit::Integer  = typemax(Int),
              maxfc::Integer  = typemax(Int),
              eps1::Real      = default_eps1,
              eps2::Real      = default_eps2,
              eta::Real       = default_eta,
              lmin::Real      = default_lmin,
              lmax::Real      = default_lmax,
              ftol::Real      = default_ftol,
              amin::Real      = default_amin,
              amax::Real      = default_amax,
              observer        = default_observer,
              verb::Integer   = false,
              output::IO      = stdout)
    # Check settings.
    mem ≥ one(mem)    || argument_error("`mem ≥ 1` must hold")
    eps1 ≥ zero(eps1) || argument_error("`eps1 ≥ 0` must hold")
    eps2 ≥ zero(eps2) || argument_error("`eps2 ≥ 0` must hold")
    eta > zero(eta)   || argument_error("`eta ≥ 0` must hold")
    lmin > zero(lmin) || argument_error("`lmin > 0` must hold")
    lmax > zero(lmax) || argument_error("`lmax > 0` must hold")
    lmin < lmax       || argument_error("`lmin < lmax` must hold")
    zero(ftol) < ftol < one(ftol) || argument_error("`0 < ftol < 1` must hold")
    amin > zero(amin) || argument_error("`amin > 0` must hold")
    amax > zero(amax) || argument_error("`amax > 0` must hold")
    amin < amax       || argument_error("`amin < amax` must hold")

    # Determine floating-point type for scalar computations (using at least
    # double-precision) and call private method with all arguments checked and
    # converted to the correct type.
    T = promote_type(Float64, eltype(x))
    args = (prj!, x, Int(mem), Int(maxit), Int(maxfc), as(T, eps1), as(T, eps2),
            as(T, eta), as(T, lmin), as(T, lmax), as(T, ftol), as(T, amin), as(T, amax),
            observer, Int(verb), output)
    if autodiff
        return _spg!((x, g) -> auto_differentiate!(fg!, x, g), args...)
    else
        return _spg!(fg!, args...)
    end
end

function _spg!(fg!, prj!, x::AbstractArray, m::Int, maxit::Int, maxfc::Int,
               eps1::T, eps2::T, eta::T, lmin::T, lmax::T, ftol::T,
               amin::T, amax::T, observer, verb::Int, output::IO) where {T<:AbstractFloat}
    # Initialization.
    t0 = time()
    iter = 0
    fcnt = 0
    pcnt = 0
    status = SEARCHING
    pgtwon = as(T, Inf)
    pginfn = as(T, Inf)

    # Allocate workspaces making a few aliases to save memory.
    #
    # 1. `x0` and `g0` are updated right after computing `s = x - x0` and
    #    `y = g - g0` the variable and gradient changes, we can use `x0`
    #    and `g0` as scratch workspaces to temporarily store `s` and `y`.
    #
    # 2. The projected gradient `pg` and the search direction can share the
    #    same workspace.
    #
    lastfv = fill!(Array{T}(undef, m), -Inf)
    g = similar(x)
    d = pg = similar(x)
    s = x0 = similar(x)
    y = g0 = similar(x)
    xbest = similar(x)

    # Project initial guess.
    prj!(x, x)
    pcnt += 1

    # Evaluate function and gradient.
    f = as(T, fg!(x, g))
    fcnt += 1

    # Initialize best solution and best function value.
    fbest = f
    copy!(xbest, x)

    # Main loop.
    fconst = false
    while true

        # Store function value for the nonmonotone line search and find minimum
        # and maximum function values since m last calls.
        if m > 1
            lastfv[(iter%m) + 1] = f
            fmin, fmax = extrema(lastfv)
            fconst = !(fmin < fmax)
        else
            fmin = fmax = f
        end

        # Compute continuous projected gradient (and its norms) as:
        # `pg = (x - prj(x - eta*g))/eta` and using `pg` as a workspace.
        combine!(pg, 1/eta, x, -1/eta, prj!(pg, combine!(pg, 1, x, -eta, g)))
        pcnt += 1
        pgtwon = norm2(T, pg)
        pginfn = norminf(T, pg)

        # Print iteration information.
        if verb > 0 && (iter % verb) == 0
            stats = Stats(; fx = f, pgtwon, pginfn, seconds = time() - t0,
                          iter, fcnt, pcnt, status)
            observer(output, stats, x, f ≤ fbest)
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
        if fconst
            # Function does not change in the last `m` iterations.
            status = FUNCTION_CONVERGENCE
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

        # Compute spectral steplength.
        if iter == 0
            # Initial steplength.
            lambda = clamp(1/pginfn, lmin, lmax)
        else
            combine!(s, x, -, x0)
            combine!(y, g, -, g0)
            sty = inner(T, s, y)
            if sty > zero(sty)
                # Safeguarded Barzilai & Borwein spectral steplength.
                sts = inner(T, s, s)
                lambda = clamp(sts/sty, lmin, lmax)
            else
                lambda = lmax
            end
        end

        # Save current point.
        copy!(x0, x)
        copy!(g0, g)
        f0 = f

        # Compute the spectral projected gradient direction and delta = ⟨g,d⟩
        prj!(x, combine!(x, 1, x0, -lambda, g0)) # x = prj(x0 - lambda*g0)
        pcnt += 1
        combine!(d, x, -, x0) # d = x - x0
        delta = inner(T, g0, d)

        # Nonmonotone line search.
        stp = one(T) # Step length for first trial.
        while true
            # Evaluate function and gradient at trial point.
            f = as(T, fg!(x, g))
            fcnt += 1

            # Compare the new function value against the best function value
            # and, if smaller, update the best function value and the
            # corresponding best point.
            if f < fbest
                fbest = f
                copy!(xbest, x)
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
            if r > zero(r) && amin*r ≤ q ≤ amax*stp*r
                stp = q/r
            else
                stp /= 2
            end

            # Compute trial point.
            combine!(x, 1, x0, stp, d) # x = x0 + stp*d
        end

        if status != SEARCHING
            # The number of function evaluations was exceeded inside the line
            # search.
            break
        end

        # Proceed with next iteration.
        iter += 1

    end

    # Report final status.
    if verb > 0
        reason = get_reason(status)
        if !issuccess(status)
            printstyled(stderr, "# WARNING: ", reason, "\n"; color=:red)
        else
            printstyled(output, "# SUCCESS: ", reason, "\n"; color=:green)
        end
    end

    # Make sure to store the best solution so far.
    fbest < f && copy!(x, xbest)

    # Return algorithm statistics.
    seconds = time() - t0
    return Stats(; fx = fbest, pgtwon, pginfn, seconds, iter, fcnt, pcnt, status)
end

"""
    spg_CUTEst(name; kwds...) -> stats, x

yields the solution to the `CUTEst` problem `name` by the SPG method. This
require to have loaded the `CUTest` package.

"""
spg_CUTEst(args...; kwds...) =
    error("invalid arguments or `CUTEst` package not yet loaded")

function default_observer(output::IO, stats::Stats, x::AbstractArray, best::Bool)
    if stats.iter == 0
        @printf(output, "# %s\n# %s\n",
                " ITER   EVAL   PROJ             F(x)              ‖PG(X)‖₂  ‖PG(X)‖_∞",
                "---------------------------------------------------------------------")
    end
    @printf(output, " %6d %6d %6d %3s %24.17e %9.2e %9.2e\n",
            stats.iter, stats.fcnt, stats.pcnt, (best ? "(*)" : "   "),
            stats.fx, stats.pgtwon, stats.pginfn)
end

@noinline argument_error(args...) = argument_error(string(args...))
argument_error(msg::AbstractString) = throw(ArgumentError(msg))

end # module
