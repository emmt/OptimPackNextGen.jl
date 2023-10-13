"""

Implements the Spectral Projected Gradient Method (Version 2: "continuous
projected gradient direction") to find the local minimizers of a given function
with convex constraints, described in:

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
    spg!

using Printf

using NumOptBase
using ..OptimPackNextGen

import OptimPackNextGen: getreason
using OptimPackNextGen: auto_differentiate!
using OptimPackNextGen.QuasiNewton: verbose
using OptimPackNextGen.VectOps

using LinearAlgebra

@enum Status begin
    TOO_MANY_EVALUATIONS = -2
    TOO_MANY_ITERATIONS  = -1
    SEARCHING            =  0
    INFNORM_CONVERGENCE  =  1
    TWONORM_CONVERGENCE  =  2
    FUNCTION_CONVERGENCE =  3
end

mutable struct Info
    f::Float64      # The final/current function value.
    fbest::Float64  # The best function value so far.
    pginfn::Float64 # ||projected grad||_inf at the final/current iteration.
    pgtwon::Float64 # ||projected grad||₂ at the final/current iteration.
    iter::Int       # The number of iterations.
    fcnt::Int       # The number of function (and gradient) evaluations.
    pcnt::Int       # The number of projections.
    status::Status  # Termination parameter.
    Info() = new(NaN, NaN, NaN, NaN, 0, 0, 0, SEARCHING)
end

LinearAlgebra.issuccess(info::Info) = issuccess(info.status)
LinearAlgebra.issuccess(status::Status) = Integer(status) > 0

const REASON = Dict{Status,String}(
    SEARCHING => "Work in progress",
    INFNORM_CONVERGENCE => "Convergence with projected gradient infinite-norm",
    TWONORM_CONVERGENCE => "Convergence with projected gradient 2-norm",
    FUNCTION_CONVERGENCE => "Function does not change in the last `m` iterations",
    TOO_MANY_ITERATIONS => "Too many iterations",
    TOO_MANY_EVALUATIONS => "Too many function evaluations")

getreason(info::Info) = getreason(info.status)
getreason(status::Status) = get(REASON, status, "unknown status")

# Default settings.
const default_eps1 = 1.0e-6
const default_eps2 = 1.0e-6
const default_eta  = 1.0
const default_lmin = 1.0e-30
const default_lmax = 1.0e+30
const default_ftol = 1.0e-4
const default_amin = 0.1
const default_amax = 0.9

"""
# Spectral Projected Gradient Method

The `spg` method implements the Spectral Projected Gradient Method (Version 2:
"continuous projected gradient direction") to find the local minimizers of a
given function with convex constraints, described in the references below. A
typical use is:

    spg(fg!, prj!, x0, m) -> x

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

Argument `x0` is the initial solution and argument `m` is the number of
previous function values to be considered in the nonmonotone line search. If `m
≤ 1`, then a monotone line search with Armijo-like stopping criterion will be
used.

Another possibility is to call:

    spg(fg!, Ω, x0, m) -> x

with `Ω` a bounded set (of type `BoundedSet`) to specify the feasible subset
for the variables `x`.

The following keywords are available:

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

* `info` is an instance of `SPG.Info` to store information about the final
  iterate.

* `verb` specifies the verbosity level. It can be a boolean to specify
  whether to print information at every iteration or an integer to print
  information every `verb` iteration(s).  No information is printed if
  `verb` is less or equal zero. The default is `verb = false`.

* `printer` specifies a subroutine to print some information at each iteration.
  This subroutine will be called as `printer(io, x, fx, info)` with `io` the
  output stream, `x` the current variables, `fx = f(x)` the corresponding
  objective function value, and `info` an instance of `SPGL.Info`.

* `io` specifes the output stream for iteration information.  It is `stdout` by
  default.

The `SPG.Info` type has the following members:

* `f` is the current function value.
* `fbest` is the best function value so far.
* `pginfn` is the infinite norm of the projected gradient.
* `pgtwon` is the Eucliddean norm of the projected gradient.
* `iter` is the number of iterations.
* `fcnt` is the number of function (and gradient) evaluations.
* `pcnt` is the number of projections.
* `status` indicates the type of termination.

Possible `status` values are:

| Status                     | Reason                                              |
|:---------------------------|:----------------------------------------------------|
| `SPG.SEARCHING`            | Work in progress                                    |
| `SPG.INFNORM_CONVERGENCE`  | Convergence with projected gradient infinite-norm   |
| `SPG.TWONORM_CONVERGENCE`  | Convergence with projected gradient 2-norm          |
| `SPG.FUNCTION_CONVERGENCE` | Function does not change in the last `m` iterations |
| `SPG.TOO_MANY_ITERATIONS`  | Too many iterations                                 |
| `SPG.TOO_MANY_EVALUATIONS` | Too many function evaluations                       |


## References

* E. G. Birgin, J. M. Martinez, and M. Raydan, "Nonmonotone spectral projected
  gradient methods on convex sets", SIAM Journal on Optimization 10,
  pp. 1196-1211 (2000).

* E. G. Birgin, J. M. Martinez, and M. Raydan, "SPG: software for
  convex-constrained optimization", ACM Transactions on Mathematical Software
  (TOMS) 27, pp. 340-349 (2001).

"""
spg(fg!, prj!, x0::AbstractArray, m::Integer; kwds...) =
    spg!(fg!, prj!, copy_variables(x0), m; kwds...)

spg!(fg!, Ω::BoundedSet{T,N}, x::AbstractArray{T,N}, m::Integer; kwds...) where {T,N} =
    spg!(fg!, Projector(Ω), x, m; kwds...)

spg!(fg!, Ω::BoundedSet, x::AbstractArray{T,N}, m::Integer; kwds...) where {T,N} =
    spg!(fg!, BoundedSet{T,N}(Ω), x, m; kwds...)

function spg!(fg!, prj!, x::AbstractArray, m::Integer;
              autodiff::Bool  = false,
              info::Info        = Info(),
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
              printer         = default_printer,
              verb::Integer   = false,
              io::IO          = stdout)
    # Check settings.
    m ≥ one(m) || argument_error("`m ≥ 1` must hold")
    eps1 ≥ zero(eps1) || argument_error("`eps1 ≥ 0` must hold")
    eps2 ≥ zero(eps2) || argument_error("`eps2 ≥ 0` must hold")
    eta > zero(eta) || argument_error("`eta ≥ 0` must hold")
    lmin > zero(lmin) || argument_error("`lmin > 0` must hold")
    lmax > zero(lmax) || argument_error("`lmax > 0` must hold")
    lmin < lmax || argument_error("`lmin < lmax` must hold")
    zero(ftol) < ftol < one(ftol) || argument_error("`0 < ftol < 1` must hold")
    amin > zero(amin) || argument_error("`amin > 0` must hold")
    amax > zero(amax) || argument_error("`amax > 0` must hold")
    amin < amax || argument_error("`amin < amax` must hold")

    # Determine floating-point type for scalar computations (using at least
    # double-precision) and call private method with all arguments checked and
    # converted to the correct type.
    T = promote_type(Float64, eltype(x))
    args = (prj!, x, Int(m), info, Int(maxit), Int(maxfc), as(T, eps1), as(T, eps2),
            as(T, eta), as(T, lmin), as(T, lmax), as(T, ftol), as(T, amin), as(T, amax),
            printer, Int(verb), io)
    if autodiff
        _spg!((x, g) -> auto_differentiate!(fg!, x, g), args...)
    else
        _spg!(fg!, args...)
    end
    return x
end

function _spg!(fg!, prj!, x::AbstractArray, m::Int, info::Info, maxit::Int, maxfc::Int,
               eps1::T, eps2::T, eta::T, lmin::T, lmax::T, ftol::T,
               amin::T, amax::T, printer, verb::Int, io::IO) where {T<:AbstractFloat}
    # Initialization.
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
    g = vcreate(x)
    d = pg = vcreate(x)
    s = x0 = vcreate(x)
    y = g0 = vcreate(x)
    xbest = vcreate(x)

    # Project initial guess.
    prj!(x, x)
    pcnt += 1

    # Evaluate function and gradient.
    f = as(T, fg!(x, g))
    fcnt += 1

    # Initialize best solution and best function value.
    fbest = f
    vcopy!(xbest, x)

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
        vcombine!(pg, 1/eta, x, -1/eta, prj!(pg, vcombine!(pg, 1, x, -eta, g)))
        pcnt += 1
        pgtwon = vnorm2(T, pg)
        pginfn = vnorminf(T, pg)

        # Print iteration information.
        if verbose(verb, iter)
            info.fbest  = fbest
            info.pgtwon = pgtwon
            info.pginfn = pginfn
            info.iter   = iter
            info.fcnt   = fcnt
            info.pcnt   = pcnt
            info.status = status
            printer(io, x, f, info)
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
            vcombine!(s, 1, x, -1, x0)
            vcombine!(y, 1, g, -1, g0)
            sty = vdot(T, s, y)
            if sty > zero(sty)
                # Safeguarded Barzilai & Borwein spectral steplength.
                sts = vdot(T, s, s)
                lambda = clamp(sts/sty, lmin, lmax)
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
        delta = vdot(T, g0, d)

        # Nonmonotone line search.
        stp = one(T) # Step length for first trial.
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
            r = (f - f0 - stp*delta)*2
            if r > zero(r) && amin*r ≤ q ≤ amax*stp*r
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
    info.fbest  = fbest
    info.pgtwon = pgtwon
    info.pginfn = pginfn
    info.iter   = iter
    info.fcnt   = fcnt
    info.pcnt   = pcnt
    info.status = status
    if verbose(verb, 0) # always print last line if verb > 0
        reason = getreason(info)
        if !issuccess(status)
            printstyled(stderr, "# WARNING: ", reason, "\n"; color=:red)
        else
            printstyled(io, "# SUCCESS: ", reason, "\n"; color=:green)
        end
    end

    # Make sure to return the best solution so far.
    fbest < f && vcopy!(x, xbest)
    return x
end

"""
     SPG.copy_variables(x)

yields a copy of the variables `x` having a *similar* array type but
floating-point element type.

"""
copy_variables(x::AbstractArray) = copyto!(similar(x, float(eltype(x))), x)

function default_printer(io::IO, x::AbstractArray, fx::Real, info::Info)
    if info.iter == 0
        @printf(io, "# %s\n# %s\n",
                " ITER   EVAL   PROJ             F(x)              ‖PG(X)‖₂  ‖PG(X)‖_∞",
                "---------------------------------------------------------------------")
    end
    @printf(io, " %6d %6d %6d %3s %24.17e %9.2e %9.2e\n",
            info.iter, info.fcnt, info.pcnt, (fx ≤ info.fbest ? "(*)" : "   "),
            fx, info.pgtwon, info.pginfn)
end

@noinline argument_error(args...) = argument_error(string(args...))
argument_error(msg::AbstractString) = throw(ArgumentError(msg))

end # module
