#
# quasinewton.jl --
#
# Limited memory quasi-Newton methods for OptimPackNextGen.
#
# -----------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2015-2021, Éric Thiébaut.
# <https://github.com/emmt/OptimPackNextGen.jl>.
#

# Improvements:
# - skip updates such that rho ≤ 0;
# - check for sufficient descent condition and implement restarts;
# - add other convergence criteria;
# - in case of early stop, revert to best solution so far;
# - better initial step than 1/vnorm2(g);

module QuasiNewton

export
    vmlmb,
    vmlmb!,
    vmlmb_CUTEst

# Imports from other packages.
using LinearAlgebra
using Printf
using TypeUtils
using NumOptBase:
    NumOptBase,
    Bound,
    BoundedSet,
    linesearch_stepmax,
    project_direction!,
    project_variables!,
    unblocked_variables!

# Imports from parent module.
using  ..OptimPackNextGen
using  ..OptimPackNextGen.LineSearches
using  ..OptimPackNextGen.VectOps
using  ..OptimPackNextGen: auto_differentiate!, copy_variables, get_tolerances
import ..OptimPackNextGen: get_reason

# Use the same floating point type for scalars as in OptimPackNextGen.
import OptimPackNextGen.Float

@enum Status begin
    LINESEARCH_ERROR     = -3
    TOO_MANY_EVALUATIONS = -2
    TOO_MANY_ITERATIONS  = -1
    SEARCHING            =  0
    CONVERGENCE_X        =  1
    CONVERGENCE_FATOL    =  2
    CONVERGENCE_FRTOL    =  3
    CONVERGENCE_FMIN     =  4
    CONVERGENCE_G        =  5
end

"""
    VMLMB.Stats{F}(; fx::Real, gnorm::Real, step::Real, seconds::Real,
                     iter::Integer, eval::Integer, rejects::Integer,
                     status::Status)

yields an immutable object collecting information returned by the VMLMB method.
Type parameter `F` is the floating-point type for `fx`, `gnorm`, and `step`.
All members are mandatory and are specified by keyword.

"""
struct Stats{F<:AbstractFloat}
    fx::F            # Objective function value.
    gnorm::F         # Euclidean norm of the (projected) gradient.
    step::F          # Line-search step length.
    seconds::Float64 # Execution time in seconds.
    iter::Int        # Number of iterations.
    eval::Int        # Number of objective function evaluations.
    rejects::Int     # Number of rejected updates.
    status::Status   # Algorithm status.
    function Stats{F}(; fx::Real, gnorm::Real, step::Real, seconds::Real,
                        iter::Integer, eval::Integer, rejects::Integer,
                        status::Status) where {F}
        return new{F}(fx, gnorm, step, seconds, iter, eval, rejects, status)
    end
end

LinearAlgebra.issuccess(stats::Stats) = issuccess(stats.status)
LinearAlgebra.issuccess(status::Status) = Integer(status) > 0

get_reason(stats::Stats) = get_reason(stats.status)
get_reason(status::Status) =
    status == LINESEARCH_ERROR     ? "Error in line-search" :
    status == TOO_MANY_EVALUATIONS ? "Too many function evaluations" :
    status == TOO_MANY_ITERATIONS  ? "Too many iterations" :
    status == SEARCHING            ? "Work in progress" :
    status == CONVERGENCE_X        ? "Convergence with `x` test satisfied" :
    status == CONVERGENCE_FATOL    ? "Convergence with `fatol` test satisfied" :
    status == CONVERGENCE_FRTOL    ? "Convergence with `frtol` test satisfied" :
    status == CONVERGENCE_FMIN     ? "Convergence with `f(x) ≤ fmin`" :
    status == CONVERGENCE_G        ? "Convergence with `g` test satisfied" :
    "Unknown status"

struct VMLMBConfig{FP<:AbstractFloat}
    method::Symbol # one of `:LBFGS`, `:BLMVM`, or `:VMLMB`
    mem::Int
    maxiter::Int
    maxeval::Int
    verb::Int
    fmin::FP
    epsilon::FP
    xatol::FP
    xrtol::FP
    fatol::FP
    frtol::FP
    gatol::FP
    grtol::FP
    autodiff::Bool
end

is_lbfgs(cfg::VMLMBConfig) = cfg.method === :LBFGS
is_blmvm(cfg::VMLMBConfig) = cfg.method === :BLMVM
is_vmlmb(cfg::VMLMBConfig) = cfg.method === :VMLMB
is_unconstrained(cfg::VMLMBConfig) = is_lbfgs(cfg)
is_bounded(cfg::VMLMBConfig) = ! is_unconstrained(cfg)
is_bounded(Ω::BoundedSet) = (NumOptBase.is_bounded_below(Ω.lower) |
                             NumOptBase.is_bounded_above(Ω.upper))

const default_xtol = (0.0, 1.0e-7)
const default_ftol = (0.0, 1.0e-8)
const default_gtol = (0.0, 1.0e-6)
const default_mem  = 5

"""
    vmlmb(fg!, x0; kwds...) -> x, stats

attempts to solve the constrained problem:

    min f(x)   subject to   x ∈ Ω   with   Ω = { x ∈ ℝⁿ | ℓ ≤ x ≤ u }

by the VMLMB method, a Variable Metric method with Limited Memory and optional
Bound constraints. Argument `fg!` implements the objective function `f(x)` and
its gradient `∇f(x)`. Argument `x0 ∈ ℝⁿ` gives an initial approximation of the
variables (its contents is left unchanged and it does not need to be feasible).
The result is a 2-tuple `(x, stats)` with `x ∈ Ω` and `stats` a structure with
information about the algorithm computations. Provided `issuccess(stats)` is
true, `x` is an approximate local minimizer of the objective function on the
feasible set `Ω`.

The caller must provide a function `fg!` to compute the value and the gradient
of the objective function as follows:

    fx = fg!(x, gx)

where `x` stores the current variables, `fx` is the value of the function at
`x` and the contents of `gx` has to be overwritten with the gradient at `x`
(when `fg!` is called, `gx` is already allocated). The best solution found so
far is returned in `x`.

Another possibility is to specify keyword `autodiff = true` and rely on
automatic differentiation to compute the gradient:

    x = vmlmb(f, x0; autodiff=true, kwds...)

where `f` is a simpler function that takes the variables `x` as a single
argument and returns the value of the objective function:

    fx = f(x)

The method [`OptimPackNextGen.auto_differentiate!`](@ref) is called to compute
the gradient of the objective function, say `f`. This method may be extended
for the specific type of `f`. An implementation of `auto_differentiate!` is
provided by `OptimPackNextGen` if the `Zygote` package is loaded.

The following keywords are available:

* `mem` specifies the amount of storage as the number of previous steps
  memorized to build an L-BFGS model of the inverse Hessian of the objective
  function. By default `mem = $default_mem`.

* `lower` and `upper` specify the lower and upper bounds for the variables. A
   bound can be a scalar to indicate that all variables have the same bound
   value. If the lower (resp. upper) bound is unspecified or set to `±∞`, the
   variables are assumed to be unbounded below (resp. above). If no bounds are
   set, VMLMB amounts to an unconstrained limited memory BFGS method (L-BFGS).

* `autodiff` is a boolean specifying whether to rely on automatic
  differentiation by calling [`OptimPackNextGen.auto_differentiate!](@ref). If
  not specified, this keyword is assumed to be `false`. You may use:

      autodiff = !applicable(fg!, x0, x0)

  to attempt to guess whether automatic differentiation is needed.

* `xtol` is a tuple of two nonnegative reals specifying respectively the
  absolute and relative tolerances for deciding convergence on the variables.
  Convergence occurs if the Euclidean norm of the the difference between
  successive iterates is less or equal `max(xtol[1], xtol[2]*vnorm2(x))`. By
  default, `xtol = $default_xtol`.

* `ftol` is a tuple of two nonnegative reals specifying respectively the
  absolute and relative errors desired in the function. Convergence occurs if
  the absolute error between `f(x)` and `f(xsol)` is less than `ftol[1]` or if
  the estimate of the relative error between `f(x)` and `f(xsol)`, where `xsol`
  is a local minimizer, is less than `ftol[2]`. By default, `ftol =
  $default_ftol`.

* `gtol` is a tuple of two nonnegative reals specifying the absolute and a
  relative thresholds for the norm of the gradient, convergence is assumed as
  soon as:

      ||g(x)|| ≤ hypot(gtol[1], gtol[2]*||g(x0)||)

  where `||g(x)||` is the Euclidean norm of the gradient at the current
  solution `x`, `||g(x0)||` is the Euclidean norm of the gradient at the
  starting point `x0`. By default, `gtol = $default_gtol`.

* `fmin` specifies a lower bound for the function. If provided, `fmin` is used
  to estimate the steepest desxecnt step length this value. The algorithm exits
  with a warning if `f(x) < fmin`.

* `maxiter` specifies the maximum number of iterations.

* `maxeval` specifies the maximum number of calls to `fg!`.

* `verb` specifies the verbosity level. It can be a boolean to specify whether
  to call the observer at every iteration or an integer to call the observer
  every `verb` iteration(s). The observer is never called if `verb` is less or
  equal zero. The default is `verb = false`.

* `observer` can be set with a callable object to print iteration information
  or inspect the current iterate, its signature is:

      observer(output::IO, stats, x)

  where `output` is the output stream, `stats` collects information about the
  current iteration (see below), and `x` is the current iterate.

* `output` specifies the output stream for printing information (`stdout` is
  used by default).

* `lnsrch` specifies the method to use for line searches (the default line
   search is `MoreThuenteLineSearch`).

* `blmvm` can be set true to emulate the BLMVM algorithm of Benson and Moré.
  This option has no effects for an uncostrained problem.

The `stats` object has the following properties:

* `stats.fx` is the objective function value at `x`.

* `stats.gnorm` is the norm of the gradient of the objective function at `x`.

* `stats.step` is the length of the line-search step to the current point.

* `stats.seconds` is the execution time in seconds.

* `stats.iter` is the number of iterations, `0` for the starting point.

* `stats.eval` is the number of function (and gradient) evaluations.

* `stats.rejects` is the number of times the computed direction was
  rejected and the L-BFGS recursion restarted.

* `stats.status` indicates the status of the algorithm.


### History

The VMLMB algorithm in
[OptimPackNextGen](https://github.com/emmt/OptimPackNextGen.jl) provides a pure
Julia implementation of the original method (Thiébaut, 2002) with some
improvements and the capability to emulate L-BFGS and BLMVM methods.

The limited memory BFGS method (L-BFGS) was first described by Nocedal (1980)
who dubbed it SQN. The method is implemented in MINPACK-2 (1995) by the FORTRAN
routine VMLM. The numerical performances of L-BFGS have been studied by Liu and
Nocedal (1989) who proved that it is globally convergent for unfiformly convex
problems with a R-linear rate of convergence. They provided the FORTRAN code
LBFGS. The BLMVM and VMLMB algorithms were proposed by Benson and Moré (2001)
and Thiébaut (2002) to account for separable bound constraints on the
variables. These two latter methods are rather different than L-BFGS-B by Byrd
at al. (1995) which has more overheads and is slower.

* J. Nocedal, "*Updating Quasi-Newton Matrices with Limited Storage*" in
  Mathematics of Computation, vol. 35, pp. 773-782 (1980).

* D.C. Liu & J. Nocedal, "*On the limited memory BFGS method for large scale
  optimization*" in Mathematical programming, vol. 45, pp. 503-528 (1989).

* R.H. Byrd, P. Lu, J. Nocedal, & C. Zhu, "*A limited memory algorithm for
  bound constrained optimization*" in SIAM Journal on Scientific Computing,
  vol. 16, pp. 1190-1208 (1995).

* S.J. Benson & J.J. Moré, "*A limited memory variable metric method in
  subspaces and bound constrained optimization problems*" in Subspaces and
  Bound Constrained Optimization Problems (2001).

* É. Thiébaut, "*Optimization issues in blind deconvolution algorithms*" in
  Astronomical Data Analysis II, Proc. SPIE 4847, pp. 174-183 (2002).

"""
function vmlmb(fg!, x0; kwds...)
    x = copy_variables(x0)
    return vmlmb!(fg!, x; kwds...), x
end

"""
    vmlmb_CUTEst(name; kwds...) -> x, stats

yields the solution to the `CUTEst` problem `name` by the VMLMB method. This
require to have loaded the `CUTest` package.

"""
vmlmb_CUTEst(arg...; kwds...) =
    error("invalid arguments or `CUTEst` package not yet loaded")

"""
     vmlmb!(fg!, x; mem=..., lower=..., upper=..., ftol=..., fmin=...) -> x, stats

finds a local minimizer of `f(x)` starting at `x` and storing the best solution
in `x`. Method `vmlmb!` is the in-place version of `vmlmb` (which to see).

"""
function vmlmb!(fg!, x::AbstractArray{T,N};
                mem::Integer = min(5, length(x)),
                lower::Bound{<:Real,N} = -Inf,
                upper::Bound{<:Real,N} = +Inf,
                autodiff::Bool = false,
                blmvm::Bool = false,
                fmin::Real = -Inf,
                maxiter::Integer = typemax(Int),
                maxeval::Integer = typemax(Int),
                xtol::NTuple{2,Real} = default_xtol,
                ftol::NTuple{2,Real} = default_ftol,
                gtol::NTuple{2,Real} = default_gtol,
                epsilon::Real = 0.0,
                verb::Integer = false,
                observer = default_observer,
                output::IO = stdout,
                lnsrch::Union{LineSearch{Float},Nothing} = nothing) where {T<:AbstractFloat,N}
    # Check settings.
    @assert mem ≥ 1
    @assert maxiter ≥ 0
    @assert maxeval ≥ 1
    @assert 0 ≤ epsilon < 1

    # Build a bounded set.
    Ω = BoundedSet{T,N}(lower, upper)

    # Determine the optimization method to emulate.
    method = if ! is_bounded(Ω)
        :LBFGS
    elseif blmvm
        :BLMVM
    else
        :VMLMB
    end

    cfg = VMLMBConfig{Float}(
        method, mem, maxiter, maxeval, verb, fmin, epsilon,
        get_tolerances(Float, "`xtol`", xtol...)...,
        get_tolerances(Float, "`ftol`", ftol...)...,
        get_tolerances(Float, "`gtol`", gtol...)...,
        autodiff)

    # Call the real method.
    stats = _vmlmb!(fg!, x, Ω, cfg, lnsrch, observer, output)
    return x, stats
end

# Provide a default line search method if needed.
function _vmlmb!(fg!, x::AbstractArray, Ω::BoundedSet, cfg::VMLMBConfig,
                 lnsrch::Nothing, observer, output::IO)
    if is_unconstrained(cfg)
        ls = MoreThuenteLineSearch{Float}(ftol=1e-3, gtol=0.9, xtol=0.1)
        return _vmlmb!(fg!, x, Ω, cfg, ls, observer, output)
    else
        ls = MoreToraldoLineSearch{Float}(ftol=1e-3, gamma=(0.1,0.5))
        return _vmlmb!(fg!, x, Ω, cfg, ls, observer, output)
    end
end

# The real worker.
function _vmlmb!(fg!, x::T, Ω::BoundedSet,
                 cfg::VMLMBConfig, lnsrch::LineSearch{FP},
                 observer, output::IO) where {T<:AbstractArray,
                                              FP<:AbstractFloat}
    t0 = time()
    STPMIN = FP(1e-20) # FIXME: should be in config
    STPMAX = FP(1e+20) # FIXME: should be in config

    act = similar(x) # FIXME: not used if not bounded
    mem = min(cfg.mem, length(x))
    fminset::Bool = (! isnan(cfg.fmin) && cfg.fmin > -Inf) # FIXME: simplify

    mark = 1    # index of most recent step and gradient difference
    m = 0       # number of memorized steps
    iter = 0    # number of algorithm iterations
    eval = 0    # number of objective function and gradient evaluations
    rejects = 0 # number of rejected search directions

    f = zero(FP)
    f0 = zero(FP)
    gd = zero(FP)
    gd0 = zero(FP)
    stp = zero(FP)
    stpmin = zero(FP)
    stpmax = zero(FP)
    smin = zero(FP)
    smax = zero(FP)
    gamma = one(FP)
    gnorm = zero(FP)
    gtest = zero(FP)

    # Variables for saving information about best point so far.
    beststp = zero(FP)
    bestf = zero(FP)
    bestgnorm = zero(FP)

    # Allocate memory for the limited memory BFGS approximation of the inverse
    # Hessian.
    g = vcreate(x) # ------------> gradient
    if is_bounded(cfg)
        p = vcreate(x) # --------> projected gradient
    end
    d = vcreate(x) # ------------> search direction
    S = Array{T}(undef, mem) # --> memorized steps
    Y = Array{T}(undef, mem) # --> memorized gradient differences
    s = vcreate(x) # ------------> for effective step
    for k in 1:mem
        S[k] = vcreate(x)
        Y[k] = vcreate(x)
    end
    rho = Array{FP}(undef, mem)
    alpha = Array{FP}(undef, mem)

    # Variable used to control the stage of the algorithm:
    #   * stage = 0 at start or restart;
    #   * stage = 1 during a line search;
    #   * stage = 2 if line search has converged;
    #   * stage = 3 if algorithm has converged;
    #   * stage = 4 if algorithm is terminated with a warning.
    stage = 0

    bounded = is_bounded(cfg)
    status = SEARCHING
    while true

        # Compute value of function and gradient, register best solution so far
        # and check for convergence based on the gradient norm.
        if bounded
            project_variables!(x, x, Ω)
        end
        if cfg.autodiff
            f = as(FP, auto_differentiate!(fg!, x, g))
        else
            f = as(FP, fg!(x, g))
        end
        eval += 1
        if bounded
            project_direction!(p, x, -, g, Ω)
        end
        if eval == 1 || f < bestf
            gnorm = vnorm2(FP, is_bounded(cfg) ? p : g)
            beststp = stp
            bestf = f
            bestgnorm = gnorm
            if eval == 1
                gtest = max(cfg.gatol, cfg.grtol*gnorm)
            end
            if gnorm ≤ gtest
                stage = 3
                status = CONVERGENCE_G
                # if gnorm == 0
                #     status = CONVERGENCE_KKT
                #     reason = "a stationary point has been found"
                # elseif is_bounded(cfg)
                #     reason = "projected gradient norm sufficiently small"
                # else
                #     reason = "gradient norm sufficiently small"
                # end
                if eval > 1
                    iter += 1 # FIXME: do this here???
                end
            end
        end
        if f ≤ cfg.fmin
            stage = 4
            status = CONVERGENCE_FMIN
        end

        if stage == 1
            # Compute effective step.
            vcombine!(s, 1, x, -1, S[mark])

            # Line search is in progress.
            if use_derivatives(lnsrch)
                if is_bounded(cfg)
                    gd = vdot(FP, g, s)/stp
                else
                    gd = -vdot(FP, g, d)
                end
            end
            state = iterate!(lnsrch, f, gd)
            if state == :SEARCHING
                stp = get_step(lnsrch)
            elseif state == :CONVERGENCE || state == :WARNING
                # Line search has converged. Increment iteration counter and
                # check for stopping conditions.
                iter += 1
                xtest = max(cfg.xatol, zero(FP))
                if cfg.xrtol > 0
                    xtest = max(xtest, cfg.xrtol*vnorm2(FP, x))
                end
                if vnorm2(FP, s) ≤ xtest
                    stage = 3
                    status = CONVERGENCE_X
                else
                    delta = max(abs(f - f0), stp*abs(gd0))
                    if delta ≤ cfg.fatol
                        stage = 3
                        status = CONVERGENCE_FATOL
                    elseif delta ≤ cfg.frtol*abs(f0)
                        stage = 3
                        status = CONVERGENCE_FRTOL
                    elseif iter ≥ cfg.maxiter
                        stage = 4
                        status = TOO_MANY_ITERATIONS
                    elseif eval ≥ cfg.maxeval
                        stage = 4
                        status = TOO_MANY_EVALUATIONS
                    else
                        stage = 2
                    end
                end
            else
                # Line seach terminated on error.
                stage = 4
                status = LINESEARCH_ERROR # FIXME throw an error
            end
        end
        if stage < 2 && eval ≥ cfg.maxeval
            stage = 4
            status = TOO_MANY_ITERATIONS
        end

        if stage != 1
            # Initial step or line search has converged.

            # Make sure the gradient norm is correct or revert to best solution
            # so far if something wrong occured.
            if stp != beststp
                if stage ≥ 3
                    stp = beststp
                    f = bestf
                    gnorm = bestgnorm
                    vcombine!(x, 1, S[mark], -stp, d)
                    if bounded
                        project_variables!(x, x, Ω)
                    end
                else
                    gnorm = vnorm2(FP, is_bounded(cfg) ? p : g)
                end
            end

            # Print some information if requested and terminate algorithm if
            # stage ≥ 3.
            if verbose(cfg.verb, iter)
                stats = Stats{FP}(; fx = f, gnorm, step = stp, seconds = time() - t0,
                                  eval, iter, rejects, status)
                observer(output, stats, x)
            end
            if stage ≥ 3
                break
            end

            # Compute next search direction.
            if stage == 2
                # Update limited memory BFGS approximation of the inverse
                # Hessian with the effective step and gradient change.
                vupdate!(S[mark], -1, x)
                vupdate!(Y[mark], -1, is_blmvm(cfg) ? p : g)
                if bounded
                    rho[mark] = vdot(FP, Y[mark], S[mark])
                    if rho[mark] > 0
                        # The update is acceptable, compute the scale.
                        gamma = rho[mark]/vdot(FP, Y[mark], Y[mark])
                    end
                end
                m = min(m + 1, mem)

                # Compute search direction.
                if is_vmlmb(cfg)
                    # VMLMB method.
                    vcopy!(d, p)
                    change = apply_lbfgs!(S, Y, rho, m, mark, d, alpha,
                                          # FIXME: speedup this?
                                          unblocked_variables!(act, x, -, d, Ω))
                else
                    # LBFGS or BLMVM method.
                    vcopy!(d, g)
                    change = apply_lbfgs!(S, Y, rho, gamma, m, mark, d, alpha)
                end
                if change
                    if bounded
                        # FIXME: speedup project_direction with the unblocked vars.?
                        project_direction!(d, x, -, d, Ω)
                    end
                    gd = -vdot(g, d)
                    reject = ! sufficient_descent(gd, cfg.epsilon, gnorm, d)
                else
                    reject = true
                end
                if reject
                    # The steepest descent will be used.
                    stage = 0
                    rejects += 1
                end

                # Circularly move the mark to the next slot.
                mark = (mark < mem ? mark + 1 : 1)
            end
            if stage == 0
                # At the initial point or computed direction was rejected.  Use
                # the steepest descent which is the projected gradient.
                vcopy!(d, is_lbfgs(cfg) ? g : p)
                gd = -gnorm^2
            end

            # Save function value, variables and (projected) gradient at start
            # of line search.
            f0 = f
            gd0 = gd
            vcopy!(S[mark], x)
            vcopy!(Y[mark], is_blmvm(cfg) ? p : g)

            # Choose initial step length.
            if stage == 2
                stp = one(FP)
            elseif fminset && cfg.fmin < f0
                stp = 2*(cfg.fmin - f0)/gd0
            else
                stp = inv(gnorm) # FIXME: use a better scale
            end
            if bounded
                # Make sure the step is not longer than necessary.
                smax = linesearch_stepmax(x, -, d, Ω)
                stp = min(stp, smax)
                stpmin = STPMIN*stp
                stpmax = min(STPMAX*stp, smax)
            else
                stpmin = STPMIN*stp
                stpmax = STPMAX*stp
            end

            # Initialize the line search.
            state = start!(lnsrch, f0, gd0, stp; stpmin=stpmin, stpmax=stpmax)
            if state != :SEARCHING
                # Something wrong happens.
                stage = 4
                status = lnsrch.status # FIXME: use a better scale
                break
            end
            stage = 1 # line search is in progress

        end

        # Compute the new iterate.
        vcombine!(x, 1, S[mark], -stp, d)

    end

    # Algorithm finished.
    if verbose(cfg.verb, 0) #always print last line if verb>0
        color = (stage > 3 ? :red : :green)
        prefix = (stage > 3 ? "WARNING: " : "CONVERGENCE: ")
        reason = get_reason(status)
        printstyled(output, "# ", prefix, reason, "\n"; color=color)
    end
    return Stats{FP}(; fx = bestf, gnorm = bestgnorm, step = beststp,
                     seconds = time() - t0, eval, iter, rejects, status)
end

"""
    sufficient_descent(gd, ε, gnorm, d)

checks whether `d` is a sufficient descent direction (Zoutenjdik condition).
Argument `gd` is the scalar product `vdot(g,d)` between the gradient `g` and
the direction `d`, `ε` is a nonnegative small value in the range `[0,1)` and
`gnorm` is the Euclidean norm of the gradient `g`.

If the Euclidean norm of `d` has already been computed, then

    sufficient_descent(gd, ε, gnorm, dnorm)

should be used instead with `dnorm = vnorm2(d)`.
"""
sufficient_descent(gd::Real, ε::Real, gnorm::Real, d) =
    sufficient_descent(gd, ε, gnorm, vnorm2(d))

sufficient_descent(gd::Real, ε::Real, gnorm::Real, dnorm::Real) =
    ε > 0 ? (gd ≤ -ε*dnorm*gnorm) : gd < 0

"""
    verbose(verb, iter)

yields whether to print information at iteration `iter` with verbose level
`verb`.

"""
verbose(verb::Integer, iter::Integer) = (verb > 0 && iter%verb == 0)

function default_observer(output::IO, stats::Stats, x::AbstractArray)
    if stats.iter == 0
        @printf(output, "#%s%s\n#%s%s\n",
                " ITER   EVAL   REJECTS",
                "          F(X)           ||G(X)||    STEP",
                "----------------------",
                "-------------------------------------------")
    end
    @printf(output, " %5d  %5d  %5d  %24.16E %9.2E %9.2E\n",
            stats.iter, stats.eval, stats.rejects, stats.fx, stats.gnorm,
            stats.step)
end

#------------------------------------------------------------------------------

"""
One of the calls:

    modif = apply_lbfgs!(S, Y, rho, gamma, m, mark, v, alpha)
    modif = apply_lbfgs!(S, Y, rho, d,     m, mark, v, alpha)
    modif = apply_lbfgs!(S, Y, rho, H0!,   m, mark, v, alpha)

computes the matrix-vector product `H*v` where `H` is the limited memory
inverse BFGS approximation. Operation is done *in-place*: on entry, argument
`v` contains the input vector `v`; on exit, argument `v` contains the
matrix-vector product `H*v`. The returned value, `modif`, is a boolean
indicating whether the initial vector was modified (BFGS updates are skipped if
`rho[k] ≤ 0`).

The linear operator `H` depends on an initial approximation of the inverse
Hessian (see below), `m` steps `S[...]` and `m` gradient differences `Y[...]`.
The most recent step and gradient difference are stored in `S[mark]` and
`Y[mark]`, respectively.

Argument `rho` contains the inner products of the steps and the gradient
differences: `rho[k] = vdot(S[k],Y[k])`. On exit rho is unchanged.

Argument `alpha` is a work vector of length at least `m`.

The initial approximation of the inverse Hessian can be specified in several
ways:

* by a scalar `gamma > 0` to assume that the initial approximation of the
  inverse Hessia is `gamma*I` where `I` is the identity;

* by a some instance of the same type as `v` (whose components are all strictly
  positive) to assume that the initial approximation of the inverse Hessian is
  a diagonal matrix whose diagonal terms are given by `d`;

* by a function `H0!` to assume that `H0!(u)` (for any instance `u` of the same
  type as `v`) applies the initial approximation of the inverse Hessian to `u`
  and stores the result in `u` (in-place operation).

"""
function apply_lbfgs!(S::Vector{T}, Y::Vector{T}, rho::Vector{FP},
                      gamma::FP, m::Int, mark::Int, v::T,
                      alpha::Vector{FP}) where {T<:AbstractArray,
                                                FP<:AbstractFloat}
    @assert gamma > 0
    apply_lbfgs!(S, Y, rho, u -> vscale!(u, gamma), m, mark, v, alpha)
end

function apply_lbfgs!(S::Vector{T}, Y::Vector{T}, rho::Vector{FP},
                      d::T, m::Int, mark::Int, v::T,
                      alpha::Vector{FP}) where {T<:AbstractArray,
                                                FP<:AbstractFloat}
    apply_lbfgs!(S, Y, rho, u -> vproduct!(u, d, u), m, mark, v, alpha)
end

function apply_lbfgs!(S::Vector{T}, Y::Vector{T}, rho::Vector{FP},
                      H0!::Function, m::Int, mark::Int, v::T,
                      alpha::Vector{FP}) where {T<:AbstractArray,
                                                FP<:AbstractFloat}
    mem = min(length(S), length(Y), length(rho), length(alpha))
    @assert 1 ≤ m ≤ mem
    @assert 1 ≤ mark ≤ mem
    modif::Bool = false
    @inbounds begin
        k = mark + 1
        for i in 1:m
            k = (k > 1 ? k - 1 : mem)
            if rho[k] > 0
                alpha[k] = vdot(FP, S[k], v)/rho[k]
                vupdate!(v, -alpha[k], Y[k])
                modif = true
            end
        end
        if modif
            H0!(v)
            for i in 1:m
                if rho[k] > 0
                    beta = vdot(FP, Y[k], v)/rho[k]
                    vupdate!(v, alpha[k] - beta, S[k])
                end
                k = (k < mem ? k + 1 : 1)
            end
        end
    end
    return modif
end

function apply_lbfgs!(S::Vector{T}, Y::Vector{T}, rho::Vector{FP},
                      m::Int, mark::Int, v::T,
                      alpha::Vector{FP}, act::T) where {T<:AbstractArray,
                                                        FP<:AbstractFloat}
    mem = min(length(S), length(Y), length(rho), length(alpha))
    @assert 1 ≤ m ≤ mem
    @assert 1 ≤ mark ≤ mem
    gamma = zero(FP)
    @inbounds begin
        k = mark + 1
        for i in 1:m
            k = (k > 1 ? k - 1 : mem)
            rho[k] = vdot(FP, act, Y[k], S[k])
            if rho[k] > zero(rho[k])
                alpha[k] = vdot(FP, act, S[k], v)/rho[k]
                vupdate!(v, -alpha[k], act, Y[k])
                if iszero(gamma)
                    gamma = rho[k]/vdot(FP, act, Y[k], Y[k])
                end
            end
        end
        if gamma != zero(gamma)
            vscale!(v, gamma)
            for i in 1:m
                if rho[k] > zero(rho[k])
                    beta = vdot(FP, act, Y[k], v)/rho[k]
                    vupdate!(v, alpha[k] - beta, act, S[k])
                end
                k = (k < mem ? k + 1 : 1)
            end
        end
    end
    return (gamma != zero(gamma))
end

end # module
