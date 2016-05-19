#
# quasi-newton.jl --
#
# Limited memory quasi-Newton methods for TiPi.
#
#------------------------------------------------------------------------------
#
# This file is part of TiPi.jl licensed under the MIT "Expat" License.
#
# Copyright (C) 2015-2016, Éric Thiébaut.
#
#------------------------------------------------------------------------------

# Improvements:
# - skip updates such that rho <= 0;
# - check for sufficient descent condition and implement restarts;
# - add other convergence criteria;
# - in case of early stop, revert to best solution so far;
# - better initial step than 1/norm2(g);

module QuasiNewton

using ..Algebra
using ..LineSearch

# Use the same floating point type for scalars as in TiPi.
import ..Float

export vmlmb, vmlmb!, EMULATE_BLMVM

const EMULATE_BLMVM = UInt(1)

"""
## VMLMB: limited memory BFGS method with optional bounds

    x = vmlmb(fg!, x0; mem=..., lower=..., upper=..., ftol=..., fmin=...)

computes a local minimizer of a function of several variables by a limited
memory variable metric method.  The caller provides a function `fg!` to compute
the value and the gradient of the function as follows:

    f = fg!(x, g)

where `x` are the current variables, `f` is the value of the function at `x`
and `g` is the gradient at `x` (`g` is already allocated as `g = similar(x0)`).
Argument `x0` gives the initial approximation of the variables (its contents is
left unchanged).  The best solution found so far is returned in `x`.

The following keywords are available:

* `mem` specifies the amount of storage.

* `ftol` is a tuple of two nonnegative reals specifying respectively the
  absolute and relative errors desired in the function.  Convergence occurs if
  the estimate of the relative error between `f(x)` and `f(xsol)`, where `xsol`
  is a local minimizer, is less than `ftol[1]` or if the absolute error between
  `f(x)` and `f(xsol)` is less than `ftol[2]`.  By default, `ftol = (0.0,1e-8)`.

* `gtol` is a tuple of two nonnegative reals specifying the absolute and a
  relative thresholds for the norm of the gradient, convergence is assumed as
  soon as:

      ||g(x)|| <= hypot(gtol[1], gtol[2]*||g(x0)||)

  where `||g(x)||` is the Euclidean norm of the gradient at the current
  solution `x`, `||g(x0)||` is the Euclidean norm of the gradient at the
  starting point `x0`.  By default, `gtol = (0.0,1e-6)`.

* `fmin` specifies a lower bound for the function.  If provided, `fmin` is used
  to estimate the steepest desxecnt step length this value.  The algorithm
  exits with a warning if `f(x) < fmin`.

* `maxiter` specifies the maximum number of iterations.

* `maxeval` specifies the maximum number of calls to `fg!`.

* `verb` specifies whether to print iteration information (`verb = false`, by
  default).

* `printer` can be set with a user defined function to print iteration
  information, its signature is:

      printer(io::IO, iter::Integer, eval::Integer, rejects::Integer,
              f::Real, gnorm::Real, stp::Real)

  where `io` is the output stream, `iter` the iteration number (`iter = 0` for
  the starting point), `eval` is the number of calls to `fg!`, `rejects` is the
  number of times the computed direction was rejected, `f` and `gnorm` are the
  value of the function and norm of the gradient at the current point, `stp` is
  the length of the step to the current point.

* `output` specifies the output stream for printing information (`STDOUT` is
  used by default).

* `lnsrch` specifies the method to use for line searches (the default
   line search is `MoreThuenteLineSearch`).

* `lower` and `upper` specify the lower and upper bounds for the variables.
   The bound can be a scalar to indicate that all variables gave the same bound
   value.  If the lower (resp. upper) bound is unspecified or set to `±∞`, the
   variables are assumed to be unbounded below (resp. above).  If no bounds are
   set, VMLMB amounts to an unconstrained limited memory BFGS method (L-BFGS).

* `blmvm` can be set true to emulate the BLMVM algorithm of Benson and Moré.
  This option has no effects for an uncostrained problem.


### History

The VMLMB algorithm in TiPi.jl provides a pure Julia implementation of the
original method with some improvements and the capability to emulate L-BFGS and
BLMVM methods.

The limited memory BFGS method (L-BFGS) was first described by Nocedal (1980)
who dubbed it SQN.  The method is implemented in MINPACK-2 (1995) by the
FORTRAN routine VMLM.  The numerical performances of L-BFGS have been studied
by Liu and Nocedal (1989) who proved that it is globally convergent for
unfiformly convex problems with a R-linear rate of convergence.  They provided
the FORTRAN code LBFGS.  The BLMVM and VMLMB algorithms were proposed by Benson
and Moré (2001) and Thiébaut (2002) to account for separable bound constraints
on the variables.  These two latter methods are rather different than L-BFGS-B
by Byrd at al. (1995) which has more overheads and is slower.

* Nocedal, J. "Updating Quasi-Newton Matrices with Limited Storage,"
  Mathematics of Computation, vol. 35, pp. 773-782 (1980).

* Liu, D. C. & Nocedal, J. "On the limited memory BFGS method for large scale
  optimization," Mathematical programming, Springer, vol. 45, pp. 503-528
  (1989).

* Byrd, R. H., Lu, P., Nocedal, J. & Zhu, C. "A limited memory algorithm for
  bound constrained optimization," SIAM Journal on Scientific Computing, SIAM,
  vol. 16, pp. 1190-1208 (1995).

* Benson, S. J. & Moré, J. J. "A limited memory variable metric method in
  subspaces and bound constrained optimization problems," in Subspaces and
  Bound Constrained Optimization Problems, (2001).

* Thiébaut, É. "Optimization issues in blind deconvolution algorithms,"
  Astronomical Data Analysis II, Proc. SPIE 4847, 174-183 (2002).

"""
function vmlmb{T}(fg!::Function, x0::T; keywords...)
    x = similar(x0)
    copy!(x, x0)
    vmlmb!(fg!, x; keywords...)
    return x
end

"""
`vmlmb!` is the in-place version of `vmlmb` (which to see):

     vmlmb!(fg!, x; mem=..., lower=..., upper=..., ftol=..., fmin=...)

finds a local minimizer of `f(x)` starting at `x` and stores the best solution
in `x`.
"""
function vmlmb!{T}(fg!::Function, x::T;
                   mem::Integer=5,
                   lower=-Inf, upper=+Inf,
                   blmvm::Bool=false,
                   fmin::Real=-Inf,
                   maxiter::Integer=typemax(Int),
                   maxeval::Integer=typemax(Int),
                   ftol::NTuple{2,Real}=(0.0,1e-8),
                   gtol::NTuple{2,Real}=(0.0,1e-6),
                   epsilon::Real=0.0,
                   verb::Bool=false,
                   printer::Function=print_iteration, output::IO=STDOUT,
                   lnsrch::AbstractLineSearch=defaultlinesearch)
    flags::UInt = (blmvm ? EMULATE_BLMVM : 0)
    vmlmb!(fg!, x, Int(mem), flags, lower, upper,
           Float(fmin), Int(maxiter), Int(maxeval),
           Float(ftol[1]), Float(ftol[2]),
           Float(gtol[1]), Float(gtol[2]),
           Float(epsilon), verb, printer, output, lnsrch)
end

# The real worker.
function vmlmb!{T}(fg!::Function, x::T, mem::Int, flags::UInt,
                   lower, upper,
                   fmin::Float, maxiter::Int, maxeval::Int,
                   fatol::Float, frtol::Float,
                   gatol::Float, grtol::Float,
                   epsilon::Float,
                   verb::Bool, printer::Function, output::IO,
                   lnsrch::AbstractLineSearch)

    @assert mem ≥ 1
    @assert maxiter ≥ 0
    @assert maxeval ≥ 1
    @assert fatol ≥ 0
    @assert frtol ≥ 0
    @assert gatol ≥ 0
    @assert grtol ≥ 0
    @assert 0 ≤ epsilon < 1

    const STPMIN = Float(1e-20)
    const STPMAX = Float(1e+20)

    # Determine the type of bounds and the emulated optimization method (0 for
    # L-BFGS, 1 for BLMVM or 2 for VMLMB).
    bounds::UInt = 0
    if isa(lower, Real)
        lo = Float(lower)
        if lower > -Inf
            bounds |= 1
        end
    elseif isa(lower, T)
        lo = lower
        bounds |= 1
    else
        error("invalid lower bound type")
    end
    if isa(upper, Real)
        hi = Float(upper)
        if upper < +Inf
            bounds |= 2
        end
    elseif isa(upper, T)
        hi = upper
        bounds |= 2
    else
        error("invalid upper bound type")
    end
    method = (bounds == 0 ? 0 : (flags & EMULATE_BLMVM) != 0 ? 1 : 2)
    if lnsrch == defaultlinesearch
        # Provide a suitable line search.
        if method == 0
            lnsrch = MoreThuenteLineSearch(ftol=1e-3, gtol=0.9, xtol= 0.1)
        else
            lnsrch = BacktrackingLineSearch(ftol=0.1, amin=0.1)
        end
    end

    mem = min(mem, length(x))
    reason::AbstractString = ""
    fminset::Bool = (! isnan(fmin) && fmin > -Inf)

    mark::Int = 1    # index of most recent step and gradient difference
    m::Int = 0       # number of memorized steps
    iter::Int = 0    # number of algorithm iterations
    eval::Int = 0    # number of objective function and gradient evaluations
    rejects::Int = 0 # number of rejected search directions

    f::Float = 0
    f0::Float = 0
    gd::Float = 0
    gd0::Float = 0
    stp::Float = 0
    stpmin::Float = 0
    stpmax::Float = 0
    smin::Float = 0
    smax::Float = 0
    gamma::Float = 1
    gnorm::Float = 0
    gtest::Float = 0

    # Variables for saving information about best point so far.
    beststp::Float = 0
    bestf::Float = 0
    bestgnorm::Float = 0

    # Allocate memory for the limited memory BFGS approximation of the inverse
    # Hessian.
    g = similar(x)     # gradient
    if method > 0
        p = similar(x) # projected gradient
    end
    d = similar(x)     # search direction
    S = Array(T, mem)  # memorized steps
    Y = Array(T, mem)  # memorized gradient differences
    for k in 1:mem
        S[k] = similar(x)
        Y[k] = similar(x)
    end
    rho = Array(Float, mem)
    alpha = Array(Float, mem)

    # Variable used to control the stage of the algorithm:
    #   * stage = 0 at start or restart;
    #   * stage = 1 during a line search;
    #   * stage = 2 if line search has converged;
    #   * stage = 3 if algorithm has converged;
    #   * stage = 4 if algorithm is terminated with a warning.
    stage::Int = 0

    while true

        # Compute value of function and gradient, register best solution so far
        # and check for convergence based on the gradient norm.
        if method > 0
            project_variables!(x, lo, hi, x)
        end
        f = fg!(x, g)
        eval += 1
        if method > 0
            project_direction!(p, lo, hi, x, -1, g)
        end
        if eval == 1 || f < bestf
            gnorm = norm2((method > 0 ? p : g))
            beststp = stp
            bestf = f
            bestgnorm = gnorm
            if eval == 1
                gtest = hypot(gatol, grtol*gnorm)
            end
            if gnorm ≤ gtest
                stage = 3
                if gnorm == 0
                    reason = "a stationary point has been found!"
                elseif method > 0
                    reason = "projected gradient sufficiently small"
                else
                    reason = "gradient sufficiently small"
                end
                if eval > 1
                    iter += 1
                end
            end
        end
        if fminset && f < fmin
            stage = 4
            reason = "f < fmin"
        end

        if stage == 1
            # Line search is in progress.
            if requires_derivative(lnsrch)
                if method > 0
                    gd = (inner(g, x) - inner(g, S[mark]))/stp
                else
                    gd = -inner(g, d)
                end
            end
            (stp, search) = iterate!(lnsrch, stp, f, gd)
            if ! search
                if check_status(lnsrch) == :CONVERGENCE
                    # Line search has converged.  Increment iteration counter
                    # and check for stopping condition.
                    iter += 1
                    delta = max(abs(f - f0), stp*abs(gd0))
                    if delta ≤ frtol*abs(f0)
                        stage = 3
                        reason = "frtol test satisfied"
                    elseif delta ≤ fatol
                        stage = 3
                        reason = "fatol test satisfied"
                    elseif iter ≥ maxiter
                        stage = 4
                        reason = "too many iterations"
                    elseif eval ≥ maxeval
                        stage = 4
                        reason = "too many evaluations"
                    else
                        stage = 2
                    end
                else
                    # Line seach terminated with a warning.
                    stage = 4
                    reason = get_reason(lnsrch)
                end
            end
        end
        if stage < 2 && eval ≥ maxeval
            stage = 4
            reason = "too many evaluations"
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
                    combine!(x, 1, S[mark], -stp, d)
                    if method > 0
                        project_variables!(x, lo, hi, x)
                    end
                else
                    gnorm = norm2((method > 0 ? p : g))
                end
            end

            # Print some information if requested and terminate algorithm if
            # stage ≥ 3.
            if verb
                printer(output, iter, eval, rejects, f, gnorm, stp)
            end
            if stage ≥ 3
                break
            end

            # Compute next search direction.
            if stage == 2
                # Update limited memory BFGS approximation of the inverse
                # Hessian with the effective step and gradient change.
                update!(S[mark], -1, x)
                update!(Y[mark], -1, (method == 1 ? p : g))
                if method < 2
                    rho[mark] = inner(Y[mark], S[mark])
                    if rho[mark] > 0
                        # The update is acceptable, compute the scale.
                        gamma = rho[mark]/inner(Y[mark], Y[mark])
                    end
                end
                m = min(m + 1, mem)

                # Compute search direction.
                if method < 2
                    copy!(d, g)
                    change = apply_lbfgs!(S, Y, rho, gamma, m, mark, d, alpha)
                else
                    copy!(d, p)
                    change = apply_lbfgs!(S, Y, rho, m, mark, d, alpha,
                                          get_free_variables(d))
                end
                if change
                    if method > 0
                        project_direction!(d, lo, hi, x, -1, d)
                    end
                    gd = -inner(g, d)
                    reject = ! sufficient_descent(gd, epsilon, gnorm, d)
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
                copy!(d, (method > 0 ? p : g))
                gd = -gnorm^2
            end

            # Save function value, variables and (projected) gradient at start
            # of line search.
            f0 = f
            gd0 = gd
            copy!(S[mark], x)
            copy!(Y[mark], (method == 1 ? p : g))

            # Choose initial step length.
            if stage == 2
                stp = 1
            else
                if fminset && fmin < f0
                    stp = 2*(fmin - f0)/gd0
                else
                    stp = 1/gnorm # FIXME: use a better scale
                end
            end
            if method > 0
                # Make sure the step is not longer than necessary.
                (smin, smax) = step_limits(lo, hi, x, -1, d)
                stp = min(stp, smax)
                stpmin = STPMIN*stp
                stpmax = min(STPMAX*stp, smax)
            else
                stpmin = STPMIN*stp
                stpmax = STPMAX*stp
            end

            # Initialize the line search.
            if ! start!(lnsrch, stp, f0, gd0, stpmin, stpmax)
                # Something wrong happens.
                stage = 4
                reason = get_reason(lnsrch)
                break
            end
            stage = 1 # line search is in progress

        end

        # Compute the new iterate.
        combine!(x, 1, S[mark], -stp, d)

    end

    # Algorithm finished.
    if verb
        color = (stage > 3 ? :red : :green)
        prefix = (stage > 3 ? "WARNING: " : "CONVERGENCE: ")
        print_with_color(color, output, "# ", prefix, reason, "\n")
    elseif stage > 3
        warn(reason)
    end
end

function check_status(lnsrch::AbstractLineSearch)
    # Check for errors.
    task = get_task(lnsrch)
    if task != :CONVERGENCE
        reason = get_reason(lnsrch)
        # FIXME: use a constant instead
        if task == :WARNING
            if reason == "rounding errors prevent progress"
                task = :CONVERGENCE
            end
        else
            error(reason)
        end
    end
    return task
end

function sufficient_descent{T}(gd::Float, ε::Float, gnorm::Float, d::T)
    ε > 0 ? (gd ≤ -ε*norm2(d)*gnorm) : gd < 0
end

function sufficient_descent(gd::Float, ε::Float, gnorm::Float, dnorm::Float)
    ε > 0 ? (gd ≤ -ε*dnorm*gnorm) : gd < 0
end

@doc """
    sufficient_descent(gd, ε, gnorm, d)

checks whether `d` is a sufficient descent direction (Zoutenjdik condition).
Argument `gd` is the scalar product `inner(g,d)` between the gradient `g` and
the direction `d`, `ε` is a nonnegative small value in the range `[0,1)` and
`gnorm` is the Euclidean norm of the gradient `g`.

If the Euclidean norm of `d` has already been computed, then

    sufficient_descent(gd, ε, gnorm, dnorm)

should be used instead with `dnorm = norm2(d)`.
""" sufficient_descent

function print_iteration(iter::Int, eval::Int, rejects::Int,
                         f::Float, gnorm::Float, step::Float)
    print_iteration(STDOUT, iter, eval, rejects, f, gnorm, step)
end

function print_iteration(io::IO, iter::Int, eval::Int, rejects::Int,
                         f::Float, gnorm::Float, step::Float)
    if iter == 0
        @printf(io, "#%s%s\n#%s%s\n",
                " ITER   EVAL   REJECTS",
                "          F(X)           ||G(X)||    STEP",
                "----------------------",
                "-------------------------------------------")
    end
    @printf(io, " %5d  %5d  %5d  %24.16E %9.2E %9.2E\n",
            iter, eval, rejects, f, gnorm, step)
end

#------------------------------------------------------------------------------

"""
One of the calls:

    modif = apply_lbfgs!(S, Y, rho, gamma, m, mark, v, alpha)
    modif = apply_lbfgs!(S, Y, rho, d,     m, mark, v, alpha)
    modif = apply_lbfgs!(S, Y, rho, H0!,   m, mark, v, alpha)

computes the matrix-vector product `H*v` where `H` is the limited memory
inverse BFGS approximation.  Operation is done *in-place*: on entry, argument
`v` contains the input vector `v`; on exit, argument `v` contains the
matrix-vector product `H*v`.  The returned value, `modif`, is a boolean
indicating whether the initial vector was modified (BFGS updates are skipped if
`rho[k] ≤ 0`).

The linear operator `H` depends on an initial approximation of the inverse
Hessian (see below), `m` steps `S[...]` and `m` gradient differences `Y[...]`.
The most recent step and gradient difference are stored in `S[mark]` and
`Y[mark]`, respectively.

Argument `rho` contains the inner products of the steps and the gradient
differences: `rho[k] = inner(S[k],Y[k])`.  On exit rho is unchanged.

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
function apply_lbfgs!{T}(S::Vector{T}, Y::Vector{T}, rho::Vector{Float},
                         gamma::Float, m::Int, mark::Int, v::T,
                         alpha::Vector{Float})
    @assert gamma > 0
    apply_lbfgs!(S, Y, rho, u -> scale!(u, gamma), m, mark, v, alpha)
end

function apply_lbfgs!{T}(S::Vector{T}, Y::Vector{T}, rho::Vector{Float},
                         d::T, m::Int, mark::Int, v::T,
                         alpha::Vector{Float})
    apply_lbfgs!(S, Y, rho, u -> multiply!(u, d, u), m, mark, v, alpha)
end

function apply_lbfgs!{T}(S::Vector{T}, Y::Vector{T}, rho::Vector{Float},
                         H0!::Function, m::Int, mark::Int, v::T,
                         alpha::Vector{Float})
    mem = min(length(S), length(Y), length(rho), length(alpha))
    @assert 1 ≤ m ≤ mem
    @assert 1 ≤ mark ≤ mem
    modif::Bool = false
    @inbounds begin
        k::Int = mark + 1
        for i in 1:m
            k = (k > 1 ? k - 1 : mem)
            if rho[k] > 0
                alpha[k] = inner(S[k], v)/rho[k]
                update!(v, -alpha[k], Y[k])
                modif = true
            end
        end
        if modif
            H0!(v)
            for i in 1:m
                if rho[k] > 0
                    beta::Float = inner(Y[k], v)/rho[k]
                    update!(v, alpha[k] - beta, S[k])
                end
                k = (k < mem ? k + 1 : 1)
            end
        end
    end
    return modif
end

function apply_lbfgs!{T}(S::Vector{T}, Y::Vector{T}, rho::Vector{Float},
                         m::Int, mark::Int, v::T,
                         alpha::Vector{Float}, sel)
    mem = min(length(S), length(Y), length(rho), length(alpha))
    @assert 1 ≤ m ≤ mem
    @assert 1 ≤ mark ≤ mem
    gamma::Float = 0
    @inbounds begin
        k::Int = mark + 1
        for i in 1:m
            k = (k > 1 ? k - 1 : mem)
            rho[k] = inner(sel, Y[k], S[k])
            if rho[k] > 0
                alpha[k] = inner(sel, S[k], v)/rho[k]
                update!(v, sel, -alpha[k], Y[k])
                if gamma == 0
                    gamma = rho[k]/inner(sel, Y[k], Y[k])
                end
            end
        end
        if gamma != 0
            scale!(v, gamma)
            for i in 1:m
                if rho[k] > 0
                    beta::Float = inner(sel, Y[k], v)/rho[k]
                    update!(v, sel, alpha[k] - beta, S[k])
                end
                k = (k < mem ? k + 1 : 1)
            end
        end
    end
    return (gamma != 0)
end

#------------------------------------------------------------------------------
end # module