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
# - skip updates such that rho <= 0
# - TODO: implement restarts
# - TODO: add other convergence criteria
# - TODO: better initial step

module QuasiNewton

using ..Algebra
using ..LineSearch

# Use the same floating point type for scalars as in TiPi.
import ..Float

export lbfgs, lbfgs!

"""
## L-BFGS: limited memory BFGS method

    x = lbfgs(fg!, x0; mem=..., frtol=..., fatol=..., fmin=...)

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

* `frtol` specifies the relative error desired in the function.  Convergence
  occurs if the estimate of the relative error between `f(x)` and `f(xsol)`,
  where `xsol` is a local minimizer, is less than `frtol`.

* `fatol` specifies the absolute error desired in the function.  Convergence
   occurs if the estimate of the absolute error between `f(x)` and `f(xsol)`,
   where `xsol` is a local minimizer, is less than `fatol`.


* `fmin` specifies a lower bound for the function.  The subroutine exits with a
  warning if `f(x) < fmin`.

* `verb` specifies whether to print iteration information (`verb = false`, by
  default).

* `printer` can be set with a user defined function to print iteration
  information, its signature is:
  ```
  printer(io::IO, iter::Integer, eval::Integer, restart::Integer,
          f::Real, gnorm::Real, stp::Real)
  ```
  where `io` is the output stream, `iter` the iteration number (`iter = 0` for
  the starting point), `eval` is the number of calls to `fg!`, `restart` is the
  number or restarts of the method, `f` and `gnorm` are the value of the
  function and norm of the gradient at the current point, `stp` is the length
  of the step to the current point.

* `output` specifies the output stream for printing information (`output =
  STDOUT` by default).

* `lnsrch` specifies the method to use for line searches (the default
   line search is `MoreThuenteLineSearch`).


### History

The limited memory BFGS method (L-BFGS) was first described by Nocedal (1980)
who dubbed it SQN.  In the MINPACK-2 project (1995), the FORTRAN routine VMLM
implements this method.  The numerical performances of L-BFGS have been studied
by Liu and Nocedal (1989) who proved that it is globally convergent for
unfiformly convex problmens with a R-linear rate of convergence.  They provided
the FORTRAN code LBFGS.

* Nocedal, J. "Updating Quasi-Newton Matrices with Limited Storage,"
  Mathematics of Computation, vol. 35, pp. 773-782 (1980).

* Liu, D. C. & Nocedal, J. "On the limited memory BFGS method for large scale
  optimization," Mathematical programming, Springer, vol. 45, pp. 503-528
  (1989).

"""
function lbfgs{T}(fg!::Function, x0::T; keywords...)
    x = similar(x0)
    copy!(x, x0)
    lbfgs!(fg!, x; keywords...)
    return x
end

"""

`lbfgs!` is the in-place version of `lbfgs` (which to see):

     lbfgs!(fg!, x; mem=..., frtol=..., fatol=..., fmin=...)

finds a local minimizer of `f(x)` starting at `x` and stores the best solution
so far in `x`.

"""
function lbfgs!{T}(fg!::Function, x::T; mem::Integer=5, frtol::Real=1e-7,
                   fatol::Real=0, fmin::Real=-Inf, verb::Bool=false,
                   printer::Function=print_iteration, output::IO=STDOUT,
                   lnsrch::AbstractLineSearch=MoreThuenteLineSearch(ftol=1e-3,
                                                                    gtol=0.9,
                                                                    xtol= 0.1))
    @assert(mem ≥ 1)
    @assert(fatol ≥ 0)
    @assert(frtol ≥ 0)

    reason::AbstractString = ""
    fminset::Bool = (! isnan(fmin) && fmin > -Inf)

    mem = min(mem, length(x))
    mark::Int = 1
    iter::Int = 0
    eval::Int = 0

    f::Float = 0
    f0::Float = 0
    gd::Float = 0
    gd0::Float = 0
    stp::Float = 0
    stpmin::Float = 0
    stpmax::Float = 0
    gamma::Float = 0    # initial scaling

    # Allocate memory
    g = similar(x)
    d = similar(x)
    S = Array(T, mem)
    Y = Array(T, mem)
    for k in 1:mem
        S[k] = similar(x)
        Y[k] = similar(x)
    end
    rho = Array(Float, mem)
    alpha = Array(Float, mem)

    # Setup line search method.
    #sftol::Float = 1e-3
    sgtol::Float = 0.9 # FIXME:
    #sxtol::Float = 0.1
    #lnsrch = MoreThuenteLineSearch(ftol=sftol, gtol=sgtol, xtol=sxtol)

    # Variable used to control the stage of the algorithm:
    #   * stage = 0 at start or restart
    #   * stage = 1 during a line search
    #   * stage = 2 if line search has converged
    #   * stage = 3 if algorithm is finished (convergence or other reasons)
    stage::Int = 0

    while true

        if stage ≤ 1
            # Compute value of function and gradient.
            f = fg!(x, g)
            eval += 1
            if fminset && f < fmin
                stage = 4
                reason = "f < fmin"
            end
        end

        if stage == 1
            # Line search is in progress.
            if requires_derivative(lnsrch)
                gd = -inner(g, d)
            end
            (stp, search) = iterate!(lnsrch, stp, f, gd)
            if ! search
                if check_status(lnsrch) == :CONVERGENCE
                    # Line search has converged.  Set stage to trigger
                    # computing a new search direction and increment iteration
                    # counter.
                    stage = 2
                    iter += 1
                else
                    # Something wrong occured.
                    # FIXME: revert to best solution so far
                    stage = 4
                    reason = get_reason(lnsrch)
                end
            end
        end

        if stage != 1
            # Initial step or line search has converged.
            gnorm = norm2(g)

            # Check for global convergence.
            if gnorm ≤ 0
                stage = 3
                reason = "a stationary point has been found"
            end
            if stage == 2
                delta = max(abs(f - f0), stp*abs(gd0))
                if delta ≤ frtol*abs(f0)
                    stage = 3
                    reason = "frtol test satisfied"
                elseif delta ≤ fatol
                    stage = 3
                    reason = "fatol test satisfied"
                end
            end

            # Print some information if requested.
            if verb
                printer(output, iter, eval, 0, f, gnorm, stp)
            end
            if stage ≥ 3
                break
            end

            # Compute next search direction.
            if stage == 2
                # Compute the step and gradient change.
                update!(S[mark], -1, x)
                update!(Y[mark], -1, g)
                rho[mark] = inner(Y[mark], S[mark])

                # Compute the scale.
                if rho[mark] > 0
                    gamma = rho[mark]/inner(Y[mark], Y[mark])
                else
                    gamma = 1 # FIXME: keep previous value?
                end

                # Compute d = H*g.
                copy!(d, g)
                mp = min(mem, iter)
                if ! apply_lbfgs!(S, Y, rho, gamma, mp, mark, d, alpha)
                    # The steepest descent is being used.
                    stage = 0
                end
                mark = (mark < mem ? mark + 1 : 1)
            else
                # Initial point or restarting, use the steepest descent.
                copy!(d, g)
            end

            # Initialize the line search.
            f0 = f
            gd0 = -inner(g, d)
            if stage == 2
                stp = 1
            else
                if fminset && fmin < f0
                    stp = 2*(fmin - f0)/gd0
                else
                    stp = 1/gnorm # FIXME: use a better scale
                end
            end
            stpmin = 0
            if fminset && fmin < f0
                # FIXME: This was in the original MINPACK-2 version but it does
                # not work so well.
                stpmax = (fmin - f0)/(sgtol*gd0)
                stp = min(stp, stpmax)
            else
                stpmax = 1e20*stp
            end
            copy!(S[mark], x) # save x0
            copy!(Y[mark], g) # save g0
            search = start!(lnsrch, stp, f0, gd0, stpmin, stpmax)
            if search
                # Line search is in progress.
                stage = 1
            else
                # Something wrong happens.
                stage = 4
                reason = get_reason(lnsrch)
            end

        end

        if stage == 1
            # Compute the new iterate.
            combine!(x, 1, S[mark], -stp, d)
        end

    end

    # Algorithm finished.
    if verb
        color = (stage > 3 ? :red : :blue)
        prefix = (stage > 3 ? "WARNING: " : "CONVERGENCE: ")
        print_with_color(color, output,
                         "# ", prefix, reason)
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

function print_iteration(iter::Int, eval::Int, restart::Int,
                         f::Float, gnorm::Float, step::Float)
    print_iteration(STDOUT, iter, eval, restart, f, gnorm, step)
end

function print_iteration(io::IO, iter::Int, eval::Int, restart::Int,
                         f::Float, gnorm::Float, step::Float)
    if iter == 0
        @printf(io, "#%s%s\n#%s%s\n",
                " ITER   EVAL  RESTARTS",
                "          F(X)           ||G(X)||    STEP",
                "----------------------",
                "-------------------------------------------")
    end
    @printf(io, " %5d  %5d  %5d  %24.16E %9.2E %9.2E\n",
            iter, eval, restart, f, gnorm, step)
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

#------------------------------------------------------------------------------
end # module
