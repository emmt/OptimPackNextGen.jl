#
# quasinewton.jl --
#
# Limited memory quasi-Newton methods for OptimPack.
#
# -----------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2015-2018, Éric Thiébaut.
# <https://github.com/emmt/OptimPackNextGen.jl>.
#

# Improvements:
# - skip updates such that rho <= 0;
# - check for sufficient descent condition and implement restarts;
# - add other convergence criteria;
# - in case of early stop, revert to best solution so far;
# - better initial step than 1/vnorm2(g);

module QuasiNewton

export
    vmlmb,
    vmlmb!,
    EMULATE_BLMVM

using Compat
using Compat.Printf

using LazyAlgebra
using ...OptimPackNextGen
using OptimPackNextGen.LineSearches
using OptimPackNextGen.SimpleBounds

# Use the same floating point type for scalars as in OptimPack.
import OptimPackNextGen.Float

const EMULATE_BLMVM = UInt(1)

# All scalar computations are done in double precision.  Thus manage to have
# `vdot` and `vnorm2` return a double precision result.

function vdot(x::AbstractArray{Float,N},
              y::AbstractArray{Float,N}) where {N}
    return LazyAlgebra.vdot(x, y)
end

function vdot(x::AbstractArray{<:Real,N},
              y::AbstractArray{<:Real,N}) where {N}
    return Float(LazyAlgebra.vdot(x, y))
end

function vdot(w::Union{AbstractArray{Float,N},AbstractVector{Int}},
              x::AbstractArray{Float,N},
              y::AbstractArray{Float,N}) where {N}
    return LazyAlgebra.vdot(w, x, y)
end

function vdot(w::Union{AbstractArray{<:Real,N},AbstractVector{Int}},
              x::AbstractArray{<:Real,N},
              y::AbstractArray{<:Real,N}) where {N}
    return Float(LazyAlgebra.vdot(w, x, y))
end

vnorm2(x::AbstractArray{Float,N}) where {N} = LazyAlgebra.vnorm2(x)
vnorm2(x::AbstractArray{<:Real,N}) where {N} = Float(LazyAlgebra.vnorm2(x))

"""
## VMLMB: limited memory BFGS method with optional bounds

    x = vmlmb(fg!, x0; mem=..., lower=..., upper=..., ftol=..., fmin=...)

computes a local minimizer of a function of several variables by a limited
memory variable metric method.  The caller provides a function `fg!` to compute
the value and the gradient of the function as follows:

    f = fg!(x, g)

where `x` are the current variables, `f` is the value of the function at `x`
and the contents of `g` has to be overwritten with the gradient at `x` (when
`fg!` is called, `g` is already allocated as `g = vcreate(x0)`).  Argument `x0`
gives the initial approximation of the variables (its contents is left
unchanged).  The best solution found so far is returned in `x`.

The following keywords are available:

* `mem` specifies the amount of storage.

* `ftol` is a tuple of two nonnegative reals specifying respectively the
  absolute and relative errors desired in the function.  Convergence occurs if
  the absolute error between `f(x)` and `f(xsol)` is less than `ftol[1]` or if
  the estimate of the relative error between `f(x)` and `f(xsol)`, where `xsol`
  is a local minimizer, is less than `ftol[2]`.  By default, `ftol =
  (0.0,1e-8)`.

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

* `output` specifies the output stream for printing information (`stdout` is
  used by default).

* `lnsrch` specifies the method to use for line searches (the default
   line search is `MoreThuenteLineSearch`).

* `lower` and `upper` specify the lower and upper bounds for the variables.
   The bound can be a scalar to indicate that all variables have the same bound
   value.  If the lower (resp. upper) bound is unspecified or set to `±∞`, the
   variables are assumed to be unbounded below (resp. above).  If no bounds are
   set, VMLMB amounts to an unconstrained limited memory BFGS method (L-BFGS).

* `blmvm` can be set true to emulate the BLMVM algorithm of Benson and Moré.
  This option has no effects for an uncostrained problem.


### History

The VMLMB algorithm in
[OptimPackNextGen](https://github.com/emmt/OptimPackNextGen.jl) provides a pure
Julia implementation of the original method (Thiébaut, 2002) with some
improvements and the capability to emulate L-BFGS and BLMVM methods.

The limited memory BFGS method (L-BFGS) was first described by Nocedal (1980)
who dubbed it SQN.  The method is implemented in MINPACK-2 (1995) by the
FORTRAN routine VMLM.  The numerical performances of L-BFGS have been studied
by Liu and Nocedal (1989) who proved that it is globally convergent for
unfiformly convex problems with a R-linear rate of convergence.  They provided
the FORTRAN code LBFGS.  The BLMVM and VMLMB algorithms were proposed by Benson
and Moré (2001) and Thiébaut (2002) to account for separable bound constraints
on the variables.  These two latter methods are rather different than L-BFGS-B
by Byrd at al. (1995) which has more overheads and is slower.

* J. Nocedal, "*Updating Quasi-Newton Matrices with Limited Storage*" in
  Mathematics of Computation, vol. 35, pp. 773-782 (1980).

* D.C. Liu & J. Nocedal, "*On the limited memory BFGS method for large scale
  optimization*" in Mathematical programming, vol. 45, pp. 503-528
  (1989).

* R.H. Byrd, P. Lu, J. Nocedal, & C. Zhu, "*A limited memory algorithm for
  bound constrained optimization*" in SIAM Journal on Scientific Computing,
  vol. 16, pp. 1190-1208 (1995).

* S.J. Benson & J.J. Moré, "*A limited memory variable metric method in
  subspaces and bound constrained optimization problems*" in Subspaces and
  Bound Constrained Optimization Problems (2001).

* É. Thiébaut, "*Optimization issues in blind deconvolution algorithms*" in
  Astronomical Data Analysis II, Proc. SPIE 4847, pp. 174-183 (2002).

"""
vmlmb(fg!::Function, x0; kwds...) = vmlmb!(fg!, vcopy(x0); kwds...)

"""
`vmlmb!` is the in-place version of `vmlmb` (which to see):

     vmlmb!(fg!, x; mem=..., lower=..., upper=..., ftol=..., fmin=...) -> x

finds a local minimizer of `f(x)` starting at `x` and stores the best solution
in `x`.
"""
function vmlmb!(fg!::Function, x::T;
                mem::Integer = min(5, length(x)),
                lower::Union{Real,T} = -Inf,
                upper::Union{Real,T} = +Inf,
                blmvm::Bool = false,
                fmin::Real = -Inf,
                maxiter::Integer = typemax(Int),
                maxeval::Integer = typemax(Int),
                ftol::NTuple{2,Real} = (0.0, 1e-8),
                gtol::NTuple{2,Real} = (0.0, 1e-6),
                epsilon::Real = 0.0,
                verb::Bool = false,
                printer::Function = print_iteration,
                output::IO = stdout,
                lnsrch::Union{LineSearch{Float},Nothing} = nothing) where {T}
    # Determine which options are used.
    flags::UInt = (blmvm ? EMULATE_BLMVM : 0)

    # Determine the type of bounds.
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

    # Determine the optimization method (0 for L-BFGS, 1 for BLMVM or 2 for
    # VMLMB).
    method = (bounds == 0 ? 0 : (flags & EMULATE_BLMVM) != 0 ? 1 : 2)

    # Provide a default line search method if needed.
    if isa(lnsrch, Nothing)
        if method == 0
            ls = MoreThuenteLineSearch(Float; ftol=1e-3, gtol=0.9, xtol=0.1)
        else
            ls = MoreToraldoLineSearch(Float; ftol=1e-3, gamma=(0.1,0.5))
        end
    else
        ls = lnsrch
    end

    # Call the real method.
    _vmlmb!(fg!, x, Int(mem), flags, lo, hi, bounds, method,
            Float(fmin), Int(maxiter), Int(maxeval),
            Float(ftol[1]), Float(ftol[2]),
            Float(gtol[1]), Float(gtol[2]),
            Float(epsilon), verb, printer, output, ls)
end

# The real worker.
function _vmlmb!(fg!::Function, x::T, mem::Int, flags::UInt,
                 lo::Union{Float, T},
                 hi::Union{Float, T},
                 bounds::UInt,
                 method::Int,
                 fmin::Float, maxiter::Int, maxeval::Int,
                 fatol::Float, frtol::Float,
                 gatol::Float, grtol::Float,
                 epsilon::Float,
                 verb::Bool, printer::Function, output::IO,
                 lnsrch::LineSearch{Float}) where {T}

    @assert mem ≥ 1
    @assert maxiter ≥ 0
    @assert maxeval ≥ 1
    @assert fatol ≥ 0
    @assert frtol ≥ 0
    @assert gatol ≥ 0
    @assert grtol ≥ 0
    @assert 0 ≤ epsilon < 1

    STPMIN = Float(1e-20)
    STPMAX = Float(1e+20)

    sel = Array{Int}(undef, 0)
    if bounds != 0
        sizehint!(sel, length(x))
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
    g = vcreate(x) # ------------> gradient
    if method > 0
        p = vcreate(x) # --------> projected gradient
    end
    d = vcreate(x) # ------------> search direction
    S = Array{T}(undef, mem) # --> memorized steps
    Y = Array{T}(undef, mem) # --> memorized gradient differences
    for k in 1:mem
        S[k] = vcreate(x)
        Y[k] = vcreate(x)
    end
    rho = Array{Float}(undef, mem)
    alpha = Array{Float}(undef, mem)

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
            project_variables!(x, x, lo, hi)
        end
        f = fg!(x, g)
        eval += 1
        if method > 0
            project_direction!(p, x, lo, hi, -1, g)
        end
        if eval == 1 || f < bestf
            gnorm = vnorm2((method > 0 ? p : g))
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
            if usederivatives(lnsrch)
                if method > 0
                    gd = (vdot(g, x) - vdot(g, S[mark]))/stp
                else
                    gd = -vdot(g, d)
                end
            end
            task = iterate!(lnsrch, stp, f, gd)
            if task == :SEARCH
                stp = getstep(lnsrch)
            elseif task == :CONVERGENCE
                # Line search has converged.  Increment iteration counter
                # and check for stopping condition.
                iter += 1
                delta = max(abs(f - f0), stp*abs(gd0))
                if delta ≤ fatol
                    stage = 3
                    reason = "fatol test satisfied"
                elseif delta ≤ frtol*abs(f0)
                    stage = 3
                    reason = "frtol test satisfied"
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
                # Line seach terminated with a warningor an error.
                stage = 4
                reason = getreason(lnsrch)
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
                    vcombine!(x, 1, S[mark], -stp, d)
                    if method > 0
                        project_variables!(x, x, lo, hi)
                    end
                else
                    gnorm = vnorm2((method > 0 ? p : g))
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
                vupdate!(S[mark], -1, x)
                vupdate!(Y[mark], -1, (method == 1 ? p : g))
                if method < 2
                    rho[mark] = vdot(Y[mark], S[mark])
                    if rho[mark] > 0
                        # The update is acceptable, compute the scale.
                        gamma = rho[mark]/vdot(Y[mark], Y[mark])
                    end
                end
                m = min(m + 1, mem)

                # Compute search direction.
                if method < 2
                    vcopy!(d, g)
                    change = apply_lbfgs!(S, Y, rho, gamma, m, mark, d, alpha)
                else
                    vcopy!(d, p)
                    change = apply_lbfgs!(S, Y, rho, m, mark, d, alpha,
                                          get_free_variables!(sel, d))
                end
                # FIXME: speedup project_direction with the free vars.?
                if change
                    if method > 0
                        project_direction!(d, x, lo, hi, -1, d)
                    end
                    gd = -vdot(g, d)
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
                vcopy!(d, (method > 0 ? p : g))
                gd = -gnorm^2
            end

            # Save function value, variables and (projected) gradient at start
            # of line search.
            f0 = f
            gd0 = gd
            vcopy!(S[mark], x)
            vcopy!(Y[mark], (method == 1 ? p : g))

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
                (smin, smax) = step_limits(x, lo, hi, -1, d)
                stp = min(stp, smax)
                stpmin = STPMIN*stp
                stpmax = min(STPMAX*stp, smax)
            else
                stpmin = STPMIN*stp
                stpmax = STPMAX*stp
            end

            # Initialize the line search.
            task = start!(lnsrch, f0, gd0, stp; stpmin=stpmin, stpmax=stpmax)
            if task != :SEARCH
                # Something wrong happens.
                stage = 4
                reason = getreason(lnsrch)
                break
            end
            stage = 1 # line search is in progress

        end

        # Compute the new iterate.
        vcombine!(x, 1, S[mark], -stp, d)

    end

    # Algorithm finished.
    if verb
        color = (stage > 3 ? :red : :green)
        prefix = (stage > 3 ? "WARNING: " : "CONVERGENCE: ")
        printstyled(output, "# ", prefix, reason, "\n"; color=color)
        #elseif stage > 3
        #@warn(reason)
    end
    return x
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


function print_iteration(iter::Integer, eval::Integer, rejects::Integer,
                         f::Real, gnorm::Real, step::Real)
    print_iteration(stdout, iter, eval, rejects, f, gnorm, step)
end

function print_iteration(io::IO, iter::Integer, eval::Integer, rejects::Integer,
                         f::Real, gnorm::Real, step::Real)
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
differences: `rho[k] = vdot(S[k],Y[k])`.  On exit rho is unchanged.

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
function apply_lbfgs!(S::Vector{T}, Y::Vector{T}, rho::Vector{Float},
                      gamma::Float, m::Int, mark::Int, v::T,
                      alpha::Vector{Float}) where {T}
    @assert gamma > 0
    apply_lbfgs!(S, Y, rho, u -> vscale!(u, gamma), m, mark, v, alpha)
end

function apply_lbfgs!(S::Vector{T}, Y::Vector{T}, rho::Vector{Float},
                      d::T, m::Int, mark::Int, v::T,
                      alpha::Vector{Float}) where {T}
    apply_lbfgs!(S, Y, rho, u -> vproduct!(u, d, u), m, mark, v, alpha)
end

function apply_lbfgs!(S::Vector{T}, Y::Vector{T}, rho::Vector{Float},
                      H0!::Function, m::Int, mark::Int, v::T,
                      alpha::Vector{Float}) where {T}
    mem = min(length(S), length(Y), length(rho), length(alpha))
    @assert 1 ≤ m ≤ mem
    @assert 1 ≤ mark ≤ mem
    modif::Bool = false
    @inbounds begin
        k::Int = mark + 1
        for i in 1:m
            k = (k > 1 ? k - 1 : mem)
            if rho[k] > 0
                alpha[k] = vdot(S[k], v)/rho[k]
                vupdate!(v, -alpha[k], Y[k])
                modif = true
            end
        end
        if modif
            H0!(v)
            for i in 1:m
                if rho[k] > 0
                    beta::Float = vdot(Y[k], v)/rho[k]
                    vupdate!(v, alpha[k] - beta, S[k])
                end
                k = (k < mem ? k + 1 : 1)
            end
        end
    end
    return modif
end

function apply_lbfgs!(S::Vector{T}, Y::Vector{T}, rho::Vector{Float},
                      m::Int, mark::Int, v::T,
                      alpha::Vector{Float}, sel) where {T}
    mem = min(length(S), length(Y), length(rho), length(alpha))
    @assert 1 ≤ m ≤ mem
    @assert 1 ≤ mark ≤ mem
    gamma::Float = 0
    @inbounds begin
        k::Int = mark + 1
        for i in 1:m
            k = (k > 1 ? k - 1 : mem)
            rho[k] = vdot(sel, Y[k], S[k])
            if rho[k] > 0
                alpha[k] = vdot(sel, S[k], v)/rho[k]
                vupdate!(v, sel, -alpha[k], Y[k])
                if gamma == 0
                    gamma = rho[k]/vdot(sel, Y[k], Y[k])
                end
            end
        end
        if gamma != 0
            vscale!(v, gamma)
            for i in 1:m
                if rho[k] > 0
                    beta::Float = vdot(sel, Y[k], v)/rho[k]
                    vupdate!(v, sel, alpha[k] - beta, S[k])
                end
                k = (k < mem ? k + 1 : 1)
            end
        end
    end
    return (gamma != 0)
end

#------------------------------------------------------------------------------
end # module
