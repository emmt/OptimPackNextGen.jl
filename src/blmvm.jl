#
# blmvm.jl -
#
# Implement Benson & Moré BLMVM algorithm in Julia.
#
#------------------------------------------------------------------------------
#
# This file is part of TiPi.jl which is licensed under the MIT "Expat" License:
#
# Copyright (C) 2015, Éric Thiébaut.
#
#------------------------------------------------------------------------------

module BLMVM

using TiPi.Algebra
using TiPi.ConvexSets

# initial_step(x, d, slen) --
#
#   Return initial step length for the first iteration or after a restart.
#   X are the current variables, D is the search direction, SLEN=[ALEN,RLEN]
#   where ALEN and RLEN are an absolute and relative step length (ALEN > 0 and
#   RLEN >= 0).
#
#   The result is: A/||D|| where ||D|| is the Euclidean norm of D and
#
#       A = RLEN*||X|| if RLEN*||X|| > 0
#         = ALEN       otherwise
#
function initial_step{T,N}(x::Array{T,N}, d::Array{T,N}, slen::NTuple{2})
    @assert(size(x) == size(d))
    dnorm = norm2(d)
    len1::Cdouble = slen[1]
    len2::Cdouble = slen[2]
    if len2 > 0
        len2 *= norm2(x)
    end
    (len2 > 0 ? len2 : len1)/dnorm
end

# FIXME: scalars should be stored as Cdouble
# FIXME: add a savememory option
# FIXME: add a savebest option
function blmvm{T<:AbstractFloat}(fg::Function, x::Array{T}, m::Integer,
                                 dom::BoxedSet{T};
                                 maxiter::Integer=-1,
                                 gtol=(0.0, 1e-4),
                                 slen=(1.0, 0.0),
                                 sftol=1e-4,
                                 verb::Integer=0)
    # Type for scalars (use at least double precision).
    Scalar = promote_type(T,Cdouble)

    # Check number of corrections to memorize.
    m = int(m)
    m < 1 && error("bad number of variable metric corrections")

    # Check options.
    #if (! is_void(xmin)) eq_nocopy, xmin, double(xmin)
    #if (! is_void(xmax)) eq_nocopy, xmax, double(xmax)
    #if (is_void(sftol)) sftol = 1e-4
    #if (is_void(maxiter)) maxiter = -1
    #if (is_void(slen)) slen = [1.0, 0.0]
    #if (is_void(gtol)) {
    #  gtol = [0.0, 1e-4]
    #} else if (identof(gtol) <= Y_DOUBLE && numberof(gtol) <= 2
    #           && min(gtol) >= 0.0) {
    #  if (numberof(gtol) == 2) {
    #    gtol = double(gtol)
    #  } else {
    #    gtol = [gtol(1), 0.0]
    #  }
    #} else {
    #  error, "bad parameter GTOL"
    #}

    # Allocate arrays for L-BFGS operator.
    S = Vector(Array{T}, m)
    Y = Vector(Array{T}, m)
    for i in 1:m
        S[i] = Array(T, size(x))
        Y[i] = Array(T, size(x))
    end
    beta = Vector(T, m)
    rho  = Vector(T, m)
    mp::Int = 0   # actual number of saved pairs
    mark::Int = 0 # total number of saved pair since start

    # The following closure returns the index where is stored the
    # (mark-j)-th correction pair.  Argument j must be in the inclusive
    # range 0:mp with mp the actual number of saved corrections.  At any
    # moment, 0 ≤ mp ≤ mark; thus mark - j ≥ 0.
    slot(j::Int) = (mark - j)%m + 1

    # Declare local variables and allocate workspaces.
    f::Scalar                # function value at x
    g    = Array(T, size(x)) # gradient at x
    x0   = Array(T, size(x)) # origin of line search
    f0::Scalar               # function value at x0
    g0   = Array(T, size(x)) # gradient at x0
    gp   = Array(T, size(x)) # projected gradient
    d    = Array(T, size(x)) # search direction
    temp = Array(T, size(x)) # temporary array
    gtest::Scalar            # gradient-based threshold for convergence
    gpnorm::Scalar           # Euclidean norm of the projected gradient
    alpha::Scalar = 0        # step length
    const GAIN = convert(Scalar, 1/2) # backtracking gain
    const ZERO = zero(Scalar)
    const ONE  = one(Scalar)
    sty::Scalar              # inner product <s,y>

    # Start the iterations of the algorithm.
    #
    # state = 0, if a line search is in progress;
    #         1, if a new iterate is available;
    #         2, if the algorithm has converged.
    state::Int = 1
    evaluations::Int = 0
    restarts::Int = 0
    iterations::Int = 0
    msg = nothing
    t0 = time_ns()*1e-9
    while true
        # Make sure X is feasible, compute function and gradient at X.
        project_variables!(x, dom, x)
        f = fg(x, g)
        evaluations += 1

        # Compute projected gradient and check for global convergence.
        project_gradient!(gp, dom, x, g)
        gpnorm = norm2(gp)
        if evaluations == 1
            gtest = gtol[1] + gtol[2]*gpnorm
        end
        if gpnorm <= gtest
            # Algorithm has converged.
            if state == 0
                iterations += 1
            end
            state = 2
        elseif state == 0
            # Line search is in progress. FIXME: re-use temp = x - x0 to update LBFGS
            combine!(temp, 1, x, -1, x0)
            if f <= f0 + sftol*inner(g0, temp)
                # Line search has converged, a new iterate is available.
                iterations += 1
                if maxiter >= 0 && iterations >= maxiter
                    msg = "WARNING: too many iterations"
                    state = 2
                else
                    state = 1
                end
            end
        end
        if verb > 0 && (state == 2 || (state == 1 && (iterations%verb) == 0))
            t = time_ns()*1e-9
            if evaluations == 1
                println("#  ITER     EVAL   TIME (s)            PENALTY           GRADIENT        STEP")
                println("-------------------------------------------------------------------------------")
            end
            @printf(" %6d  %6d  %9.3f  %24.16e  %12.6e  %12.6e\n",
                    iterations, evaluations, t - t0, f, gpnorm, alpha)
        end
        if state == 2
            # Algorithm terminated.
            if msg != nothing
                println(msg)
            end
            return x
        elseif state == 0
            # Previous step was too long.
            alpha *= GAIN
        else
            # A new search direction is required.
            if iterations >= 1
                # Update L-BFGS approximation of the Hessian.
                k = slot(0)
                combine!(S[k], ONE, x,  -ONE, x0) # FIXME: already done in TEMP
                combine!(Y[k], ONE, gp, -ONE, gp0)
                sty = inner(S[k], Y[k])
                if sty <= ZERO
                    # Skip update (may result in loosing one correction pair).
                    mp = min(mp, m - 1)
                else
                    # Update mark and number of saved corrections.
                    gamma = sty/inner(Y[k], Y[k])
                    rho[k] = ONE/sty
                    mark += 1
                    mp = min(mp + 1, m)
                end
            end
            if mp >= 1
                # Apply the L-BFGS two-loop recursion to compute a search
                # direction.
                scale!(d, -ONE, g)
                for j in 1:+1:mp
                    k = slot(j)
                    beta[k] = rho[k]*inner(d, S[k])
                    update!(d, -beta[k], Y[k])
                end
                scale!(d, gamma)
                for j in mp:-1:1
                    k = slot(j)
                    update!(d, beta[k] - rho[k]*inner(d, Y[k]), S[k])
                end
                project_direction!(temp, dom, x, d)
                if inner(temp, g) >= 0.0
                    # Not a descent direction.
                    restarts += 1
                    mp = 0
                else
                    alpha = ONE
                end
            end
            if mp < 1
                # Use steepest descent.
                scale!(d, -ONE, g)
                alpha = initial_step(x, d, slen)
            end

            # Start line search.
            state = 0
            alpha = shortcut_step(alpha, dom, x, d)
            # FIXME: alpha may be zero
            f0 = f
            copy!(x0, x)
            copy!(g0, g)
            copy!(gp0, gp)
        end
        combine!(x, ONE, x0, alpha, d)
    end
end

end # module

# Local Variables:
# mode: Julia
# tab-width: 8
# indent-tabs-mode: nil
# fill-column: 79
# coding: utf-8
# ispell-local-dictionary: "american"
# End:
