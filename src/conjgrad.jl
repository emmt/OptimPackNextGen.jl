#
# conjgrad.jl --
#
# Linear conjugate gradients for TiPi.
#
#------------------------------------------------------------------------------
#
# This file is part of TiPi.jl licensed under the MIT "Expat" License.
#
# Copyright (C) 2015, Éric Thiébaut & Jonathan Léger.
#
#------------------------------------------------------------------------------

abstract LinearOperator

import TiPi.Algebra: inner, update!, combine!

# A LinearProblem stores the components of the normal equations.
type LinearProblem{T<:AbstractFloat,N}
    lhs::LinearOperator            # left hand side matrix
    rhs::Array{T,N}                # right hand side vector
end

function apply!(op::LinearOperator, q, p)
    error("this function must be specialized")
end

conjgrad(p::LinearProblem, x0, tol, maxiter) = conjgrad(p.lhs, p.rhs, x0, tol, maxiter)
conjgrad!(p::LinearProblem, x, tol, maxiter) = conjgrad!(p.lhs, p.rhs, x, tol, maxiter)

# conjgrad - conjugate gradient algorithm.
#
# This routine iteratively solves the linear system:
#
#     A.x = b
#
# Arguments:
#    A   - The left-hand-side operator.
#    b   - The right-hand-side vector of the normal equations.
#    x0  - The initial guess for the solution `x`.
#    tol - The tolerance(s) for convergence, can be one or two values: `atol`
#           (as if `rtol` = 0) or [`atol`, `rtol`] where `atol` and `rtol` are
#           the absolute and relative tolerances.  Convergence occurs when the
#           Euclidean norm of the residuals is less or equal the largest of
#           `atol` and `rtol` times the Eucliden norm of the initial
#           residuals.
#    maxiter - The maximum number of iterations.
#
function conjgrad(A::LinearOperator, b, x0, tol, maxiter)
    x = copy(x0)
    conjgrad!(A, b, x, tol, maxiter)
    return x
end

function conjgrad!{T<:AbstractFloat,N}(A::LinearOperator, b::Array{T,N}, x::Array{T,N},
                                       tol, maxiter)
    # Initialization.
    @assert(size(b) == size(x))
    p = Array(T, size(x))
    q = Array(T, size(x))
    r = Array(T, size(x))
    apply!(A, r, x)
    combine!(r, 1, b, -1, r)
    rho = inner(r, r)
    if length(tol) == 1
        epsilon = tol[1]
    elseif length(tol) == 2
        epsilon = max(tol[1], tol[2]*sqrt(rho))
    else
        error("bad tolerance")
    end

    # Conjugate gradient iterations.
    k = 0
    rho0 = zero(T)
    while true
        k += 1
        if sqrt(rho) <= epsilon
            # Normal convergence.
            break
        elseif k > maxiter
            warn("too many ($(maxiter)) conjugate gradient iterations")
            break
        end
        if k == 1
            # First iteration.
            copy!(p, r)
        else
            combine!(p, 1, r, (rho/rho0), p)
        end
        apply!(A, q, p)
        gamma = inner(p, q)
        if gamma <= zero(T)
            error("left-hand-side operator A is not positive definite")
            break
        end
        alpha = rho/gamma
        update!(x, +alpha, p)
        update!(r, -alpha, q)
        rho0 = rho
        rho = inner(r, r)
    end
end

# Local Variables:
# mode: Julia
# tab-width: 8
# indent-tabs-mode: nil
# fill-column: 79
# coding: utf-8
# ispell-local-dictionary: "american"
# End:
