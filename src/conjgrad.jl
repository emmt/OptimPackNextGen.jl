#
# conjgrad.jl --
#
# Linear conjugate gradients for TiPi.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#


# A NormalEquations stores the components of the normal equations.
type NormalEquations{T}
    lhs::SelfAdjointOperator{T}    # left hand side matrix
    rhs::T                         # right hand side vector
end

function conjgrad{T}(p::NormalEquations{T}, x0::T, tol, maxiter::Integer)
    conjgrad(p.lhs, p.rhs, x0, tol, maxiter)
end

function conjgrad!{T}(p::NormalEquations{T}, x::T, tol, maxiter::Integer)
    conjgrad!(p.lhs, p.rhs, x, tol, maxiter)
end

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
function conjgrad{T}(A::SelfAdjointOperator{T}, b::T, x0::T, tol,
                     maxiter::Integer)
    x = vcreate(x0)
    vcopy!(x, x0)
    conjgrad!(A, b, x, tol, maxiter)
    return x
end

function conjgrad!{T}(A::SelfAdjointOperator{T}, b::T, x::T, tol,
                      maxiter::Integer)
    # Initialization.
    p = vcreate(x)
    q = vcreate(x)
    r = vcreate(x)
    apply_direct!(r, A, x)
    vcombine!(r, 1, b, -1, r)
    rho::Float = 0
    rho0::Float = 0
    alpha::Float = 0
    gamma::Float = 0
    epsilon::Float = 0
    rho = vdot(r, r)
    if length(tol) == 1
        epsilon = tol[1]
    elseif length(tol) == 2
        epsilon = tol[1] + tol[2]*sqrt(rho)
    else
        error("bad tolerance")
    end

    # Conjugate gradient iterations.
    k = 0
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
            vcopy!(p, r)
        else
            vcombine!(p, 1, r, (rho/rho0), p)
        end
        apply_direct!(q, A, p)
        gamma = vdot(p, q)
        if gamma <= 0
            error("left-hand-side operator A is not positive definite")
            break
        end
        alpha = rho/gamma
        vupdate!(x, +alpha, p)
        vupdate!(r, -alpha, q)
        rho0 = rho
        rho = vdot(r, r)
    end
end
