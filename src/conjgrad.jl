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
immutable NormalEquations{T}
    lhs::SelfAdjointOperator{T}    # left hand side matrix
    rhs::T                         # right hand side vector
end

doc"""
# Linear Conjugate Gradient Algorithm

This method iteratively solves the linear system:

    A⋅x = b

where `A` is the so-called left-hand-side operator (which must be positive
definite) of the normal equations, `b` is the right-hand-side vector of the
normal equations and `x` are the unknowns.

To use this method, initial value, say `x0`, for the variables may be provided:

    x = conjgrad(A, b, x0)

the intial variables may be used to store the final result:

    conjgrad!(A, b, x) -> x

will overwrite the contents of the initial variables `x` with the final result
and will return it.  If no initial variables are specified, the default is to
start with all variables set to zero.

Alternatively, the input of the algorithm may be an instance, say `eq`, of
`NormalEquations`:

    x = conjgrad(eq, x0)
    conjgrad!(eq, x) -> x


There are several keywords to control the algorithm:

* `tol` specifies the tolerances for convergence, it is a tuple of two values
  `(atol, rtol)` where `atol` and `rtol` are the absolute and relative
  tolerances.  Convergence occurs when the Euclidean norm of the residuals is
  less or equal the largest of `atol` and `rtol` times the Eucliden norm of the
  initial residuals.  By default, `tol = (0.0, 1.0e-3)`.

* `maxiter` specifies the maximum number of iterations.

"""
function conjgrad end

function conjgrad{T}(A::SelfAdjointOperator{T}, b::T; kws...)
    conjgrad!(A, b, vfill!(vcreate(b), 0); kws...)
end

function conjgrad{T}(A::SelfAdjointOperator{T}, b::T, x0::T; kws...)
    conjgrad!(A, b, vcopy(x0); kws...)
end

function conjgrad!{T}(A::SelfAdjointOperator{T}, b::T, x::T;
                      tol::NTuple{2,Real}=(0.0,1e-3),
                      maxiter::Integer=typemax(Int))
    # Initialization.
    @assert tol[1] ≥ 0
    @assert tol[2] ≥ 0
    if vdot(x,x) > 0 # cheap trick to check whether x is non-zero
        r = apply_direct!(vcreate(x), A, x)
        vcombine!(r, 1, b, -1, r)
    else
        r = vcopy(b)
    end
    rho::Float = vdot(r, r)
    rho0::Float = 0
    alpha::Float = 0
    gamma::Float = 0
    epsilon::Float = epsilon = Float(tol[1]) + Float(tol[2])*sqrt(rho)

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
            p = vcopy(r)
            q = vcreate(x)
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
    return x
end

function conjgrad{T}(eq::NormalEquations{T}, args...; kws...)
    conjgrad(eq.lhs, eq.rhs, args...; kws...)
end

function conjgrad!{T}(eq::NormalEquations{T}, args...; kws...)
    conjgrad!(eq.lhs, eq.rhs, args...; kws...)
end
