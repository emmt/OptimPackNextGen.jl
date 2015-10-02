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

# A LinearProblem stores the components of the normal equations.
type LinearProblem{T<:FloatingPoint,N}
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

function conjgrad!{T<:FloatingPoint,N}(A::LinearOperator, b::Array{T,N}, x::Array{T,N},
                                       tol, maxiter)
    # Initialization.
    @assert(size(b) == size(x))
    p = Array(T, size(x))
    q = Array(T, size(x))
    r = Array(T, size(x))
    apply!(A, r, x)
    axpby!(r, 1, b, -1, r)
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
            axpby!(p, 1, r, (rho/rho0), p)
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

###############################################################################
# LINEAR ALGEBRA

# update!(x,a,p) - Update "vector" x by doing: x += a*p.
function update!{T<:FloatingPoint,N}(x::Array{T,N}, a::T, p::Array{T,N})
    @assert(size(x) == size(p))
    @simd for i in 1:length(x)
        @inbounds x[i] += a*p[i]
    end
end

# inner(x,y) - Compute the inner product of "vectors" x and y.
function inner{T<:FloatingPoint,N}(x::Array{T,N}, y::Array{T,N})
    @assert(size(x) == size(y))
    s::T = 0
    @simd for i in 1:length(x)
        @inbounds s += x[i]*y[i]
    end
    return s
end

function axpby!{T<:FloatingPoint,N}(dst::Array{T,N},
                                    alpha::Real, x::Array{T,N},
                                    beta::Real,  y::Array{T,N})
    @assert(size(x) == size(dst))
    @assert(size(y) == size(dst))
    const n = length(dst)
    const a::T = alpha
    const b::T = beta
    if a == zero(T)
        if b == zero(T)
            @simd for i in 1:n
                @inbounds dst[i] = zero(T)
            end
        elseif b == one(T)
            @simd for i in 1:n
                @inbounds dst[i] = y[i]
            end
        elseif b == -one(T)
            @simd for i in 1:n
                @inbounds dst[i] = -y[i]
            end
        else
            @simd for i in 1:n
                @inbounds dst[i] = b*y[i]
            end
        end
    elseif a == one(T)
        if b == zero(T)
            @simd for i in 1:n
                @inbounds dst[i] = x[i]
            end
        elseif b == one(T)
            @simd for i in 1:n
                @inbounds dst[i] = x[i] + y[i]
            end
        elseif b == -one(T)
            @simd for i in 1:n
                @inbounds dst[i] = x[i] - y[i]
            end
        else
            @simd for i in 1:n
                @inbounds dst[i] = x[i] + b*y[i]
            end
        end
    elseif a == -one(T)
        if b == zero(T)
            @simd for i in 1:n
                @inbounds dst[i] = -x[i]
            end
        elseif b == one(T)
            @simd for i in 1:n
                @inbounds dst[i] = y[i] - x[i]
            end
        elseif b == -one(T)
            @simd for i in 1:n
                @inbounds dst[i] = -x[i] - y[i]
            end
        else
            @simd for i in 1:n
                @inbounds dst[i] = b*y[i] - x[i]
            end
        end
    else
        if b == zero(T)
            @simd for i in 1:n
                @inbounds dst[i] = a*x[i]
            end
        elseif b == one(T)
            @simd for i in 1:n
                @inbounds dst[i] = a*x[i] + y[i]
            end
        elseif b == -one(T)
            @simd for i in 1:n
                @inbounds dst[i] = a*x[i] - y[i]
            end
        else
            @simd for i in 1:n
                @inbounds dst[i] = a*x[i] + b*y[i]
            end
        end
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
