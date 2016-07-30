#
# interp.jl --
#
# Linear interpolation.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#

import Base: sparse, eltype

immutable Interpolator{T<:AbstractFloat,N} <: LinearOperator{AbstractArray{T,N},
                                                             AbstractArray{T,1}}
    J::Vector{Int}
    A::Vector{T}
    nrows::Int
    ncols::Int
    width::Int
    dims::NTuple{N,Int} # dimensions of result
end

eltype{T,N}(op::Interpolator{T,N}) = T
output_size{T,N}(op::Interpolator{T,N}) = op.dims
input_size{T,N}(op::Interpolator{T,N}) = (op.ncols,)

coefficients(op::Interpolator) = op.A
columns(op::Interpolator) = op.J
function rows(op::Interpolator)
    const nrows = op.nrows
    const width = op.width
    @assert length(op.A) == width*nrows
    @assert length(op.J) == width*nrows
    I = Array(Int, width*nrows)
    K = 1:width
    for i in 1:nrows
        for k in K
            @inbounds I[k] = i
        end
        K += width
    end
    return I
end

"""
# Linear interpolator

A linear interpolator is created by:

    op = Interpolator(ker, grd, pos)

which yields a linear interpolator suitable for interpolating with the kernel
`ker` a function sampled on the grid `grd` at positions `pos`.

Then `y = call(op, x)` or `y = op(x)` yields the interpolated values for
interpolation weights `x`.  The shape of `y` is the same as that of `pos`.

Formally, this amounts to computing:

    y[i] = sum_j ker((pos[i] - grd[j])/step(grd))*x[j]

with `step(grd)` the (constant) step size between the nodes of the grid `grd`
and `grd[j]` the `j`-th position of the grid.

"""
function Interpolator{T<:AbstractFloat}(ker::Kernels.Kernel{T},
                                        grd::Range,
                                        pos::AbstractArray)
    const nrows = length(pos)
    const ncols = length(grd)
    const width = length(ker)
    const nmax = nrows*width
    const s::T = T(width/2)
    const delta::T = T(step(grd))
    const offset::T = T(first(grd)) - delta

    J = Array(Int, nmax)
    A = Array(T, nmax)
    n::Int = 0
    @inbounds begin
        for i in eachindex(pos)
            t = (T(pos[i]) - offset)/delta
            k0 = floor(Int, t - s)
            for k in k0+1:k0+width
                j = max(1, min(k, ncols))
                n += 1
                J[n] = j
                A[n] = ker(t - T(k))
            end
        end
    end
    @assert n == nmax
    return Interpolator(J, A, nrows, ncols, width, size(pos))
end

function checksize{T,N}(op::Interpolator{T,N},
                        out::AbstractArray{T,N},
                        inp::AbstractArray{T,1})
    number = op.nrows*op.width
    length(op.J) == number || error("bad number of interpolator indices")
    length(op.A) == number || error("bad number of interpolator weights")
    @assert length(inp) == op.ncols
    @assert size(out) == op.dims
    length(out) == op.nrows || error("bug: bad number of \"rows\"")
end

function apply_direct{T,N}(op::Interpolator{T,N},
                           src::AbstractArray{T,1})
    dst = Array(T, output_size(op))
    apply_direct!(dst, op, src)
    return dst
end

function apply_direct!{T,N}(dst::AbstractArray{T,N},
                            op::Interpolator{T,N},
                            src::AbstractArray{T,1})
    checksize(op, dst, src)
    const width = op.width
    const nrows = op.nrows
    const ncols = op.ncols
    A = op.A
    J = op.J
    K = 1:width
    @inbounds begin
        for i in 1:nrows
            s = zero(T)
            for k in K
                j = J[k]
                1 ≤ j ≤ ncols || error("corrupted interpolator table")
                s += A[k]*src[j]
            end
            dst[i] = s
            K += width
        end
    end
end

function apply_adjoint{T,N}(op::Interpolator{T,N},
                            src::AbstractArray{T,N})
    dst = Array(T, input_size(op))
    apply_adjoint!(dst, op, src)
    return dst
end

function apply_adjoint!{T,N}(dst::AbstractArray{T,1},
                             op::Interpolator{T,N},
                             src::AbstractArray{T,N})
    checksize(op, src, dst)
    is(dst, src) && error("operation cannot be done in-place")
    fill!(dst, zero(T))
    const width = op.width
    const nrows = op.nrows
    const ncols = op.ncols
    A = op.A
    J = op.J
    K = 1:width
    @inbounds begin
        for i in 1:nrows
            s = src[i]
            for k in K
                j = J[k]
                1 ≤ j ≤ ncols || error("corrupted interpolator table")
                dst[j] += A[k]*s
            end
            K += width
        end
    end
end

# Convert to a sparse matrix.
function sparse(op::Interpolator)
    const nrows = op.nrows
    const ncols = op.ncols
    const width = op.width
    I = Array(Int, nrows*width)
    n::Int = 0
    @inbounds begin
        for i in 1:nrows
            k0::Int = width*(i - 1)
            for k in k0+1:k0+width
                I[k] = i
            end
        end
    end
    sparse(I, op.J, op.A, nrows, ncols)
end

function AtWA{T,N}(op::Interpolator{T,N}, wgt::AbstractArray{T,N})
    ncols = op.ncols
    AtWA!(Array(T, ncols, ncols), op, wgt)
end

function AtA{T,N}(op::Interpolator{T,N})
    ncols = op.ncols
    AtA!(Array(T, ncols, ncols), op)
end

function AtA!{T,N}(dst::AbstractArray{T,2}, op::Interpolator{T,N})
    const nrows = op.nrows
    const ncols = op.ncols
    const width = op.width
    @assert size(dst) == (ncols, ncols)
    fill!(dst, zero(T))
    A::Vector{T} = coefficients(op)
    J::Vector{Int} = columns(op)
    K = 1:width
    @assert length(J) == length(A)
    @inbounds begin
        for i in 1:nrows
            for k in K
                1 ≤ J[k] ≤ ncols || error("corrupted interpolator table")
            end
            for k1 in K
                j1 = J[k1]
                fact = A[k1]
                for k2 in K
                    j2 = J[k2]
                    dst[j1,j2] += A[k2]*fact
                end
            end
            K += width
        end
    end

    return dst
end

function AtWA!{T,N}(dst::AbstractArray{T,2}, op::Interpolator{T,N},
                    wgt::AbstractArray{T,N})
    const nrows = op.nrows
    const ncols = op.ncols
    const width = op.width
    @assert size(dst) == (ncols, ncols)
    @assert size(wgt) == output_size(op)
    fill!(dst, zero(T))
    A::Vector{T} = coefficients(op)
    J::Vector{Int} = columns(op)
    K = 1:width
    @assert length(J) == length(A)
    @inbounds begin
        for i in 1:nrows
            for k in K
                1 ≤ J[k] ≤ ncols || error("corrupted interpolator table")
            end
            wgt_i = wgt[i]
            for k1 in K
                j1 = J[k1]
                fact = wgt_i*A[k1]
                for k2 in K
                    j2 = J[k2]
                    dst[j1,j2] += A[k2]*fact
                end
            end
            K += width
        end
    end
    return dst
end

# Default regularization levels.
const RGL_EPS = 1e-9
const RGL_MU = 0.0

doc"""

    fit(A, y[, w][; epsilon=1e-9, mu=0.0]) -> x

performs a linear fit of `y` by the model `A*x` with `A` a linear interpolator.
The returned value `x` minimizes:

    sum(w.*(A*x - y).^2)

where `w` are some weights.  If `w` is not specified, all weights are assumed
to be equal to one; otherwise `w` must be an array of nonnegative values and of
same size as `y`.

Keywords `epsilon` and `mu` may be specified to regularize the solution and
minimize:

    sum(w.*(A*x - y).^2) + rho*(epsilon*norm(x)^2 + mu*norm(D*x)^2)

where `D` is a finite difference operator, `rho` is the maximum diagonal
element of `A'*diag(w)*A` and `norm` is the Euclidean norm.

"""
function fit{T,N}(A::Interpolator{T,N}, y::AbstractArray{T,N};
                  epsilon::Real = RGL_EPS, mu::Real = RGL_MU)
    @assert size(y) == output_size(A)
    @assert size(w) == size(y)

    # Compute RHS vector A'*W*y with W = diag(w).
    rhs = A'*(w.*y)

    # Compute LHS matrix A'*W*A with W = diag(w).
    lhs = AtWA(A, w)

    # Regularize a bit.
    regularize!(lhs, epsilon, mu)

    # Solve the linear equations.
    cholfact!(lhs,:U,Val{true})\rhs
end

function fit{T,N}(A::Interpolator{T,N}, y::AbstractArray{T,N};
                  epsilon::Real = RGL_EPS, mu::Real = RGL_MU)
    @assert size(y) == output_size(A)
    @assert size(w) == size(y)

    # Compute RHS vector A'*y.
    rhs = A'*y

    # Compute LHS matrix A'*W*A with W = diag(w).
    lhs = AtA(A)

    # Regularize a bit.
    regularize!(lhs, epsilon, mu)

    # Solve the linear equations.
    cholfact!(lhs,:U,Val{true})\rhs
end

doc"""
    regularize(A, ϵ, μ) -> R

regularizes the symmetric matrix `A` to produce the matrix:

    R = A + ρ*(ϵ*I + μ*D'*D)

where `I` is the identity, `D` is a finite difference operator and `ρ` is the
maximum diagonal element of `A`.  The in-place version:

    regularize!(A, ϵ, μ) -> A

stores the regularized matrix in `A` (and returns it).

"""
function regularize end

regularize{T<:AbstractFloat}(A::AbstractArray{T,2}, args...) =
    regularize!(copy!(Array(T, size(A)), A), args...)

function regularize!{T<:AbstractFloat}(A::AbstractArray{T,2},
                                       eps::Real = RGL_EPS,
                                       mu::Real = RGL_MU)
    regularize!(A, T(eps), T(mu))
end

function regularize!{T<:AbstractFloat}(A::AbstractArray{T,2}, eps::T, mu::T)
    local rho::T
    @assert eps ≥ zero(T)
    @assert mu ≥ zero(T)
    @assert size(A,1) == size(A,2)
    const n = size(A,1)
    if eps > zero(T) || mu > zero(T)
        rho = A[1,1]
        for j in 2:n
            d = A[j,j]
            rho = max(rho, d)
        end
        rho > zero(T) || error("we have a problem!")
    end
    if eps > zero(T)
        q = eps*rho
        for j in 1:n
            A[j,j] += q
        end
    end
    if mu > zero(T)
        q = rho*mu
        if n ≥ 2
            r = q + q
            A[1,1] += q
            A[2,1] -= q
            for i in 2:n-1
                A[i-1,i] -= q
                A[i,  i] += r
                A[i+1,i] -= q
            end
            A[n-1,n] -= q
            A[n,  n] += q
        elseif n == 1
            A[1,1] += q
        end
    end
    return A
end

@doc @doc(regularize) regularize!
