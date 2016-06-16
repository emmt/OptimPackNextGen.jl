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
    @inbounds begin
        k0 = 0
        for i in 1:nrows
            s = zero(T)
            for k in k0+1:k0+width
                j = J[k]
                1 ≤ j ≤ ncols || error("corrupted interpolator table")
                s += A[k]*src[j]
            end
            dst[i] = s
            k0 += width
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
    const width = op.width
    const nrows = op.nrows
    const ncols = op.ncols
    A = op.A
    J = op.J
    is(dst, src) && error("operation cannot be done in-place")
    fill!(dst, zero(T))
    @inbounds begin
        k0 = 0
        for i in 1:nrows
            s = src[i]
            for k in k0+1:k0+width
                j = J[k]
                1 ≤ j ≤ ncols || error("corrupted interpolator table")
                dst[j] += A[k]*s
            end
            k0 += width
        end
    end
end

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
