#
# interpol.jl --
#
# Linear interpolation.
#
#------------------------------------------------------------------------------
#
# This file is part of TiPi.jl licensed under the MIT "Expat" License.
#
# Copyright (C) 2016, Éric Thiébaut &  Thibault Wanner.
#
#------------------------------------------------------------------------------

import Base.sparse

type Interpolator{T<:AbstractFloat,N}
    J::Vector{Int}
    A::Vector{T}
    nrows::Int
    ncols::Int
    width::Int
    dims::NTuple{N,Int} # dimensions of result
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
    const s::T = T(width)/T(2)
    const delta::T = T(step(grd))
    const offset::T = T(first(grd)) - delta

    J = Array(Int, nmax)
    A = Array(T, nmax)
    n::Int = 0
    @inbounds begin
        for i in 1:nrows
            t = (T(pos[i]) - offset)/delta
            kmin = floor(Int, t - s) + 1
            for k in kmin:kmin+width-1
                j = max(1, min(k, ncols))
                n += 1
                J[n] = j
                A[n] = ker(t - T(k))
            end
        end
    end
    @assert(n == nmax)
    return Interpolator(J, A, nrows, ncols, width, size(pos))
end

function call{T<:AbstractFloat,N}(op::Interpolator{T,N},
                                  x::AbstractArray{T,1})
    @assert(length(x) == op.ncols)
    y = Array(T, op.dims)
    @assert(length(y) == op.nrows)
    const width::Int = op.width
    @inbounds begin
        for i in 1:op.nrows
            k0::Int = width*(i - 1)
            s::T = zero(T)
            for k in k0+1:k0+width
                j = op.J[k]
                a = op.A[k]
                s += a*x[j]
            end
            y[i] = s
        end
    end
    return y
end

function call{T<:AbstractFloat,N,R<:Real}(op::Interpolator{T,N},
                                          x::AbstractArray{R,1})
    op(convert(Array{T,1}, x))
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
