#
# finitediff.jl --
#
# Finite differences.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#
module FiniteDifferences

# FIXME: generalize to more dimensions
# FIXME: use @generated, @nexpr, etc. instead of *stupid* hand generated code
# FIXME: add different weights along different dimensions
# FIXME: add other directions than principal ones

using Base.Cartesian

using TiPi
importall TiPi.Algebra

export D, Dt, DtD, D!, Dt!, DtD!,
       OperatorD, OperatorDtD


function D{T<:Real,N}(x::AbstractArray{T,N})
    D!(Array(T, N, size(x)...), x)
end

function Dt{T<:Real,N}(x::AbstractArray{T,N})
    Dt!(Array(T, size(x)[2:end]), x)
end

function DtD{T<:Real,N}(x::AbstractArray{T,N})
    DtD!(Array(T, size(x)), x)
end

immutable OperatorD{E,F} <: LinearOperator{E,F} end
immutable OperatorDtD{E} <: LinearOperator{E,E} end

OperatorD{T,N}(::Array{T,N}) = OperatorD(T, N)
OperatorD{T}(::Type{T}, N::Integer) = OperatorD{Array{T,N+1},Array{T,N}}()

apply_direct{E,F}(::OperatorD{E,F}, x::F) = D(x)
apply_adjoint{E,F}(::OperatorD{E,F}, x::E) = Dt(x)

OperatorDtD{T,N}(::Array{T,N}) = OperatorDtD(T, N)
OperatorDtD{T}(::Type{T}, N::Integer) = OperatorDtD{Array{T,N}}()

apply_direct{E}(::OperatorDtD{E}, x::E) = DtD(x)
apply_adjoint{E}(::OperatorDtD{E}, x::E) = DtD(x)

#------------------------------------------------------------------------ N = 1

function D!{T<:Real}(y::AbstractArray{T,2},
                     x::AbstractArray{T,1},
                     incr::Bool=false)
    @assert size(y) == (1, size(x)...)
    imax = length(x)
    range = 1:imax
    if incr
        @inbounds @simd for i in range
            i1 = min(i + 1, imax)
            y[1,i] += x[i1] - x[i]
        end
    else
        @inbounds @simd for i in range
            i1 = min(i + 1, imax)
            y[1,i] = x[i1] - x[i]
        end
    end
    return y
end

function Dt!{T<:Real}(y::AbstractArray{T,1},
                      x::AbstractArray{T,2},
                      incr::Bool=false)
    @assert size(x) == (1, size(y)...)
    imax = length(y)
    range = 1:imax
    incr || fill!(y, zero(T))
    @inbounds @simd for i in range
        i1 = min(i + 1, imax)
        d1 = x[1,i]
        y[i]  -= d1
        y[i1] += d1
    end
    return y
end

function DtD!{T<:Real}(y::AbstractArray{T,1},
                       x::AbstractArray{T,1},
                       incr::Bool=false)
    @assert size(x) == size(y)
    imax = length(x)
    range = 1:imax
    incr || fill!(y, zero(T))
    @inbounds @simd for i in range
        i1 = min(i + 1, imax)
        d1 = x[i1] - x[i]
        y[i]  -= d1
        y[i1] += d1
    end
    return y
end

#------------------------------------------------------------------------ N = 2

function D!{T<:Real}(y::AbstractArray{T,3},
                     x::AbstractArray{T,2},
                     incr::Bool=false)
    @assert size(y) == (2, size(x)...)
    range = CartesianRange(size(x))
    imax = last(range)
    s1 = CartesianIndex((1,0))
    s2 = CartesianIndex((0,1))
    if incr
        @inbounds @simd for i in range
            i1 = min(i + s1, imax)
            i2 = min(i + s2, imax)
            xi = x[i]
            d1 = x[i1] - xi
            d2 = x[i2] - xi
            y[1,i] += d1
            y[2,i] += d2
        end
    else
        @inbounds @simd for i in range
            i1 = min(i + s1, imax)
            i2 = min(i + s2, imax)
            xi = x[i]
            d1 = x[i1] - xi
            d2 = x[i2] - xi
            y[1,i] = d1
            y[2,i] = d2
        end
    end
    return y
end

function Dt!{T<:Real}(y::AbstractArray{T,2},
                      x::AbstractArray{T,3},
                      incr::Bool=false)
    @assert size(x) == (2, size(y)...)
    range = CartesianRange(size(y))
    imax = last(range)
    s1 = CartesianIndex((1,0))
    s2 = CartesianIndex((0,1))
    incr || fill!(y, zero(T))
    @inbounds @simd for i in range
        i1 = min(i + s1, imax)
        i2 = min(i + s2, imax)
        d1 = x[1,i]
        d2 = x[2,i]
        y[i] -= d1 + d2
        y[i1] += d1
        y[i2] += d2
    end
    return y
end

function DtD!{T<:Real}(y::AbstractArray{T,2},
                       x::AbstractArray{T,2},
                       incr::Bool=false)
    @assert size(x) == size(y)
    range = CartesianRange(size(y))
    imax = last(range)
    s1 = CartesianIndex((1,0))
    s2 = CartesianIndex((0,1))
    incr || fill!(y, zero(T))
    @inbounds @simd for i in range
        i1 = min(i + s1, imax)
        i2 = min(i + s2, imax)
        xi = x[i]
        d1 = x[i1] - xi
        d2 = x[i2] - xi
        y[i] -= d1 + d2
        y[i1] += d1
        y[i2] += d2
    end
    return y
end

#------------------------------------------------------------------------ N = 3

function D!{T<:Real}(y::AbstractArray{T,4},
                     x::AbstractArray{T,3},
                     incr::Bool=false)
    @assert size(y) == (3, size(x)...)
    range = CartesianRange(size(x))
    imax = last(range)
    s1 = CartesianIndex((1,0,0))
    s2 = CartesianIndex((0,1,0))
    s3 = CartesianIndex((0,0,1))
    if incr
        @inbounds @simd for i in range
            i1 = min(i + s1, imax)
            i2 = min(i + s2, imax)
            i3 = min(i + s3, imax)
            xi = x[i]
            d1 = x[i1] - xi
            d2 = x[i2] - xi
            d3 = x[i3] - xi
            y[1,i] += d1
            y[2,i] += d2
            y[3,i] += d3
        end
    else
        @inbounds @simd for i in range
            i1 = min(i + s1, imax)
            i2 = min(i + s2, imax)
            i3 = min(i + s3, imax)
            xi = x[i]
            d1 = x[i1] - xi
            d2 = x[i2] - xi
            d3 = x[i3] - xi
            y[1,i] = d1
            y[2,i] = d2
            y[3,i] = d3
        end
    end
    return y
end

function Dt!{T<:Real}(y::AbstractArray{T,3},
                      x::AbstractArray{T,4},
                      incr::Bool=false)
    @assert size(x) == (3, size(y)...)
    range = CartesianRange(size(y))
    imax = last(range)
    s1 = CartesianIndex((1,0,0))
    s2 = CartesianIndex((0,1,0))
    s3 = CartesianIndex((0,0,1))
    incr || fill!(y, zero(T))
    @inbounds @simd for i in range
        i1 = min(i + s1, imax)
        i2 = min(i + s2, imax)
        i3 = min(i + s3, imax)
        d1 = x[1,i]
        d2 = x[2,i]
        d3 = x[3,i]
        y[i] -= d1 + d2 + d3
        y[i1] += d1
        y[i2] += d2
        y[i3] += d3
    end
    return y
end

function DtD!{T<:Real}(y::AbstractArray{T,3},
                       x::AbstractArray{T,3},
                       incr::Bool=false)
    @assert size(x) == size(y)
    range = CartesianRange(size(y))
    imax = last(range)
    s1 = CartesianIndex((1,0,0))
    s2 = CartesianIndex((0,1,0))
    s3 = CartesianIndex((0,0,1))
    incr || fill!(y, zero(T))
    @inbounds @simd for i in range
        i1 = min(i + s1, imax)
        i2 = min(i + s2, imax)
        i3 = min(i + s3, imax)
        xi = x[i]
        d1 = x[i1] - xi
        d2 = x[i2] - xi
        d3 = x[i3] - xi
        y[i] -= d1 + d2 + d3
        y[i1] += d1
        y[i2] += d2
        y[i3] += d3
    end
    return y
end

end #module
