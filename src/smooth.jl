#
# smooth.jl --
#
# Quadratic smoothness cost functions for TiPi.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#

import Base.call

immutable QuadraticSmoothness <: AbstractCost end

function cost{T<:AbstractFloat}(alpha::Float, ::QuadraticSmoothness,
                                x::Array{T,2})
    alpha == zero(Float) && return zero(Float)
    err::Float = 0
    dims = size(x)
    dim1 = dims[1]
    dim2 = dims[2]
    for i2 in 1:dim2
        i2n = (i2 < dim2 ? i2 + 1 : dim2)
        for i1 in 1:dim1
            i1n = (i1 < dim1 ? i1 + 1 : dim1)
            dx1 = x[i1n,i2] - x[i1,i2]
            dx2 = x[i1,i2n] - x[i1,i2]
            err += dx1*dx1 + dx2*dx2
        end
    end
    return alpha*err
end

function cost!{T<:AbstractFloat}(alpha::Float, ::QuadraticSmoothness,
                                 x::Array{T,2}, g::Array{T,2}, clr::Bool)
    @assert size(g) == size(x)
    clr && vfill!(g, 0)
    alpha == zero(Float) && return zero(Float)
    err::Float = 0
    const dims = size(x)
    const dim1 = dims[1]
    const dim2 = dims[2]
    const beta = T(alpha + alpha)
    for i2 in 1:dim2
        i2n = (i2 < dim2 ? i2 + 1 : dim2)
        for i1 in 1:dim1
            i1n = (i1 < dim1 ? i1 + 1 : dim1)
            dx1 = x[i1n,i2] - x[i1,i2]
            dx2 = x[i1,i2n] - x[i1,i2]
            err += dx1*dx1 + dx2*dx2
            dx1 *= beta
            dx2 *= beta
            g[i1,i2] -= (dx1 + dx2)
            g[i1n,i2] += dx1
            g[i1,i2n] += dx2
        end
    end
    return alpha*err
end
