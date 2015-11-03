#
# smooth.jl --
#
# Quadratic smoothness cost functions for TiPi.
#
#------------------------------------------------------------------------------
#
# This file is part of TiPi.jl licensed under the MIT "Expat" License.
#
# Copyright (C) 2015, Éric Thiébaut & Jonathan Léger.
#
#------------------------------------------------------------------------------

import Base.call

type QuadraticSmoothness{N} <: AbstractCost
end

function cost{T<:AbstractFloat}(alpha::Real,
                                ::QuadraticSmoothness{2},
                                x::Array{T,2})
    alpha == 0 && return zero(Cdouble)
    err::Cdouble = 0
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
    return convert(Cdouble,alpha)*err
end

function cost!{T<:AbstractFloat}(alpha::Real,
                                 ::QuadraticSmoothness{2},
                                 x::Array{T,2},
                                 g::Array{T,2},
                                 clr::Bool)
    @assert(size(g) == size(x))
    clr && fill!(g, zero(T))
    alpha == 0 && return zero(Cdouble)
    err::Cdouble = 0
    const dims = size(x)
    const dim1 = dims[1]
    const dim2 = dims[2]
    const beta = convert(T, alpha + alpha)
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
    return convert(Cdouble,alpha)*err
end

# Local Variables:
# mode: Julia
# tab-width: 8
# indent-tabs-mode: nil
# fill-column: 79
# coding: utf-8
# ispell-local-dictionary: "american"
# End:
