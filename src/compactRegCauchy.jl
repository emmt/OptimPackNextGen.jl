

#
# compactRegCauchy.jl --
#
# cost functions implementing a compactness a priori using the Cauchy function for TiPi.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#

import Base.call

immutable CompactRegCauchy <: AbstractCost
    a::Cdouble
    center::NTuple{2,Cdouble}
    pixDim::NTuple{2,Cdouble}
    imgDim::NTuple{2,Int}
    w::Array{Cdouble,2}
    #offset::NTuple{N,Cdouble} #translation
    #rotMat::Array{Cdouble,2} #an N*N matrix that is the rotation
    #isotropic::Bool #maybe later
    function HyperbolicEdgePreserving(_a::Cdouble,
                                      _center::NTuple{2,Cdouble},
                                      _pixDim::NTuple{2,Cdouble},
                                      _imgDim::NTuple{2,Int})
        a > 0.0 || error("radius must be strictly positive")
        prod(_pixDim) > 0.0 || error("pixel dimensions must be strictly positive")
        prod(_imgDim) > 0 || error("image dimensions must be strictly positive")
        #create the weight map
        _w=Array(Cdouble,_imgDim);
        a2=a*a
        for i2 in 1:dim2
            i2f2=param.pixDim[2]*Float(i2)*param.pixDim[2]*Float(i2)
            for i1 in 1:dim1
                i1f=param.pixDim[1]*Float(i1)
                _w[i1,i2] = (1.0 + 2.0*(i1f*i1f + i2f2)/a2)
            end
        end
        new(_a,_center,_pixDim,_imgDim,_w)
    end
end

function cost{T<:AbstractFloat}(alpha::Float, param::CompactRegCauchy,
                                x::Array{T,2})
    alpha == zero(Float) && return zero(Float)
    return alpha*sum(param.w.*x.*x)
end

function cost!{T<:AbstractFloat}(alpha::Float, param::CompactRegCauchy,
                                 x::Array{T,2}, g::Array{T,2}, clr::Bool)
    @assert size(g) == size(x)
    clr && vfill!(g, 0)
    alpha == zero(Float) && return zero(Float)
    if clr
        g[:] = param.w[:].*x[:]
    else
        g[:] += param.w[:].*x[:]
    end
    return alpha*sum(param.w.*x.*x)
end

#err::Float = 0
#dims = size(x)
#dim1 = dims[1]
#dim2 = dims[2]
#a2=param.a*param.a
#for i2 in 1:dim2
#    i2f2=param.pixDim[2]*Float(i2)*param.pixDim[2]*Float(i2)
#    for i1 in 1:dim1
#        i1f=param.pixDim[1]*Float(i1)
#        err += (1.0 + 2.0*(i1f*i1f + i2f2)/a2)*x[i1,i2]
#    end
#end
    