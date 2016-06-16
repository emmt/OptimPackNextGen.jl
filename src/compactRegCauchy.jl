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
    theta0::Cdouble           #rad. characteristic angle of the object
    thetaC::NTuple{2,Cdouble} #rad. Well, it is quite obvious, this is the center of the object if it is known (a good guess is the center of the FOV)
    dTheta::Cdouble           #rad the angular resolution of the pixels (assume that pixel are isotropic) 
    imgDim::NTuple{2,Int}     #pix the pixel size of the FOV
    w::Array{Cdouble,2}       #unitless the weights
    #offset::NTuple{N,Cdouble} #translation
    #rotMat::Array{Cdouble,2} #an N*N matrix that is the rotation
    #isotropic::Bool #maybe later
    function CompactRegCauchy(_theta0::Cdouble,
                             _thetaC::NTuple{2,Cdouble},
                             _dTheta::Cdouble,
                             _imgDim::NTuple{2,Int})
        _theta0 > 0.0 || error("radius must be strictly positive")
        _dTheta > 0.0 || error("pixel dimension must be strictly positive")
        prod(_imgDim) > 0 || error("image dimensions must be strictly positive")
        #create the weight map
        _w=Array(Cdouble,_imgDim);
        a2=_theta0*_theta0/(_dTheta*_dTheta) #pix^2
        for i2 in 1:_imgDim[2]
            i2f2 = _dTheta*Float(i2) - _thetaC[2] #rad projection onto the first basis vector
            i2f2 *= i2f2 #rad^2
            for i1 in 1:_imgDim[1]
                i1f=_dTheta*Float(i1) - _thetaC[1] #rad projection onto the second basis vector
                _w[i1,i2] = (1.0 + 2.0*(i1f*i1f + i2f2)/a2)
            end
        end
        #normalize the weights by  Z=\int_{\Omega} w(\theta) d\theta where \Omega is the angular portion of space covered by the FOV
        Z=prod(_imgDim)*sum(_w)*_dTheta*_dTheta
        new(_theta0,_thetaC,_dTheta,_imgDim,_w/Z)
    end
end

#function update_radius!(param::CompactRegCauchy,aNew::Cdouble)
#    aNew > 0.0 || error("radius must be strictly positive")
#    param.a=aNew
#    #create the weight map
#    a2=aNew*aNew
#    for i2 in 1:param.imgDim[2]
#        i2f2 = param.pixDim[2]*Float(i2) - param.center[2]
#        i2f2 *= i2f2
#        for i1 in 1:param.imgDim[1]
#            i1f=param.pixDim[1]*Float(i1) - param.center[1]
#            param.w[i1,i2] = (1.0 + 2.0*(i1f*i1f + i2f2)/a2)
#        end
#    end
#    nothing
#end

function cost{T<:AbstractFloat}(alpha::Float, param::CompactRegCauchy,
                                x::Array{T,2})
    alpha == zero(Float) && return zero(Float)
    return alpha*vdot(x, param.w.*x)
end

function cost!{T<:AbstractFloat}(alpha::Float, param::CompactRegCauchy,
                                 x::Array{T,2}, g::Array{T,2}, clr::Bool)
    @assert size(g) == size(x)
    if alpha == zero(Float)
        clr && vfill!(g, 0)
        return zero(Float)
    end
    temp = param.w.*x
    if clr
        vscale!(g, 2*alpha, temp)
    else
        vupdate!(g, 2*alpha, temp)
    end
    return alpha*vdot(x, temp)
end
