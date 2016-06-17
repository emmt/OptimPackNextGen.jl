#
# deconv.jl --
#
# Regularized deconvolution for TiPi.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#

module Deconvolution

import Base: size, convert, length, ndims, get, getindex

import Base: DFT, FFTW
import Base.FFTW: fftwNumber, fftwReal, fftwComplex

using TiPi
importall TiPi.Algebra

#import TiPi: conjgrad!, conjgrad

export fast_deconv, quad_deconv

doc"""

    fast_deconv(psf, dat, rgl)

performs a fast deconvolution of the input data `dat` (blurred and noisy),
assuming `psf` is the point spread function (PSF) and for a regularization
level `rgl`.  The result has the same dimensions as `dat`.

The method is a simplified Wiener filter tuned by the parameter `rgl` which must be
nonnegative, the higher the smoother the result.

The PSF is assumed to be geometrically centered (same conventions as
`TiPi.ifftshift`).  The PSF is automatically zero-padded if its dimensions are
smaller than those of the data.

Keyword `F` can be set with an instance of `TiPi.FFTOperator` suitable for
computing the discrete Fourier transform of the data.  The default is:

    F = TiPi.FFTOperator(dat)

"""
function fast_deconv{T<:fftwNumber,N}(psf::Array{T,N},
                                      dat::Array{T,N},
                                      rgl::Real;
                                      F::FFTOperator=FFTOperator(dat))
    @assert rgl ≥ 0
    rgl = convert(real(T), rgl)
    h, z, wrk = prepare_fast_deconv(psf, dat, F)
    const scale = convert(real(T), 1/length(dat))
    @inbounds @simd for k in 1:length(z)
        z[k] *= scale/(abs2(h[k]) + rgl)*conj(h[k])
    end
    return apply_adjoint!(wrk, F, z)
end

function fast_deconv{T<:fftwNumber,R<:fftwReal,N}(psf::Array{T,N},
                                                  dat::Array{T,N},
                                                  rgl::Array{R,N};
                                                  F::FFTOperator=FFTOperator(dat))
    R === real(T) || throw(ArgumentError("bad regularization weights type"))
    size(rgl) != output_size(F) || throw(ArgumentError("bad regularization weights size"))
    h, z, wrk = prepare_fast_deconv(psf, dat, F)
    const scale = convert(R, 1/length(dat))
    @inbounds @simd for k in 1:length(z)
        z[k] *= scale/(abs2(h[k]) + rgl[k])*conj(h[k])
    end
    return apply_adjoint!(wrk, F, z)
end

function prepare_fast_deconv{T<:fftwNumber,N}(psf::Array{T,N},
                                              dat::Array{T,N},
                                              F::FFTOperator)
    padding = false
    for i in 1:N
        if size(psf, i) < size(dat, i)
            padding = true
        end
        if size(psf, i) > size(dat, i)
            error("dimensions of PSF must not be larger than those of data")
        end
    end
    wrk = ifftshift((padding ? zeropad(psf, size(dat)) : psf))
    return F(wrk), F(dat), wrk
end

#---------------------------------------------------------------------------


function data_size{T,N}(dat::AbstractArray{T,N},
                        wgt::AbstractArray{T,N})
    bbox = BoundingBox(wgt, 0)
    @assert length(bbox) > 0 "weights are zero everywhere!"
    return size(bbox)
end
data_size{T,N}(dat::AbstractArray{T,N}) = size(dat)
data_size{T,N}(dat::AbstractArray{T,N}, ::Void) = size(dat)

function psf_size{T,N}(psf::AbstractArray{T,N})
    bbox = BoundingBox(psf, 0)
    @assert length(bbox) > 0 "PSF is zero everywhere!"
    return size(bbox)
end

function default_size{N,T}(psf::AbstractArray{T,N},
                           dat::AbstractArray{T,N},
                           wgt::Union{AbstractArray{T,N},Void},
                           out::Union{AbstractArray{T,N},Void})
    psfsiz = psf_size(psf)
    datsiz = data_size(dat, wgt)
    minsiz = ntuple(d -> datsiz[d] + psfsiz[d] - 1, N)
    if ! is(out, nothing)
        minsiz = max(minsiz, size(out))
    end
    return ntuple(d -> goodfftdim(minsiz[d]), N)
end

function padding{T,N}(arr::Array{T,N}, dims::NTuple{N,Int})
    answer = false
    for i in 1:N
        dims[i] < 1 && error("bad dimension")
        dims[i] < size(arr,i) && error("too small dimension")
        if dims[i] > size(arr,i)
            answer = true
        end
    end
    return answer
end

countbads{T<:Integer,N}(::AbstractArray{T,N}) = 0
function countbads{T<:Real,N}(A::AbstractArray{T,N})
    nbads = 0
    @inbounds for i in eachindex(A)
        nbads += (isfinite(A[i]) ? 0 : 1)
    end
    return nbads
end

function privatecopy{T,N}(src::AbstractArray{T,N})
    cpy = Array(T, size(src))
    @inbounds @simd for i in eachindex(cpy, src)
        cpy[i] = src[i]
    end
    return cpy
end

function quad_deconv{T<:fftwReal,N}(psf::Array{T,N},
                                    dat::Array{T,N},
                                    rgl::Real;
                                    init::Union{Array{T,N},Void}=nothing,
                                    wgt::Union{Array{T,N},Void}=nothing,
                                    fftwflags::Integer=FFTW.ESTIMATE,
                                    timelimit::Real=FFTW.NO_TIMELIMIT,
                                    lower=-Inf, upper=+Inf,
                                    kws...)
    @assert rgl ≥ 0
    µ = convert(T, rgl)::T

    # Check weights and data.
    wgt, dat = fix_weighted_data(wgt, dat; bad=0)

    # Determine the size of the result.
    dims = default_size(psf, dat, wgt, init)

    # Pad the data and the weights if needed and create the weighting
    # operator.
    if padding(dat, dims)
        if is(wgt, nothing)
            wgt = ones(T, size(dat))
        end
        wgt = zeropad(wgt, dims)
        dat = zeropad(dat, dims)
    end
    y = dat
    W = (is(wgt, nothing) ? Identity(typeof(y)) :
         DiagonalOperator(wgt)) :: LinearEndomorphism{typeof(y)}

    # Create the convolution operator.  The array `x` will also serve to store
    # the resulting image.
    if padding(psf, dims)
        x = psf = zeropad(psf, dims)
    else
        x = Array(T, dims)
    end
    H = CirculantConvolution(psf, shift=true, flags=fftwflags,
                             timelimit=timelimit)

    # Initial solution.
    if is(init, nothing)
        vfill!(x, 0)
    else
        if padding(init, dims)
            # FIXME: find a better value for padding than zero.
            vcopy!(x, zeropad(init, dims))
        else
            vcopy!(x, init)
        end
    end

    # Regularization.
    D = OperatorD(x)
    DtD = OperatorDtD(x)

    # LHS and RHS terms of the normal equations.
    A = H'*W*H + µ*DtD
    b = H'*W*y

    eq = NormalEquations(A, b)

    if isinf(lower) && isinf(upper) && lower < upper
        # Use linear conjugate gradients.
        x = conjgrad!(eq, x; kws...)
    else
        # Use quasi-Newton method with bound constraints.
        x = vmlmb!((x, g) -> cost!(eq, x, g), x;
                   lower=lower, upper=upper, kws...)
    end
    return x
end


end # module
