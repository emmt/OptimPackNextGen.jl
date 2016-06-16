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

import Base: size, convert

import Base: DFT, FFTW
import Base.FFTW: fftwNumber, fftwReal, fftwComplex

using TiPi
importall TiPi.Algebra

export fast_deconv

doc"""

    fast_deconv(dat, psf, rgl)

Perform a fast deconvolution using a simplified Wiener filter.  Argument `dat`
is the input data (blurred and noisy).  The result has the same dimensions as
`dat`.  Argument `psf` is the point spread function (PSF).  The PSF is assumed
to be geometrically centered (same conventions as `TiPi.ifftshift`).  The PSF
is automatically zero-padded if its dimensions are smaller than those of the
data.  Argument `rgl` is the regularization level, its value should be
nonnegative, the higher the smoother the result.

Keyword `F` can be set with an instance of `TiPi.FFTOperator` suitable for
computing the discrete Fourier transform of the data.  The default is:

    F = TiPi.FFTOperator(dat)

"""
function fast_deconv{T<:fftwNumber,N}(dat::Array{T,N},
                                      psf::Array{T,N},
                                      rgl::Real;
                                      F::FFTOperator=FFTOperator(dat))
    @assert rgl ≥ 0
    rgl = convert(real(T), rgl)
    h, z, wrk = prepare_fast_deconv(dat, psf, F)
    const scale = convert(real(T), 1/length(dat))
    @inbounds @simd for k in 1:length(z)
        z[k] *= scale/(abs2(h[k]) + rgl)*conj(h[k])
    end
    return apply_adjoint!(wrk, F, z)
end

function fast_deconv{T<:fftwNumber,R<:fftwReal,N}(dat::Array{T,N},
                                                  psf::Array{T,N},
                                                  rgl::Array{R,N};
                                                  F::FFTOperator=FFTOperator(dat))
    R === real(T) || throw(ArgumentError("bad regularization weights type"))
    size(rgl) == size(dat) || throw(ArgumentError("bad regularization weights size"))
    h, z, wrk = prepare_fast_deconv(dat, psf, F)
    const scale = convert(R, 1/length(dat))
    @inbounds @simd for k in 1:length(z)
        z[k] *= scale/(abs2(h[k]) + rgl[k])*conj(h[k])
    end
    return apply_adjoint!(wrk, F, z)
end

function prepare_fast_deconv{T<:fftwNumber,N}(dat::Array{T,N},
                                              psf::Array{T,N},
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

end # module
