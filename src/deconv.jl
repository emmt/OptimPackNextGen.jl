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

module Deconv

import Base: size, convert

import TiPi: Float, MDA,
             AbstractCost, cost, cost!,
             default_weights, pad, zeropad,
             NormalEquations,
             LinearOperator, SelfAdjointOperator, FFTOperator,
             apply_direct, apply_direct!,
             apply_adjoint, apply_adjoint!,
             input_size, output_size,
             vcreate, vupdate!, vcombine!, vcopy!, vdot, vscale, vscale!

import Base: DFT, FFTW
import Base.FFTW: fftwNumber, fftwReal, fftwComplex

doc"""

    fast_deconv(dat, psf, rgl)

Perform a fast deconvolution using a simplified Wiener filter.  Argument `dat`
is the input data (blurred and noisy).  The result has the same dimensions as
`dat`.  Argument `psf` is the point spread function.  It is assumed to be
geometrically centered (same convetions as `TiPi.ifftshift`).  It is
automatically zero-padded if its dimensions are smaller than those of the data.
Argument `rgl` is the regularization level, its value should be nonnegative,
the higher the smoother the result.

Keyword `F` can be set with an instance of `TiPi.FFTOperator` suitable for
computing the discrete Fourier transform of the data.  The default is:

    F = TiPi.FFTOperator(dat)

"""
function fast_deconv{T<:fftwNumber,N}(dat::Array{T,N}, psf::Array{T,N},
                                      rgl::Real;
                                      F::Union{FFTOperator,Void}=nothing)
    @assert rgl ≥ 0
    if fast_deconv_check_dims(dat, psf)
        psf = zeropad(psf, size(dat))
    end
    if is(F, nothing)
        F = FFTOperator(dat)
    end
    tmp = ifftshift(psf) # will also serve to store the resulting image
    h = F(tmp)
    z = F(dat)
    const alpha = convert(real(T), rgl)
    const scale = convert(real(T), 1/length(dat))
    @simd for k in 1:length(z)
        @inbounds z[k] *= scale/(abs2(h[k]) + alpha)*conj(h[k])
    end
    return apply_adjoint!(tmp, F, z)
end

function fast_deconv{T<:fftwReal,N}(dat::Array{T,N}, psf::Array{T,N},
                                    rgl::Array{T,N}; kws...)
    _fast_deconv_helper(dat, psf, rgl; kws...)
end

function fast_deconv{T<:fftwReal,N}(dat::Array{Complex{T},N},
                                    psf::Array{Complex{T},N},
                                    rgl::Array{T,N}; kws...)
    _fast_deconv_helper(dat, psf, rgl; kws...)
end

typealias RealComplexArray{T,N} Union{Array{T,N},Array{Complex{T},N}}

function _fast_deconv_helper{T<:fftwReal,N}(dat::RealComplexArray{T,N},
                                            psf::RealComplexArray{T,N},
                                            rgl::Array{T,N};
                                            F::Union{FFTOperator,Void}=nothing)
    if fast_deconv_check_dims(dat, psf)
        psf = zeropad(psf, size(dat))
    end
    if is(F, nothing)
        F = FFTOperator(dat)
    end
    @assert size(rgl) == output_size(F)
    tmp = ifftshift(psf) # will also serve to store the resulting image
    h = F(tmp)
    z = F(dat)
    const scale = convert(T, 1/length(dat))
    @simd for k in 1:length(z)
        @inbounds z[k] *= scale/(abs2(h[k]) + rgl[k])*conj(h[k])
    end
    return apply_adjoint!(tmp, F, z)
end

function fast_deconv_check_dims{R,S,N}(dat::Array{R,N}, psf::Array{S,N})
    padding = false
    for i in 1:N
        if size(psf, i) < size(dat, i)
            padding = true
        end
        if size(psf, i) > size(dat, i)
            error("dimensions of PSF must not be larger than those of data")
        end
    end
    return padding
end

function deconv_check_dims{T,N}(dat::Array{T,N}, psf::Array{T,N},
                                dims::NTuple{N,Int})
    for i in 1:N
        dims[i] < 1 && error("bad dimension")
        dims[i] < max(size(dat,i) + size(psf,i)) && error("too small dimension")
    end
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

function quad_deconv{T<:fftwNumber,N}(dat::Array{T,N}, psf::Array{T,N},
                                      rgl::Real;
                                      wgt::Union{Array{T,N},Void}=nothing,
                                      dims::Union{NTuple{N,Int},Void}=nothing,
                                      fftwflags::Integer=FFTW.ESTIMATE,
                                      timelimits::Real=FFTW.NO_TIME_LIMIT,
                                      kws...)
    @assert rgl ≥ 0
    if is(dims, nothing)
        if ! is(F, nothing)
            dims = input_dims(F)
            deconv_check_dims(dat, psf, dims)
        else
            dims = ntuple(i -> goodfftdim(size(dat,i) + size(psf,i) - 1), N)
        end
    else
        deconv_check_dims(dat, psf, dims)
    end

    W::SelfAdjointOperator{Array{T,N}}
    if is(wgt, nothing)
        if padding(dat, dims)
            W = ScalingOperator(pad(default_weights(dat)))
            dat = pad(dat, dims)
        else
            W = Identity(Array{T,N})
        end
    else
        @assert size(wgt) == size(dat)
        check_weights(dat, wgt) # FIXME:
        if padding(dat, dims)
            dat = pad(dat, dims)
            wgt = pad(wgt, dims)
        end
        W = ScalingOperator(wgt)
    end

    work::Array{T,N} # will also serve to store the resulting image
    if padding(psf, dims)
        psf = pad(psf, dims)
        work = psf
    else
        work = Array(T, dims)
    end
    H = CirculantConvolution(psf, shift=true, flags=fftwflags,
                             timelimits=timelimits)

    ip = QuadraticInverseProblem(H, dat, W; µ=rgl)
    vfill!(sol, 0)
    solve!(ip, sol; kws...)
end

type DeconvolutionParam{T<:AbstractFloat,N} <: AbstractCost
    # Settings from the data.
    msk::Array{Bool,N}             # mask of valid data, same size as X
    y::Array{T,N}                  # data
    wgt::Array{T,N}                # weights, same size as Y

    # Model operator.
    mtf::Array{Complex{T},N}       # modulation transfer function,
                                   # same size as X
    function DeconvolutionParam(msk::Array{Bool,N},
                                y::Array{T,N},
                                wgt::Array{T,N},
                                mtf::Array{Complex{T},N})
        @assert size(msk) == size(mtf)
        @assert size(wgt) == size(y)
        # FIXME: we may ensure that there are as many true values in the mask
        # as the length of wgt and y
        new(msk, y, wgt, mtf)
    end
end

function fix_data!{T<:AbstractFloat,N}(dat::Array{T,N}, wgt::Array{T,N};
                                       bad::Real=NaN)

    @assert size(wgt) == size(dat)
    bad::T = bad
    cnt = 0
    @inbounds begin
        for i in 1:length(dat)
            if isnan(wgt[i]) || isinf(wgt[i]) || wgt[i] < zero(T)
                error("invalid weights")
            end
            if isinf(dat[i])
                if wgt[i] > zero(T)
                    error("invalid infinite data value(s) with non-zero weights")
                end
                dat[i] = zero(T) # to avoid problems later
            elseif isnan(dat[i])
                wgt[i] = zero(T)
                dat[i] = zero(T) # to avoid problems later
            elseif wgt[i] > zero(T)
                cnt += 1
            end
        end
    end
    return cnt
end

function compute_mtf{T<:AbstractFloat,N}(psf::Array{T,N},
                                         dims::NTuple{N,Int}, flags::UInt)
    # Check PSF values.
    s::T = zero(T)
    @inbounds begin
        for i in 1:length(psf)
            if isnan(psf[i]) || isinf(psf[i])
                error("invalid PSF value(s)")
            end
            s += psf[i]
        end
    end

    # Normalize PSF if requested.
    if (flags & NORMALIZE) != 0 && s != 1
        if (flags & VERBOSE) != 0
            warn("the PSF is not normalized ($s)")
        end
        s != 0 || error("sum(PSF) = 0")
        psf = scale(one(T)/s, psf)
    end

    # Compute the MTF.
    if (flags & USE_RFFT) != 0
        mtf = rfft(ifftshift(pad(zero(T), psf, dims)))
    else
        mtf = ifftshift(pad(zero(Complex{T}), psf, dims))
        fft!(mtf)
    end
    return mtf
end

function compute_mtf{T<:AbstractFloat,N}(psf::Array{Complex{T},N},
                                         dims::NTuple{N,Int}, flags::UInt)
    # Check flags.
    if (flags & NORMALIZE) != 0
        error("normalization of complex PSF not implemented")
    end
    if (flags & USE_RFFT) != 0
        error("real-complex transform not possible with complex PSF")
    end

    # Check PSF values.
    @inbounds begin
        for i in 1:length(psf)
            re = psf[i].re
            im = psf[i].im
            if isnan(re) || isinf(re) || isnan(im) || isinf(im)
                error("invalid PSF value(s)")
            end
        end
    end

    # Compute the MTF.
    mtf = ifftshift(pad(zero(Complex{T}), psf, dims))
    fft!(mtf)
    return mtf
end

function deconvparam{T<:AbstractFloat,N}(h::Array{T,N},
                                         y::Array{T,N},
                                         xdims::NTuple{N,Int};
                                         keywords...)
    deconvparam(h, y, default_weights(y), xdims; keywords...)
end

function deconvparam{T<:AbstractFloat,N}(h::Array{T,N},
                                         y::Array{T,N},
                                         w::Array{T,N},
                                         xdims::NTuple{N,Int};
                                         normalize::Bool=false,
                                         verbose::Bool=false)

    # Check weights and data.  Make a mask for the valid data.
    for k in 1:N
        max(size(y,k), size(h,k)) <= xdims[k] || error("output $(k)-th dimension too small")
        size(w, k) == size(y, k) || error("incompatible $(k)-th dimension of weights")
    end

    (cnt, msk) = buildmask(y, w)

    # Compute the MTF.
    mtf = compute_mtf(h, xdims; normalize=true)

    # Discard non-significant data and weights.
    if cnt != length(y)
        wt = Array(T, cnt)
        yt = Array(T, cnt)
        j = 0
        for i in 1:length(msk)
            if msk[i]
                j += 1
                wt[j] = w[i]
                yt[j] = y[i]
            end
        end
        w = wt
        y = yt
    end

    # False-pad the the mask of valid data.
    msk = pad(false, msk, xdims)

    DeconvolutionParam{T,N}(msk, y, w, mtf)

end


function cost{T,N}(alpha::Real, param::DeconvolutionParam{T,N},
                   x::Array{T,N})

    # Integrate un-weighted cost.
    @assert size(x) == size(param.msk)

    # Short circuit if weight is zero.
    if alpha == 0
        return 0.0
    end

    msk = param.msk
    wgt = param.wgt
    y = param.y
    h = param.mtf
    z = Array(Complex{T}, size(h))

    # Every FFT if done in-place in the workspace z
    const n = length(z)
    const m = length(y)
    @inbounds begin
        @simd for i in 1:n
            z[i] = x[i]
        end
    end
    fft!(z)
    # FIXME: check whether z[i] *= h[i] or, better, z *= h is as fast
    for i in 1:n
        z_re = z[i].re
        z_im = z[i].im
        h_re = h[i].re
        h_im = h[i].im
        z[i] = complex(h_re*z_re - h_im*z_im,
                       h_re*z_im + h_im*z_re)
    end
    bfft!(z)
    const scl::T = 1/n
    j = 0
    err::Cdouble = 0
    @inbounds begin
        for i in 1:n
            if msk[i]
                j < m || throw(BoundsError())
                j += 1
                r = scl*z[i].re - y[j]
                err += wgt[j]*r*r
            end
        end
    end
    return alpha*err
end

function cost!{T,N}(alpha::Real, param::DeconvolutionParam{T,N},
                    x::Array{T,N}, g::Array{T,N}, clr::Bool=false)
    # Minimal checking.
    @assert size(x) == size(g)
    @assert size(x) == size(param.h)

    # Clear gradient if requested.
    clr && vfill!(g, 0)

    # Short circuit if weight is zero.
    alpha == 0 && return 0.0

    # Integrate cost and gradient.
    msk = param.msk
    wgt = param.wgt
    y = param.y
    h = param.mtf
    z = Array(Complex{T}, size(h))

    # Every FFT if done in-place in the workspace z
    const n = length(z)
    const m = length(y)
    _copy!(z, x)
    fft!(z)
    _multiply!(z, h, false)
    bfft!(z)
    scl::T = 1/n
    j = 0
    err::Cdouble = 0
    @inbounds begin
        @simd for i in 1:n
            if msk[i]
                j += 1
                r = scl*z[i].re - y[j]
                wr = wgt[j]*r
                err += wr*r
                z[i] = wr
            else
                z[i] = 0
            end
        end
    end
    fft!(z)
    _multiply!(z, h, true)
    bfft!(z)
    scl = 2*alpha/n
    @inbounds begin
        @simd for i in 1:n
            g[i] += scl*z[i].re
        end
    end

    return alpha*err
end

type DeconvolutionHessian{T<:AbstractFloat,N} <: SelfAdjointOperator{Array{T,N}}

    # Settings from the data.
    msk::Array{Bool,N}             # mask of valid data, same size as X
    wgt::Array{T}                  # statistical weights

    # Regularization parameters.
    alpha::Vector{Float64}         # regularization weights
    other::Vector{Vector{Int}}     # index indirection tables for
                                   # finite differences along each dimension
    # Model operator.
    mtf::Array{Complex{T},N}       # modulation transfer function, same size as X
    z::Array{Complex{T},N}         # workspace vector, same size as X
end


function init{T<:AbstractFloat,N}(h::Array{T,N}, y::Array{T,N}, alpha)
    dims = ntuple(N, i -> goodfftdim(size(h,i) + size(y,i) - 1))
    return init(h, y, ones(T, size(y)), dims, alpha)
end

function init{T<:AbstractFloat,N}(h::Array{T,N}, y::Array{T,N},
                                  xdims::NTuple{N,Int}, alpha)
    return init(h, y, default_weights(y), xdims, alpha)
end


function nearestother(dim::Int)
    other = Array(Int, dim)
    for i in 1:dim-1
        other[i] = i+1
    end
    other[dim] = dim
    return other
end

function init{S<:Real,T<:AbstractFloat,N}(h::Array{T,N},
                                          y::Array{T,N},
                                          w::Array{T,N},
                                          xdims::NTuple{N,Int},
                                          alpha::Vector{S};
                                          normalize::Bool=false,
                                          verbose::Bool=false)
    @assert length(alpha) == N
    size(w) == size(y) || error("incompatible $(k)-th dimension of weights")

    a = Array(Float64, N)
    other = Array(Vector{Int}, N)
    for k in 1:N
        if max(size(y,k), size(h,k)) > xdims[k]
            error("output "*nth(k)*" dimension too small")
        end
        if isnan(alpha[k]) || isinf(alpha[k]) || alpha[k] < zero(S)
            error("invalid regularization weights")
        end
        a[k] = alpha[k]
        other[k] = nearestother(xdims[k])
    end
    (cnt, msk) = buildmask(y, w)

    # Compute the MTF.
    mtf = compute_mtf(h, xdims; normalize=normalize, verbose=verbose)

    # Compute the RHS vector: b = H'.W.y
    wy = Array(T, size(y))
    for i in 1:length(y)
        wy[i] = (w[i] > zero(T) ? w[i]*y[i] : zero(T))
    end
    z = pad(zero(Complex{T}), wy, xdims)
    fft!(z)
    _multiply!(z, h, true)
    bfft!(z)
    b = Array(T, xdims)
    scl::T = 1/length(z)
    for i in 1:length(z)
        b[i] = scl*z[i].re
    end

    # Discard non-significant weights
    if cnt != length(w)
        tmp = Array(T, cnt)
        j = 0
        for i in 1:length(msk)
            if msk[i]
                j += 1
                tmp[j] = w[i]
            end
        end
        @assert j == cnt
        w = tmp
    end

    # False-pad the the mask of valid data.
    msk = pad(false, msk, xdims)

    A = DeconvolutionHessian(msk, w, a, other, mtf, z)
    NormalEquations(A, b)
end

function apply_direct!{T,N}(dst::Array{T,N},
                            op::DeconvolutionHessian{T,N},
                            src::Array{T,N})

    @assert size(src) == size(dst)

    msk = op.msk
    wgt = op.wgt
    z = op.z
    h = op.mtf

    #########################
    # Compute dst = H'.W.H.src
    #########################

    # Every FFT is done in-place in the workspace z
    const n = length(z)
    for i in 1:n
        z[i] = src[i]
    end
    fft!(z)
    _multiply!(z, h, false)
    bfft!(z)
    const scl::T = 1/(n*n)
    j = 0
    for i in 1:n
        if msk[i]
            j += 1
            z[i] = scl*wgt[j]*z[i].re
        else
            z[i] = 0
        end
    end
    fft!(z)
    _multiply!(z, h, true)
    bfft!(z)
    for i in 1:n
        dst[i] = z[i].re
    end


    #########################
    # Do dst += mu*R.src
    #########################
    DtD!(op.alpha, op.other, dst, src)

end

function DtD!{S<:Real, T<:AbstractFloat}(alpha::Vector{S},
                                         other::Array{Vector{Int}},
                                         q::Array{T,1},
                                         p::Array{T,1})
    @assert length(alpha) == 1
    @assert length(other) == 1
    #vfill!(q, zero(T))
    alpha1::T = alpha[1]
    other1 = other[1]
    for i in 1:length(p) # for each input element
        j = other1[i] # index of neighbor
        t = alpha1*(p[j] - p[i])
        q[i] -= t
        q[j] += t
    end
end

function DtD!{S<:Real, T<:AbstractFloat}(alpha::Vector{S},
                                         other::Vector{Vector{Int}},
                                         q::Array{T,2}, p::Array{T,2})
    @assert length(alpha) == 2
    @assert length(other) == 2
    #vfill!(q, zero(T))
    dim1 = size(p, 1)
    dim2 = size(p, 2)
    alpha1 = alpha[1]
    alpha2 = alpha[2]
    other1 = other[1]
    other2 = other[2]
    @assert length(other1) == dim1
    @assert length(other2) == dim2
    for i2 in 1:dim2
        i2o = other2[i2]
        for i1 in 1:dim1
            i1o = other1[i1]
            t1 = alpha1*(p[i1o,i2] - p[i1,i2])
            t2 = alpha2*(p[i1,i2o] - p[i1,i2])
            q[i1,i2] -= (t1 + t2)
            q[i1o,i2] += t1
            q[i1,i2o] += t2
        end
    end
end

#------------------------------------------------------------------------------

# FIXME: should be part of utils.jl

function _copy!{T,N}(dst::Array{T,N}, src::Array{T,N})
    if !is(dst, src)
        @assert size(dst) == size(src)
        @inbounds begin
            @simd for i in 1:length(dst)
                dst[i] = src[i]
            end
        end
    end
end

function _copy!{D,S,N}(dst::Array{D,N}, src::Array{S,N})
    @assert size(dst) == size(src)
    @inbounds begin
        @simd for i in 1:length(dst)
            dst[i] = src[i]
        end
    end
end

"""
    _copy!(dst, src)

copies the values of the source `src` into the destination `dst`.  An error is
thrown if the source and destination do not have the same dimensions.  The
types of their elements may be different.

""" _copy!

function _scale!{T<:AbstractFloat,N}(dst::Array{T,N},
                                     scl::T,
                                     src::Array{T,N})
    @assert size(dst) == size(src)
    @inbounds begin
        @simd for i in 1:length(dst)
            dst[i] = scl*src[i]
        end
    end
end

function _scale!{T<:AbstractFloat,N}(dst::Array{Complex{T},N},
                                     scl::T,
                                     src::Array{Complex{T},N})
    @assert size(dst) == size(src)
    @inbounds begin
        @simd for i in 1:length(dst)
            dst[i] = scl*src[i]
        end
    end
end

function _scale!{T<:AbstractFloat,N}(dst::Array{T,N},
                                     scl::T,
                                     src::Array{Complex{T},N})
    @assert size(dst) == size(src)
    @inbounds begin
        @simd for i in 1:length(dst)
            dst[i] = scl*src[i].re
        end
    end
end

function _scale!{T<:AbstractFloat,N}(dst::Array{Complex{T},N},
                                     scl::T,
                                     src::Array{T,N})
    @assert size(dst) == size(src)
    @inbounds begin
        @simd for i in 1:length(dst)
            dst[i] = scl*src[i].re
        end
    end
end

"""
    _scale!(dst, alpha, src)

copies the values of the source `src` scaled by the factor `alpha` into the
destination `dst`.  An error is thrown if the source and destination do not
have the same dimensions.  The types of their elements may be different.

""" _scale!

# Fast in-place multiplication by the MTF.
function _multiply!{T,N}(arr::Array{Complex{T},N},
                         mtf::Array{Complex{T},N},
                         conjugate::Bool=false)
    @assert size(arr) == size(mtf)
    if conjugate
        @inbounds begin
            @simd for i in 1:length(arr)
                arr_re = arr[i].re
                arr_im = arr[i].im
                mtf_re = mtf[i].re
                mtf_im = mtf[i].im
                arr[i] = complex(mtf_re*arr_re + mtf_im*arr_im,
                                 mtf_re*arr_im - mtf_im*arr_re)
            end
        end
    else
        @inbounds begin
            @simd for i in 1:length(arr)
                arr_re = arr[i].re
                arr_im = arr[i].im
                mtf_re = mtf[i].re
                mtf_im = mtf[i].im
                arr[i] = complex(mtf_re*arr_re - mtf_im*arr_im,
                                 mtf_re*arr_im + mtf_im*arr_re)
            end
        end
    end
end

"""
    _multiply!(arr, mtf)
    _multiply!(arr, mtf, false)

stores in `arr` the elementwise multiplication of `arr` by `mtf`, while:

    _multiply!(arr, mtf, true)

stores in `arr` the elementwise multiplication of `arr` by `conj(mtf)`.  An
error is thrown if the arrays do not have the same dimensions.

""" _multiply!

end # module
