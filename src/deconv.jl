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

# FIXME: use real-complex FFT
module Deconv

using ..Algebra
import ..Algebra: apply_direct, apply_adjoint, apply_direct!

import TiPi: cost, cost!, AbstractCost
import TiPi: defaultweights, pad, zeropad
import TiPi.MDA

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

function buildmask{T<:AbstractFloat,N}(y::Array{T,N}, w::Array{T,N})
    @assert size(w) == size(y)
    msk = Array(Bool, size(y))
    cnt = 0
    for i in 1:length(y)
        if isnan(w[i]) || isinf(w[i]) || w[i] < zero(T)
            error("invalid weights")
        end
        if isnan(y[i]) && w[i] > zero(T)
            error("invalid data must have zero-weight")
        end
        if isinf(y[i])
            error("invalid data value(s)")
        end
        msk[i] = (w[i] > zero(T))
        if msk[i]
            cnt += 1
        end
    end
    return (cnt, msk)
end

function compute_mtf{T<:AbstractFloat,N}(h::Array{T,N},
                                         dims::NTuple{N,Int};
                                         normalize::Bool=false,
                                         verbose::Bool=false)
    # Check PSF values.
    s::T = zero(T)
    for i in 1:length(h)
        if isnan(h[i]) || isinf(h[i])
            error("invalid PSF value(s)")
        end
        s += h[i]
    end

    # Normalize PSF if requested.
    if normalize && s != 1
        verbose && warn("the PSF is not normalized ($s)")
        s != 0 || error("sum(PSF) = 0")
        h = scale(convert(T,1/s), h)
    end

    # Compute the MTF.
    mtf = ifftshift(pad(zero(Complex{T}), h, dims))
    fft!(mtf)
    return mtf
end

function deconvparam{T<:AbstractFloat,N}(h::Array{T,N},
                                         y::Array{T,N},
                                         xdims::NTuple{N,Int};
                                         normalize::Bool=false,
                                         verbose::Bool=false)
    deconvparam(h, y, defaultweights(y), xdims;
                normalize=normalize, verbose=verbose)
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
    @assert(size(x) == size(param.msk))

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
    for i in 1:n
        z[i] = x[i]
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
    for i in 1:n
        if msk[i]
            j += 1
            r = scl*z[i].re - y[j]
            err += wgt[j]*r*r
        end
    end
    return alpha*err
end

function cost!{T,N}(alpha::Real, param::DeconvolutionParam{T,N},
                    x::Array{T,N}, g::Array{T,N}, clr::Bool=false)
    # Minimal checking.
    @assert(size(x) == size(g))

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
    for i in 1:n
        z[i] = x[i]
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
    scl::T = 1/n
    j = 0
    err::Cdouble = 0
    for i in 1:n
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
    fft!(z)
    # FIXME: check whether z[i] *= conj(h[i]) or, better, z *= conj(h) is as fast
    for i in 1:n
        z_re = z[i].re
        z_im = z[i].im
        h_re = h[i].re
        h_im = h[i].im
        z[i] = complex(h_re*z_re + h_im*z_im,
                       h_re*z_im - h_im*z_re)
    end
    bfft!(z)
    scl = 2*alpha/n
    for i in 1:n
        g[i] += scl*z[i].re
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
    return init(h, y, defaultweights(y), xdims, alpha)
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
    @assert(length(alpha) == N)
    size(w) == size(y) || error("incompatible $(k)-th dimension of weights")

    a = Array(Float64, N)
    other = Array(Vector{Int}, N)
    for k in 1:N
        if max(size(y,k), size(h,k)) > xdims[k]
            error("output $(k)-th dimension too small")
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
    for i in 1:length(z)
        z_re = z[i].re
        z_im = z[i].im
        h_re = mtf[i].re
        h_im = mtf[i].im
        z[i] = complex(h_re*z_re + h_im*z_im,
                       h_re*z_im - h_im*z_re)
    end
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
        @assert(j == cnt)
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

    # Every FFT if done in-place in the workspace z
    const n = length(z)
    for i in 1:n
        z[i] = src[i]
    end
    fft!(z)
    for i in 1:n
        z_re = z[i].re
        z_im = z[i].im
        h_re = h[i].re
        h_im = h[i].im
        z[i] = complex(h_re*z_re - h_im*z_im,
                       h_re*z_im + h_im*z_re)
    end
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
    for i in 1:n
        z_re = z[i].re
        z_im = z[i].im
        h_re = h[i].re
        h_im = h[i].im
        z[i] = complex(h_re*z_re + h_im*z_im,
                       h_re*z_im - h_im*z_re)
    end
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
    @assert(length(alpha) == 1)
    @assert(length(other) == 1)
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
    @assert(length(alpha) == 2)
    @assert(length(other) == 2)
    #vfill!(q, zero(T))
    dim1 = size(p, 1)
    dim2 = size(p, 2)
    alpha1 = alpha[1]
    alpha2 = alpha[2]
    other1 = other[1]
    other2 = other[2]
    @assert(length(other1) == dim1)
    @assert(length(other2) == dim2)
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

end # module
