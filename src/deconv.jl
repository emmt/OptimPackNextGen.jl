#
# deconv.jl --
#
# Regularized deconvolution for TiPi.
#
#------------------------------------------------------------------------------
#
# This file is part of TiPi.jl licensed under the MIT "Expat" License.
#
# Copyright (C) 2015, Éric Thiébaut & Jonathan Léger.
#
#------------------------------------------------------------------------------

module Deconv

import TiPi: apply!, LinearOperator, LinearProblem
import TiPi: defaultweights, pad, zeropad

type DeconvolutionHessian{T<:AbstractFloat,N} <: LinearOperator

    # Settings from the data.
    msk::Array{Bool,N}             # mask of valid data, same size as X
    wgt::Array{T,N}                # weights, same size as Y

    # Regularization parameters.
    alpha::Vector{Float64}         # regularization weights
    other::Vector{Vector{Int}}     # index indirection tables for
                                   # finite differences along each dimension
    # Model operator.
    mtf::Array{Complex{T},N}       # modulation transfer function, same size as X
    z::Array{Complex{T},N}         # workspace vector, same size as X
end


function init{T<:AbstractFloat,N}(h::Array{T,N}, y::Array{T,N}, alpha)
    dims = ntuple(N, i -> fftbestdim(size(h,i) + size(y,i) - 1))
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

function init{S<:AbstractFloat,T<:AbstractFloat,N}(h::Array{T,N},
                                                   y::Array{T,N},
                                                   w::Array{T,N},
                                                   xdims::NTuple{N,Int},
                                                   alpha::Vector{S})
    @assert(length(alpha) == N)
    a = Array(Float64, N)
    other = Array(Vector{Int}, N)
    for k in 1:N
        max(size(y,k), size(h,k)) <= xdims[k] || error("output $(k)-th dimension too small")
        size(w, k) == size(y, k) || error("incompatible $(k)-th dimension of weights")
        if isnan(alpha[k]) || isinf(alpha[k]) || alpha[k] < zero(S)
            error("invalid regularization weights")
        end
        a[k] = alpha[k]
        other[k] = nearestother(xdims[k])
    end
    msk = Array(Bool, size(y))
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
    end
    s::T = zero(T)
    for i in 1:length(h)
        if isnan(h[i]) || isinf(h[i])
            error("invalid PSF value(s)")
        end
        s += h[i]
    end
    if s != 1
        warn("the PSF is not normalized ($s)")
    end


    # Compute the MTF
    mtf = fftshift(pad(zero(Complex{T}), h, xdims))
    fft!(mtf)

    # Compute the RHS vector: b = H'.W.y
    wy = Array(T, size(y))
    for i in 1:length(y)
        wy[i] = (w[i] > zero(T) ? w[i]*y[i] : zero(T))
    end
    z = pad(zero(Complex{T}), wy, xdims)
    println("z: $(typeof(z))")
    fft!(z)
    for i in 1:length(z)
        z_re = z[i].re
        z_im = z[i].im
        h_re = mtf[i].re
        h_im = mtf[i].im
        z[i] = complex(h_re*z_re - h_im*z_im,
                       h_re*z_im + h_im*z_re)
    end
    bfft!(z)
    b = Array(T, xdims)
    scl::T = 1/length(z)
    for i in 1:length(z)
        b[i] = scl*z[i].re
    end

    # False-pad the the mask of valid data.
    msk = pad(false, msk, xdims)

    A = DeconvolutionHessian(msk, w, a, other, mtf, z)
    LinearProblem(A, b)
end

function apply!{T<:AbstractFloat,N}(op::DeconvolutionHessian,
                                    q::Array{T,N}, p::Array{T,N})
    @assert(size(p) == size(q))

    msk = op.msk
    wgt = op.wgt
    z = op.z
    h = op.mtf

    #########################
    # Compute q = H'.W.H.p
    #########################

    # Every FFT if done in-place in the workspace z
    const n = length(z)
    for i in 1:n
        z[i] = p[i]
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
        q[i] = z[i].re
    end


    #########################
    # Do q += mu*R.p
    #########################
    DtD!(op.alpha, op.other, q, p)

end

function DtD!{S<:AbstractFloat, T<:AbstractFloat}(alpha::Vector{S},
                                                  other::Array{Vector{Int}},
                                                  q::Array{T,1},
                                                  p::Array{T,1})
    @assert(length(alpha) == 1)
    @assert(length(other) == 1)
    alpha1::T = alpha[1]
    other1 = other[1]
    for i in 1:length(p) # for each input element
        j = other1[i] # index of neighbor
        t = alpha1*(p[j] - p[i])
        q[i] -= t
        q[j] += t
    end
end

function DtD!{S<:AbstractFloat, T<:AbstractFloat}(alpha::Vector{S},
                                                  other::Vector{Vector{Int}},
                                                  q::Array{T,2}, p::Array{T,2})
    @assert(length(alpha) == 2)
    @assert(length(other) == 2)
    dim1 = size(p, 1)
    dim2 = size(p, 2)
    alpha1 = alpha[1]
    alpha2 = alpha[2]
    other1 = other[1]
    other2 = other[2]
    @assert(length(other1) == dim1)
    @assert(length(other2) == dim2)
    for i2 in 1:dim2
        j2 = other2[i2]
        for i1 in 1:dim1
            j1 = other1[i1]
            t1 = alpha1*(p[j1,i2] - p[i1,i2])
            t2 = alpha2*(p[i1,j2] - p[i1,i2])
            q[i1,i2] -= t1 + t2
            q[j1,i2] += t1
            q[i1,j2] += t2
        end
    end
end

end # module

# Local Variables:
# mode: Julia
# tab-width: 8
# indent-tabs-mode: nil
# fill-column: 79
# coding: utf-8
# ispell-local-dictionary: "american"
# End:
