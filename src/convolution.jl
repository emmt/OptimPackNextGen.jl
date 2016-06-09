#
# convolution.jl --
#
# Convolution and circulant convolution for TiPi.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#

module Convolution

import Base: eltype, size, ndims

import TiPi: Endomorphism,
             input_size, output_size,
             input_ndims, output_ndims,
             input_eltype, output_eltype,
             apply_direct, apply_direct!,
             apply_adjoint, apply_adjoint!

import Base: DFT, FFTW
import Base.FFTW: fftwNumber, fftwReal, fftwComplex

export CirculantConvolution

type CirculantConvolution{T<:fftwNumber,C<:fftwComplex,N} <: Endomorphism{Array{T,N}}
    xdims::NTuple{N,Int}      # input dimensions
    zdims::NTuple{N,Int}      # complex dimensions
    mtf::Array{C,N}           # pre-scaled modulation transfer function
    forward::DFT.Plan{T}      # plan for forward transform
    backward::DFT.Plan{C}     # plan for backward transform
end

# Basic methods for a linear operator on Julia's arrays.
input_size(H::CirculantConvolution) = H.xdims
output_size(H::CirculantConvolution) = H.xdims
input_size(H::CirculantConvolution, i::Integer) = H.xdims[i]
output_size(H::CirculantConvolution, i::Integer) = H.xdims[i]
input_ndims{T,C,N}(H::CirculantConvolution{T,C,N}) = N
output_ndims{T,C,N}(H::CirculantConvolution{T,C,N}) = N
input_eltype{T,C,N}(H::CirculantConvolution{T,C,N}) = T
output_eltype{T,C,N}(H::CirculantConvolution{T,C,N}) = T

# Basic methods for an array.
eltype{T,C,N}(H::CirculantConvolution{T,C,N}) = T
size{T,C,N}(H::CirculantConvolution{T,C,N}) = ntuple(i -> H.xdims[(i ≤ N ? i : i - N)], 2*N)
size{T,C,N}(H::CirculantConvolution{T,C,N}, i::Integer) = H.xdims[(i ≤ N ? i : i - N)]
ndims{T,C,N}(H::CirculantConvolution{T,C,N}) = 2*N

doc"""
`check_flags(flags)` checks whether `flags` is an allowed bitwise-or
combination of FFTW planner flags (see
http://www.fftw.org/doc/Planner-Flags.html) and return the filtered flags.
"""
function check_flags(flags::Integer)
    planning = flags & (FFTW.ESTIMATE | FFTW.MEASURE | FFTW.PATIENT |
                        FFTW.EXHAUSTIVE | FFTW.WISDOM_ONLY)
    if flags != planning
        throw(ArgumentError("only FFTW planning flags can be specified"))
    end
    UInt32(planning)
end

doc"""
# Circulant convolution operator

The circulant convolution operator `H` and its adjoint are given by:

    H  = (1/n)*F'*diag(mtf)*F
    H' = (1/n)*F'*diag(conj(mtf))*F

with `n` the number of elements, `F` the discrete Fourier transform operator
and `mtf` the modulation transfer function.

The operator `H` can be created by:

    H = CirculantConvolution(psf, flags=FFTW.ESTIMATE, timelimit=Inf)

where `psf` is the point spread function (PSF).  Keyword `flags` is a
bitwise-or of FFTW planner flags, defaulting to `FFTW.ESTIMATE`.  Keyword
`timelimit` specifies a rough upper bound on the allowed planning time, in
seconds.

If the operator is to be used many times (as in interative methods), it is
recommended to use at least `flags=FFTW.MEASURE` which generally yields faster
transforms compared to the default `flags=FFTW.ESTIMATE`.

The operator can be used as a regular linear operator: `H(x)` or `H*x` to
compute the convolution of `x` and `H'(x)` or `H'*x` to apply the adjoint of
`H` to `x`.

For a slight improvement of performances, an array `y` to store the result of
the operation can be provided:

    apply_direct!(y, H, x)   or  apply!(y, H, x)
    apply_adjoint!(y, H, x)  or  apply!(y, H', x)

If provided, `y` must be at a different memory location than `x`.

""" CirculantConvolution

# Create a circular convolution operator for complex arrays (see
# doc/convolution.md for explanations).
function CirculantConvolution{T<:fftwReal,N}(psf::Array{Complex{T},N};
                                             flags::Integer=FFTW.ESTIMATE,
                                             kws...)
    # Check arguments and get dimensions.
    planning = check_flags(flags)
    n = length(psf)
    dims = size(psf)

    # Allocate array for the scaled MTF, will also be used
    # as a scratch array for planning which may destroy its input.
    mtf = similar(psf)

    # Compute the plans with FFTW flags suitable for out-of-place forward
    # transform and in-place backward transform.
    forward = plan_fft(mtf; flags=(planning | FFTW.PRESERVE_INPUT), kws...)
    backward = plan_bfft!(mtf; flags=(planning | FFTW.DESTROY_INPUT), kws...)

    # Compute the scaled MTF.
    A_mul_B!(mtf, forward, psf)
    scale!(mtf, T(1/n))

    # Build the operator.
    CirculantConvolution{Complex{T},Complex{T},N}(dims, dims, mtf,
                                                  forward, backward)
end

# Create a circular convolution operator for real arrays.
function CirculantConvolution{T<:fftwReal,N}(psf::Array{T,N};
                                             flags::Integer=FFTW.ESTIMATE,
                                             normalize::Bool=false, kws...)
    # Check arguments and compute dimensions.
    planning = check_flags(flags)
    n = length(psf)
    xdims = size(psf)
    zdims = ntuple(i -> (i == 1 ? div(xdims[i],2) + 1 : xdims[i]), N)

    # Allocate temporary array for the scaled MTF and, if needed, a scratch
    # array for planning which may destroy its input.
    mtf = Array(Complex{T}, zdims)
    if planning == FFTW.ESTIMATE || planning == FFTW.WISDOM_ONLY
        tmp = psf
    else
        tmp = Array(T, xdims)
    end

    # Compute the plans with suitable FFTW flags.  The forward transform (r2c)
    # must preserve its input, while the backward transform (c2r) may destroy
    # it (in fact there are no input-preserving algorithms for
    # multi-dimensional c2r transforms).
    forward = plan_rfft(tmp; flags=(planning | FFTW.PRESERVE_INPUT), kws...)
    backward = plan_brfft(mtf, xdims[1]; flags=(planning | FFTW.DESTROY_INPUT),
                          kws...)

    # Compute the scaled MTF.
    A_mul_B!(mtf, forward, psf)
    if normalize
        if real(mtf[1]) <= zero(T)
            throw(ArgumentError("cannot normalize: sum(PSF) ≤ 0"))
        end
        scl = T(real(mtf[1])/n)
    else
        scl = T(1/n)
    end
    if scl != one(T)
        scale!(mtf, scl)
    end

    # Build operator.
    CirculantConvolution{T,Complex{T},N}(xdims, zdims, mtf,
                                         forward, backward)
end

for adj in (false, true)
    f = (adj ? :apply_adjoint : :apply_direct)
    f! = Symbol(string(f, "!"))

    for R in (Float32, Float64)
        C = Complex{R}

        @eval function $f{N}(H::CirculantConvolution{$C,$C,N},
                             x::Array{$C,N})
            $f!(Array($C, size(x)), H, x)
        end

        @eval function $f!{N}(y::Array{$C,N},
                              H::CirculantConvolution{$C,$C,N},
                              x::Array{$C,N})
            A_mul_B!(y, H.forward, x)  # out-of-place forward FFT of x in y
            _multiply!(y, H.mtf, $adj) # in-place multiply y by mtf/n
            A_mul_B!(y, H.backward, y) # in-place backward FFT of y
            return y
        end

        @eval function $f{N}(H::CirculantConvolution{$R,Complex{$R},N},
                             x::Array{$R,N})
            $f!(Array($R, size(x)), H, x)
        end

        @eval function $f!{N}(y::Array{$R,N},
                              H::CirculantConvolution{$R,Complex{$R},N},
                              x::Array{$R,N})
            z = Array(Complex{$R}, H.zdims) # allocate temporary
            A_mul_B!(z, H.forward, x)       # out-of-place forward FFT of x in z
            _multiply!(z, H.mtf, $adj)      # in-place multiply z by mtf/n
            A_mul_B!(y, H.backward, z)      # out-of-place backward FFT of z in y
            return y
        end

    end

end

#------------------------------------------------------------------------------
# Fast in-place multiplication by the MTF.

doc"""
    _multiply!(arr, mtf)
    _multiply!(arr, mtf, false)

stores in `arr` the elementwise multiplication of `arr` by `mtf`, while:

    _multiply!(arr, mtf, true)

stores in `arr` the elementwise multiplication of `arr` by `conj(mtf)`.  An
error is thrown if the arrays do not have the same dimensions.

""" function _multiply! end

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

end # module
