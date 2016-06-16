#
# fft.jl --
#
# Implement FFT operator for TiPi.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#

module FFT

import Base: DFT, FFTW
import Base.FFTW: fftwNumber, fftwReal, fftwComplex

using TiPi
importall TiPi.Algebra

export FFTOperator

immutable Forward end
immutable Backward end

type FFTOperator{T<:fftwNumber,C<:fftwComplex,N} <:
    LinearOperator{Array{C,N},Array{T,N}}
    inp_dims::NTuple{N,Int}           # input dimensions
    out_dims::NTuple{N,Int}           # output dimensions
    forward::Union{Void,DFT.Plan{T}}  # plan for forward transform
    backward::Union{Void,DFT.Plan{C}} # plan for backward transform
    temp::Union{Void,Array{C,N}}      # temporary copy of input complex array
                                      # in c2r (complex-to-real transforms)
    planning::UInt32
    timelimit::Float64
    scale::T
end

# Real-to-complex FFT.
function FFTOperator{T<:fftwReal,N}(::Type{T},
                                    dims::NTuple{N,Int};
                                    flags::Integer=FFTW.ESTIMATE,
                                    timelimit::Real=FFTW.NO_TIMELIMIT)
    planning = check_flags(flags)
    number = check_dimensions(dims)
    zdims = ntuple(i -> (i == 1 ? div(dims[i],2) + 1 : dims[i]), N)
    FFTOperator{T,Complex{T},N}(dims, zdims, nothing, nothing, nothing,
                                planning, Float64(timelimit),
                                convert(T, 1/number))
end

# Complex-to-complex FFT.
function FFTOperator{T<:fftwReal,N}(::Type{Complex{T}},
                                    dims::NTuple{N,Int};
                                    flags::Integer=FFTW.ESTIMATE,
                                    timelimit::Real=FFTW.NO_TIMELIMIT)
    planning = check_flags(flags)
    number = check_dimensions(dims)
    FFTOperator{Complex{T},Complex{T},N}(dims, dims, nothing, nothing, nothing,
                                         planning, Float64(timelimit),
                                         convert(T, 1/number))
end

function FFTOperator{T<:fftwNumber,N}(arr::Array{T,N}; kws...)
    FFTOperator(eltype(arr), size(arr); kws...)
end

function temporary{T,C,N}(F::FFTOperator{T,C,N}, x::Array{T,N},
                          ::Type{Forward})
    @assert size(x) == input_size(F)
    if (F.planning & (FFTW.ESTIMATE | FFTW.WISDOM_ONLY)) != F.planning
        return Array(T, input_size(F))
    else
        return x
    end
end

function temporary{T,C,N}(F::FFTOperator{T,C,N}, x::Array{C,N},
                          ::Type{Backward})
    @assert size(x) == output_size(F)
    if (F.planning & (FFTW.ESTIMATE | FFTW.WISDOM_ONLY)) != F.planning
        return Array(C, output_size(F))
    else
        return x
    end
end

doc"""
`get_plan!(y, F, x, dir)` check arguments, allocate needed ressources
for applying the FFT operator `F` to input `x` and output `y` in direction
`dir` (either `Forward` or `Backward`) and returns the corresponding plan.
""" function get_plan! end

# Get plan for the forward complex-to-complex transform.
function get_plan!{C<:Complex,N}(y::Array{C,N}, F::FFTOperator{C,C,N},
                                 x::Array{C,N}, ::Type{Forward})
    @assert size(x) == input_size(F)
    @assert size(y) == output_size(F)
    if is(F.forward, nothing)
        F.forward = plan_fft(temporary(F, x, Forward);
                             flags=(F.planning | FFTW.PRESERVE_INPUT),
                             timelimit=F.timelimit)
    end
    return F.forward
end

# Get plan for the backward complex-to-complex transform.
function get_plan!{C<:Complex,N}(y::Array{C,N}, F::FFTOperator{C,C,N},
                                 x::Array{C,N}, ::Type{Backward})
    @assert size(x) == output_size(F)
    @assert size(y) == input_size(F)
    if is(F.backward, nothing)
        F.backward = plan_bfft(temporary(F, x, Backward);
                               flags=(F.planning | FFTW.PRESERVE_INPUT),
                               timelimit=F.timelimit)
    end
    return F.backward
end

# Get plan for the forward real-to-complex transform.
function get_plan!{T<:Real,C<:Complex,N}(y::Array{C,N},
                                         F::FFTOperator{T,C,N},
                                         x::Array{T,N}, ::Type{Forward})
    @assert size(x) == input_size(F)
    @assert size(y) == output_size(F)
    if is(F.forward, nothing)
        F.forward = plan_rfft(temporary(F, x, Forward);
                              flags=(F.planning | FFTW.PRESERVE_INPUT),
                              timelimit=F.timelimit)
    end
    return F.forward
end

# Get plan for the backward real-to-complex transform.
function get_plan!{T<:Real,C<:Complex,N}(y::Array{T,N},
                                         F::FFTOperator{T,C,N},
                                         x::Array{C,N}, ::Type{Backward})
    @assert size(x) == output_size(F)
    @assert size(y) == input_size(F)
    if is(F.temp, nothing)
        F.temp = Array(C, output_size(F))
    elseif size(F.temp) != output_size(F)
        error("corrupted FFT operator data")
    end
    if is(F.backward, nothing)
        F.backward = plan_brfft(F.temp, input_size(F, 1);
                                flags=F.planning,
                                timelimit=F.timelimit)
    end
    return F.backward
end

# Apply direct transform.
function apply_direct{T,C,N}(F::FFTOperator{T,C,N}, x::Array{T,N})
    apply_direct!(Array(C, output_size(F)), F, x)
end

# Apply complex-to-complex and real-to-complex forward transform.
function apply_direct!{T,C,N}(y::Array{C,N}, F::FFTOperator{T,C,N},
                              x::Array{T,N})
    A_mul_B!(y, get_plan!(y, F, x, Forward), x)
end

# Apply adjoint transform.
function apply_adjoint{T,C,N}(F::FFTOperator{T,C,N}, x::Array{C,N})
    apply_adjoint!(Array(T, F.inp_dims), F, x)
end

# Apply complex-to-complex backward transform.
function apply_adjoint!{C<:Complex,N}(y::Array{C,N}, F::FFTOperator{C,C,N},
                                      x::Array{C,N})
    A_mul_B!(y, get_plan!(y, F, x, Backward), x)
end

# Apply real-to-complex backward transform.
function apply_adjoint!{T<:Real,C<:Complex,N}(y::Array{T,N},
                                              F::FFTOperator{T,C,N},
                                              x::Array{C,N})
    plan = get_plan!(y, F, x, Backward)
    temp = F.temp
    if pointer(x) != pointer(temp)
        @simd for i in 1:length(temp)
            @inbounds temp[i] = x[i]
        end
    end
    A_mul_B!(y, plan, temp)
end

# Apply inverse transform.
apply_inverse{T,C,N}(F::FFTOperator{T,C,N}, x::Array{C,N}) =
    apply_inverse!(Array(T, F.inp_dims), F, x)

# Apply complex-to-complex inverse transform with destination given.
function apply_inverse!{C<:Complex,N}(y::Array{C,N}, F::FFTOperator{C,C,N},
                                      x::Array{C,N})
    scale!(A_mul_B!(y, get_plan!(y, F, x, Backward), x), F)
end

# Apply real-to-complex inverse transform with destination given.
function apply_inverse!{T<:Real,C<:Complex,N}(y::Array{T,N},
                                              F::FFTOperator{T,C,N},
                                              x::Array{C,N})
    A_mul_B!(y, get_plan!(y, F, x, Backward), scale!(F.temp, F, x))
end

# Apply adjoint of inverse transform.
function apply_inverse_adjoint{T,C,N}(F::FFTOperator{T,C,N}, x::Array{T,N})
    apply_inverse_adjoint!(Array(C, output_size(F)), F, x)
end

# Apply complex-to-complex and real-to-complex forward transform.
function apply_inverse_adjoint!{T,C,N}(y::Array{C,N}, F::FFTOperator{T,C,N},
                                       x::Array{T,N})
    scale!(A_mul_B!(y, get_plan!(y, F, x, Forward), x), F)
end

input_size(F::FFTOperator) = F.inp_dims
input_size(F::FFTOperator, i::Integer) = F.inp_dims[i]
output_size(F::FFTOperator) = F.out_dims
output_size(F::FFTOperator, i::Integer) = F.out_dims[i]
input_ndims{T,C,N}(F::FFTOperator{T,C,N}) = N
output_ndims{T,C,N}(F::FFTOperator{T,C,N}) = N
input_eltype{T,C,N}(F::FFTOperator{T,C,N}) = T
output_eltype{T,C,N}(F::FFTOperator{T,C,N}) = C

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
`check_dimensions(dims)` checks whether the list of dimensions `dims` is
correct and returns the corresponding total number of elements."""
function check_dimensions{N}(dims::NTuple{N,Int})
    number = 1
    for i in 1:length(dims)
        dim = dims[i]
        if dim < 1
            error("invalid dimension(s)")
        end
        number *= dim
    end
    return number
end

function scale!{S,T,C,N}(dst::Array{S,N}, F::FFTOperator{T,C,N})
    @inbounds begin
        const scale = F.scale
        @simd for i in 1:length(dst)
            dst[i] *= scale
        end
    end
    return dst
end

function scale!{S,T,C,N}(dst::Array{S,N}, F::FFTOperator{T,C,N}, src::Array{S,N})
    @assert size(src) == size(dst)
    @inbounds begin
        const scale = F.scale
        @simd for i in 1:length(dst)
            dst[i] = scale*src[i]
        end
    end
    return dst
end

end # module
