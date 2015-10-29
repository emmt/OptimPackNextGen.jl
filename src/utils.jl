#
# utils.jl --
#
# General purpose routines for TiPi.
#
#------------------------------------------------------------------------------
#
# This file is part of TiPi.jl licensed under the MIT "Expat" License.
#
# Copyright (C) 2015, Éric Thiébaut & Jonathan Léger.
#
#------------------------------------------------------------------------------

function defaultweights{T<:AbstractFloat,N}(y::Array{T,N})
    wgt = Array(T, size(y))
    for i in 1:length(wgt)
        wgt[i] = (isnan(y[i]) ? zero(T) : one(T))
    end
    return wgt
end

fftbestdim{T<:Integer}(n::T) = nextprod([2,3,5], n)

zeropad{T,N}(arr::Array{T,N}, dims::NTuple{N,Int}) = pad(zero(T), arr, dims)

function zeropad{T,N}(arr::Array{T,N}, dims::Integer...)
    length(dims) == N || error("incompatible number of dimensions")
    pad(zero(T), arr, ntuple(N, i -> int(dims[i])))
end

function pad{S,T,N}(val::S, arr::Array{T,N}, dims::Integer...)
    length(dims) == N || error("incompatible number of dimensions")
    pad(val, arr, ntuple(N, i -> int(dims[i])))
end

function pad{S,T,N}(val::S, src::Array{T,N}, dstDims::NTuple{N,Int})
    if N < 1 || length(dstDims) != N
        error("bad number of dimensions")
    end

    # Compte offsets to preserve the position of the center (as specified
    # by the conventions).
    srcDims = size(src)
    r = Array(UnitRange{Int}, N)
    for k in 1:N
        srcDim = srcDims[k]
        dstDim = dstDims[k]
        if dstDim < srcDim
            error("dimensions of result must be at least as large as those of the input")
        end
        offset = div(dstDim,2) - div(srcDim,2)
        r[k] = offset + 1 : offset + srcDim
    end

    # Create destination array (initially filled with the padding value)
    # and block-copy the source.
    dst = Array(S, dstDims)
    fill!(dst, val)
    if N == 1
        dst[r[1]] = src
    elseif N == 2
        dst[r[1], r[2]] = src
    elseif N == 3
        dst[r[1], r[2], r[3]] = src
    elseif N == 4
        dst[r[1], r[2], r[3], r[4]] = src
    elseif N == 5
        dst[r[1], r[2], r[3], r[4], r[5]] = src
    elseif N == 6
        dst[r[1], r[2], r[3], r[4], r[5], r[6]] = src
    elseif N == 7
        dst[r[1], r[2], r[3], r[4], r[5], r[6], r[7]] = src
    elseif N == 8
        dst[r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]] = src
    elseif N == 9
        dst[r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]] = src
    elseif N == 10
        dst[r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]] = src
    else
        error("too many dimensions")
    end
    return dst
end


# Local Variables:
# mode: Julia
# tab-width: 8
# indent-tabs-mode: nil
# fill-column: 79
# coding: utf-8
# ispell-local-dictionary: "american"
# End:
