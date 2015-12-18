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

"""
### Get good dimension length for the FFT

```
    goodfftdim(len)
```
returns the smallest integer which is greater or equal `len` and which is a
multiple of powers of 2, 3 and/or 5.

"""
goodfftdim(n::Integer) = nextprod([2,3,5], n)

"""
### Generate Discrete Fourier Transform frequency indexes or frequencies

Syntax:
```
    k = fftfreq(dim)
    f = fftfreq(dim, step)
```

With a single argument, the function returns a vector of `dim` values set with
the frequency indexes:
```
    k = [0, 1, 2, ..., n-1, -n, ..., -2, -1]   if dim = 2*n
    k = [0, 1, 2, ..., n,   -n, ..., -2, -1]   if dim = 2*n + 1
```
depending whther `dim` is even or odd.  These rules are compatible to what is
assumed by `fftshift` (which to see) in the sense that:
```
    fftshift(fftfreq(dim)) = [-n, ..., -2, -1, 0, 1, 2, ...]
```

With two arguments, `step` is the sample spacing in the direct space and the
result is a floating point vector with `dim` elements set with the frequency
bin centers in cycles per unit of the sample spacing (with zero at the start).
For instance, if the sample spacing is in seconds, then the frequency unit is
cycles/second.  This is equivalent to:
```
     fftfreq(dim)/(dim*step)
```

See also: `fft`, `fftshift`.
"""
function fftfreq(dim::Integer)
    dim = Int(dim)
    n = div(dim, 2)
    f = Array(Int, dim)
    for k in 1:dim-n
        f[k] = k - 1
    end
    for k in dim-n+1:dim
        f[k] = k - (1 + dim)
    end
    return f
end

function fftfreq(dim::Integer, step::Real)
    dim = Int(dim)
    scl = Cdouble(1/(dim*step))
    n = div(dim, 2)
    f = Array(Cdouble, dim)
    for k in 1:dim-n
        f[k] = (k - 1)*scl
    end
    for k in dim-n+1:dim
        f[k] = (k - (1 + dim))*scl
    end
    return f
end

zeropad{T,N}(arr::Array{T,N}, dims::NTuple{N,Int}) = pad(zero(T), arr, dims)

function zeropad{T,N}(arr::Array{T,N}, dims::Integer...)
    length(dims) == N || error("incompatible number of dimensions")
    pad(zero(T), arr, ntuple(i -> int(dims[i]), N))
end

function pad{S,T,N}(val::S, arr::Array{T,N}, dims::Integer...)
    length(dims) == N || error("incompatible number of dimensions")
    pad(val, arr, ntuple(i -> int(dims[i]), N))
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
