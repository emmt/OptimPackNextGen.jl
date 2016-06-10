#
# utils.jl --
#
# General purpose methods for TiPi.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#

doc"""
`nth(n)` yields a human readable string for `n` integer like `1-st`, `2-nd`,
etc.
"""
function nth(n::Integer)
    k = abs(n)%10
    k == 1 ? string(n,"-th") :
    k == 2 ? string(n,"-nd") :
    k == 3 ? string(n,"-rd") :
    string(n,"-th")
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

dimlist{N}(dims::NTuple{N,Int}) = dims
dimlist{N}(dims::NTuple{N,Integer}) = ntuple(i -> Int(dims[i]), N)
dimlist(dims::Integer...) = ntuple(i -> Int(dims[i]), length(dims))
dimlist{N,T}(dims::Array{T,N}) = ntuple(i -> Int(dims[i]), length(dims))

function subrange(small::Int, large::Int)
    # Compte offsets to preserve the position of the center (as specified
    # by the conventions).
    1 ≤ small ≤ large || throw(BoundsError())
    offset = div(large, 2) - div(small, 2)
    offset + 1 : offset + small
end

subrange(small::Integer, large::Integer) = subrange(Int(small), Int(large))

function subrange{N}(small::NTuple{N,Integer}, large::NTuple{N,Integer})
    ntuple(i -> subrange(small[i], large[i]), N)
end

function crop{T,N}(src::AbstractArray{T,N}, dims::NTuple{N,Int})
    src[subrange(dims, size(src))...]
end

function crop{T,N}(src::AbstractArray{T,N}, dims::NTuple{N,Integer})
    crop(src, dimlist(dims))
end

function crop{T,N}(src::AbstractArray{T,N}, dims::Integer...)
    length(dims) == N || error("incompatible number of dimensions")
    crop(src, dimlist(dims))
end

function crop!{S,T,N}(dst::AbstractArray{S,N}, src::AbstractArray{T,N})
    copy!(dst, src[subrange(size(dst), size(src))...])
end

function pad{S,T,N}(val::S, src::AbstractArray{T,N}, dims::NTuple{N,Int})
    dst = Array(S, dims)
    fill!(dst, val)
    dst[subrange(size(src), dims)...] = src
    return dst
end

function pad{S,T,N}(val::S, src::AbstractArray{T,N}, dims::NTuple{N,Integer})
    pad(val, src, dimlist(dims))
end

function pad{S,T,N}(val::S, src::Array{T,N}, dims::Integer...)
    length(dims) == N || error("incompatible number of dimensions")
    pad(val, src, dimlist(dims))
end

function paste!{S,T,N}(dst::AbstractArray{S,N}, src::AbstractArray{T,N})
    dst[subrange(size(src), size(dst))...] = src
end

zeropad{T,N}(src::Array{T,N}, dims::NTuple{N,Integer}) = pad(zero(T), src, dims)

function zeropad{T,N}(src::Array{T,N}, dims::Integer...)
    length(dims) == N || error("incompatible number of dimensions")
    zeropad(src, dims)
end

"""
    crop(src, dims)

yields a subarray of dimensions `dims` consisting in the central region of the
source array `src`.  The cropped region must have as many dimensions as the
source and all the dimensions of the cropped region must be less or equal the
corresponding dimension of the source.

The in-place version:

    crop!(dst, src)

is equivalent to:

    copy(dst, crop(src, size(dst)))

### See Also
pad, zeropad, paste!.
""" crop

@doc @doc(crop) crop!

"""
    pad(val, src, dims)

yields an array of dimensions `dims` whose values are copied from the source
array `src` in the central region and set to `val` elsewhere.  The data type of
`val` determines the type of the elements of the result.  The dimensions `dims`
must be larger or equal the corresponding dimension of the source.

Zero-padding is done by:

    zeropad(src, dims)

and amounts to:

    pad(zero(eltype(src), src, dims)

### See Also
crop, crop!, paste!.
""" pad

@doc @doc(pad) zeropad

"""
    paste!(dst, src)

copies `src` into the central region of `dst` (leaving unchanged the other
elements of `dst`).  The two arrays must have the same number of dimensions and
all dimensions of the destination must be larger or equal the corresponding
dimension of the source.

### See Also
crop, crop!, pad, zeropad.
""" paste!

doc"""

    box = bounding_box(a)

yields the bounding-box of the non-zero values in array `a`.  The result is a
tuple of index bounds:

    (i1min, i1max)                   # for a 1D array
    (i1min, i1max, i2min, i2max)     # for a 2D array
    etc.

one pair for each dimension of `a`; if `i1min > i1max`, then the bounding-box
is empty.  An additional scalar argument `b` can be provided to find the
bounding-box of the region where the elements of `a` are different from `b`:

    box = bounding_box(a, b)

""" function bounding_box end

# also work for T = Bool because zero(Bool) = false.
function bounding_box{T,N}(a::AbstractArray{T,N})
    bounding_box(a, zero(T))
end

function bounding_box{T,N,B}(a::AbstractArray{T,N}, b::B)
    bounding_box(a, T(b))
end

function bounding_box{T}(a::AbstractArray{T,1}, b::T)
    n1 = length(a)
    i1min, i1max = n1 + 1, 0
    @inbounds begin
        for i1 in 1:n1
            if a[i1] != b
                i1min = i1
                break
            end
        end
        for i1 in n1:-1:i1min
            if a[i1] != b
                i1max = i1
                break
            end
        end
    end
    return (i1min, i1max)
end

function bounding_box{T}(a::AbstractArray{T,2}, b::T)
    i1min, i1max = size(a,1) + 1, 0
    i2min, i2max = size(a,2) + 1, 0
    @inbounds begin
        for i2 in 1:size(a,2)
            si1min, si1max = bounding_box(slice(a,:,i2), b)
            if si1min ≤ si1max
                i1min = min(i1min, si1min)
                i1max = max(i1max, si1max)
                i2min = min(i2min, i2)
                i2max = max(i2max, i2)
            end
        end
    end
    return (i1min, i1max, i2min, i2max)
end

function bounding_box{T}(a::AbstractArray{T,3}, b::T)
    i1min, i1max = size(a,1) + 1, 0
    i2min, i2max = size(a,2) + 1, 0
    i3min, i3max = size(a,3) + 1, 0
    @inbounds begin
        for i3 in 1:size(a,3)
            si1min, si1max, si2min, si2max = bounding_box(slice(a,:,:,i3), b)
            if si1min ≤ si1max
                i1min = min(i1min, si1min)
                i1max = max(i1max, si1max)
                i2min = min(i2min, si2min)
                i2max = max(i2max, si2max)
                i3min = min(i3min, i3)
                i3max = max(i3max, i3)
            end
        end
    end
    return (i1min, i1max, i2min, i2max, i3min, i3max)
end

function bounding_box{T}(a::AbstractArray{T,4}, b::T)
    i1min, i1max = size(a,1) + 1, 0
    i2min, i2max = size(a,2) + 1, 0
    i3min, i3max = size(a,3) + 1, 0
    i4min, i3max = size(a,4) + 1, 0
    @inbounds begin
        for i4 in 1:size(a,4)
            si1min, si1max, si2min, si2max, si3min, si3max =
                bounding_box(slice(a,:,:,:,i4), b)
            if si1min ≤ si1max
                i1min = min(i1min, si1min)
                i1max = max(i1max, si1max)
                i2min = min(i2min, si2min)
                i2max = max(i2max, si2max)
                i3min = min(i3min, si3min)
                i3max = max(i3max, si3max)
                i4min = min(i4min, i4)
                i4max = max(i4max, i4)
            end
        end
    end
    return (i1min, i1max, i2min, i2max, i3min, i3max, i4min, i4max)
end

#------------------------------------------------------------------------------
