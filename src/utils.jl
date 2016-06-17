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

import Base: size, convert, length, ndims, get, getindex,
             start, done, next, range, first, last

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

doc"""
# Containers and Marked Objects

Julia typing system can be exploited to "mark" some object instances so that
they are seen as another specific type.  For instance, this feature is used to
mark linear operators as being transposed (so that they behave as their
adjoint) or inverted (so that they behave as their inverse).

These marked objects have a single member: the object that is marked.  This
single member can be retrieved by the `contents` method.  The following piece
of code shows the idea:

    immutable MarkedType{T}
        data::T
    end
    MarkedType{T}(obj::T) = MarkedType{T}(obj)
    contents(obj::MarkedType) = obj.data

More generally, the `contents` method can be used to retrieve the contents of a
"container" object:

    contents(container)

"""
function contents end

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

#------------------------------------------------------------------------------
# BOUNDING BOXES

doc"""
# Bounding Boxes

A bounding-box is meant to represent a rectangular region of interest in an
array-like object.

It is possible to automatically detect the region with significant values
of an array `A`:

    BoundingBox(A, b)
    BoundingBox(A)

where `b` is the value of the elements of `A` that shloub be considered as
being outside of the region of interest.  If not specified, `b =
zero(eltype(A))`.

Some array-like methods are implemented for a bounding box `bbox`:

* `length(bbox)` yields the number of elements enclosed in the bounding box.

* `size(bbox)` yields the dimensions of the region enclosed by the bounding box.
  `size(bbox, i)` yields the `i`-th dimension of this region.

* `ndims(bbox)` yields the number of dimensions of the region enclosed by the
  bounding box.

Bounding boxes can be used as a `CartesianRange` to iterate in the positions of
the bounding boxes:

    for i in box
        A[i] = ...
    end

Bounding boxes are indexable:

    bbox[i]

yields the `UnitRange` of indices corresponding to the `i`-th dimension of the
region enclosed by the bounding box.

Finally:

    first(bbox)  :: CartesianIndex
    last(bbox)   :: CartesianIndex
    range(bbox)  :: CartesianRange

yields the Cartesian index corresponding the the first and last corner of the
bounding box and Cartesian range of the whole region.

"""
immutable BoundingBox{N}
    rng::CartesianRange{CartesianIndex{N}}
    siz::NTuple{N,Int}
    len::Int
end

# FIXME: this constructor is not very efficient
function BoundingBox{N}(rng::CartesianRange{CartesianIndex{N}})
    tmp = last(rng) - start(rng)
    siz = ntuple(d->max(tmp[d] + 1, 0), N)
    len = 1
    for d in 1:N
        len *= siz[d]
    end
    BoundingBox{N}(rng, siz, len)
end

BoundingBox{N}(siz::NTuple{N,Int}) = BoundingBox(CartesianRange(siz))

function BoundingBox{T,N}(a::AbstractArray{T,N}, b=zero(T))
    tup = bounding_box(a, b)
    @assert length(tup) == 2*N
    first = CartesianIndex(ntuple(d->tup[2*d-1], N))
    last = CartesianIndex(ntuple(d->tup[2*d], N))
    BoundingBox(CartesianRange(first, last))
end

convert{N}(::Type{CartesianRange{CartesianIndex{N}}}, bbox::BoundingBox{N}) = bbox.rng
size(bbox::BoundingBox) = bbox.siz
size(bbox::BoundingBox, i::Integer) = bbox.siz[i]
length(bbox::BoundingBox) = bbox.len
ndims{N}(::BoundingBox{N}) = N
getindex{N}(bbox::BoundingBox{N}, i::Integer) = first(bbox)[i]:last(bbox)[i]
get{N}(bbox::BoundingBox{N}, i::Integer, def) = (1 ≤ i ≤ N ? bbox[i] : def)
start(bbox::BoundingBox) = start(bbox.rng)
done(bbox::BoundingBox, state) = done(bbox.rng, state)
next(bbox::BoundingBox, state) = next(bbox.rng, state)
first(bbox::BoundingBox) = first(bbox.rng)
last(bbox::BoundingBox) = last(bbox.rng)
range(bbox::BoundingBox) = bbox.rng

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

function bounding_box{T,N}(a::AbstractArray{T,N}, b)
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

# FIXME: there is a better algorithm for rank > 1
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
