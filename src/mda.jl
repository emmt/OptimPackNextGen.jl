#
# mda.jl --
#
# Implement reading/writing MDA data files in Julia.
#
#------------------------------------------------------------------------------
#
# This file is part of TiPi.jl licensed under the MIT "Expat" License.
#
# Copyright (C) 2015, Éric Thiébaut & Jonathan Léger.
#
#------------------------------------------------------------------------------

# MDA (for *Multi-Dimensional Array*) is a simple file format to store
# multi-dimensional (rectangular) arrays of various types and dimensions.  The
# format adds very little overhead to describe the array and supports
# reading and writing for different byte orders and to/from files or streams
# (reading/writing is sequential and does not require seekable streams).
#
# Usage:
#
#     MDA.read(name)  - reads the data from file `name`
#     MDA.read(inp)   - read the next MDA array from input stream `inp`
#
#     MDA.write(arr, name, order) - writes an array `arr` to file `name`
#                                   with given byte order (default is to use
#                                   native byte order)
#     MDA.write(arr, out, order)  - writes an array `arr` to output stream
#                                   `out` with given byte order (default is
#                                   to use native byte order)

# Note: The reading/writing is ~ 10 times faster with `bswap` than
#       with my own byte-swapping with unrolled loops (even with the
#       @inbounds macro) and the memory footprint is ~ 5 times
#       smaller.  However Yorick is ~ 7 times faster with its built-in
#       capabilities.  Without byte swapping Julia and Yorick are
#       equally fast.

module MDA

# All implemented types (the complex ones are non-standard):
IMPLEMENTED_TYPES = (( 1, Int8,       "signed 8-bit integer"),
                     ( 2, UInt8,      "unsigned 8-bit integer"),
                     ( 3, Int16,      "signed 16-bit integer"),
                     ( 4, UInt16,     "unsigned 16-bit integer"),
                     ( 5, Int32,      "signed 32-bit integer"),
                     ( 6, UInt32,     "unsigned 32-bit integer"),
                     ( 7, Int64,      "signed 64-bit integer"),
                     ( 8, UInt64,     "unsigned 64-bit integer"),
                     ( 9, Float32,    "32-bit floating-point"),
                     (10, Float64,    "64-bit floating-point"),
                     (11, Complex64,  "64-bit complex"),
                     (12, Complex128, "128-bit complex"))

const TYPE_TABLE = Dict{DataType,Int}()
for (code, T, descr) in IMPLEMENTED_TYPES
    TYPE_TABLE[T] = code
end

# We define a specific type to change the signatures of the read/write
# methods and thus overload them.
immutable ByteOrder
    signature::UInt32
    function ByteOrder(value::UInt32)
        if value != 0x04030201 && value != 0x01020304
            error("unsupported byte order")
        end
        return new(value)
    end
end

const BIG_ENDIAN = ByteOrder(0x01020304)
const LITTLE_ENDIAN = ByteOrder(0x04030201)
const NATIVE_BYTE_ORDER = ByteOrder(ENDIAN_BOM)

ByteOrder() = NATIVE_BYTE_ORDER

function ByteOrder(name::AbstractString)
    name == "native" ? NATIVE_BYTE_ORDER :
    name == "big"    ? BIG_ENDIAN        :
    name == "little" ? LITTLE_ENDIAN     :
    error("unknown byte order ($name)")
end

function ByteOrder(symb::Symbol)
    name == :native ? NATIVE_BYTE_ORDER :
    name == :big    ? BIG_ENDIAN        :
    name == :little ? LITTLE_ENDIAN     :
    error("unknown byte order ($symb)")
end

function write(arr::Array, name::AbstractString, order::Union{AbstractString,Symbol})
    write(arr, name, ByteOrder(order))
end

function write{T,N}(arr::Array{T,N}, name::AbstractString,
                    order::ByteOrder=NATIVE_BYTE_ORDER)
    haskey(TYPE_TABLE, T) || error("unsupported data type")
    code = TYPE_TABLE[T]
    dims = size(arr)
    if maximum(dims) > typemax(UInt32)
        error("dimensions are too large (integer overflow)")
    end
    hdr = Array(UInt32, 1 + N)
    hdr[1] = UInt32(0x4D444100 | (code << 4) | N)
    for k in 1:N
        hdr[k + 1] = UInt32(dims[k])
    end
    swap = swapping(order)
    Base.open(name, "w") do out
        write(out, hdr, swap)
        write(out, arr, swap)
    end
end

function read(name::AbstractString)
    global IMPLEMENTED_TYPES, BIG_ENDIAN, LITTLE_ENDIAN
    ident = Array(UInt8, 4)
    Base.open(name, "r") do inp
        Base.read!(inp, ident)
        if ident[1] == 0x4D && ident[2] == 0x44 && ident[3] == 0x41
            info = UInt(ident[4])
            order = BIG_ENDIAN
        elseif ident[4] == 0x4D && ident[3] == 0x44 && ident[2] == 0x41
            info = UInt(ident[1])
            order = LITTLE_ENDIAN
        else
            error("not a MDA file/stream")
        end
        typeId = int((info >> 4) & 0xF)
        rank = int(info & 0xF)
        if typeId <= 0 || typeId > length(IMPLEMENTED_TYPES)
            error("illegal data type in MDA file/stream")
        end
        dataType = IMPLEMENTED_TYPES[typeId][2]
        dims = read(inp, order, UInt32, rank)
        return read(inp, order, dataType, int(dims)...)
    end
end

function swapping(order::ByteOrder)
    if order.signature == BIG_ENDIAN.signature || order.signature == LITTLE_ENDIAN.signature
        return swap = false
    else
        swap = (sizeof(T) > 1)
    end
end

# Write the contents of the array `arr` to the `stream` in binary
# format with or without byte swapping depending on byte order in
# `order`.
function write{T<:Real,N}(stream::IO, arr::Array{T,N}, swap::Bool)
    # Figure out whether byte swapping is required.
    if swap && sizeof(T) > 1
        # Write with byte swapping.
        bufsiz = div(8192, sizeof(T))
        number = length(arr)
        if number <= bufsiz
            # Write all at once.
            Base.write(stream, map(bswap, arr))
        else
            # Write data by pieces.
            if N != 1
                arr = reshape(arr, number)
            end
            offset = 0
            while offset < number
                n = min(bufsiz, number - offset)
                Base.write(stream, map(bswap, arr[offset + 1 : offset + n]))
                offset += n
            end
        end
    else
        # No byte swapping is needed.
        Base.write(stream, arr)
        return
    end
    nothing
end

function write{T<:Real}(stream::IO, arr::Array{Complex{T}}, swap::Bool)
    write(stream, reinterpret(T, arr, (2*length(arr),)), swap)
end

# Read an array of element type `T` and dimensions `dims` from the
# `stream` in binary format with byte swapping if required by the byte
# ordering in `order`.  The array is returned.
function read{T}(stream::IO, order::ByteOrder, ::Type{T}, dims::Int...)
    return read(stream, order, T, dims)
end
function read{T,N}(stream::IO, order::ByteOrder, ::Type{T},
                   dims::NTuple{N,Int})
    arr = Array(T, dims)
    read!(stream, order, arr)
    return arr
end

# Read the contents of the array `arr` from `stream` in binary
# format with byte swapping if required by the byte ordering in
# `order`.  The contents of the array is modified.
function read!{T<:Real,N}(stream::IO, order::ByteOrder, arr::Array{T,N})
    # Read all data.
    Base.read!(stream, arr)
    if order.signature != ENDIAN_BOM && sizeof(T) > 1
        # Swap bytes if storage byte order is different from native
        # byte order and element size is greater than 1.
        map!(bswap, arr)
    end
end
function read!{T<:Real,N}(stream::IO, order::ByteOrder,
                          arr::Array{Complex{T},N})
    read!(stream, order, reinterpret(T, arr, (2*length(arr),)))
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
