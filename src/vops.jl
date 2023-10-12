module VectOps

export
    as,
    vcombine!,
    vcopy!,
    vcopy,
    vcreate,
    vdot,
    vnorm1,
    vnorm2,
    vnorminf,
    vproduct!,
    vscale!,
    vupdate!

import NumOptBase

vcreate(x::AbstractArray) = similar(x)

vcopy(x::AbstractArray) = copyto!(similar(x), x)

function vcopy!(dst::AbstractArray, src::AbstractArray)
    NumOptBase.check_axes(dst, src)
    return copyto!(dst, src)
end

const vscale!   = NumOptBase.scale!
const vcombine! = NumOptBase.combine!
const vupdate!  = NumOptBase.update!
const vproduct! = NumOptBase.multiply!

function vdot(x::AbstractArray{<:Number,N},
              y::AbstractArray{<:Number,N}) where {N}
    return NumOptBase.inner(x, y)
end

function vdot(::Type{T},
              x::AbstractArray{<:Number,N},
              y::AbstractArray{<:Number,N}) where {T<:Number, N}
    return as(T, vdot(x, y))
end

function vdot(w::AbstractArray{<:Number,N},
              x::AbstractArray{<:Number,N},
              y::AbstractArray{<:Number,N}) where {N}
    return NumOptBase.inner(w, x, y)
end

function vdot(::Type{T},
              w::AbstractArray{<:Number,N},
              x::AbstractArray{<:Number,N},
              y::AbstractArray{<:Number,N}) where {T<:Number, N}
    return as(T, vdot(w, x, y))
end

for norm in (:norm1, :norm2, :norminf)
    vnorm = Symbol("v", norm)
    @eval begin
        $vnorm(x::AbstractArray) = NumOptBase.$norm(x)
        $vnorm(::Type{T}, x::AbstractArray) where {T<:Number} = as(T, $vnorm(x))
    end
end

# FIXME: Use TypeUtils.
as(::Type{T}, x::T) where {T} = x
as(::Type{T}, x::Any) where {T} = convert(T, x)::T

end # module
