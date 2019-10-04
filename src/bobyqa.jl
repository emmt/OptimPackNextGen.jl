#
# bobyqa.jl --
#
# Julia interface to Mike Powell's BOBYQA method.
#
# -----------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License:
#
# Copyright (C) 2015-2019, Éric Thiébaut
# <https://github.com/emmt/OptimPackNextGen.jl>.
#

module Bobyqa

export
    bobyqa,
    bobyqa!

using Compat
using Compat.Printf

import
    ..AbstractContext,
    ..AbstractStatus,
    ..getreason,
    ..getstatus,
    ..grow!,
    ..iterate,
    ..restart

# The dynamic library implementing the method.
import .._libbobyqa
const DLL = _libbobyqa

# Status returned by most functions of the library.
struct Status <: AbstractStatus
    _code::Cint
end

const SUCCESS              = Status( 0)
const BAD_NVARS            = Status(-1)
const BAD_NPT              = Status(-2)
const BAD_RHO_RANGE        = Status(-3)
const BAD_SCALING          = Status(-4)
const TOO_CLOSE            = Status(-5)
const ROUNDING_ERRORS      = Status(-6)
const TOO_MANY_EVALUATIONS = Status(-7)
const STEP_FAILED          = Status(-8)

# Get a textual explanation of the status returned by BOBYQA.
function getreason(status::Status)
    ptr = ccall((:bobyqa_reason, DLL), Ptr{UInt8}, (Cint,), status._code)
    if ptr == C_NULL
        error("unknown BOBYQA status: ", status._code)
    end
    unsafe_string(ptr)
end

# `_wrklen(...)` yields the number of elements in BOBYQA workspace.
_wrklen(n::Integer, npt::Integer) = _wrklen(Int(n), Int(npt))
_wrklen(n::Int, npt::Int) = (npt + 5)*(npt + n) + div(3*n*(n + 5),2)
_wrklen(x::AbstractVector{<:AbstractFloat}, npt::Integer) =
    _wrklen(length(x), npt)
function _wrklen(x::AbstractVector{<:AbstractFloat},
                 scl::AbstractVector{<:AbstractFloat},
                 npt::Integer)
    return _wrklen(x, npt) + 3*length(scl)
end

# `_work(...)` yields a large enough workspace for NEWUOA.
_work(x::AbstractVector{<:AbstractFloat}, npt::Integer) =
    Vector{Cdouble}(undef, _wrklen(x, npt))
function _work(x::AbstractVector{<:AbstractFloat},
               scl::AbstractVector{<:AbstractFloat},
               npt::Integer)
    return Vector{Cdouble}(undef, _wrklen(x, scl, npt))
end

# Wrapper for the objective function in BOBYQA, the actual objective function
# is provided by the client data as a `jl_value_t*` pointer.
function _objfun(n::Cptrdiff_t, xptr::Ptr{Cdouble}, fptr::Ptr{Cvoid})::Cdouble
    x = unsafe_wrap(Array, xptr, n)
    f = unsafe_pointer_to_objref(fptr)
    return Cdouble(f(x))
end

# With precompilation, `__init__()` carries on initializations that must occur
# at runtime like `@cfunction` which returns a raw pointer.
const _objfun_c = Ref{Ptr{Cvoid}}()
function __init__()
    _objfun_c[] = @cfunction(_objfun, Cdouble,
                             (Cptrdiff_t, Ptr{Cdouble}, Ptr{Cvoid}))
end

function optimize!(f::Function, x::DenseVector{Cdouble},
                   xl::DenseVector{Cdouble}, xu::DenseVector{Cdouble},
                   rhobeg::Real, rhoend::Real;
                   scale::DenseVector{Cdouble}=Array{Cdouble}(undef, 0),
                   maximize::Bool = false,
                   npt::Integer = 2*length(x) + 1,
                   check::Bool = false,
                   verbose::Integer = 0,
                   maxeval::Integer = 30*length(x),
                   work::Vector{Cdouble} = _work(x, scale, npt))
    n = length(x)
    length(xl) == n || error("bad length for inferior bound")
    length(xu) == n || error("bad length for superior bound")
    nscl = length(scale)
    if nscl == 0
        sclptr = Ptr{Cdouble}(0)
    elseif nscl == n
        sclptr = pointer(scale)
    else
        error("bad number of scaling factors")
    end
    grow!(work, _wrklen(x, scale, npt))
    status = Status(ccall((:bobyqa_optimize, DLL), Cint,
                          (Cptrdiff_t, Cptrdiff_t, Cint, Ptr{Cvoid}, Any,
                           Ptr{Cdouble}, Ptr{Cdouble},
                           Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Cdouble,
                           Cptrdiff_t, Cptrdiff_t, Ptr{Cdouble}),
                          n, npt, (maximize ? Cint(1) : Cint(0)),
                          _objfun_c[], f, x, xl, xu, sclptr, rhobeg, rhoend,
                          verbose, maxeval, work))
    if check && status != SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

optimize(f::Function, x0::AbstractVector{<:Real}, args...; kwds...) =
    optimize!(f, copyto!(Array{Cdouble}(undef, length(x0)), x0),
              args...; kwds...)

minimize!(args...; kwds...) = optimize!(args...; maximize=false, kwds...)
maximize!(args...; kwds...) = optimize!(args...; maximize=true, kwds...)

minimize(args...; kwds...) = optimize(args...; maximize=false, kwds...)
maximize(args...; kwds...) = optimize(args...; maximize=true, kwds...)

function bobyqa!(f::Function, x::DenseVector{Cdouble},
                 xl::DenseVector{Cdouble}, xu::DenseVector{Cdouble},
                 rhobeg::Real, rhoend::Real;
                 npt::Integer = 2*length(x) + 1,
                 verbose::Integer = 0,
                 maxeval::Integer = 30*length(x),
                 check::Bool = true,
                 work::Vector{Cdouble} = _work(x, npt))
    n = length(x)
    length(xl) == n || error("bad length for inferior bound")
    length(xu) == n || error("bad length for superior bound")
    grow!(work, _wrklen(x, npt))
    status = Status(ccall((:bobyqa, DLL), Cint,
                          (Cptrdiff_t, Cptrdiff_t, Ptr{Cvoid}, Any,
                           Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
                           Cdouble, Cdouble, Cptrdiff_t, Cptrdiff_t,
                           Ptr{Cdouble}),
                          n, npt, _objfun_c[], f, x, xl, xu,
                          rhobeg, rhoend, verbose, maxeval, work))
    if check && status != SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

bobyqa(f::Function, x0::DenseVector{Cdouble}, args...; kwds...) =
    bobyqa!(f, copy(x0), args...; kwds...)

end # module Bobyqa
