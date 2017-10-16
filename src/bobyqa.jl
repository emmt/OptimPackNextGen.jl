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
# Copyright (C) 2015-2017, Éric Thiébaut.
#

module Bobyqa

export
    bobyqa,
    bobyqa!

# FIXME: with Julia 0.5 all relative (prefixed by .. or ...) symbols must be
#        on the same line as `import`
import ..AbstractStatus, ..AbstractContext, ..getreason, ..getstatus, ..iterate, ..restart,  .._libbobyqa

# The dynamic library implementing the method.
const _LIB = _libbobyqa

# Status returned by most functions of the library.
immutable Status <: AbstractStatus
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
    ptr = ccall((:bobyqa_reason, _LIB), Ptr{UInt8}, (Cint,), status._code)
    if ptr == C_NULL
        error("unknown BOBYQA status: ", status._code)
    end
    unsafe_string(ptr)
end

# Yield the number of elements in BOBYQA workspace.
_wslen(n::Integer, npt::Integer) =
    (npt + 5)*(npt + n) + div(3*n*(n + 5),2)

# Wrapper for the objective function in BOBYQA, the actual objective function
# is provided by the client data.
function _objfun(n::Cptrdiff_t, xptr::Ptr{Cdouble}, fptr::Ptr{Void})
    x = unsafe_wrap(Array, xptr, n)
    f = unsafe_pointer_to_objref(fptr)
    convert(Cdouble, f(x))::Cdouble
end

# With precompilation, `__init__()` carries on initializations that must occur
# at runtime like `cfunction` which returns a raw pointer.
const _objfun_c = Ref{Ptr{Void}}()
function __init__()
    _objfun_c[] = cfunction(_objfun, Cdouble,
                            (Cptrdiff_t, Ptr{Cdouble}, Ptr{Void}))
end

function optimize!(f::Function, x::DenseVector{Cdouble},
                   xl::DenseVector{Cdouble}, xu::DenseVector{Cdouble},
                   rhobeg::Real, rhoend::Real;
                   scale::DenseVector{Cdouble}=Array{Cdouble}(0),
                   maximize::Bool=false,
                   npt::Integer=2*length(x) + 1,
                   check::Bool=false,
                   verbose::Integer=0,
                   maxeval::Integer=30*length(x))
    n = length(x)
    length(xl) == n || error("bad length for inferior bound")
    length(xu) == n || error("bad length for superior bound")
    nw = _wslen(n, npt)
    nscl = length(scale)
    if nscl == 0
        sclptr = convert(Ptr{Cdouble}, C_NULL)
    elseif nscl == n
        sclptr = pointer(scale)
        nw += 3*n
    else
        error("bad number of scaling factors")
    end
    work = Array{Cdouble}(nw)
    status = Status(ccall((:bobyqa_optimize, _LIB), Cint,
                          (Cptrdiff_t, Cptrdiff_t, Cint, Ptr{Void},
                           Ptr{Void}, Ptr{Cdouble}, Ptr{Cdouble},
                           Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Cdouble,
                           Cptrdiff_t, Cptrdiff_t, Ptr{Cdouble}),
                          n, npt, (maximize ? Cint(1) : Cint(0)),
                          _objfun_c[], pointer_from_objref(f),
                          x, xl, xu, sclptr, rhobeg, rhoend,
                          verbose, maxeval, work))
    if check && status != SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

optimize(f::Function, x0::DenseVector{Cdouble}, args...; kwds...) =
    optimize!(f, copy(x0), args...; kwds...)

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
                 check::Bool = true)
    n = length(x)
    length(xl) == n || error("bad length for inferior bound")
    length(xu) == n || error("bad length for superior bound")
    work = Array{Cdouble}(_wslen(n, npt))
    status = Status(ccall((:bobyqa, _LIB), Cint,
                          (Cptrdiff_t, Cptrdiff_t, Ptr{Void}, Ptr{Void},
                           Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
                           Cdouble, Cdouble, Cptrdiff_t, Cptrdiff_t,
                           Ptr{Cdouble}),
                          n, npt, _objfun_c[],
                          pointer_from_objref(f), x, xl, xu,
                          rhobeg, rhoend, verbose, maxeval, work))
    if check && status != SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

bobyqa(f::Function, x0::DenseVector{Cdouble}, args...; kwds...) =
    bobyqa!(f, copy(x0), args...; kwds...)

function runtests()
    # The test function.
    function f(x::DenseVector{Cdouble})
        fx = 0.0
        n = length(x)
        for i in 4:2:n
            for j in 2:2:i-2
                tempa = x[i - 1] - x[j - 1]
                tempb = x[i] - x[j]
                temp = tempa*tempa + tempb*tempb
                temp = max(temp,1e-6)
                fx += 1.0/sqrt(temp)
            end
        end
        return fx
    end

    # Run the tests.
    bdl = -1.0
    bdu =  1.0
    rhobeg = 0.1
    rhoend = 1e-6
    for m in (5,10)
        q = 2.0*pi/m
        n = 2*m
        x = Array{Cdouble}(n)
        xl = Array{Cdouble}(n)
        xu = Array{Cdouble}(n)
        for i in 1:n
            xl[i] = bdl
            xu[i] = bdu
        end
        for jcase in 1:2
            if jcase == 2
                npt = 2*n + 1
            else
                npt = n + 6
            end
            @printf("\n\n     2D output with M =%4ld,  N =%4ld  and  NPT =%4ld\n",
                    m, n, npt)
            for j in 1:m
                temp = q*j
                x[2*j - 1] = cos(temp)
                x[2*j]     = sin(temp)
            end
            fx = bobyqa!(f, x, xl, xu, rhobeg, rhoend, npt=npt,
                         verbose=2, maxeval=500000)[3]
            @printf("\n***** least function value: %.15e\n", fx)
        end
    end
end

end # module Bobyqa
