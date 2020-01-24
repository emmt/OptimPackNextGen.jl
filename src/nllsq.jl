#
# nllsq.jl --
#
# Non-linear least squares fit with Mike Powell's NEWUOA method.
#
# ----------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License:
#
# Copyright (C) 2018-2019, Éric Thiébaut.
#
# ----------------------------------------------------------------------------
module NonLinearLeastSquares

export
    nllsq,
    nllsq!

using ArrayTools
using Printf
using ..Powell

"""
```julia
nllsq([w,] y, f, p0, x) -> pmin, cmin
```

performs a non-linear (weighted) least fit of the data `y` by the model
`f(p,x)` where `f` is a function, `p` are the model parameters (a vector of
floating point values) and `x` are the explanatory variables (they are left
unmodifed by the method and are just passed to the function). Argument `p0`
specifies the initial values of the parameters.  The best parameters `pmin`
and the corresponding objective function `cmin` are returned.  The objective
function is defined as:

```
c(p) = ‖y - f(p,x)‖²
```

or as:

```
c(p) = ‖y - f(p,x)‖²_W
```

with `W = diag(w)` if the weights `w` are specified.

Keyword `rho` can be set with `(rhobeg,rhoend)` the initial and final size of
the trust region relative to the scaling of the parameters.

Keyword `scale` can be set with the typical size of the parameters.  By
default, `scale[i] = max(abs(p0[i]), sqrt(eps))`.

Other keywords are passed to `newuoa`.

See also: [`newuoa`](@ref).
"""
function nllsq(y::AbstractArray{<:Real,N},
               f::Function,
               p0::AbstractVector{<:Real},
               x; kwds...) where {N}
    return nllsq!(y, f, copyparameters(p0), x; kwds...)
end

function nllsq(w::AbstractArray{<:Real,N},
               y::AbstractArray{<:Real,N},
               f::Function,
               p0::AbstractVector{<:Real},
               x; kwds...) where {N}
    return nllsq!(w, y, f, copyparameters(p0), x; kwds...)
end

function nllsq!(y::AbstractArray{<:Real,N},
                f::Function,
                p0::DenseVector{Cdouble},
                x;
                rho::NTuple{2,Real} = (0.1, 1e-6),
                scale::AbstractVector{<:Real} = defaultscale(p0),
                kwds...) where {N}
    status, pmin, cmin = Newuoa.minimize!(p -> _nllsq(y, f, p, x),
                                          p0, rho[1], rho[2]; check=false,
                                          scale=scale, kwds...)
    check(status)
    return pmin, cmin
end

function nllsq!(w::AbstractArray{<:Real,N},
                y::AbstractArray{<:Real,N},
                f::Function,
                p0::AbstractVector{Cdouble},
                x;
                rho::NTuple{2,Real} = (0.1, 1e-6),
                scale::AbstractVector{<:Real} = defaultscale(p0),
                kwds...) where {N}
    status, pmin, cmin = Newuoa.minimize!(p -> _nllsq(w, y, f, p, x),
                                          p0, rho[1], rho[2]; check=false,
                                          scale=scale, kwds...)
    check(status)
    return pmin, cmin
end

# Default parameter scaling factors.
function defaultscale(p0::AbstractVector{<:Real})
    n = length(p0)
    scl = Array{Cdouble}(undef, n)
    sml = sqrt(eps(Cdouble))
    @inbounds for i in 1:n
        scl[i] = max(abs(p0[i]), sml)
    end
    return scl
end

# Copy parameters so that they are writable and compatible with NEWUOA.
copyparameters(p0::AbstractVector{<:Real}) =
    copyto!(Array{Cdouble}(undef, length(p0)), p0)


# Check NEWUOA status.
function check(status::Newuoa.Status)
    if status != Newuoa.SUCCESS
        msg = Newuoa.getreason(status)
        if (status == Newuoa.ROUNDING_ERRORS ||
            status == Newuoa.TOO_MANY_EVALUATIONS)
            @warn msg
        else
            error(msg)
        end
    end
end


# Objective functions.

function _nllsq(y::AbstractArray{<:Real,N},
                f::Function,
                p::AbstractVector{<:Real},
                x) where {N}
    m = f(p, x)
    local err::Cdouble = 0
    @inbounds @simd for i in safe_indices(m, y)
        mi, yi = m[i], y[i]
        err += (yi - mi)^2
    end
    return err
end

function _nllsq(w::AbstractArray{<:Real,N},
                y::AbstractArray{<:Real,N},
                f::Function,
                p::AbstractVector{<:Real},
                x) where {N}
    m = f(p, x)
    local err::Cdouble = 0
    @inbounds @simd for i in safe_indices(m, w, y)
        mi, wi, yi = m[i], w[i], y[i]
        err += wi*(yi - mi)^2
    end
    return err
end

end
