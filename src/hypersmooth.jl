#
# hypersmooth.jl --
#
# Hyperbolic edge-preserving regularization for TiPi.
#
#------------------------------------------------------------------------------
#
# This file is part of TiPi.jl licensed under the MIT "Expat" License.
#
# Copyright (C) 2015, Éric Thiébaut & Jonathan Léger.
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Data conversion utilities
# =========================

typealias Flt Cdouble

flt(x) = convert(Flt, x)

flt_tuple(a::Real) = (flt(a),)
flt_tuple{N,T}(a::NTuple{N,T}) = ntuple(i -> flt(a[i]), N)
flt_tuple{T}(a::Array{T,1}) = ntuple(i -> flt(a[i]), length(a))

#------------------------------------------------------------------------------
#
# Hyperbolic Edge-Preserving
# ==========================
#
# This edge-preserving regularization is a smooth (i.e. differentiable) cost
# function which can be used to implement edge-preserving smoothness.  The
# current implementation is isotropic.  With a very low threshold, this
# regularization can be seen as an hyperbolic approximation of the total
# variation (TV).
#
# The quadratic norm of the spatial gradient is the sum along all the
# dimensions of the average squared differences along the dimension.  A weight
# is applied along each dimension to account for the regularization weight and
# for unequal sampling along the dimensions.
#
# More specifically, the regularization writes:
#
#     alpha*(sqrt(sd1 + sd2 + ... + sdn + eps^2) - eps)
#
# with `alpha` > 0 the regularization weight, `n` the rank and `sdk` the
# average squared scaled differences along the `k`-th dimension.  The removal
# of `eps` is just to have the cost equal to zero for a flat array.
#
# For instance, in 2D, the total variation is computed for each 2x2 blocs with:
#
#     sd1 = mu1^2*((x2 - x1)^2 + (x4 - x3)^2)/2
#         = w1*((x2 - x1)^2 + (x4 - x3)^2)
#     sd2 = mu2^2*((x3 - x1)^2 + (x4 - x2)^2)/2
#         = w2*((x3 - x1)^2 + (x4 - x2)^2)
#
# where multiplication by `mu1` and `mu2` is to scale the finite differences
# along each dimension and division by 2 is to average (there are 2 possible
# finite differences in a 2x2 bloc along a given dimension).  The weights `w1`
# and `w2` and `s` are given by:
# ```
#     w1 = mu1^2/2
#     w2 = mu2^2/2
#     s = eps^2
# ```
# FIXME: The sum should done in double precision whatever the type?
# FIXME: Use metaprogramming to code all this ;-)
# FIXME: save N multiplications in the isotropic case
# FIXME: make specialized code when some of the mu's are zero (because
#        that's how regularizing along not all direction will be
#        implemented
# FIXME: optimize clear operation to avoid one pass through gx (done in 1D)

immutable HyperbolicEdgePreserving{N} <: AbstractCost
    eps::Cdouble
    mu::NTuple{N,Cdouble}
    isotropic::Bool
    function HyperbolicEdgePreserving(eps::Cdouble,
                                      mu::NTuple{N,Cdouble})
        eps > 0.0 || error("threshold must be strictly positive")
        (mumin, mumax) = extrema(mu)
        mumin >= 0.0 || error("weights must be nonnegative")
        isotropic = (mumin == mumax)
        new(eps, mu, isotropic)
    end
end

function HyperbolicEdgePreserving(eps::Real, mu::Real)
    HyperbolicEdgePreserving{1}(flt(eps), (flt(mu),))
end

function HyperbolicEdgePreserving(eps::Real, mu::Vector{Real})
    N = length(mu)
    HyperbolicEdgePreserving{N}(flt(eps), ntuple(i -> flt(mu[i]), N))
end

function HyperbolicEdgePreserving(eps::Real, mu::NTuple)
    N = length(mu)
    HyperbolicEdgePreserving{N}(flt(eps), ntuple(i -> flt(mu[i]), N))
end

#------------------------------------------------------------------------------
# 1D case

# Note that the weight `alpha` is not taken into account when summing the cost
# (this is done at the end) while `alpha` is taken into account when
# integrating the gradient `gx`.

function cost{T}(alpha::Real, param::HyperbolicEdgePreserving{1},
                 x::Array{T,1})
    # Short circuit if weight is zero.
    if alpha == 0 || param.mu[1] == 0
        return 0.0
    end

    # Integrate un-weighted cost.
    const dim1 = size(x)[1]
    const s = convert(T, (param.eps/param.mu[1])^2)
    fx::T = zero(T)
    for i1 in 2:dim1
        @inbounds fx += sqrt((x[i1] - x[i1-1])^2 + s)
    end

    # Remove the "bias" and scale the cost.
    return (param.mu[1]*fx - (dim1 - 1)*param.eps)*alpha
end

function cost!{T}(alpha::Real, param::HyperbolicEdgePreserving{1},
                  x::Array{T,1}, gx::Array{T,1}, clr::Bool=false)
    # Minimal checking.
    @assert(size(x) == size(gx))

    # Short circuit if weight is zero.
    if alpha == 0 || param.mu[1] == 0
        clr && fill!(gx, 0)
        return 0.0
    end

    # Integrate un-weighted cost and gradient.
    const dim1 = size(x)[1]
    const s = convert(T, (param.eps/param.mu[1])^2)
    const beta = convert(T, alpha*param.mu[1])
    fx::T = zero(T)
    if clr
        tp = zero(T)
        @simd for i1 in 2:dim1
            @inbounds y = x[i1] - x[i1-1]
            r = sqrt(y*y + s)
            fx += r
            p = beta*(y/r)
            @inbounds gx[i1-1] = tp - p
            tp = p
        end
        @inbounds gx[end] = tp
    else
        @simd for i1 in 2:dim1
            @inbounds y = x[i1] - x[i1-1]
            r = sqrt(y*y + s)
            fx += r
            p = beta*(y/r)
            @inbounds gx[i1-1] -= p
            @inbounds gx[i1]   += p
        end
    end

    # Remove the "bias" and scale the cost.
    return ((fx*param.mu[1]) - (dim1 - 1)*param.eps)*alpha
end

#------------------------------------------------------------------------------
#
# 2D case:
#
#     In the code, (i1,i2) are the 2D coordinates of the 4th point (see figure
#     below) of the region of interest.
#
#            ^
#            |
#            +----+----+            X1 = x[i1-1, i2-1]
#         i2 | x3 | x4 |            x2 = x[i1,   i2-1]
#            +----+----+            x3 = x[i1-1, i2]
#       i2-1 | x1 | x2 |            x4 = x[i1,   i2]
#            +----+----+-->
#             i1-1  i1
#
function cost{T}(alpha::Real, param::HyperbolicEdgePreserving{2},
                 x::Array{T,2})
    if alpha == 0
        return 0.0
    end
    dims = size(x)
    const dim1 = dims[1]
    const dim2 = dims[2]
    const s = convert(T, (param.eps)^2)
    fx = zero(T)
    if param.isotropic
        # Same weights along all directions.
        const w = convert(T, param.mu[1]^2/2)
        for i2 in 2:dim2
            @inbounds x2 = x[1,i2-1]
            @inbounds x4 = x[1,i2]
            @simd for i1 in 2:dim1
                # Move to next 2x2 bloc.
                x1 = x2; @inbounds x2 = x[i1,i2-1]
                x3 = x4; @inbounds x4 = x[i1,i2]

                # Compute hyperbolic approximation of L2 norm of the
                # spatial gradient.
                fx += sqrt(((x2 - x1)^2 + (x4 - x3)^2 +
                            (x3 - x1)^2 + (x4 - x2)^2)*w + s)
            end
        end
    else
        # Not same weights along all directions.
        const w1 = convert(T, param.mu[1]^2/2)
        const w2 = convert(T, param.mu[2]^2/2)
        for i2 in 2:dim2
            @inbounds x2 = x[1,i2-1]
            @inbounds x4 = x[1,i2]
            @simd for i1 in 2:dim1
                # Move to next 2x2 bloc.
                x1 = x2; @inbounds x2 = x[i1,i2-1]
                x3 = x4; @inbounds x4 = x[i1,i2]

                # Compute hyperbolic approximation of L2 norm of the
                # spatial gradient.
                fx += sqrt(((x2 - x1)^2 + (x4 - x3)^2)*w1 +
                           ((x3 - x1)^2 + (x4 - x2)^2)*w2 + s)
            end
        end
    end

    # Remove the "bias" and scale the cost.
    return (fx - (dim1 - 1)*(dim2 - 1)*param.eps)*alpha
end

function cost!{T}(alpha::Real, param::HyperbolicEdgePreserving{2},
                  x::Array{T,2}, gx::Array{T,2}, clr::Bool=false)
    @assert(size(x) == size(gx))
    clr && fill!(gx, 0)
    alpha == 0 && return 0.0
    dims = size(x)
    const dim1 = dims[1]
    const dim2 = dims[2]
    const s = convert(T, (param.eps)^2)
    fx = zero(T)
    if param.isotropic
        # Same weights along all directions.
        const w = convert(T, param.mu[1]^2/2)
        const q = convert(T, alpha*w)
        for i2 in 2:dim2
            @inbounds x2 = x[1,i2-1]
            @inbounds x4 = x[1,i2]
            @simd for i1 in 2:dim1
                # Move to next 2x2 bloc.
                x1 = x2; @inbounds x2 = x[i1,i2-1]
                x3 = x4; @inbounds x4 = x[i1,i2]

                # Compute horizontal and vertical differences.
                x2_x1 = x2 - x1
                x3_x1 = x3 - x1
                x4_x2 = x4 - x2
                x4_x3 = x4 - x3

                # Compute hyperbolic approximation of L2 norm of the
                # spatial gradient.
                r = sqrt((x2_x1^2 + x4_x3^2 + x3_x1^2 + x4_x2^2)*w + s)
                fx += r

                # Integrate the gradient of the cost.
                p = q/r
                @inbounds gx[i1-1, i2-1] -= (x2_x1 + x3_x1)*p
                @inbounds gx[i1,   i2-1] += (x2_x1 - x4_x2)*p
                @inbounds gx[i1-1, i2  ] -= (x4_x3 - x3_x1)*p
                @inbounds gx[i1,   i2  ] += (x4_x3 + x4_x2)*p
            end
        end
    else
        # Not same weights along all directions.
        const w1 = convert(T, param.mu[1]^2/2)
        const w2 = convert(T, param.mu[2]^2/2)
        const a = convert(T, alpha)
        for i2 in 2:dim2
            @inbounds x2 = x[1,i2-1]
            @inbounds x4 = x[1,i2]
            @simd for i1 in 2:dim1
                # Move to next 2x2 bloc.
                x1 = x2; @inbounds x2 = x[i1,i2-1]
                x3 = x4; @inbounds x4 = x[i1,i2]

                # Compute horizontal and vertical differences.
                x2_x1 = x2 - x1
                x4_x3 = x4 - x3
                x3_x1 = x3 - x1
                x4_x2 = x4 - x2

                # Compute hyperbolic approximation of L2 norm of the
                # spatial gradient.
                r = sqrt((x2_x1^2 + x4_x3^2)*w1 +
                         (x3_x1^2 + x4_x2^2)*w2 + s)
                fx += r

                # Integrate the gradient of the cost.
                q = a/r
                p1 = w1*q
                x2_x1 *= p1
                x4_x3 *= p1
                p2 = w2*q
                x3_x1 *= p2
                x4_x2 *= p2
                @inbounds gx[i1-1, i2-1] -= x2_x1 + x3_x1
                @inbounds gx[i1,   i2-1] += x2_x1 - x4_x2
                @inbounds gx[i1-1, i2  ] -= x4_x3 - x3_x1
                @inbounds gx[i1,   i2  ] += x4_x3 + x4_x2
            end
        end
    end

    # Remove the "bias" and scale the cost.
    return (fx - (dim1 - 1)*(dim2 - 1)*param.eps)*alpha
end

#------------------------------------------------------------------------------
#
# Notations for 3-d volume x[i1,i2,i3]:
#
#                 i3  i2
#                  | /
#                  |/              x1 = x[i1-1, i2-1, i3-1]
#        x7--------x8---> i1       x2 = x[i1  , i2-1, i3-1]
#       /:        /|               x3 = x[i1-1, i2  , i3-1]
#      / :       / |               x4 = x[i1  , i2  , i3-1]
#     x5--------x6 |               x5 = x[i1-1, i2-1, i3  ]
#     |  x3.....|..x4              x6 = x[i1  , i2-1, i3  ]
#     | '       | /                x7 = x[i1-1, i2  , i3  ]
#     |'        |/                 x8 = x[i1  , i2  , i3  ]
#     x1--------x2
#
function cost{T}(alpha::Real, param::HyperbolicEdgePreserving{3},
                 x::Array{T,3})
    alpha == 0 && return 0.0
    dims = size(x)
    const dim1 = dims[1]
    const dim2 = dims[2]
    const dim3 = dims[3]
    const s = convert(T, (param.eps)^2)
    fx = zero(T)
    if param.isotropic
        # Same weights along all directions.
        const w = convert(T, param.mu[1]^2/4)
        for i3 in 2:dim3
            for i2 in 2:dim2
                # Get part of 2x2x2 bloc such that 8th point is at
                # coordinates [1,i2,i3].
                @inbounds x2 = x[1, i2-1, i3-1]
                @inbounds x4 = x[1, i2  , i3-1]
                @inbounds x6 = x[1, i2-1, i3  ]
                @inbounds x8 = x[1, i2  , i3  ]
                @simd for i1 in 2:dim1
                    # Move to next 2x2x2 bloc.
                    x1 = x2; @inbounds x2 = x[i1, i2-1, i3-1]
                    x3 = x4; @inbounds x4 = x[i1, i2  , i3-1]
                    x5 = x6; @inbounds x6 = x[i1, i2-1, i3  ]
                    x7 = x8; @inbounds x8 = x[i1, i2  , i3  ]

                    # Compute differences along 1st dimension.
                    x2_x1 = x2 - x1
                    x4_x3 = x4 - x3
                    x6_x5 = x6 - x5
                    x8_x7 = x8 - x7

                    # Compute differences along 2nd dimension.
                    x3_x1 = x3 - x1
                    x4_x2 = x4 - x2
                    x7_x5 = x7 - x5
                    x8_x6 = x8 - x6

                    # Compute differences along 3rd dimension.
                    x5_x1 = x5 - x1
                    x6_x2 = x6 - x2
                    x7_x3 = x7 - x3
                    x8_x4 = x8 - x4

                    # Compute hyperbolic approximation of L2 norm of
                    # the spatial gradient.
                    fx += sqrt((x2_x1^2 + x4_x3^2 + x6_x5^2 +
                                x8_x7^2 + x3_x1^2 + x4_x2^2 +
                                x7_x5^2 + x8_x6^2 + x5_x1^2 +
                                x6_x2^2 + x7_x3^2 + x8_x4^2)*w + s)
                end
            end
        end
    else
        # Not same weights along all directions.
        const w1 = convert(T, param.mu[1]^2/4)
        const w2 = convert(T, param.mu[2]^2/4)
        const w3 = convert(T, param.mu[3]^2/4)
        for i3 in 2:dim3
            for i2 in 2:dim2
                # Get part of 2x2x2 bloc such that 8th point is at
                # coordinates [1,i2,i3].
                @inbounds x2 = x[1, i2-1, i3-1]
                @inbounds x4 = x[1, i2  , i3-1]
                @inbounds x6 = x[1, i2-1, i3  ]
                @inbounds x8 = x[1, i2  , i3  ]
                @simd for i1 in 2:dim1
                    # Move to next 2x2x2 bloc.
                    x1 = x2; @inbounds x2 = x[i1, i2-1, i3-1]
                    x3 = x4; @inbounds x4 = x[i1, i2  , i3-1]
                    x5 = x6; @inbounds x6 = x[i1, i2-1, i3  ]
                    x7 = x8; @inbounds x8 = x[i1, i2  , i3  ]

                    # Compute differences along 1st dimension.
                    x2_x1 = x2 - x1
                    x4_x3 = x4 - x3
                    x6_x5 = x6 - x5
                    x8_x7 = x8 - x7
                    r1 = x2_x1^2 + x4_x3^2 + x6_x5^2 + x8_x7^2

                    # Compute differences along 2nd dimension.
                    x3_x1 = x3 - x1
                    x4_x2 = x4 - x2
                    x7_x5 = x7 - x5
                    x8_x6 = x8 - x6
                    r2 = x3_x1^2 + x4_x2^2 + x7_x5^2 + x8_x6^2

                    # Compute differences along 3rd dimension.
                    x5_x1 = x5 - x1
                    x6_x2 = x6 - x2
                    x7_x3 = x7 - x3
                    x8_x4 = x8 - x4
                    r3 = x5_x1^2 + x6_x2^2 + x7_x3^2 + x8_x4^2

                    # Compute hyperbolic approximation of L2 norm of
                    # the spatial gradient.
                    fx += sqrt(w1*r1 + w2*r2 + w3*r3 + s)
                end
            end
        end
    end

    # Remove the "bias" and scale the cost.
    return (fx - (dim1 - 1)*(dim2 - 1)*(dim3 - 1)*param.eps)*alpha
end

function cost!{T}(alpha::Real, param::HyperbolicEdgePreserving{3},
                  x::Array{T,3}, gx::Array{T,3}, clr::Bool=false)
    @assert(size(x) == size(gx))
    clr && fill!(gx, 0)
    alpha == 0 && return 0.0
    dims = size(x)
    const dim1 = dims[1]
    const dim2 = dims[2]
    const dim3 = dims[3]
    const s = convert(T, (param.eps)^2)
    fx = zero(T)
    if param.isotropic
        # Same weights along all directions.
        const w = convert(T, param.mu[1]^2/4)
        const q = convert(T, alpha*w)
        for i3 in 2:dim3
            for i2 in 2:dim2
                # Get part of 2x2x2 bloc such that 8th point is at
                # coordinates [1,i2,i3].
                @inbounds x2 = x[1, i2-1, i3-1]
                @inbounds x4 = x[1, i2  , i3-1]
                @inbounds x6 = x[1, i2-1, i3  ]
                @inbounds x8 = x[1, i2  , i3  ]
                @simd for i1 in 2:dim1
                    # Move to next 2x2x2 bloc.
                    x1 = x2; @inbounds x2 = x[i1, i2-1, i3-1]
                    x3 = x4; @inbounds x4 = x[i1, i2  , i3-1]
                    x5 = x6; @inbounds x6 = x[i1, i2-1, i3  ]
                    x7 = x8; @inbounds x8 = x[i1, i2  , i3  ]

                    # Compute differences along 1st dimension.
                    x2_x1 = x2 - x1
                    x4_x3 = x4 - x3
                    x6_x5 = x6 - x5
                    x8_x7 = x8 - x7

                    # Compute differences along 2nd dimension.
                    x3_x1 = x3 - x1
                    x4_x2 = x4 - x2
                    x7_x5 = x7 - x5
                    x8_x6 = x8 - x6

                    # Compute differences along 3rd dimension.
                    x5_x1 = x5 - x1
                    x6_x2 = x6 - x2
                    x7_x3 = x7 - x3
                    x8_x4 = x8 - x4

                    # Compute hyperbolic approximation of L2 norm of
                    # the spatial gradient.
                    r = sqrt((x2_x1^2 + x4_x3^2 + x6_x5^2 +
                              x8_x7^2 + x3_x1^2 + x4_x2^2 +
                              x7_x5^2 + x8_x6^2 + x5_x1^2 +
                              x6_x2^2 + x7_x3^2 + x8_x4^2)*w + s)
                    fx += r

                    # Integrate the gradient of the cost.
                    p = q/r
                    @inbounds gx[i1-1, i2-1, i3-1] -= (x2_x1 + x3_x1 + x5_x1)*p
                    @inbounds gx[i1  , i2-1, i3-1] += (x2_x1 - x4_x2 - x6_x2)*p
                    @inbounds gx[i1-1, i2  , i3-1] -= (x4_x3 - x3_x1 + x7_x3)*p
                    @inbounds gx[i1  , i2  , i3-1] += (x4_x3 + x4_x2 - x8_x4)*p
                    @inbounds gx[i1-1, i2-1, i3  ] -= (x6_x5 + x7_x5 - x5_x1)*p
                    @inbounds gx[i1  , i2-1, i3  ] += (x6_x5 - x8_x6 + x6_x2)*p
                    @inbounds gx[i1-1, i2  , i3  ] -= (x8_x7 - x7_x5 - x7_x3)*p
                    @inbounds gx[i1  , i2  , i3  ] += (x8_x7 + x8_x6 + x8_x4)*p
                end
            end
        end
    else
        # Not same weights along all directions.
        const w1 = convert(T, param.mu[1]^2/4)
        const w2 = convert(T, param.mu[2]^2/4)
        const w3 = convert(T, param.mu[3]^2/4)
        const a = convert(T, alpha)
        for i3 in 2:dim3
            for i2 in 2:dim2
                # Get part of 2x2x2 bloc such that 8th point is at
                # coordinates [1,i2,i3].
                @inbounds x2 = x[1, i2-1, i3-1]
                @inbounds x4 = x[1, i2  , i3-1]
                @inbounds x6 = x[1, i2-1, i3  ]
                @inbounds x8 = x[1, i2  , i3  ]
                @simd for i1 in 2:dim1
                    # Move to next 2x2x2 bloc.
                    x1 = x2; @inbounds x2 = x[i1, i2-1, i3-1]
                    x3 = x4; @inbounds x4 = x[i1, i2  , i3-1]
                    x5 = x6; @inbounds x6 = x[i1, i2-1, i3  ]
                    x7 = x8; @inbounds x8 = x[i1, i2  , i3  ]

                    # Compute differences along 1st dimension.
                    x2_x1 = x2 - x1
                    x4_x3 = x4 - x3
                    x6_x5 = x6 - x5
                    x8_x7 = x8 - x7
                    r1 = x2_x1^2 + x4_x3^2 + x6_x5^2 + x8_x7^2

                    # Compute differences along 2nd dimension.
                    x3_x1 = x3 - x1
                    x4_x2 = x4 - x2
                    x7_x5 = x7 - x5
                    x8_x6 = x8 - x6
                    r2 = x3_x1^2 + x4_x2^2 + x7_x5^2 + x8_x6^2

                    # Compute differences along 3rd dimension.
                    x5_x1 = x5 - x1
                    x6_x2 = x6 - x2
                    x7_x3 = x7 - x3
                    x8_x4 = x8 - x4
                    r3 = x5_x1^2 + x6_x2^2 + x7_x3^2 + x8_x4^2

                    # Compute hyperbolic approximation of L2 norm of
                    # the spatial gradient.
                    r = sqrt(w1*r1 + w2*r2 + w3*r3 + s)
                    fx += r

                    # Integrate the gradient of the cost.
                    q = a/r
                    p1 = w1*q
                    x2_x1 *= p1
                    x4_x3 *= p1
                    x6_x5 *= p1
                    x8_x7 *= p1
                    p2 = w2*q
                    x3_x1 *= p2
                    x4_x2 *= p2
                    x7_x5 *= p2
                    x8_x6 *= p2
                    p3 = w3*q
                    x5_x1 *= p3
                    x6_x2 *= p3
                    x7_x3 *= p3
                    x8_x4 *= p3
                    @inbounds gx[i1-1, i2-1, i3-1] -= x2_x1 + x3_x1 + x5_x1
                    @inbounds gx[i1  , i2-1, i3-1] += x2_x1 - x4_x2 - x6_x2
                    @inbounds gx[i1-1, i2  , i3-1] -= x4_x3 - x3_x1 + x7_x3
                    @inbounds gx[i1  , i2  , i3-1] += x4_x3 + x4_x2 - x8_x4
                    @inbounds gx[i1-1, i2-1, i3  ] -= x6_x5 + x7_x5 - x5_x1
                    @inbounds gx[i1  , i2-1, i3  ] += x6_x5 - x8_x6 + x6_x2
                    @inbounds gx[i1-1, i2  , i3  ] -= x8_x7 - x7_x5 - x7_x3
                    @inbounds gx[i1  , i2  , i3  ] += x8_x7 + x8_x6 + x8_x4
                end
            end
        end
    end

    # Remove the "bias" and scale the cost.
    return (fx - (dim1 - 1)*(dim2 - 1)*(dim3 - 1)*param.eps)*alpha
end

# Local Variables:
# mode: Julia
# tab-width: 8
# indent-tabs-mode: nil
# fill-column: 79
# coding: utf-8
# ispell-local-dictionary: "american"
# End:
