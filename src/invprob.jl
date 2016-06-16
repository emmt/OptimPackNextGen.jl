#
# invprob.jl --
#
# Inverse problems for TiPi.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#

module InverseProblems

import Base: size, convert

import TiPi: Float, MDA,
             AbstractCost, cost, cost!,
             Hessian,
             default_weights, pad, zeropad,
             NormalEquations,
             LinearOperator, SelfAdjointOperator, FakeLinearOperator,
             Identity, ScalingOperator, DiagonalOperator,
             is_fake,
             apply, apply!,
             apply_direct, apply_direct!,
             apply_adjoint, apply_adjoint!,
             input_size, output_size,
             input_type, output_type,
             vcreate,
             vupdate!,
             vcombine, vcombine!,
             vcopy, vcopy!,
             vdot,
             vscale, vscale!

doc"""
# General Quadratic Inverse Problem

A general quadratic inverse problem writes:

    x = argmin_x { (1/2) ‖H⋅x - y‖²_W + (µ/2) ‖D⋅x‖² }

where `x` are the unknowns, `H` is a linear model, `y` are the data, `W` are
statistical weights, 'µ > 0' a tuning parameter and `D` a linear operator
implementing the regularization.  The Euclidean and weighted L2 norms are
defined by:

    ‖v‖²   = ⟨v,v⟩
    ‖v‖²_W = ⟨v,W⋅v⟩

with `⟨u,v⟩ = vdot(u,v)` the inner product of `u` and `v`.


## Create an instance of a quadratic inverse problem

    ip = QuadraticInverseProblem(H, y, W=..., µ=..., D=..., DtD=...)


## Solving a quadratic inverse problem

To solve a quadratic inverse problem (by an iterative optimization method),
there are two possibilities:

1. there are no additional constraints, then use the linear conjugate gradient
   algorithm:

    x = solve(:conjgrad, ip, xinit=x0, maxiter=.., tol=...)

2. there are bound constraints, then use VMLMB algorithm:

    x = solve(:vmlmb, ip, xinit=x0, lower=..., upper=...,
              maxiter=.., tol=..., mem=...)


## Quadratic Denoising

For denoising problems, the linear model is just the identity: `H = I`, assuming
`T` is the type of the data (and also of the unknowns)

    ip = QuadraticInverseProblem(Identity(typeof(y)), y, W=DiagonalOperator(w),
                                 µ = 0.01, D = ...)

"""
type QuadraticInverseProblem{E,F,G} <: AbstractCost
    H::LinearOperator{E,F}
    y::E
    W::SelfAdjointOperator{E}
    µ::Float
    D::LinearOperator{G,F}
    DtD::LinearOperator{F,F}
    function QuadraticInverseProblem{E,F,G}(H::LinearOperator{E,F},
                                            y::E,
                                            W::SelfAdjointOperator{E},
                                            µ::Float,
                                            D::LinearOperator{G,F},
                                            DtD::SelfAdjointOperator{F})
        @assert µ ≥ 0
        new(H, y, W, µ, D, DtD)
    end
end

QuadraticInverseProblem{E,F}(H::LinearOperator{E,F}, y::E; kws...) =
    QuadraticInverseProblem(H, y, Identity(E); kws...)

QuadraticInverseProblem{E,F}(H::LinearOperator{E,F}, y::E, ::Void; kws...) =
    QuadraticInverseProblem(H, y, Identity(E); kws...)

QuadraticInverseProblem{E,F}(H::LinearOperator{E,F}, y::E, w::E; kws...) =
    QuadraticInverseProblem(H, y, DiagonalOperator(w); kws...)

function QuadraticInverseProblem{E,F,G}(H::LinearOperator{E,F},
                                        y::E,
                                        W::SelfAdjointOperator{E};
                                        µ::Real=0.0,
                                        D::LinearOperator{G,F}=FakeLinearOperator(Any,F),
                                        DtD::SelfAdjointOperator{F}=FakeLinearOperator(F))
    return QuadraticInverseProblem{E,F,G}(H, y, W, Float(µ), D, DtD)
end

function tune!(ip::QuadraticInverseProblem, µ::Real)
    @assert µ ≥ 0
    ip.µ = Float(µ)
    return ip
end

tune(ip::QuadraticInverseProblem) = ip.µ

convert{E,F,G}(::Type{NormalEquations{F}}, ip::QuadraticInverseProblem{E,F,G}) =
    NormalEquations(ip)

function NormalEquations{E,F,G}(ip::QuadraticInverseProblem{E,F,G})
    # Get all members to simplify code below.
    H, y, W, µ, D, DtD = ip.H, ip.y, ip.W, ip.µ, ip.D, ip.DtD

    b = H'⋅W⋅y
    A = Hessian(ip, F)
    return NormalEquations{F}(A, b)
end

function apply_direct!{E,F,G}(y::E,
                              A::Hessian{QuadraticInverseProblem{E,F,G},F},
                              x::E)
    # Get all members to simplify code below.
    ip = contents(A)
    H, y, W, µ, D, DtD = ip.H, ip.y, ip.W, ip.µ, ip.D, ip.DtD

    # Form y = H'⋅W⋅H⋅x + µ D'⋅D⋅x taking care of H⋅x being possibly x.
    r = H⋅x
    if is(r, x)
        r = apply(W, r)
    else
        apply!(r, W, r)
    end
    apply!(y, H', r)
    if µ > 0
        if ! is_fake(DtD)
            vupdate!(y, µ, DtD⋅x)
        elseif ! is_fake(D)
            vupdate!(y, µ, D'⋅D⋅x)
        else
            vupdate!(y, µ, x)
        end
    end
    return y
end


function cost{E,F,G}(alpha::Real, ip::QuadraticInverseProblem{E,F,G}, x::F)
    # Short circuit if global multiplier is zero.
    if alpha == 0
        return Float(0)
    end

    # Get all members to simplify code below.
    H, y, W, µ, D, DtD = ip.H, ip.y, ip.W, ip.µ, ip.D, ip.DtD

    # Form "residuals" r = H⋅x - y and weighted residuals Wr = W⋅(H⋅x - y),
    # taking care of H⋅x being possibly x.
    r = H⋅x
    if is(r, x)
        r = vcombine(1, r, -1, y)
    else
        vupdate!(r, -1, y)
    end
    Wr = W⋅r

    # Compute likelihood and regularization.
    lkl::Float = vdot(r,Wr)
    rgl::Float = 0
    if µ > 0
        if ! is_fake(D)
            Dx = D⋅x
            rgl = vdot(Dx, Dx)
        elseif ! is_fake(DtD)
            rgl = vdot(x, DtD⋅x)
        else
            rgl = vdot(x, x)
        end
    end
    return Float(alpha/2)*(lkl + µ*rgl)
end

function cost!{E,F,G}(alpha::Real, ip::QuadraticInverseProblem{E,F,G}, x::F,
                      g::F, clr::Bool=false)
    # Short circuit if global multiplier is zero.
    if alpha == 0
        clr && vfill!(g, 0)
        return Float(0)
    end

    # Get all members to simplify code below.
    H, y, W, µ, D, DtD = ip.H, ip.y, ip.W, ip.µ, ip.D, ip.DtD

    # Form "residuals" r = H⋅x - y and weighted residuals Wr = W⋅(H⋅x - y),
    # taking care of H⋅x being possibly x.
    r = H⋅x
    if is(r, x)
        r = vcombine(1, r, -1, y)
    else
        vupdate!(r, -1, y)
    end
    Wr = W⋅r

    # Compute likelihood and regularization.
    lkl::Float = vdot(r,Wr)
    if clr
        vscale!(g, alpha, H'⋅Wr)
    else
        vupdate!(g, alpha, H'⋅Wr)
    end
    rgl::Float = 0
    if µ > 0
        if ! is_fake(DtD)
            DtDx = DtD⋅x
            rgl = vdot(x, DtDx)
            vupdate!(g, alpha*µ, DtDx)
        elseif ! is_fake(D)
            Dx = D⋅x
            rgl = vdot(Dx, Dx)
            vupdate!(g, alpha*µ, D'⋅Dx)
        else
            rgl = vdot(x, x)
            vupdate!(g, alpha*µ, x)
        end
    end
    return Float(alpha/2)*(lkl + µ*rgl)
end

function solve!{E,F,G}(ip::QuadraticInverseProblem{E,F,G}, x::F;
                       µ::Real=tune(ip), method::Symbol=:vmlmb, kws...)
    tune!(ip, mu)
    if method === :vmlmb
        TiPi.vmlmb!((x, g) -> cost!(ip, x, g), x; kws...)
    elseif method === :conjgrad
        TiPi.conjgrad!(NormalEquations(ip), x; kws...)
    else
        error("invalid optimization method")
    end
    return x
end

solve{E,F,G}(ip::QuadraticInverseProblem{E,F,G}, x0::F; kws...) =
    solve!(ip, vcopy(x0); kws...)

end # module
