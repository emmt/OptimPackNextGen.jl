#
# TiPi.jl --
#
# Toolkit for Inverse Problems and Imaging in Julia.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#

module TiPi

# The `apply` method is marked as deprecated but it is widely used in TiPi to
# apply an operator, so we import it from `Base`.  The `apply!` method is not
# defined elsewhere, so we export it.
import Base: apply
export apply!

export MDA,
       goodfftdim,
       fftfreq,
       BoundingBox,
       crop,
       crop!,
       pad,
       zeropad,
       paste!,
       NormalEquations,
       conjgrad,
       conjgrad!,
       AbstractCost,
       cost,
       cost!,
       check_gradient,
       prox,
       prox!,
       vmlmb,
       vmlmb!,
       Hessian,
       CirculantConvolution,
       HyperbolicEdgePreserving,
       QuadraticCost,
       QuadraticSmoothness,
       CompactRegCauchy,
       Algebra,
       Operator,
       NonlinearOperator,
       NonlinearEndomorphism,
       LinearOperator,
       LinearEndomorphism,
       SelfAdjointOperator,
       Identity,
       DiagonalOperator,
       RankOneOperator,
       ScalingOperator,
       CroppingOperator,
       ZeroPaddingOperator,
       FakeLinearOperator,
       is_fake,
       FFTOperator,
       fast_deconv,
       OperatorD, OperatorDtD

doc"""
`Float` is the type of all floating point scalar, it is currently an alias to
`Float64`.
"""
typealias Float Float64

# FIXME: this is dirty?
abstract AbstractCost
function cost  end
function cost! end

include("utils.jl")
include("algebra.jl"); importall .Algebra
include("fft.jl"); importall .FFT
include("convolution.jl"); importall .Convolution
include("AffineTransforms.jl")
include("kernels.jl")
include("interp.jl")
include("mda.jl")
include("lnsrch.jl")
include("quasi-newton.jl"); importall .QuasiNewton
include("step_globmin.jl")
include("spg.jl"); importall .SPG
include("weights.jl")
include("cost.jl")
include("smooth.jl")
include("hypersmooth.jl")
include("finitediff.jl"); importall .FiniteDifferences
include("invprob.jl"); importall .InverseProblems
include("deconv.jl"); importall .Deconvolution

include("compactRegCauchy.jl")

end # module
