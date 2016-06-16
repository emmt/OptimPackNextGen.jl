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

export MDA
export goodfftdim,
       fftfreq,
       bounding_box,
       crop,
       crop!,
       pad,
       zeropad,
       paste!,
       cost,
       cost!,
       check_gradient,
       prox,
       prox!,
       AbstractCost, Hessian,
       CirculantConvolution,
       HyperbolicEdgePreserving,
       QuadraticCost,
       QuadraticSmoothness,
       CompactRegCauchy

doc"""
`Float` is the type of all floating point scalar, it is currently an alias to
`Float64`.
"""
typealias Float Float64

include("utils.jl")

include("algebra.jl")
import .Algebra: LinearOperator,
                 LinearEndomorphism,
                 SelfAdjointOperator,
                 Identity,
                 NormalEquations,
                 DiagonalOperator,
                 RankOneOperator,
                 ScalingOperator,
                 CroppingOperator,
                 ZeroPaddingOperator,
                 FakeLinearOperator,
                 FakeLinearEndomorphism,
                 FakeSelfAdjointOperator,
                 is_fake,
                 input_eltype,
                 input_ndims,
                 input_size,
                 input_type,
                 output_eltype,
                 output_ndims,
                 output_size,
                 output_type,
                 apply,
                 apply!,
                 apply_direct,
                 apply_direct!,
                 apply_adjoint,
                 apply_adjoint!,
                 apply_inverse,
                 apply_inverse!,
                 apply_inverse_adjoint,
                 apply_inverse_adjoint!,
                 check_operator,
                 is_identity,
                 vcombine,
                 vcombine!,
                 vcopy,
                 vcopy!,
                 vcreate,
                 vdot,
                 vfill!,
                 vnorm1,
                 vnorm2,
                 vnorminf,
                 vproduct,
                 vproduct!,
                 vscale,
                 vscale!,
                 vswap!,
                 vupdate!,
                 conjgrad,
                 project_variables!,
                 project_direction!,
                 step_limits,
                 get_free_variables

include("fft.jl")
import .FFT: FFTOperator

include("convolution.jl")
using .Convolution

include("AffineTransforms.jl")
include("kernels.jl")
include("interp.jl")
include("mda.jl")
include("lnsrch.jl")
include("quasi-newton.jl")
include("step_globmin.jl")
include("weights.jl")
include("cost.jl")
include("smooth.jl")
include("hypersmooth.jl")

include("invprob.jl")
import .InverseProblems: QuadraticInverseProblem,
                         solve, solve!

include("deconv.jl")
include("compactRegCauchy.jl")

end # module
