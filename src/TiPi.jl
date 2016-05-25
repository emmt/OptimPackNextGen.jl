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
       AbstractCost,
       HyperbolicEdgePreserving,
       QuadraticCost,
       QuadraticSmoothness

"""
`Float` is the type of all floating point scalar, it is currently an alias to
`Cdouble` which is itself an alias to `Float64`.
"""
typealias Float Cdouble

include("utils.jl")
include("algebra.jl")
using .Algebra
for sym in names(TiPi.Algebra)
    @eval export $sym
end
include("AffineTransforms.jl")
include("kernels.jl")
include("interp.jl")
include("mda.jl")
include("lnsrch.jl")
include("quasi-newton.jl")
include("step_globmin.jl")
include("cost.jl")
include("smooth.jl")
include("hypersmooth.jl")
include("deconv.jl")

end # module
