#
# OptimPackNextGen.jl --
#
# Package for numerical optimization in Julia.
#
# ------------------------------------------------------------------------------
#
# This file is part of OptimPack.jl which is licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2015-2017, Éric Thiébaut.
#

isdefined(Base, :__precompile__) && __precompile__()

module OptimPackNextGen

export
    conjgrad,
    conjgrad!,
    vmlmb,
    vmlmb!,
    spg,
    spg!,
    globmin,
    globmax,
    getreason

doc"""
`Float` is the type of all floating point scalars, it is currently an alias to
`Cdouble` which is itself an alias to `Float64`.
"""
typealias Float Cdouble

function getreason end

include("algebra.jl"); importall .Algebra
include("conjgrad.jl")
include("lnsrch.jl")
include("quasinewton.jl"); importall .QuasiNewton
include("step.jl"); importall .Step
include("spg.jl"); importall .SPG

end # module
