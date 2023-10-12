#
# OptimPackNextGen.jl --
#
# Package for numerical optimization in Julia.
#
# -----------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2015-2018, Éric Thiébaut,
# <https://github.com/emmt/OptimPackNextGen.jl>.
#

module OptimPackNextGen

export
    # Linear Conjugate Gradient method:
    # FIXME: conjgrad!, conjgrad,
    #
    # Separable bounds (re-exported from NumOptBase):
    Projector, Bound, BoundedSet,

    # Spectral Projected Gradient (SPG) method:
    SPG, spg, spg!,

    # Brent's methods:
    fmin,
    fzero,

    # Powell's methods:
    Powell,

    # Non-Linear Least SQuares (NLLSQ):
    nllsq!, nllsq,

    # Variable Metric with Limited Memory and Bounds (VMLMB) method:
    vmlmb!, vmlmb,

    # Trust region step:
    gqtpar!, gqtpar,

    # Miscellaneous:
    getreason,
    issuccess

using LinearAlgebra
using NumOptBase

if !isdefined(Base, :get_extension)
    using Requires
end

"""

`Float` is the type of all floating point scalars, it is currently an alias to
`Cdouble` which is itself an alias to `Float64`.

"""
const Float = Cdouble

include("wrappers.jl")

include("utils.jl")

include("vops.jl")

include("gqtpar.jl")
import .MoreSorensen: gqtpar, gqtpar!

include("linesearches.jl")
import .LineSearches: getreason

include("quasinewton.jl")
import .QuasiNewton: vmlmb!, vmlmb, EMULATE_BLMVM

include("brent.jl")
import .Brent: fmin, fzero

include("powell.jl")

include("nllsq.jl")
import .NonLinearLeastSquares: nllsq, nllsq!

include("bradi.jl")

include("step.jl")

include("spg.jl")
import .SPG: spg, spg!

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" include(
            "../ext/OptimPackNextGenZygoteExt.jl")
    end
end

end # module
