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
    SPG, spg, spg!, spg_CUTEst,

    # Brent's methods:
    fmin,
    fzero,

    # Powell's methods:
    Powell,

    # Non-Linear Least SQuares (NLLSQ):
    nllsq!, nllsq,

    # Variable Metric with Limited Memory and Bounds (VMLMB) method:
    vmlmb, vmlmb!, vmlmb_CUTEst,

    # Trust region step:
    gqtpar!, gqtpar,

    # Miscellaneous:
    ObjectiveFunction,
    get_reason,
    issuccess

using LinearAlgebra
using NumOptBase
using TypeUtils

"""

`Float` is the type of all floating point scalars, it is currently an alias to
`Cdouble` which is itself an alias to `Float64`.

"""
const Float = Cdouble

include("wrappers.jl")

include("utils.jl")

# FIXME: include("gqtpar.jl")
# FIXME: import .MoreSorensen: gqtpar, gqtpar!

include("linesearches.jl")

include("quasinewton.jl")
import .QuasiNewton: vmlmb, vmlmb!, vmlmb_CUTEst

include("brent.jl")
import .Brent: fmin, fzero

# FIXME: include("nllsq.jl")
# FIXME: import .NonLinearLeastSquares: nllsq, nllsq!

include("bradi.jl")

include("step.jl")

include("spg.jl")
import .SPG: spg, spg!, spg_CUTEst

@static if !isdefined(Base, :get_extension)
    using Requires
    function __init__()
        if !isdefined(Base, :get_extension)
            @require CUTEst = "1b53aba6-35b6-5f92-a507-53c67d53f819" include(
                "../ext/OptimPackNextGenCUTEstExt.jl")
            @require NLPModels = "a4795742-8479-5a88-8948-cc11e1c8c1a6" include(
                "../ext/OptimPackNextGenNLPModelsExt.jl")
            @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" include(
                "../ext/OptimPackNextGenZygoteExt.jl")
        end
    end
end

end # module
