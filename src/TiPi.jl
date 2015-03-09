#
# TiPi.jl --
#
# Toolkit for Inverse Problems and Imaging in Julia.
#
#------------------------------------------------------------------------------
#
# This file is part of TiPi.jl licensed under the MIT "Expat" License.
#
# Copyright (C) 2015, Éric Thiébaut & Jonathan Léger.
#
#------------------------------------------------------------------------------

module TiPi

export cost, cost!, CostParam, HyperbolicEdgePreserving

include("cost.jl")
include("hypersmooth.jl")

end # module

# Local Variables:
# mode: Julia
# tab-width: 8
# indent-tabs-mode: nil
# fill-column: 79
# coding: utf-8
# ispell-local-dictionary: "american"
# End:
