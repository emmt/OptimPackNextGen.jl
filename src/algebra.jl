#
# algebra.jl --
#
# Provide linear operators and implement operations on *vectors*.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#

module Algebra

# Use the same floating point type for scalars as in TiPi.
import TiPi.Float

include("operators.jl")
include("vectors.jl")
include("conjgrad.jl")

end # module
