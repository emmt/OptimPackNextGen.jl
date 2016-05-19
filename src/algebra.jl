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

export LinearOperator,
       SelfAdjointOperator,
       Identity,
       apply_direct,
       apply_adjoint,
       inner,
       norm1,
       norm2,
       normInf,
       swap!,
       update!,
       combine!,
       project_variables!,
       project_direction!,
       step_limits,
       get_free_variables

# Use the same floating point type for scalars as in TiPi.
import ..Float

include("operators.jl")
include("vectors.jl")

end # module
