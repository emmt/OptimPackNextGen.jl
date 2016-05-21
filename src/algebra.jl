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
       Endomorphism,
       SelfAdjointOperator,
       Identity,
       NormalEquations,
       apply_direct,
       apply_direct!,
       apply_adjoint,
       apply_adjoint!,
       vcombine!,
       vcopy!,
       vcreate,
       vdot,
       vfill!,
       vnorm1,
       vnorm2,
       vnorminf,
       vproduct!,
       vscale!,
       vswap!,
       vupdate!,
       conjgrad,
       project_variables!,
       project_direction!,
       step_limits,
       get_free_variables

# Use the same floating point type for scalars as in TiPi.
import ..Float

include("operators.jl")
include("vectors.jl")
include("conjgrad.jl")

end # module
