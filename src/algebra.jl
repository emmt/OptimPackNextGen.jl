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

export Operator,
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
       input_eltype,
       input_ndims,
       input_size,
       input_type,
       output_eltype,
       output_ndims,
       output_size,
       output_type,
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
       get_free_variables,
       NormalEquations,
       conjgrad,
       conjgrad!

include("operators.jl")
include("vectors.jl")
include("conjgrad.jl")

end # module
