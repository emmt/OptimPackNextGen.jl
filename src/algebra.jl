#
# algebra.jl --
#
# Provide operations on *vectors*.
#
# -----------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2015-2017, Éric Thiébaut.
#

module Algebra

using Compat

# Use the same floating point type for scalars as in OptimPack.
import OptimPackNextGen.Float

export
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
    project_variables!,
    project_direction!,
    step_limits,
    get_free_variables,
    get_free_variables!

include("vectors.jl")

end # module
