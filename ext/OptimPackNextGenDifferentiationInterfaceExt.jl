# 
# OptimPackNextGenDifferentiationInterfaceExt.jl --
# 
# This module provides an interface for automatic differentiation using the 
# DifferentiationInterface package.
#
# The module defines a struct `OptimPackNextGen.AutoDiffObjectiveFunction` 
# which wraps a function `f`, a preparation object `prep`, and a backend `backend`.
#
module OptimPackNextGenDifferentiationInterfaceExt

if isdefined(Base, :get_extension)
    using DifferentiationInterface
    using OptimPackNextGen
else
    using ..DifferentiationInterface
    using ..OptimPackNextGen
end

struct AutoDiffObjectiveFunction{F,P,B} <: OptimPackNextGen.AbstractObjectiveFunction
	f::F
	prep::P
	backend::B
end

"""
	OptimPackNextGen.AutoDiffObjectiveFunction(f, backend)

Creates an `AutoDiffObjectiveFunction` for the given function `f` and backend `backend` without preparation.
"""
OptimPackNextGen.AutoDiffObjectiveFunction(f,backend) = AutoDiffObjectiveFunction(f, nothing, backend)

"""
	(f::OptimPackNextGen.AutoDiffObjectiveFunction)(x, gx)
	
Computes the value and gradient of the objective function at `x`, storing the gradient in `gx`.
Returns the value of the objective function.
"""
(f::OptimPackNextGen.AutoDiffObjectiveFunction)(x, gx) = first(DifferentiationInterface.value_and_gradient!(f.f, gx, f.prep, f.backend, x))
(f::OptimPackNextGen.AutoDiffObjectiveFunction{F,Nothing,B})(x, gx) where {F,B} = first(DifferentiationInterface.value_and_gradient!(f.f, gx, f.backend, x))

end # module
