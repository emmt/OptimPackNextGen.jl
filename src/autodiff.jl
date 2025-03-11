"""
    AutoDiffObjectiveFunction{F,P,B}

A callable structure wrapping an objective function  `f`  that used automatic differentiation
for computing gradients.

Fields:
- `f::F`: The objective function.
- `prep::P`: prep  object (cache) used in differentiation (can be `nothing`).
- `backend::B`: The backend used for differentiation.
"""

abstract type AbstractObjectiveFunction <: Function end


"""
	AutoDiffObjectiveFunction{F,P,B}

A callable structure wrapping an objective function  `f`  that used automatic differentiation
for computing gradients.

Fields:
- `f::F`: The objective function.
- `prep::P`: prep  object (cache) used in differentiation (can be `nothing`).
- `backend::B`: The backend used for differentiation.
"""

struct AutoDiffObjectiveFunction{F,P,B} <: AbstractObjectiveFunction
	f::F
	prep::P
	backend::B
end

AutoDiffObjectiveFunction(arg...; kwds...) =
     error("`DifferentiationInterface` package must be loaded first")

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63" include(
            "../ext/OptimPackNextGenDifferentiationInterfaceExt.jl")
    end
end
