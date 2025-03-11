module RosenbrokTests

using OptimPackNextGen
using Test
using Printf
using ForwardDiff
using DifferentiationInterface
using Zygote
using MoonCake

backendZ = AutoZigote()
backendFD = AutoForwardDiff()
backendM = AutoMooncake(; config=nothing)

rosenbrock_fg! = OptimPackNextGen.Examples.rosenbrock_fg!
rosenbrock_f = OptimPackNextGen.Examples.rosenbrock_f

VERBOSE = true

function rosenbrock_init!(x0::Array{T,1}) where {T<:Real}
	x0[1:2:end] .= -1.2
	x0[2:2:end] .=  1.0
	return nothing
end

function rosenbrock_test(n::Integer=20, m::Integer=3; single::Bool=false)
  T = (single ? Float32 : Float64)
  x0 = Array{T}(undef, n)
  rosenbrock_init!(x0)
  lbfgs(rosenbrock_fg!, x0, m, verb=VERBOSE)
end

function nonnegative!(dst::AbstractArray{T,N},
                      src::AbstractArray{T,N}) where {T,N}
    vmin = zero(T)
    @inbounds @simd for i in eachindex(dst, src)
        val = src[i]
        dst[i] = (val < vmin ? vmin : val)
    end
    return dst
end

# Run tests in double and single precisions.
for (T, prec) in ((Float64, "double"), (Float32, "single"))

    n = 20
    x0 = Array{T}(undef, n)
    xsol = ones(T,n)
    atol = 1e-3
    rosenbrock_init!(x0)
    prepFD = prepare_gradient(rosenbrock_f, backendFD, zero(x0))
    prepZ = prepare_gradient(rosenbrock_f, backendZ, zero(x0))
    prepM = prepare_gradient(rosenbrock_f, backendM, zero(x0))
    
    fgFD! = AutoDiffObjectiveFunction(rosenbrock_f, prepFD, backendFD)
    fgZ! = AutoDiffObjectiveFunction(rosenbrock_f, prepZ, backendZ)
    fgM! = AutoDiffObjectiveFunction(rosenbrock_f, prepM, backendM)

    #@printf("\nTesting NLCG in %s precision\n", prec)
    #x1 = nlcg(rosenbrock_fg!, x0, verb=VERBOSE)
    #err = maximum(abs.(x1 .- xsol))
    #@printf("Maximum absolute error: %.3e\n", err)
    #@test err < atol
    VERBOSE = 10 # test verbose != Bool
    @printf("\nTesting VMLMB in %s precision with Oren & Spedicato scaling\n", prec)
    x2 = vmlmb(rosenbrock_fg!, x0, verb=VERBOSE)
    err = maximum(abs.(x2 .- xsol))
    @printf("Maximum absolute error: %.3e\n", err)
    @test err < atol

    @printf("\nTesting VMLMB with automatic differentiation with ForwardDiff in %s precision with Oren & Spedicato scaling\n", prec)
    x2a = vmlmb(fgFD!, x0, verb=VERBOSE)
    err = maximum(abs.(x2a .- xsol))
    @printf("Maximum absolute error: %.3e\n", err)
    @test err < atol

    @printf("\nTesting VMLMB with automatic differentiation with Zygote in %s precision with Oren & Spedicato scaling\n", prec)
    x2b = vmlmb(fgZ!, x0, verb=VERBOSE)
    err = maximum(abs.(x2b .- xsol))
    @printf("Maximum absolute error: %.3e\n", err)
    @test err < atol

    @printf("\nTesting VMLMB with automatic differentiation with MoonCake in %s precision with Oren & Spedicato scaling\n", prec)
    x2c = vmlmb(fgM!, x0, verb=VERBOSE)
    err = maximum(abs.(x2c .- xsol))
    @printf("Maximum absolute error: %.3e\n", err)
    @test err < atol

    VERBOSE = true
    @printf("\nTesting VMLMB in %s precision with Oren & Spedicato scaling\n", prec)
    x3 = vmlmb(rosenbrock_fg!, x0, verb=VERBOSE, mem=15)
    err = maximum(abs.(x3 .- xsol))
    @printf("Maximum absolute error: %.3e\n", err)
    @test err < atol

    @printf("\nTesting VMLMB with automatic differentiation in %s precision with Oren & Spedicato scaling\n", prec)
    x3a = vmlmb(fgFD!, x0, verb=VERBOSE, mem=15)
    err = maximum(abs.(x3a .- xsol))
    @printf("Maximum absolute error: %.3e\n", err)
    @test err < atol

    @printf("\nTesting VMLMB in %s precision with nonnegativity\n", prec)
    x4 = vmlmb(rosenbrock_fg!, x0, verb=VERBOSE, lower=0)
    err = maximum(abs.(x4 .- xsol))
    @printf("Maximum absolute error: %.3e\n", err)
    @test err < atol

    @printf("\nTesting VMLMB with automatic differentiation in %s precision with nonnegativity\n", prec)
    x4a = vmlmb(fgFD!, x0, verb=VERBOSE, lower=0)
    err = maximum(abs.(x4a .- xsol))
    @printf("Maximum absolute error: %.3e\n", err)
    @test err < atol

    #@printf("\nTesting VMLM in %s precision with Barzilai & Borwein scaling\n", prec)
    #x5 = vmlmb(rosenbrock_fg!, x0, verb=VERBOSE,
    #           scaling=OptimPackNextGen.SCALING_BARZILAI_BORWEIN)
    #err = maximum(abs.(x5 .- xsol))
    #@printf("Maximum absolute error: %.3e\n", err)
    #@test err < atol

    @printf("\nTesting SPG in %s precision with nonnegativity\n", prec)
    x6 = spg(rosenbrock_fg!, nonnegative!, x0, 10; verb=VERBOSE)
    @printf("\nTesting SPG in %s precision with automatic differentiation and nonnegativity\n", prec)
    x7 = spg(fgFD!, nonnegative!, x0, 10; verb=VERBOSE, autodiff=true)
end

end # module

nothing
