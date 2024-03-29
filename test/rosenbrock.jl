module RosenbrokTests

using OptimPackNextGen
using Test
using Printf
using Zygote

VERBOSE = true

function rosenbrock_init!(x0::Array{T,1}) where {T<:Real}
  x0[1:2:end] .= -1.2
  x0[2:2:end] .=  1.0
  return nothing
end

function rosenbrock_fg!(x::Array{T,1}, gx::Array{T,1}) where {T<:Real}
  local c1::T = 1
  local c2::T = 2
  local c10::T = 10
  local c200::T = 200
  x1 = x[1:2:end]
  x2 = x[2:2:end]
  t1 = c1 .- x1
  t2 = c10*(x2 - x1.*x1)
  g2 = c200*(x2 - x1.*x1)
  gx[1:2:end] = -c2*(x1 .* g2 + t1)
  gx[2:2:end] = g2
  return sum(t1.*t1) + sum(t2.*t2)
end

function rosenbrock_f(x::Array{T,1}) where {T<:Real}
  local c1::T = 1
  local c2::T = 2
  local c10::T = 10
  local c200::T = 200
  x1 = x[1:2:end]
  x2 = x[2:2:end]
  t1 = c1 .- x1
  t2 = c10*(x2 - x1.*x1)
  return sum(t1.*t1) + sum(t2.*t2)
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

    @printf("\nTesting VMLMB with automatic differentiation in %s precision with Oren & Spedicato scaling\n", prec)
    x2a = vmlmb(rosenbrock_f, x0, verb=VERBOSE, autodiff=true)
    err = maximum(abs.(x2 .- xsol))
    @printf("Maximum absolute error: %.3e\n", err)
    @test err < atol

    VERBOSE = true
    @printf("\nTesting VMLMB in %s precision with Oren & Spedicato scaling\n", prec)
    x3 = vmlmb(rosenbrock_fg!, x0, verb=VERBOSE, mem=15)
    err = maximum(abs.(x3 .- xsol))
    @printf("Maximum absolute error: %.3e\n", err)
    @test err < atol

    @printf("\nTesting VMLMB with automatic differentiation in %s precision with Oren & Spedicato scaling\n", prec)
    x3a = vmlmb(rosenbrock_f, x0, verb=VERBOSE, mem=15, autodiff=true)
    err = maximum(abs.(x3 .- xsol))
    @printf("Maximum absolute error: %.3e\n", err)
    @test err < atol

    @printf("\nTesting VMLMB in %s precision with nonnegativity\n", prec)
    x4 = vmlmb(rosenbrock_fg!, x0, verb=VERBOSE, lower=0)
    err = maximum(abs.(x4 .- xsol))
    @printf("Maximum absolute error: %.3e\n", err)
    @test err < atol

    @printf("\nTesting VMLMB with automatic differentiation in %s precision with nonnegativity\n", prec)
    x4a = vmlmb(rosenbrock_f, x0, verb=VERBOSE, lower=0, autodiff=true)
    err = maximum(abs.(x4 .- xsol))
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
    x7 = spg(rosenbrock_f, nonnegative!, x0, 10; verb=VERBOSE, autodiff=true)
end

end # module

nothing
