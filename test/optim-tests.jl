if ! isdefined(:OptimPackNextGen)
    include("../src/OptimPackNextGen.jl")
end

module Test

using Base.Test
using OptimPackNextGen.Algebra
using OptimPackNextGen.LineSearch
using OptimPackNextGen.QuasiNewton

function rosenbrock_init!{T<:Real}(x0::Array{T,1})
  x0[1:2:end] = -1.2
  x0[2:2:end] =  1.0
  return nothing
end

function rosenbrock_fg!{T<:Real}(x::Array{T,1}, gx::Array{T,1})
  const c1 = T(1)
  const c2 = T(2)
  const c10 = T(10)
  const c200 = T(200)
  x1 = x[1:2:end]
  x2 = x[2:2:end]
  t1 = c1 - x1
  t2 = c10*(x2 - x1.*x1)
  g2 = c200*(x2 - x1.*x1)
  gx[1:2:end] = -c2*(x1.*g2 + t1)
  gx[2:2:end] = g2
  return sum(t1.*t1) + sum(t2.*t2)
end

function rosenbrock_test(n::Integer=20, m::Integer=3; single::Bool=false)
  T = (single ? Float32 : Float64)
  x0 = Array(T, n)
  rosenbrock_init!(x0)
  vmlmb(rosenbrock_fg!, x0, m, verb=true)
end

# Run tests in double and single precisions.
n = 20
for (T, prec) in ((Float64, "double"), (Float32, "single"))

    x0 = Array(T, n)
    rosenbrock_init!(x0)

    # First run tests in verbose mode (also serve for pre-compilation and
    # warmup).
    backtrack = BacktrackingLineSearch(amin=0.01)
    @printf("\nTesting L-BFGS in %s precision and with Moré & Thuente line search\n", prec)
    x = vmlmb(rosenbrock_fg!, x0, verb=true, fmin=0)
    #@test_approx_eq_eps x1 ones(T,n) 1e-3
    @printf("\nTesting L-BFGS in %s precision and with Armijo's line search\n", prec)
    x = vmlmb(rosenbrock_fg!, x0, verb=true, fmin=0, lnsrch=backtrack)
    #@test_approx_eq_eps x ones(T,n) 1e-3
    @printf("\nTesting BLMVM in %s precision and with Armijo's line search\n", prec)
    x = vmlmb(rosenbrock_fg!, x0, verb=true, fmin=0, lower=0, blmvm=true)
    #@test_approx_eq_eps x ones(T,n) 1e-3
    @printf("\nTesting VMLMB in %s precision and with Armijo's line search\n", prec)
    x = vmlmb(rosenbrock_fg!, x0, verb=true, fmin=0, lower=0)
    #@test_approx_eq_eps x ones(T,n) 1e-3

    # Then run tests for timings.
    @printf("\nTesting L-BFGS in %s precision and with Moré & Thuente line search\n", prec)
    @time x = vmlmb(rosenbrock_fg!, x0, verb=false, fmin=0)
    @printf("\nTesting L-BFGS in %s precision and with Armijo's line search\n", prec)
    @time x = vmlmb(rosenbrock_fg!, x0, verb=false, fmin=0, lnsrch=backtrack)
    @printf("\nTesting BLMVM in %s precision and with Armijo's line search\n", prec)
    @time x = vmlmb(rosenbrock_fg!, x0, verb=false, fmin=0, lower=0, blmvm=true)
    @printf("\nTesting VMLMB in %s precision and with Armijo's line search\n", prec)
    @time x = vmlmb(rosenbrock_fg!, x0, verb=false, fmin=0, lower=0)

end

nothing

end # module
