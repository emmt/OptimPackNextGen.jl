if ! isdefined(:OptimPackNextGen)
    include("../src/OptimPackNextGen.jl")
end

module OptimTest

using Test
using LazyAlgebra
using OptimPackNextGen.LineSearches
using OptimPackNextGen.QuasiNewton

function rosenbrock_init!(x0::Vector{<:Real})
  x0[1:2:end] .= -1.2
  x0[2:2:end] .=  1.0
  return nothing
end

function rosenbrock_fg!(x::Vector{T}, gx::Vector{T}) where {T<:Real}
  const c1 = T(1)
  const c2 = T(2)
  const c10 = T(10)
  const c200 = T(200)
  x1 = x[1:2:end]
  x2 = x[2:2:end]
  t1 = c1 .- x1
  t2 = c10*(x2 - x1 .* x1)
  g2 = c200*(x2 - x1 .* x1)
  gx[1:2:end] = -c2*(x1 .* g2 + t1)
  gx[2:2:end] = g2
  return sum(t1 .* t1) + sum(t2 .* t2)
end

function rosenbrock_test(n::Integer=20, m::Integer=3; single::Bool=false)
  T = (single ? Float32 : Float64)
  x0 = Array{T}(undef, n)
  rosenbrock_init!(x0)
  vmlmb(rosenbrock_fg!, x0, m, verb=true)
end

# Run tests in double and single precisions.
n = 20
for (T, prec) in ((Float64, "double"), (Float32, "single"))

    x0 = Array{T}(undef, n)
    rosenbrock_init!(x0)
    armijo = ArmijoLineSearch()
    moretoraldo = MoreToraldoLineSearch(gamma=(0.1,0.5))
    morethuente = MoreThuenteLineSearch(ftol=1e-3, gtol=0.9, xtol=0.1)

    # First run tests in verbose mode (also serve for pre-compilation and
    # warmup).
    @printf("\nTesting L-BFGS in %s precision and with Armijo's line search\n", prec)
    x = vmlmb(rosenbrock_fg!, x0; verb=true, fmin=0, lnsrch=armijo)
    #@test_approx_eq_eps x ones(T,n) 1e-3
    @printf("\nTesting L-BFGS in %s precision and with Moré & Toraldo line search\n", prec)
    x = vmlmb(rosenbrock_fg!, x0; verb=true, fmin=0, lnsrch=moretoraldo)
    #@test_approx_eq_eps x ones(T,n) 1e-3
    @printf("\nTesting L-BFGS in %s precision and with Moré & Thuente line search\n", prec)
    x = vmlmb(rosenbrock_fg!, x0; verb=true, fmin=0, lnsrch=morethuente)
    #@test_approx_eq_eps x1 ones(T,n) 1e-3
    @printf("\nTesting BLMVM in %s precision and with Armijo's line search\n", prec)
    x = vmlmb(rosenbrock_fg!, x0; verb=true, fmin=0, lower=0, lnsrch=armijo, blmvm=true)
    #@test_approx_eq_eps x ones(T,n) 1e-3
    @printf("\nTesting BLMVM in %s precision and with Moré & Toraldo line search\n", prec)
    x = vmlmb(rosenbrock_fg!, x0; verb=true, fmin=0, lower=0, lnsrch=moretoraldo, blmvm=true)
    #@test_approx_eq_eps x ones(T,n) 1e-3
    @printf("\nTesting VMLMB in %s precision and with Armijo's line search\n", prec)
    x = vmlmb(rosenbrock_fg!, x0; verb=true, fmin=0, lower=0, lnsrch=armijo)
    #@test_approx_eq_eps x ones(T,n) 1e-3
    @printf("\nTesting VMLMB in %s precision and with Moré & Toraldo line search\n", prec)
    x = vmlmb(rosenbrock_fg!, x0; verb=true, fmin=0, lower=0, lnsrch=moretoraldo)
    #@test_approx_eq_eps x ones(T,n) 1e-3

    # Then run tests for timings.
    @printf("\nTesting L-BFGS in %s precision and with Armijo's line search\n", prec)
    @time x = vmlmb(rosenbrock_fg!, x0; verb=false, fmin=0, lnsrch=armijo)
    @printf("\nTesting L-BFGS in %s precision and with Moré & Toraldo line search\n", prec)
    @time x = vmlmb(rosenbrock_fg!, x0; verb=false, fmin=0, lnsrch=moretoraldo)
    @printf("\nTesting L-BFGS in %s precision and with Moré & Thuente line search\n", prec)
    @time x = vmlmb(rosenbrock_fg!, x0; verb=false, fmin=0, lnsrch=morethuente)


    @printf("\nTesting BLMVM in %s precision and with Armijo's line search\n", prec)
    @time x = vmlmb(rosenbrock_fg!, x0; verb=false, fmin=0, lower=0, lnsrch=armijo, blmvm=true)
    @printf("\nTesting BLMVM in %s precision and with Moré & Toraldo line search\n", prec)
    @time x = vmlmb(rosenbrock_fg!, x0; verb=false, fmin=0, lower=0, lnsrch=moretoraldo, blmvm=true)

    @printf("\nTesting VMLMB in %s precision and with Armijo's line search\n", prec)
    @time x = vmlmb(rosenbrock_fg!, x0; verb=false, fmin=0, lower=0, lnsrch=armijo)
    @printf("\nTesting VMLMB in %s precision and with Moré & Toraldo line search\n", prec)
    @time x = vmlmb(rosenbrock_fg!, x0; verb=false, fmin=0, lower=0, lnsrch=moretoraldo)

end

nothing

end # module
