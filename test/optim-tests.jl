module Test

using Base.Test

include("../src/TiPi.jl")
using .TiPi.LineSearch
using .TiPi.QuasiNewton

function rosenbrock_init!{T<:Real}(x0::Array{T,1})
  x0[1:2:end] = -1.2
  x0[2:2:end] =  1.0
  return nothing
end

function rosenbrock_fg!{T<:Real}(x::Array{T,1}, gx::Array{T,1})
  const c1::T = 1
  const c2::T = 2
  const c10::T = 10
  const c200::T = 200
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
  lbfgs(rosenbrock_fg!, x0, m, verb=true)
end

# Run tests in double and single precisions.
for (T, prec) in ((Float64, "double"), (Float32, "single"))

    x0 = Array(T, 20)
    rosenbrock_init!(x0)

    @printf("\nTesting L-BFGS in %s precision and with Moré & Thuente line search\n", prec)
    x1 = lbfgs(rosenbrock_fg!, x0, verb=true)
    #@test_approx_eq_eps x1 ones(T,20) 1e-3

    @printf("\nTesting L-BFGS in %s precision and with Armijo's line search\n", prec)
    x2 = lbfgs(rosenbrock_fg!, x0, verb=true, lnsrch=BacktrackingLineSearch(amin=0.01))
    #@test_approx_eq_eps x1 ones(T,20) 1e-3

    #@printf("\nTesting NLCG in %s precision\n", prec)
    #x1 = OptimPack.nlcg(rosenbrock_fg!, x0, verb=true)
    #@test_approx_eq_eps x1 ones(T,20) 1e-3
    #
    #@printf("\nTesting VMLMB in %s precision with Oren & Spedicato scaling\n", prec)
    #x2 = OptimPack.vmlmb(rosenbrock_fg!, x0, verb=true)
    #                    #scaling=OptimPack.SCALING_OREN_SPEDICATO)
    #@test_approx_eq_eps x2 ones(T,20) 1e-3
    #
    #@printf("\nTesting VMLMB in %s precision with Oren & Spedicato scaling\n", prec)
    #x3 = OptimPack.vmlmb(rosenbrock_fg!, x0, verb=true, mem=15)
    #                    #scaling=OptimPack.SCALING_OREN_SPEDICATO)
    #@test_approx_eq_eps x3 ones(T,20) 1e-3
    #
    #@printf("\nTesting VMLMB in %s precision with nonnegativity\n", prec)
    #x4 = OptimPack.vmlmb(rosenbrock_fg!, x0, verb=true, lower=0)
    #@test_approx_eq_eps x4 ones(T,20) 1e-3

    #@printf("\nTesting VMLM in %s precision with Barzilai & Borwein scaling\n", prec)
    #x3 = OptimPack.vmlmb(rosenbrock_fg!, x0, verb=true,
    #                     scaling=OptimPack.SCALING_BARZILAI_BORWEIN)
    #@test_approx_eq_eps x3 ones(T,20) 1e-4

end

nothing

end # module
