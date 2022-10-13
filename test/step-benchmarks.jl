module BenchmarkingStep
using BenchmarkTools
using OptimPackNextGen: Step

print("\n# Simple parabola:")
(xbest, fbest, xtol, n, st) = @btime Step.minimize($(Step.testParabola), -1, 2, maxeval=100,
                                                   verb=false, atol=0, rtol=0)
println("x = $xbest ± $xtol, f(x) = $fbest, ncalls = $n, status = $st")

print("\n# Brent's 5th function:")
(xbest, fbest, xtol, n, st) = @btime Step.minimize($(Step.testBrent5), -10, 10, maxeval=1000,
                                                   verb=false, atol=0, rtol=0)
println("x = $xbest ± $xtol, f(x) = $fbest, ncalls = $n, status = $st")

print("\n# Michalewicz's 1st function:")
(xbest, fbest, xtol, n, st) = @btime Step.minimize($(Step.testMichalewicz1), -1, 2, maxeval=1000,
                                                   verb=false, atol=0, rtol=0)
println("x = $xbest ± $xtol, f(x) = $fbest, ncalls = $n, status = $st")

print("\n# Michalewicz's 2nd function:")
(xbest, fbest, xtol, n, st) = @btime Step.maximize($(Step.testMichalewicz2), 0, $pi, maxeval=1000,
                                                   verb=false, atol=0, rtol=0)
println("x = $xbest ± $xtol, f(x) = $fbest, ncalls = $n, status = $st")

end # module
