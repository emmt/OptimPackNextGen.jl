# conjgrad-tests.jl -
#
# Tests for conjugate gradient algorithm.
#

if ! isdefined(:OptimPackNextGen)
    include("../src/OptimPackNextGen.jl")
end

module ConjGradTests

using Base.Test
using OptimPackNextGen

verb = true
@testset "conjgrad" begin
    for (T, prec) in ((Float16, 5e-2),
                      (Float32, 1e-5),
                      (Float64, 1e-7))
        # FIXME: Due to the randomness, the tests mail fail...
        dims = (70,39)
        H = randn(T, dims)
        H *= T(1/maximum(abs(H)))
        x = randn(T, dims[2])
        y = H*x + T(0.01)*randn(T, dims[1])
        A = H'*H
        A!(dst, src) = (dst[:] = A*src; dst)
        b = H'*y
        x0 = A\b
        err = maximum(abs(conjgrad(A!, b; verb=verb, ftol=1e-16,
                                   maxiter=length(b)) - x0))

        verb && println(T, " --> ", err, "\n")
        @test err ≤ prec
    end
end

end # module
