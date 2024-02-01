module OptimPackNextGenTests

using OptimPackNextGen
using Test
using Printf
using Zygote

VERBOSE = true

function banner(str::AbstractString)
    len = 75
    line = repeat('*', len)
    println()
    println(line)
    println("*** ", str, " ", repeat('*', max(3, len - length(str) - 5)))
    println(line)
end

include("fzero-tests.jl")
@testset "Brent fzero" FzeroTests.runtests(; verb=false)

include("fmin-tests.jl")
@testset "Brent fmin " FminTests.runtests(:fmin, :fmax; verb=false)
@testset "BraDi      " FminTests.runtests(:bradi; verb=false)
#@testset "STEP       " FminTests.runtests(:step; verb=false)

include("rosenbrock.jl")
include("bradi-tests.jl")

end # module

nothing
