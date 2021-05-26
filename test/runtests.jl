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

include("rosenbrock.jl")
include("bradi-tests.jl")

if true
    include("cobyla-tests.jl")
    banner("Standard tests")
    CobylaTests.runtests()
    banner("Tests with scale=0.7")
    CobylaTests.runtests(scale=0.7)
    banner("Tests with reverse-communication")
    CobylaTests.runtests(revcom=true)
end

if true
    include("newuoa-tests.jl")
    banner("Standard NEWUOA tests")
    NewuoaTests.runtests()
    banner("NEWUOA tests with scale=0.7")
    NewuoaTests.runtests(scale=0.7)
    banner("NEWUOA tests with reverse-communication")
    NewuoaTests.runtests(revcom=true)
end

if true
    include("bobyqa-tests.jl")
    banner("Standard BOBYQA tests")
    BobyqaTests.runtests()
end

end # module

nothing
