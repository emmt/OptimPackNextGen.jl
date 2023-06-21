# brent-tests.jl -
#
# Tests for Brent's algorithms.
#

module BrentTests

using Test

include("fzero-tests.jl")
@testset "Brent fzero" FzeroTests.runtests(; verb=false)

include("fmin-tests.jl")
@testset "Brent fmin " FminTests.runtests(:fmin, :fmax; verb=false)

end # module
