# vectors-tests.jl -
#
# Tests for vectorized operations.
#

if ! isdefined(:OptimPackNextGen)
    include("../src/OptimPackNextGen.jl")
end

module VectorsTests

using Base.Test
using OptimPackNextGen.Algebra

@testset "vfill" begin
    for T in (Float16, Float32, Float64)
        dims = (3,4,5)
        a = randn(T, dims)
        @test (0,0) == extrema(vfill!(a, 0) .- zeros(T,dims))
        @test (0,0) == extrema(vfill!(a, 1) .- ones(T,dims))
        @test (0,0) == extrema(vfill!(a,pi) .- fill!(similar(a), pi))
    end
end

@testset "vscale" begin
    for T in (Float16, Float32, Float64)
        dims = (3,4,5)
        a = randn(T, dims)
        b = vcreate(a)
        @test (0,0) == extrema(vscale!(b, 0,a) .- 0*a)
        @test (0,0) == extrema(vscale!(b, 1,a) .- a)
        @test (0,0) == extrema(vscale!(b,-1,a) .+ a)
        @test (0,0) == extrema(vscale!(b,pi,a) .- T(pi)*a)
    end
end

@testset "vupdate" begin
    for T in (Float16, Float32, Float64)
        dims = (3,4,5)
        a = randn(T, dims)
        d = randn(T, dims)
        b = vcreate(a)
        @test (0,0) == extrema(vupdate!(vcopy!(b,a), 0,d) .- a)
        @test (0,0) == extrema(vupdate!(vcopy!(b,a), 1,d) .- (a .+ d))
        @test (0,0) == extrema(vupdate!(vcopy!(b,a),-1,d) .- (a .- d))
        @test (0,0) == extrema(vupdate!(vcopy!(b,a),pi,d) .- (a .+ T(pi)*d))
    end
end

@testset "vproduct" begin
    for T in (Float16, Float32, Float64)
        dims = (3,4,5)
        a = randn(T, dims)
        b = randn(T, dims)
        c = vcreate(a)
        @test (0,0) == extrema(vproduct!(c,a,b) .- (a .* b))
    end
end

@testset "vcombine" begin
    for T in (Float16, Float32, Float64)
        dims = (3,4,5)
        a = randn(T, dims)
        b = randn(T, dims)
        d = vcreate(a)
        α = randn(T)
        β = randn(T)
        @test (0,0) == extrema(vcombine!(d, 0,a, 0,b) .- zeros(T,dims))
        @test (0,0) == extrema(vcombine!(d, 0,a, 1,b) .- b)
        @test (0,0) == extrema(vcombine!(d, 0,a,-1,b) .+ b)
        @test (0,0) == extrema(vcombine!(d, 0,a,pi,b) .- T(pi)*b)
        @test (0,0) == extrema(vcombine!(d, 1,a, 0,b) .- a)
        @test (0,0) == extrema(vcombine!(d,-1,a, 0,b) .+ a)
        @test (0,0) == extrema(vcombine!(d,pi,a, 0,b) .- T(pi)*a)
        @test (0,0) == extrema(vcombine!(d, 1,a, 1,b) .- (a .+ b))
        @test (0,0) == extrema(vcombine!(d,-1,a, 1,b) .- (b .- a))
        @test (0,0) == extrema(vcombine!(d, 1,a,-1,b) .- (a .- b))
        @test (0,0) == extrema(vcombine!(d,-1,a,-1,b) .+ (a .+ b))
        @test (0,0) == extrema(vcombine!(d, α,a, β,b) .- (T(α)*a + T(β)*b))
    end
end

end # module
