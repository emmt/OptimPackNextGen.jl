module TestingSPG

using OptimPackNextGen
using Test
using Printf
using Zygote

VERBOSE = true

"""
    iterable(x) -> itr

yields an iterable object: `x`, if object `x` is iterable; `(x,)`, otherwise.

"""
iterable(x) = is_iterable(x) ? x : (x,)

"""
    is_iterable(x) -> bool

yields whether method `iterate` is implemented for object `x`.

"""
is_iterable(x) = is_iterable_type(typeof(x))

"""
    is_iterable_type(T) -> bool

yields whether method `iterate` is implemented for objects of type `T`.

"""
@generated is_iterable_type(::Type{T}) where {T} =
    # We use a generated function here so that we only pay the price of the
    # introspection once per given type.
    hasmethod(iterate, Tuple{T})

function rosenbrock_init!(x0::Array{T,1}) where {T<:Real}
    x0[1:2:end] .= -1.2
    x0[2:2:end] .=  1.0
    return x0
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

function nonnegative!(dst::AbstractArray{T,N},
                      src::AbstractArray{T,N}) where {T,N}
    vmin = zero(T)
    @inbounds @simd for i in eachindex(dst, src)
        val = src[i]
        dst[i] = (val < vmin ? vmin : val)
    end
    return dst
end

function runtests(; floats = (Float64, Float32), n::Integer = 20)
    # Run tests in double and single precisions.
    @testset "SPG for Rosenbrock function" for T in iterable(floats)
        eps = sizeof(T)*8 < 64 ? 1e-4 : 1e-6
        x0 = rosenbrock_init!(Array{T}(undef, n))
        info = SPG.Info()
        kwds = (info=info, verb=VERBOSE, eps1=eps, eps2=eps)
        N = ndims(x0)

        println("\n# Testing SPG with $T floats and nonnegativity.")
        x6 = spg(rosenbrock_fg!, BoundedSet{T,N}(0, +Inf), x0, 10; kwds...)
        @test issuccess(info)

        println("\n# Testing SPG with $T floats, automatic differentiation, and nonnegativity.")
        x7 = spg(rosenbrock_f, nonnegative!, x0, 10; kwds..., autodiff=true)
        @test issuccess(info)
    end
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    TestingSPG.runtests()
end
