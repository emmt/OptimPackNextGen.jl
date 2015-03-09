using TiPi
using Base.Test

function reldif(a::Real, b::Real)
    a == b && return 0.0
    2*abs(a - b)/(abs(a) + abs(b))
end

function reldif{T<:Real}(a::Array{T}, b::Array{T})
    dn = enorm(a - b)
    dn == 0 && return 0.0
    2*dn/(enorm(a) + enorm(b))
end

function enorm{T<:Real}(a::Array{T})
    s::Cdouble = 0.0
    for ai in a
        s += convert(Cdouble, ai)^2
    end
    return sqrt(s)
end

function xtime_print(str, val, tbg)
    t, b, g = tbg
    if 0 < g
        @printf("%s %.15E [elapsed time: %.3E seconds, %d bytes allocated, %.2f%% gc time]\n",
                str, val, t/1e9, b, 100*g/t)
    else
        @printf("%s %.15E [elapsed time: %.3E seconds, %d bytes allocated]\n",
                str, val, t/1e9, b)
    end
end

macro xtime(ex)
    quote
        local b0 = Base.gc_bytes()
        local t0 = Base.time_ns()
        local g0 = Base.gc_time_ns()
        local val = $(esc(ex))
        local g1 = Base.gc_time_ns()
        local t1 = Base.time_ns()
        local b1 = Base.gc_bytes()
        val, (t1-t0, b1-b0, g1-g0)
    end
end

function testcost{T}(alpha::Real, param::HyperbolicEdgePreserving,
                     x::Array{T})
    res = @xtime(cost(alpha, param, x))
    dims = size(x)
    rank = length(dims)
    if rank == 1
        ref = @xtime begin
            mu1 = convert(T, param.mu[1])
            eps = convert(T, param.eps)
            dim1 = dims[1]
            x1 = x[1:end-1]
            x2 = x[2:end]
            alpha*(sum(sqrt((mu1)^2*(x2 - x1).^2 + eps^2)) - length(x1)*eps)
        end
    elseif rank == 2
        ref = @xtime begin
            mu1 = convert(T, param.mu[1])
            mu2 = convert(T, param.mu[2])
            eps = convert(T, param.eps)
            dim1 = dims[1]
            dim2 = dims[2]
            x1 = x[ 1:end-1 , 1:end-1 ]
            x2 = x[ 2:end   , 1:end-1 ]
            x3 = x[ 1:end-1 , 2:end   ]
            x4 = x[ 2:end   , 2:end   ]
            alpha*(sum(sqrt((mu1^2/2)*((x2 - x1).^2 + (x4 - x3).^2) +
                            (mu2^2/2)*((x3 - x1).^2 + (x4 - x2).^2) + eps^2))
                   - length(x1)*eps)
        end
    elseif rank == 3
        ref = @xtime begin
            mu1 = convert(T, param.mu[1])
            mu2 = convert(T, param.mu[2])
            mu3 = convert(T, param.mu[3])
            eps = convert(T, param.eps)
            dim1 = dims[1]
            dim2 = dims[2]
            x1 = x[ 1:end-1 , 1:end-1 , 1:end-1 ]
            x2 = x[ 2:end   , 1:end-1 , 1:end-1 ]
            x3 = x[ 1:end-1 , 2:end   , 1:end-1 ]
            x4 = x[ 2:end   , 2:end   , 1:end-1 ]
            x5 = x[ 1:end-1 , 1:end-1 , 2:end   ]
            x6 = x[ 2:end   , 1:end-1 , 2:end   ]
            x7 = x[ 1:end-1 , 2:end   , 2:end   ]
            x8 = x[ 2:end   , 2:end   , 2:end   ]
            alpha*(sum(sqrt((mu1^2/4)*((x2 - x1).^2 + (x4 - x3).^2 +
                                       (x6 - x5).^2 + (x8 - x7).^2) +
                            (mu2^2/4)*((x3 - x1).^2 + (x4 - x2).^2 +
                                       (x7 - x5).^2 + (x8 - x6).^2) +
                            (mu3^2/4)*((x5 - x1).^2 + (x6 - x2).^2 +
                                       (x7 - x3).^2 + (x8 - x4).^2) + eps^2))
                   - length(x1)*eps)
        end
    end
    err = reldif(res[1], ref[1])/(Base.eps(T)*sqrt(length(x)))
    xtime_print("   reference   =", ref...)
    xtime_print("   code result =", res...)
    @printf("   normalized error = %.3E\n", err)
    return err
end

function testcost!{T}(alpha::Real, param::HyperbolicEdgePreserving,
                      x::Array{T})
    dims = size(x)
    gx = Array(T, dims)
    gxref = Array(T, dims)
    res = @xtime(cost!(alpha, param, x, gx, true))
    fill!(gxref, 0)
    rank = length(dims)
    if rank == 1
        ref = @xtime begin
            mu1 = convert(T, param.mu[1])
            eps = convert(T, param.eps)
            dim1 = dims[1]
            x1 = x[1:end-1]
            x2 = x[2:end]
            x2_x1 = x2 - x1
            r = sqrt(mu1^2*x2_x1.^2 + eps^2)
            fx = alpha*(sum(r) - length(x1)*eps)
            q = (alpha*mu1^2)./r.*x2_x1
            gxref[1:end-1] -= q
            gxref[2:end  ] += q
            fx
        end
    elseif rank == 2
        ref = @xtime begin
            mu1 = convert(T, param.mu[1])
            mu2 = convert(T, param.mu[2])
            eps = convert(T, param.eps)
            dim1 = dims[1]
            dim2 = dims[2]
            x1 = x[ 1:end-1 , 1:end-1 ]
            x2 = x[ 2:end   , 1:end-1 ]
            x3 = x[ 1:end-1 , 2:end   ]
            x4 = x[ 2:end   , 2:end   ]
            r = sqrt((mu1^2/2)*((x2 - x1).^2 + (x4 - x3).^2) +
                     (mu2^2/2)*((x3 - x1).^2 + (x4 - x2).^2) + eps^2)
            fx = alpha*(sum(r) - length(x1)*eps)
            q = alpha./r
            q1 = (mu1^2/2)*q
            q2 = (mu2^2/2)*q
            gxref[ 1:end-1 , 1:end-1 ] += q1.*(x1 - x2) + q2.*(x1 - x3)
            gxref[ 2:end   , 1:end-1 ] += q1.*(x2 - x1) + q2.*(x2 - x4)
            gxref[ 1:end-1 , 2:end   ] += q1.*(x3 - x4) + q2.*(x3 - x1)
            gxref[ 2:end   , 2:end   ] += q1.*(x4 - x3) + q2.*(x4 - x2)
            fx
        end
    elseif rank == 3
        ref = @xtime begin
            mu1 = convert(T, param.mu[1])
            mu2 = convert(T, param.mu[2])
            mu3 = convert(T, param.mu[3])
            eps = convert(T, param.eps)
            dim1 = dims[1]
            dim2 = dims[2]
            x1 = x[ 1:end-1 , 1:end-1 , 1:end-1 ]
            x2 = x[ 2:end   , 1:end-1 , 1:end-1 ]
            x3 = x[ 1:end-1 , 2:end   , 1:end-1 ]
            x4 = x[ 2:end   , 2:end   , 1:end-1 ]
            x5 = x[ 1:end-1 , 1:end-1 , 2:end   ]
            x6 = x[ 2:end   , 1:end-1 , 2:end   ]
            x7 = x[ 1:end-1 , 2:end   , 2:end   ]
            x8 = x[ 2:end   , 2:end   , 2:end   ]
            r = sqrt((mu1^2/4)*((x2 - x1).^2 + (x4 - x3).^2 +
                                (x6 - x5).^2 + (x8 - x7).^2) +
                     (mu2^2/4)*((x3 - x1).^2 + (x4 - x2).^2 +
                                (x7 - x5).^2 + (x8 - x6).^2) +
                     (mu3^2/4)*((x5 - x1).^2 + (x6 - x2).^2 +
                                (x7 - x3).^2 + (x8 - x4).^2) + eps^2)
            fx = alpha*(sum(r) - length(x1)*eps)
            q = alpha./r
            q1 = (mu1^2/4)*q
            q2 = (mu2^2/4)*q
            q3 = (mu3^2/4)*q
            gxref[ 1:end-1 , 1:end-1 , 1:end-1 ] += q1.*(x1 - x2) + q2.*(x1 - x3) + q3.*(x1 - x5)
            gxref[ 2:end   , 1:end-1 , 1:end-1 ] += q1.*(x2 - x1) + q2.*(x2 - x4) + q3.*(x2 - x6)
            gxref[ 1:end-1 , 2:end   , 1:end-1 ] += q1.*(x3 - x4) + q2.*(x3 - x1) + q3.*(x3 - x7)
            gxref[ 2:end   , 2:end   , 1:end-1 ] += q1.*(x4 - x3) + q2.*(x4 - x2) + q3.*(x4 - x8)
            gxref[ 1:end-1 , 1:end-1 , 2:end   ] += q1.*(x5 - x6) + q2.*(x5 - x7) + q3.*(x5 - x1)
            gxref[ 2:end   , 1:end-1 , 2:end   ] += q1.*(x6 - x5) + q2.*(x6 - x8) + q3.*(x6 - x2)
            gxref[ 1:end-1 , 2:end   , 2:end   ] += q1.*(x7 - x8) + q2.*(x7 - x5) + q3.*(x7 - x3)
            gxref[ 2:end   , 2:end   , 2:end   ] += q1.*(x8 - x7) + q2.*(x8 - x6) + q3.*(x8 - x4)
            fx
        end
    end
    fx_err = reldif(res[1], ref[1])/(Base.eps(T)*sqrt(length(x)))
    gx_err = reldif(gx, gxref)
    xtime_print("   reference   =", ref...)
    xtime_print("   code result =", res...)
    @printf("   normalized error for f(x) = %.3E\n", fx_err)
    @printf("   relative error for g(x) = %.3E\n", gx_err)
    return fx_err
end

# Weight and threshold for all cases:
alpha = 1.0 + rand()
eps = 0.02

# 1-D tests
println("\nRunning 1D tests...")
dims = (178,)
x = rand(dims) - 0.5
mu = (1.7,)
param = HyperbolicEdgePreserving(eps, mu)
@test testcost(alpha, param, x) <= 0.3
testcost!(alpha, param, x)

# 2-D tests
println("\nRunning 2D tests...")
dims = (174, 200)
x = rand(dims) - 0.5

println(" - Isotropic case:")
mu = (1.2, 1.2)
param = HyperbolicEdgePreserving(eps, mu)
@test testcost(alpha, param, x) <= 0.3
testcost!(alpha, param, x)

println(" - Anisotropic case:")
mu = (1.7, 1.2)
param = HyperbolicEdgePreserving(eps, mu)
@test testcost(alpha, param, x) <= 0.3
testcost!(alpha, param, x)

# 3-D tests
println("\nRunning 3D tests...")
dims = (174, 200, 108)
x = rand(dims) - 0.5

println(" - Isotropic case:")
mu = (1.2, 1.2, 1.2)
param = HyperbolicEdgePreserving(eps, mu)
@test testcost(alpha, param, x) <= 0.3
@test testcost!(alpha, param, x) <= 0.3

println(" - Anisotropic case:")
mu = (1.7, 1.2, 0.9)
param = HyperbolicEdgePreserving(eps, mu)
@test testcost(alpha, param, x) <= 0.3
@test testcost!(alpha, param, x) <= 0.3
