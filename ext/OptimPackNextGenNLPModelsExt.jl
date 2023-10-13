module OptimPackNextGenNLPModelsExt

if isdefined(Base, :get_extension)
    using NLPModels
    using OptimPackNextGen
    import OptimPackNextGen: BoundedSet, ObjectiveFunction
    using NumOptBase
    import NumOptBase: Projector
else
    using ..NLPModels
    using ..OptimPackNextGen
    import ..OptimPackNextGen: BoundedSet, ObjectiveFunction
    using ..NumOptBase
    import ..NumOptBase: Projector
end

function (objfun::ObjectiveFunction{<:AbstractNLPModel})(x::AbstractArray)
    return obj(objfun, x)
end

function (objfun::ObjectiveFunction{<:AbstractNLPModel})(x::AbstractArray{T,N},
                                                         g::AbstractArray{T,N}) where {T,N}
    grad!(parent(objfun), x, g)
    return objfun(x)
end

# Implement (part of) the NLPModels API.
for func in (:get_x0, :get_lvar, :get_uvar, :get_name, :get_nvar, :get_ncon,
             :neval_obj, :neval_grad,
             :has_bounds, :bound_constrained,
             :has_equalities, :equality_constrained,
             :has_inequalities, :inequality_constrained,
             :linearly_constrained, :unconstrained,
             )
    @eval NLPModels.$func(objfun::ObjectiveFunction{<:AbstractNLPModel}) =
        $func(parent(objfun))
end
for func in (:obj, :grad)
    @eval function NLPModels.$func(objfun::ObjectiveFunction{<:AbstractNLPModel},
                                   x::AbstractArray)
        return $func(parent(objfun), x)
    end
end
for func in (:(grad!),)
    @eval function NLPModels.$func(objfun::ObjectiveFunction{<:AbstractNLPModel},
                                   x::AbstractArray{T,N},
                                   g::AbstractArray{T,N}) where {T,N}
        return $func(parent(objfun), x, g)
    end
end

# FIXME: These methods are type-unstable because the type of bound constraints
#        cannot be inferred from the type of the argument.
function Projector(arg::Union{ObjectiveFunction{<:AbstractNLPModel},
                              AbstractNLPModel})
    return Projector(BoundedSet(arg))
end
function BoundedSet(arg::Union{ObjectiveFunction{<:AbstractNLPModel},
                               AbstractNLPModel})
    x0 = get_x0(arg)
    T = eltype(x0)
    N = ndims(x0)
    if has_bounds(arg)
        return BoundedSet{T,N}(get_lvar(arg), get_uvar(arg))
    else
        return BoundedSet{T,N}(-Inf, +Inf)
    end
end

end # module
