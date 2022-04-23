module Lib

# Path to the shared libraries.
@static if false
    import OptimPack_jll
    const libopk    = OptimPack_jll.libopk
    const libbobyqa = OptimPack_jll.libbobyqa
    const libcobyla = OptimPack_jll.libcobyla
    const libnewuoa = OptimPack_jll.libnewuoa
else
    const libopk    = "/home/eric/apps/lib/libopk.so.3"
    const libbobyqa = "/home/eric/apps/lib/libbobyqa.so.2"
    const libcobyla = "/home/eric/apps/lib/libcobyla.so.2"
    const libnewuoa = "/home/eric/apps/lib/libnewuoa.so.2"
end

# typedef double bobyqa_objfun ( const opk_index n , const double * x , void * data )
const bobyqa_objfun = Cvoid

"""
    bobyqa_status

Status for BOBYQA routines.

This type enumerate the possible values returned by [`bobyqa`](@ref)(), bobyqa\\_get\\_status() and bobyqa\\_iterate().

| Enumerator                        | Note                                             |
| :-------------------------------- | :----------------------------------------------- |
| BOBYQA\\_SUCCESS                  | Algorithm converged                              |
| BOBYQA\\_BAD\\_NVARS              | Bad number of variables                          |
| BOBYQA\\_BAD\\_NPT                | NPT is not in the required interval              |
| BOBYQA\\_BAD\\_RHO\\_RANGE        | Bad trust region radius parameters               |
| BOBYQA\\_BAD\\_SCALING            | Bad scaling factor(s)                            |
| BOBYQA\\_TOO\\_CLOSE              | Insufficient space between the bounds            |
| BOBYQA\\_ROUNDING\\_ERRORS        | Too much cancellation in a denominator           |
| BOBYQA\\_TOO\\_MANY\\_EVALUATIONS | Maximum number of function evaluations exceeded  |
| BOBYQA\\_STEP\\_FAILED            | A trust region step has failed to reduce Q       |
"""
@enum bobyqa_status::Int32 begin
    BOBYQA_SUCCESS = 0
    BOBYQA_BAD_NVARS = -1
    BOBYQA_BAD_NPT = -2
    BOBYQA_BAD_RHO_RANGE = -3
    BOBYQA_BAD_SCALING = -4
    BOBYQA_TOO_CLOSE = -5
    BOBYQA_ROUNDING_ERRORS = -6
    BOBYQA_TOO_MANY_EVALUATIONS = -7
    BOBYQA_STEP_FAILED = -8
end

"""
Integer data type used for array indices in OptimPack.
"""
const opk_index = Cptrdiff_t

function bobyqa(n, npt, objfun, data, x, xl, xu, rhobeg, rhoend, iprint, maxfun, w)
    @ccall libbobyqa.bobyqa(n::opk_index, npt::opk_index, objfun::Ptr{Cvoid}, data::Any, x::Ptr{Cdouble}, xl::Ptr{Cdouble}, xu::Ptr{Cdouble}, rhobeg::Cdouble, rhoend::Cdouble, iprint::opk_index, maxfun::opk_index, w::Ptr{Cdouble})::bobyqa_status
end

"""
    opk_bool

Possible boolean (logical) values.
"""
@enum opk_bool::UInt32 begin
    OPK_FALSE = 0
    OPK_TRUE = 1
end

function bobyqa_optimize(n, npt, maximize, objfun, data, x, xl, xu, scl, rhobeg, rhoend, iprint, maxfun, w)
    @ccall libbobyqa.bobyqa_optimize(n::opk_index, npt::opk_index, maximize::opk_bool, objfun::Ptr{Cvoid}, data::Any, x::Ptr{Cdouble}, xl::Ptr{Cdouble}, xu::Ptr{Cdouble}, scl::Ptr{Cdouble}, rhobeg::Cdouble, rhoend::Cdouble, iprint::opk_index, maxfun::opk_index, w::Ptr{Cdouble})::bobyqa_status
end

function bobyqa_reason(status)
    @ccall libbobyqa.bobyqa_reason(status::bobyqa_status)::Cstring
end

function bobyqa_test()
    @ccall libbobyqa.bobyqa_test()::Cvoid
end

# typedef double cobyla_calcfc ( opk_index n , opk_index m , const double x [ ] , double con [ ] , void * data )
"""
Prototype of the function assumed by the COBYLA algorithm.

Prototype of the objective function assumed by the COBYLA routine. The returned value is the function value at `x`, `n` is the number of variables, `m` is the number of constraints, `x` are the current values of the variables and `con` is to store the `m` constraints. `data` is anything needed by the function (unused by COBYLA itself).
"""
const cobyla_calcfc = Cvoid

"""
    cobyla_status

Status for COBYLA routines.

This type enumerate the possible values returned by [`cobyla`](@ref)(), [`cobyla_get_status`](@ref)() and [`cobyla_iterate`](@ref)().

| Enumerator                        | Note                                     |
| :-------------------------------- | :--------------------------------------- |
| COBYLA\\_INITIAL\\_ITERATE        | Only used internally                     |
| COBYLA\\_ITERATE                  | User requested to compute F(X) and C(X)  |
| COBYLA\\_SUCCESS                  | Algorithm converged                      |
| COBYLA\\_BAD\\_NVARS              | Bad number of variables                  |
| COBYLA\\_BAD\\_NCONS              | Bad number of constraints                |
| COBYLA\\_BAD\\_RHO\\_RANGE        | Invalid trust region parameters          |
| COBYLA\\_BAD\\_SCALING            | Bad scaling factor(s)                    |
| COBYLA\\_ROUNDING\\_ERRORS        | Rounding errors prevent progress         |
| COBYLA\\_TOO\\_MANY\\_EVALUATIONS | Too many evaluations                     |
| COBYLA\\_BAD\\_ADDRESS            | Illegal address                          |
| COBYLA\\_CORRUPTED                | Corrupted workspace                      |
"""
@enum cobyla_status::Int32 begin
    COBYLA_INITIAL_ITERATE = 2
    COBYLA_ITERATE = 1
    COBYLA_SUCCESS = 0
    COBYLA_BAD_NVARS = -1
    COBYLA_BAD_NCONS = -2
    COBYLA_BAD_RHO_RANGE = -3
    COBYLA_BAD_SCALING = -4
    COBYLA_ROUNDING_ERRORS = -5
    COBYLA_TOO_MANY_EVALUATIONS = -6
    COBYLA_BAD_ADDRESS = -7
    COBYLA_CORRUPTED = -8
end

"""
    cobyla(n, m, fc, data, x, rhobeg, rhoend, iprint, maxfun, work, iact)

Minimize a function of many variables subject to inequality constraints.

The [`cobyla`](@ref) algorithm minimizes an objective function `f(x)` subject to `m` inequality constraints on `x`, where `x` is a vector of variables that has `n` components. The algorithm employs linear approximations to the objective and constraint functions, the approximations being formed by linear interpolation at `n+1` points in the space of the variables. We regard these interpolation points as vertices of a simplex. The parameter `rho` controls the size of the simplex and it is reduced automatically from `rhobeg` to `rhoend`. For each `rho` the subroutine tries to achieve a good vector of variables for the current size, and then `rho` is reduced until the value `rhoend` is reached. Therefore `rhobeg` and `rhoend` should be set to reasonable initial changes to and the required accuracy in the variables respectively, but this accuracy should be viewed as a subject for experimentation because it is not guaranteed. The subroutine has an advantage over many of its competitors, however, which is that it treats each constraint individually when calculating a change to the variables, instead of lumping the constraints together into a single penalty function. The name of the subroutine is derived from the phrase "Constrained Optimization BY Linear Approximations".

The user must set the values of `n`, `m`, `rhobeg` and `rhoend`, and must provide an initial vector of variables in `x`. Further, the value of `iprint` should be set to 0, 1, 2 or 3, which controls the amount of printing during the calculation. Specifically, there is no output if `iprint=0` and there is output only at the end of the calculation if `iprint=1`. Otherwise each new value of `rho` and `sigma` is printed. Further, the vector of variables and some function information are given either when `rho` is reduced or when each new value of `f(x)` is computed in the cases `iprint=2` or `iprint=3` respectively. Here `sigma` is a penalty parameter, it being assumed that a change to `x` is an improvement if it reduces the merit function:

f(x) + sigma*max(0.0, -C1(x), -C2(x), ..., -CM(x)),

where `C1`, `C2`, ..., `CM` denote the constraint functions that should become nonnegative eventually, at least to the precision of `rhoend`. In the printed output the displayed term that is multiplied by `sigma` is called `maxcv`, which stands for "MAXimum Constraint Violation". The argument `maxfun` is the address of an integer variable that must be set by the user to a limit on the number of calls of `fc`, the purpose of this routine being given below. The value of `maxfun` will be altered to the number of calls of `fc` that are made. The arguments `work` and `iact` provide real and integer arrays that are used as working space. Their lengths must be at least `n*(3*n+2*m+11)+4*m+6` and `m+1` respectively.

In order to define the objective and constraint functions, we require a function `fc` that has the following prototype:

REAL fc(INTEGER n, INTEGER m, const REAL x[], REAL con[], void* data);

The values of `n` and `m` are fixed and have been defined already, while `x` is now the current vector of variables. The function should return the value of the objective function at `x` and store constraint functions at `x` in `con[0]`, `con[1]`, ..., `con[m-1]`. Argument `data` will be set with whatever value has been provided when [`cobyla`](@ref) was called. Note that we are trying to adjust `x` so that `f(x)` is as small as possible subject to the constraint functions being nonnegative.

### Parameters
* `n`: - The number of variables.
* `m`: - The number of constraints.
* `fc`: - The objective function.
* `data`: - Anything needed by the objective function.
* `x`: - On entry, the initial variables; on exit, the final variables.
* `rhobeg`: - The initial trust region radius.
* `rhoend`: - The final trust region radius.
* `maxfun`: - On entry, the maximum number of calls to `fc`; on exit, the actual number of calls to `fc`.
* `iprint`: - The level of verbosity.
* `maxfun`: - The maximum number of calls to `fc`.
* `work`: - Workspace array with at least `n*(3*n+2*m+11)+4*m+6` elements. On successful exit, the value of the objective function and of the worst constraint at the final `x` are stored in `work[0]` and `work[1]` respectively.
* `iact`: - Workspace array with at least `m+1` elements. On successful exit, the actual number of calls to `fc` is stored in `iact[0]`.
### Returns
`COBYLA_SUCCESS` is returned when the algorithm is successful; any other value indicates an error (use [`cobyla_reason`](@ref) to have an explanation).
"""
function cobyla(n, m, fc, data, x, rhobeg, rhoend, iprint, maxfun, work, iact)
    @ccall libcobyla.cobyla(n::opk_index, m::opk_index, fc::Ptr{Cvoid}, data::Any, x::Ptr{Cdouble}, rhobeg::Cdouble, rhoend::Cdouble, iprint::opk_index, maxfun::opk_index, work::Ptr{Cdouble}, iact::Ptr{opk_index})::cobyla_status
end

"""
    cobyla_optimize(n, m, maximize, fc, data, x, scl, rhobeg, rhoend, iprint, maxfun, work, iact)

Minimize or maximize a function of many variables subject to inequality constraints with optional scaling.

This function is a variant of [`cobyla`](@ref) which attempts to minimize or maximize an objective function of many variables subject to inequality constraints. The scaling of the variables is important for the success of the algorithm and the `scale` argument (if not `NULL`) should specify the relative size of each variable. If specified, `scale` is an array of `n` strictly nonnegative values, such that `scale[i]*rho` (with `rho` the trust region radius) is the size of the trust region for the `i`-th variable. Thus `scale[i]*rhobeg` is the typical step size for the `i`-th variable at the beginning of the algorithm and `scale[i]*rhoend` is the typical precision on the `i`-th variable at the end. If `scale` is not specified, a unit scaling for all the variables is assumed.

### Parameters
* `n`: - The number of variables.
* `m`: - The number of constraints.
* `maximize`: - Non-zero to attempt to maximize the objective function; zero to attempt to minimize it.
* `fc`: - The objective function.
* `data`: - Anything needed by the objective function.
* `x`: - On entry, the initial variables; on exit, the final variables.
* `scale`: - An array of `n` scaling factors, all strictly positive. Can be `NULL` to perform no scaling of the variables.
* `rhobeg`: - The initial trust region radius.
* `rhoend`: - The final trust region radius.
* `maxfun`: - The maximum number of calls to `fc`.
* `iprint`: - The level of verbosity.
* `maxfun`: - On entry, the maximum number of calls to `fc`; on exit, the actual number of calls to `fc`.
* `work`: - Workspace array with at least `n*(3*n+2*m+11)+4*m+6` elements. On successful exit, the value of the objective function and of the worst constraint at the final `x` are stored in `work[0]` and `work[1]` respectively.
* `iact`: - Workspace array with at least `m+1` elements. On successful exit, the actual number of calls to `fc` is stored in `iact[0]`.
### Returns
`COBYLA_SUCCESS` is returned when the algorithm is successful; any other value indicates an error (use [`cobyla_reason`](@ref) to have an explanation).
"""
function cobyla_optimize(n, m, maximize, fc, data, x, scl, rhobeg, rhoend, iprint, maxfun, work, iact)
    @ccall libcobyla.cobyla_optimize(n::opk_index, m::opk_index, maximize::opk_bool, fc::Ptr{Cvoid}, data::Any, x::Ptr{Cdouble}, scl::Ptr{Cdouble}, rhobeg::Cdouble, rhoend::Cdouble, iprint::opk_index, maxfun::opk_index, work::Ptr{Cdouble}, iact::Ptr{opk_index})::cobyla_status
end

mutable struct cobyla_context_ end

const cobyla_context = cobyla_context_

function cobyla_create(n, m, rhobeg, rhoend, iprint, maxfun)
    @ccall libcobyla.cobyla_create(n::opk_index, m::opk_index, rhobeg::Cdouble, rhoend::Cdouble, iprint::opk_index, maxfun::opk_index)::Ptr{cobyla_context}
end

function cobyla_delete(ctx)
    @ccall libcobyla.cobyla_delete(ctx::Ptr{cobyla_context})::Cvoid
end

function cobyla_iterate(ctx, f, x, c)
    @ccall libcobyla.cobyla_iterate(ctx::Ptr{cobyla_context}, f::Cdouble, x::Ptr{Cdouble}, c::Ptr{Cdouble})::cobyla_status
end

function cobyla_restart(ctx)
    @ccall libcobyla.cobyla_restart(ctx::Ptr{cobyla_context})::cobyla_status
end

function cobyla_get_status(ctx)
    @ccall libcobyla.cobyla_get_status(ctx::Ptr{cobyla_context})::cobyla_status
end

function cobyla_get_nevals(ctx)
    @ccall libcobyla.cobyla_get_nevals(ctx::Ptr{cobyla_context})::opk_index
end

function cobyla_get_rho(ctx)
    @ccall libcobyla.cobyla_get_rho(ctx::Ptr{cobyla_context})::Cdouble
end

function cobyla_get_last_f(ctx)
    @ccall libcobyla.cobyla_get_last_f(ctx::Ptr{cobyla_context})::Cdouble
end

function cobyla_reason(status)
    @ccall libcobyla.cobyla_reason(status::cobyla_status)::Cstring
end

# typedef double newuoa_objfun ( const opk_index n , const double * x , void * data )
const newuoa_objfun = Cvoid

"""
    newuoa_status

Status for NEWUOA routines.

This type enumerate the possible values returned by [`newuoa`](@ref)(), [`newuoa_get_status`](@ref)() and [`newuoa_iterate`](@ref)().

| Enumerator                        | Note                                                                                        |
| :-------------------------------- | :------------------------------------------------------------------------------------------ |
| NEWUOA\\_INITIAL\\_ITERATE        | Only used internaly                                                                         |
| NEWUOA\\_ITERATE                  | Caller is requested to evaluate the objective function and call [`newuoa_iterate`](@ref)()  |
| NEWUOA\\_SUCCESS                  | Algorithm converged                                                                         |
| NEWUOA\\_BAD\\_NVARS              | Bad number of variables                                                                     |
| NEWUOA\\_BAD\\_NPT                | NPT is not in the required interval                                                         |
| NEWUOA\\_BAD\\_RHO\\_RANGE        | Invalid RHOBEG/RHOEND                                                                       |
| NEWUOA\\_BAD\\_SCALING            | Bad scaling factor(s)                                                                       |
| NEWUOA\\_ROUNDING\\_ERRORS        | Too much cancellation in a denominator                                                      |
| NEWUOA\\_TOO\\_MANY\\_EVALUATIONS | Maximum number of function evaluations exceeded                                             |
| NEWUOA\\_STEP\\_FAILED            | Trust region step has failed to reduce quadratic approximation                              |
| NEWUOA\\_BAD\\_ADDRESS            | Illegal NULL address                                                                        |
| NEWUOA\\_CORRUPTED                | Corrupted or misused workspace                                                              |
"""
@enum newuoa_status::Int32 begin
    NEWUOA_INITIAL_ITERATE = 2
    NEWUOA_ITERATE = 1
    NEWUOA_SUCCESS = 0
    NEWUOA_BAD_NVARS = -1
    NEWUOA_BAD_NPT = -2
    NEWUOA_BAD_RHO_RANGE = -3
    NEWUOA_BAD_SCALING = -4
    NEWUOA_ROUNDING_ERRORS = -5
    NEWUOA_TOO_MANY_EVALUATIONS = -6
    NEWUOA_STEP_FAILED = -7
    NEWUOA_BAD_ADDRESS = -8
    NEWUOA_CORRUPTED = -9
end

function newuoa(n, npt, objfun, data, x, rhobeg, rhoend, iprint, maxfun, work)
    @ccall libnewuoa.newuoa(n::opk_index, npt::opk_index, objfun::Ptr{Cvoid}, data::Any, x::Ptr{Cdouble}, rhobeg::Cdouble, rhoend::Cdouble, iprint::opk_index, maxfun::opk_index, work::Ptr{Cdouble})::newuoa_status
end

"""
    newuoa_optimize(n, npt, maximize, objfun, data, x, scl, rhobeg, rhoend, iprint, maxfun, work)

Optimize a function of many variables without derivatives.

This function seeks the least (or the most) value of a function `f(x)` of many variables `x[0]`, `x[1]`, ..., `x[n-1], by a trust region method that forms quadratic models by interpolation. There can be some freedom in the interpolation conditions, which is taken up by minimizing the Frobenius norm of the change to the second derivative of the quadratic model, beginning with a zero matrix.

Arguments `rhobeg` and `rhoend` must be set to the initial and final values of a trust region radius, so both must be positive with `0 < rhoend <= rhobeg`. Typically `rhobeg` should be about one tenth of the greatest expected change to a variable, and `rhoend` should indicate the accuracy that is required in the final values of the variables. The proper scaling of the variables is important for the success of the algorithm and the optional `scale` argument should be specified if the typical precision is not the same for all variables. If specified, `scale` is an array of same dimensions as `x` with strictly nonnegative values, such that `scale[i]*rho` (with `rho` the trust region radius) is the size of the trust region for the i-th variable. If `scale` is not specified, a unit scaling for all the variables is assumed.

### Parameters
* `n`: - The number of variables which must be at least 2.
* `npt`: - The number of interpolation conditions. Its value must be in the interval `[n + 2, (n + 1)*(n + 2)/2]`. The recommended value is `2*n + 1`.
* `maximize`: - If true, maximize the function; otherwise, minimize it.
* `objfun`: - The objective function. Called as `objfun(n,x,data)`, it returns the value of the objective function for the variables `x[0]`, `x[1]`, ..., `x[n-1]`. Argument `data` is anything else needed by the objective function.
* `data`: - Anything needed by the objective function. This address is passed to the objective fucntion each time it is called.
* `x`: - On entry, the initial variables; on return, the solution.
* `scale`: - Scaling factors for the variables. May be `NULL` to use the same unit scaling factors for all variables. Otherwise, must all be strictly positive.
* `rhobeg`: - The initial radius of the trust region.
* `rhoend`: - The final radius of the trust region.
* `iprint`: - The amount of printing, its value should be set to 0, 1, 2 or 3. Specifically, there is no output if `iprint = 0` and there is output only at the return if `iprint = 1`. Otherwise, each new value of `rho` is printed, with the best vector of variables so far and the corresponding value of the objective function. Further, each new value of `f(x)` with its variables are output if `iprint = 3`.
* `maxfun`: - The maximum number of calls to the objective function.
* `work`: - A workspace array. If `scl` is `NULL`, its length must be at least `(npt + 13)*(npt + n) + 3*n*(n + 3)/2` and at least `(npt + 13)*(npt + n) + 3*n*(n + 3)/2 + n` otherwise. On exit, `work[0]` is set with `f(x)` the value of the objective function at the solution.
### Returns
On success, the returned value is `NEWUOA_SUCCESS`; a different value is returned on error (see [`newuoa_reason`](@ref) for an explanatory message).
"""
function newuoa_optimize(n, npt, maximize, objfun, data, x, scl, rhobeg, rhoend, iprint, maxfun, work)
    @ccall libnewuoa.newuoa_optimize(n::opk_index, npt::opk_index, maximize::opk_bool, objfun::Ptr{Cvoid}, data::Any, x::Ptr{Cdouble}, scl::Ptr{Cdouble}, rhobeg::Cdouble, rhoend::Cdouble, iprint::opk_index, maxfun::opk_index, work::Ptr{Cdouble})::newuoa_status
end

function newuoa_reason(status)
    @ccall libnewuoa.newuoa_reason(status::newuoa_status)::Cstring
end

mutable struct newuoa_context_ end

const newuoa_context = newuoa_context_

function newuoa_create(n, npt, rhobeg, rhoend, iprint, maxfun)
    @ccall libnewuoa.newuoa_create(n::opk_index, npt::opk_index, rhobeg::Cdouble, rhoend::Cdouble, iprint::opk_index, maxfun::opk_index)::Ptr{newuoa_context}
end

function newuoa_delete(ctx)
    @ccall libnewuoa.newuoa_delete(ctx::Ptr{newuoa_context})::Cvoid
end

function newuoa_iterate(ctx, f, x)
    @ccall libnewuoa.newuoa_iterate(ctx::Ptr{newuoa_context}, f::Cdouble, x::Ptr{Cdouble})::newuoa_status
end

function newuoa_restart(ctx)
    @ccall libnewuoa.newuoa_restart(ctx::Ptr{newuoa_context})::newuoa_status
end

function newuoa_get_status(ctx)
    @ccall libnewuoa.newuoa_get_status(ctx::Ptr{newuoa_context})::newuoa_status
end

function newuoa_get_nevals(ctx)
    @ccall libnewuoa.newuoa_get_nevals(ctx::Ptr{newuoa_context})::opk_index
end

function newuoa_get_rho(ctx)
    @ccall libnewuoa.newuoa_get_rho(ctx::Ptr{newuoa_context})::Cdouble
end

function newuoa_test()
    @ccall libnewuoa.newuoa_test()::Cvoid
end

"""
    opk_object_

Basic object structure.

The definition of this structure must be exposed so that others types can "inherit" from it. However, the contents must be left unchanged otherwise unexpected results may occur.
"""
struct opk_object_
    finalize::Ptr{Cvoid}
    references::opk_index
end

"""
Opaque basic object type.
"""
const opk_object = opk_object_

"""
    opk_hold_object(obj)

Hold a reference on an object.

This function increments the reference count of its argument and returns it. If the argument is `NULL`, nothing is done. Every call to this function must be balanced by a call to `[`opk_drop_object`](@ref)()`.

It is a good practice to hold a reference whenever a persistent structure (e.g., another object) remembers the object. To limit the number of casts, the macro `[`OPK_HOLD`](@ref)()` can be used instead.

### Parameters
* `obj`: - The object to lock (can be NULL).
### Returns
Its argument (with one more reference if not NULL).
"""
function opk_hold_object(obj)
    @ccall libopk.opk_hold_object(obj::Ptr{opk_object})::Ptr{opk_object}
end

"""
    opk_drop_object(obj)

Drop a reference on an object.

This function decrements the reference count of its argument and delete it if there are no other references. If the argument is `NULL`, nothing is done. To effectively delete an object, there must be a call to this function for every call to `[`opk_hold_object`](@ref)()` on this object plus one (to release the reference by the creator of the object).

### Parameters
* `obj`: - The object to release (can be `NULL`).
"""
function opk_drop_object(obj)
    @ccall libopk.opk_drop_object(obj::Ptr{opk_object})::Cvoid
end

"""
    opk_get_object_references(obj)

Get the number of references on an object.

### Parameters
* `obj`: - The object (can be `NULL`).
### Returns
The number of references set on the object. If the object address is `NULL`, the result is zero; otherwise, the result is greater of equal one.
"""
function opk_get_object_references(obj)
    @ccall libopk.opk_get_object_references(obj::Ptr{opk_object})::opk_index
end

struct opk_vspace_operations_
    description::Cstring
    finalize_space::Ptr{Cvoid}
    create::Ptr{Cvoid}
    finalize_vector::Ptr{Cvoid}
    peek::Ptr{Cvoid}
    poke::Ptr{Cvoid}
    _import::Ptr{Cvoid}
    _export::Ptr{Cvoid}
    fill::Ptr{Cvoid}
    norm1::Ptr{Cvoid}
    norm2::Ptr{Cvoid}
    norminf::Ptr{Cvoid}
    dot::Ptr{Cvoid}
    dot3::Ptr{Cvoid}
    copy::Ptr{Cvoid}
    swap::Ptr{Cvoid}
    scale::Ptr{Cvoid}
    product::Ptr{Cvoid}
    axpby::Ptr{Cvoid}
    axpbypcz::Ptr{Cvoid}
    boxprojvar::Ptr{Cvoid}
    boxprojdir::Ptr{Cvoid}
    boxfreevar::Ptr{Cvoid}
    boxsteplim::Ptr{Cvoid}
end

"""
Table of methods for vector spaces.
"""
const opk_vspace_operations = opk_vspace_operations_

struct opk_vspace_
    base::opk_object
    ops::Ptr{opk_vspace_operations}
    size::opk_index
end

"""
Opaque vector space type. This sub-type inherits from [`opk_object`](@ref).
"""
const opk_vspace = opk_vspace_

"""
    opk_vector_

Structure implementing a basic vector type.

The [`opk_vector`](@ref) structure is intentionally exposed to let different implementations coexist (although in separate codes). If one want to implement a sub-type of the vector type, it is sufficient to define a new structure whose first member is an [`opk_vector`](@ref). The function `[`opk_allocate_vector`](@ref)()` **must** be used to allocate the whole structure. For instance: ~~~~~~~~~~{.c} typedef struct { [`opk_vector`](@ref) base; double* data; } my\\_vector;

void finalize\\_vector(my\\_vector* v) { if (v->data != NULL) { free(v->data); } }

my\\_vector* new\\_vector(int n) { my\\_vector* v = (my\\_vector*)[`opk_allocate_vector`](@ref)(space, sizeof(my\\_vector)); if (v != NULL) { v->data = (double*)malloc(n*sizeof(double)); if (v->data == NULL) { [`OPK_DROP`](@ref)(v); return NULL; } } return v; } ~~~~~~~~~~

OptimPack routines only require the address of such vectors and treat them as opaque structures.

| Field | Note                                       |
| :---- | :----------------------------------------- |
| base  | Base type (must be the first member).      |
| owner | Vector space to which the vector belongs.  |
"""
struct opk_vector_
    base::opk_object
    owner::Ptr{opk_vspace}
end

"""
Opaque vector type. This sub-type inherits from [`opk_object`](@ref).
"""
const opk_vector = opk_vector_

struct opk_lnsrch_operations_
    finalize::Ptr{Cvoid}
    start::Ptr{Cvoid}
    iterate::Ptr{Cvoid}
    use_deriv::opk_bool
end

"""
` LowLineSearch  Implementing line search methods.`

` LowLevel`

@{
"""
const opk_lnsrch_operations = opk_lnsrch_operations_

"""
    opk_status

Values returned by OptimPack routines.

`OPK_SUCCESS` indicates that the routine was successfull, any other value indicate a failure or a warning.
"""
@enum opk_status::UInt32 begin
    OPK_SUCCESS = 0
    OPK_INVALID_ARGUMENT = 1
    OPK_INSUFFICIENT_MEMORY = 2
    OPK_ILLEGAL_ADDRESS = 3
    OPK_NOT_IMPLEMENTED = 4
    OPK_CORRUPTED_WORKSPACE = 5
    OPK_BAD_SPACE = 6
    OPK_OUT_OF_BOUNDS_INDEX = 7
    OPK_NOT_STARTED = 8
    OPK_NOT_A_DESCENT = 9
    OPK_STEP_CHANGED = 10
    OPK_STEP_OUTSIDE_BRACKET = 11
    OPK_STPMIN_GT_STPMAX = 12
    OPK_STPMIN_LT_ZERO = 13
    OPK_STEP_LT_STPMIN = 14
    OPK_STEP_GT_STPMAX = 15
    OPK_FTOL_TEST_SATISFIED = 16
    OPK_GTOL_TEST_SATISFIED = 17
    OPK_XTOL_TEST_SATISFIED = 18
    OPK_STEP_EQ_STPMAX = 19
    OPK_STEP_EQ_STPMIN = 20
    OPK_ROUNDING_ERRORS_PREVENT_PROGRESS = 21
    OPK_NOT_POSITIVE_DEFINITE = 22
    OPK_BAD_PRECONDITIONER = 23
    OPK_INFEASIBLE_BOUNDS = 24
    OPK_WOULD_BLOCK = 25
    OPK_UNDEFINED_VALUE = 26
    OPK_TOO_MANY_EVALUATIONS = 27
    OPK_TOO_MANY_ITERATIONS = 28
    OPK_MAX_STATUS = 29
end

"""
    opk_lnsrch_task

Possible values returned by {

```c++
 opk_lnsrch_start} and {@link
 opk_lnsrch_iterate}.
 These values are for the caller to decide what to do next.  In case of error
 or warning, {@link opk_lnsrch_get_status} can be used to query more
 information.


```

| Enumerator                 | Note                                   |
| :------------------------- | :------------------------------------- |
| OPK\\_LNSRCH\\_ERROR       | An error occurred.                     |
| OPK\\_LNSRCH\\_SEARCH      | Line search in progress.               |
| OPK\\_LNSRCH\\_CONVERGENCE | Line search has converged.             |
| OPK\\_LNSRCH\\_WARNING     | Line search terminated with warnings.  |
"""
@enum opk_lnsrch_task::Int32 begin
    OPK_LNSRCH_ERROR = -1
    OPK_LNSRCH_SEARCH = 0
    OPK_LNSRCH_CONVERGENCE = 1
    OPK_LNSRCH_WARNING = 2
end

"""
    opk_lnsrch_

| Field     | Note                                                      |
| :-------- | :-------------------------------------------------------- |
| base      | Base type (must be the first member).                     |
| ops       | Table of line search methods.                             |
| stp       | Current step length.                                      |
| stpmin    | Lower bound for the step.                                 |
| stpmax    | Upper bound for the step.                                 |
| finit     | Function value at the start of the search.                |
| ginit     | Directional derivative value at the start of the search.  |
| status    | Last status.                                              |
| task      | Current pending task.                                     |
| searching | True if search is in progress.                            |
"""
struct opk_lnsrch_
    base::opk_object
    ops::Ptr{opk_lnsrch_operations}
    stp::Cdouble
    stpmin::Cdouble
    stpmax::Cdouble
    finit::Cdouble
    ginit::Cdouble
    status::opk_status
    task::opk_lnsrch_task
    searching::Cint
end

"""
Opaque line search type. This sub-type inherits from [`opk_object`](@ref).
"""
const opk_lnsrch = opk_lnsrch_

struct opk_operator_operations_
    finalize::Ptr{Cvoid}
    apply_direct::Ptr{Cvoid}
    apply_adjoint::Ptr{Cvoid}
    apply_inverse::Ptr{Cvoid}
end

"""
` LowOperators  Implementing operators acting on vectors.`

` LowLevel`

@{
"""
const opk_operator_operations = opk_operator_operations_

struct opk_operator_
    base::opk_object
    ops::Ptr{opk_operator_operations}
    inpspace::Ptr{opk_vspace}
    outspace::Ptr{opk_vspace}
end

"""
Opaque operator type. This sub-type inherits from [`opk_object`](@ref).
"""
const opk_operator = opk_operator_

"""
    opk_convexset_

Private structure to store the base of an instance derived from a convex set.

| Field | Note                                   |
| :---- | :------------------------------------- |
| base  | Base type (must be the first member).  |
| space | Variable space.                        |
"""
struct opk_convexset_
    base::opk_object
    space::Ptr{opk_vspace}
    finalize::Ptr{Cvoid}
    projvar::Ptr{Cvoid}
    projdir::Ptr{Cvoid}
    freevar::Ptr{Cvoid}
    steplim::Ptr{Cvoid}
end

"""
Opaque structure to represent an instance of a convex set.
"""
const opk_convexset = opk_convexset_

"""
    opk_get_reason(status)

Retrieve a textual description for a given status.

### Parameters
* `status`: - The status code.
### Returns
A pointer to a string describing the status or an empty string, "", if the status does not correspond to any known status.
"""
function opk_get_reason(status)
    @ccall libopk.opk_get_reason(status::opk_status)::Cstring
end

"""
    opk_guess_status(code)

Retrieve OptimPack status from C library error code.

This function is needed to figure out the kind of errors for the few routines which do not return a status (mostly the ones which create objects).

### Parameters
* `code`: The error code, usually `errno`.
### Returns
The OptimPack status corresponding to the error.
"""
function opk_guess_status(code)
    @ccall libopk.opk_guess_status(code::Cint)::opk_status
end

"""
    opk_copy_string(dst, size, src)

Copy a string.

### Parameters
* `dst`: - The destination buffer to copy the soruce (can be `NULL`).
* `size`: - The number of available bytes in `buf`.
* `src`: - The source string; `NULL` is considered as being the same as an empty string "".
### Returns
The minimum number of bytes required to store the source string (including the terminating '\\0' character).
"""
function opk_copy_string(dst, size, src)
    @ccall libopk.opk_copy_string(dst::Cstring, size::Csize_t, src::Cstring)::Csize_t
end

"""
    opk_get_integer_constant(name, ptr)

Get a constant given its name.

This function is mostly need for OptimPack wrappers in other languages than C to avoid hardcoding the constants of the library. For now this function is not particularly efficient so it should be sparsely used.

### Parameters
* `name`: - The name of the constant; for instance: `"OPK\\_SUCCESS"`.
* `ptr`: - The address where to store the value of the constant.
### Returns
`OPK_SUCCESS` or `OPK_INVALID_ARGUMENT` if the name is unknown.
"""
function opk_get_integer_constant(name, ptr)
    @ccall libopk.opk_get_integer_constant(name::Cstring, ptr::Ptr{Clong})::opk_status
end

"""
    opk_eltype

Type of the variables in a conventional array.

| Enumerator   | Note                                                    |
| :----------- | :------------------------------------------------------ |
| OPK\\_FLOAT  | Variables are single precision floating point numbers.  |
| OPK\\_DOUBLE | Variables are double precision floating point numbers.  |
"""
@enum opk_eltype::UInt32 begin
    OPK_FLOAT = 0
    OPK_DOUBLE = 1
end

# typedef void opk_free_proc ( void * )
"""
Prototype of function to release client data.
"""
const opk_free_proc = Cvoid

"""
    opk_new_simple_double_vector_space(size)

Create a vector space for array of double's in conventional memory.

This particular type of vector space deals with arrays of values stored contiguously and accessible as conventional arrays. This include arrays allocated from the heap, dynamically allocated with `malloc()`, arrays in shared memory, memory mapped files, etc.

To create vectors belonging to this kind of vector space, as for any type of vector spaces, it is possible to call `[`opk_vcreate`](@ref)()` but it is also possible to call `[`opk_wrap_simple_double_vector`](@ref)()` to wrap an existing array (of the correct size and type of course) into a vector.

### Parameters
* `size`: - The number of elements of the vectors of the space.
### Returns
A new vector space or `NULL` in case of errors.
"""
function opk_new_simple_double_vector_space(size)
    @ccall libopk.opk_new_simple_double_vector_space(size::opk_index)::Ptr{opk_vspace}
end

"""
    opk_wrap_simple_double_vector(vspace, data, free_client_data, client_data)

Wrap an existing array into a simple vector.

This function creates a new vector whose elements are stored into an array provided by the caller. The caller is responsible of ensuring that the memory is sufficiently large (the array has at least `vspace->size` elements) and correctly aligned.

When the vector is destroyed, the function `free\\_client\\_data()`, if not `NULL`, is called with argument `client_data` to release ressources. Then the container is freed. If function `free\\_client\\_data()` is `NULL`, it is assumed that the caller is responsible of releasing the data when no longer needed.

A typical usage is: ~~~~~~~~~~{.c} #define N 1000 [`opk_vspace`](@ref)* vspace = [`opk_new_simple_double_vector_space`](@ref)(N);

double heap\\_array[N]; [`opk_vector`](@ref)* v1 = [`opk_wrap_simple_double_vector`](@ref)(vspace, heap\\_array, NULL, NULL);

double* dynamic\\_array = (double*)malloc(N*sizeof(double)); [`opk_vector`](@ref)* v2 = [`opk_wrap_simple_double_vector`](@ref)(vspace, dynamic\\_array, free, dynamic\\_array); ~~~~~~~~~~

which creates two vectors, `v1` and `v2`, which are respectively wrapped around an array allocated on the heap and around a dynamically allocated array.

In the above example, the `client_data` and the `data` are the same but the possible distinction is needed to allow for using of various kind of objects which contain an array of values that can be wrapped into a vector. For objects of type `object_type`, we can do something like: ~~~~~~~~~~{.c} object\\_type* obj = ...; [`opk_vspace`](@ref)* vspace = [`opk_new_simple_double_vector_space`](@ref)(get\\_number(obj)); [`opk_vector`](@ref)* v = [`opk_wrap_simple_double_vector`](@ref)(vspace, get\\_data(obj), delete\\_object, (void*)obj); ~~~~~~~~~~ where `get\\_number()` returns the number of elements stored in the data part of the object, `get\\_data()` returns the address of these elements, and `delete\\_object()` delete the object. Of course, if one prefers to keep the control on the object management, passing `NULL` for the `free\\_client\\_data()` function is always possible.

### Parameters
* `vspace`: - The vector space which will own the vector.
* `data`: - The array of values, must have at least `vspace->size` elements.
* `free_client_data`: - Function called to release ressources. If not `NULL`, it is called with argument `client_data` when the vector is destroyed.
* `client_data`: - Anything required by the `free\\_client\\_data()` method.
### Returns
A new vector of `vspace`, `NULL` in case of error.
"""
function opk_wrap_simple_double_vector(vspace, data, free_client_data, client_data)
    @ccall libopk.opk_wrap_simple_double_vector(vspace::Ptr{opk_vspace}, data::Ptr{Cdouble}, free_client_data::Ptr{Cvoid}, client_data::Ptr{Cvoid})::Ptr{opk_vector}
end

function opk_get_simple_double_vector_data(v)
    @ccall libopk.opk_get_simple_double_vector_data(v::Ptr{opk_vector})::Ptr{Cdouble}
end

function opk_get_simple_double_vector_client_data(v)
    @ccall libopk.opk_get_simple_double_vector_client_data(v::Ptr{opk_vector})::Ptr{Cvoid}
end

function opk_get_simple_double_vector_free_client_data(v)
    @ccall libopk.opk_get_simple_double_vector_free_client_data(v::Ptr{opk_vector})::Ptr{Cvoid}
end

"""
    opk_rewrap_simple_double_vector(vect, new_data, new_free_client_data, new_client_data)

Re-wrap an array into an existing simple vector.

This functions replaces the contents of a simple wrapped vector. It is assumed that the vector `vect` is a wrapped vector, that the new data `new_data` is correctly aligned and large enough. If the former `free\\_client\\_data()` method of the wrapped vector `vect` is not `NULL` and if either the new `free\\_client\\_data()` method or the new `client_data` differ from the former ones, then the former `free\\_client\\_data()` method is applied to the former `client_data`.

Re-wrapping is considered as a hack which merely saves the time needed to allocate a container for a wrapped vector. It is the caller responsibility to ensure that all the assumptions hold. In many cases dropping the old vector and wrapping the arguments into a new vector is safer.

### Parameters
* `vect`: - The vector to re-wrap.
* `new_data`: - The new array of values.
* `new_free_client_data`: - The new method to free client data.
* `new_client_data`: - The new client data.
### Returns
`OPK_SUCCESS` or `OPK_FAILURE`. In case of failure, global variable `errno` is set to: `EFAULT` if `vect` or `new_data` are `NULL`, `EINVAL` if `vect` is not a vector of the correct kind.
"""
function opk_rewrap_simple_double_vector(vect, new_data, new_free_client_data, new_client_data)
    @ccall libopk.opk_rewrap_simple_double_vector(vect::Ptr{opk_vector}, new_data::Ptr{Cdouble}, new_free_client_data::Ptr{Cvoid}, new_client_data::Ptr{Cvoid})::Cint
end

"""
    opk_new_simple_float_vector_space(size)

Create a vector space for array of float's in conventional memory.

See `[`opk_new_simple_double_vector_space`](@ref)()` for a description.
"""
function opk_new_simple_float_vector_space(size)
    @ccall libopk.opk_new_simple_float_vector_space(size::opk_index)::Ptr{opk_vspace}
end

"""
    opk_wrap_simple_float_vector(vspace, data, free_client_data, client_data)

Wrap an existing array into a simple vector.

See `[`opk_wrap_simple_double_vector`](@ref)()` for a description.
"""
function opk_wrap_simple_float_vector(vspace, data, free_client_data, client_data)
    @ccall libopk.opk_wrap_simple_float_vector(vspace::Ptr{opk_vspace}, data::Ptr{Cfloat}, free_client_data::Ptr{Cvoid}, client_data::Ptr{Cvoid})::Ptr{opk_vector}
end

function opk_get_simple_float_vector_data(v)
    @ccall libopk.opk_get_simple_float_vector_data(v::Ptr{opk_vector})::Ptr{Cfloat}
end

function opk_get_simple_float_vector_client_data(v)
    @ccall libopk.opk_get_simple_float_vector_client_data(v::Ptr{opk_vector})::Ptr{Cvoid}
end

function opk_get_simple_float_vector_free_client_data(v)
    @ccall libopk.opk_get_simple_float_vector_free_client_data(v::Ptr{opk_vector})::Ptr{Cvoid}
end

function opk_rewrap_simple_float_vector(v, new_data, new_free_client_data, new_client_data)
    @ccall libopk.opk_rewrap_simple_float_vector(v::Ptr{opk_vector}, new_data::Ptr{Cfloat}, new_free_client_data::Ptr{Cvoid}, new_client_data::Ptr{Cvoid})::Cint
end

"""
    opk_vcreate(vspace)

Create a vector instance.

This functions creates a new vector of a given vector space. The contents of the vector is undefined. The caller holds a reference on the returned vector which has to be released with the function `[`opk_drop_object`](@ref)()` or with the macro `[`OPK_DROP`](@ref)()`.

### Parameters
* `vspace`: - The vector space which owns the vector to create.
### Returns
A new vector of the vector space; `NULL` in case of error.
"""
function opk_vcreate(vspace)
    @ccall libopk.opk_vcreate(vspace::Ptr{opk_vspace})::Ptr{opk_vector}
end

"""
    opk_vprint(file, name, vect, nmax)

Print vector contents.

### Parameters
* `file`: - The output file stream, `stdout` is used if `NULL`.
* `name`: - The name of the vector, can be `NULL`.
* `nmax`: - The maximum number of elements to print. The vector size is used if this parameter is not strictly positive.
"""
function opk_vprint(file, name, vect, nmax)
    @ccall libopk.opk_vprint(file::Ptr{Libc.FILE}, name::Cstring, vect::Ptr{opk_vector}, nmax::opk_index)::Cvoid
end

"""
    opk_vpeek(vect, k, ptr)

Fetch a specific vector component.

This function is by no means intended to be efficient and should be avoided except for debugging purposes.

### Parameters
* `vect`: - A vector.
* `k`: - The index of the compoent to peek.
* `ptr`: - The address to store the component value (as a double precision floating point).
### Returns
A standard status.
"""
function opk_vpeek(vect, k, ptr)
    @ccall libopk.opk_vpeek(vect::Ptr{opk_vector}, k::opk_index, ptr::Ptr{Cdouble})::opk_status
end

"""
    opk_vpoke(vect, k, value)

Set the value of a specific vector component.

This function is by no means intended to be efficient and should be avoided except for debugging purposes.

### Parameters
* `vect`: - A vector.
* `k`: - The index of the component to set.
* `value`: - The value to store in the component (as a double precision floating point).
### Returns
A standard status.
"""
function opk_vpoke(vect, k, value)
    @ccall libopk.opk_vpoke(vect::Ptr{opk_vector}, k::opk_index, value::Cdouble)::opk_status
end

"""
    opk_vimport(dst, src, type, n)

Copy the values of a conventional array into a vector.

### Parameters
* `dst`: - The destination vector.
* `src`: - The source array.
* `type`: - The type of the elements of the source array.
* `n`: - The number of elements in the source array.
### Returns
A standard status. The number of elements of the source must match those of the destination.
"""
function opk_vimport(dst, src, type, n)
    @ccall libopk.opk_vimport(dst::Ptr{opk_vector}, src::Ptr{Cvoid}, type::opk_eltype, n::opk_index)::opk_status
end

"""
    opk_vexport(dst, type, n, src)

Copy the values of a vector into a conventional array.

### Parameters
* `dst`: - The destination array.
* `type`: - The type of the elements of the destination array.
* `n`: - The number of elements in the destination array.
* `src`: - The source vector.
### Returns
A standard status. The number of elements of the source must match those of the destination.
"""
function opk_vexport(dst, type, n, src)
    @ccall libopk.opk_vexport(dst::Ptr{Cvoid}, type::opk_eltype, n::opk_index, src::Ptr{opk_vector})::opk_status
end

"""
    opk_vzero(vect)

Fill a vector with zeros.

This functions set all elements of a vector to zero.

### Parameters
* `vect`: - The vector to fill.
"""
function opk_vzero(vect)
    @ccall libopk.opk_vzero(vect::Ptr{opk_vector})::Cvoid
end

"""
    opk_vfill(vect, alpha)

Fill a vector with a given value.

This functions set all elements of a vector to the given value.

### Parameters
* `vect`: - The vector to fill.
* `alpha`: - The value.
"""
function opk_vfill(vect, alpha)
    @ccall libopk.opk_vfill(vect::Ptr{opk_vector}, alpha::Cdouble)::Cvoid
end

"""
    opk_vcopy(dst, src)

Copy vector contents.

This functions copies the contents of the source vector into the destination vector. Both vectors must belong to the same vector space.

### Parameters
* `dst`: - The destination vector.
* `src`: - The source vector.
"""
function opk_vcopy(dst, src)
    @ccall libopk.opk_vcopy(dst::Ptr{opk_vector}, src::Ptr{opk_vector})::Cvoid
end

"""
    opk_vscale(dst, alpha, src)

Scale a vector by a scalar.

This functions multiplies the elements of the source vector by the given value and stores the result into the destination vector. Both vectors must belong to the same vector space. The operation is optimized for specfific values of `alpha`: with `alpha = 1`, the operation is the same as `[`opk_vcopy`](@ref)()`; with `alpha = 0`, the operation is the same as `[`opk_vzero`](@ref)()`.

### Parameters
* `dst`: - The destination vector.
* `alpha`: - The scale factor.
* `src`: - The source vector.
"""
function opk_vscale(dst, alpha, src)
    @ccall libopk.opk_vscale(dst::Ptr{opk_vector}, alpha::Cdouble, src::Ptr{opk_vector})::Cvoid
end

"""
    opk_vswap(x, y)

Exchange vector contents.

This functions exchanges the contents of two vectors. Both vectors must belong to the same vector space.

### Parameters
* `x`: - A vector.
* `y`: - Another vector.
"""
function opk_vswap(x, y)
    @ccall libopk.opk_vswap(x::Ptr{opk_vector}, y::Ptr{opk_vector})::Cvoid
end

"""
    opk_vdot(x, y)

Compute the inner product of two vectors.

This functions computes the inner product, also known as scalar product, of two vectors. Both vectors must belong to the same vector space.

### Parameters
* `x`: - A vector.
* `y`: - Another vector.
### Returns
The inner product of the two vectors, that is the sum of the product of their elements.
"""
function opk_vdot(x, y)
    @ccall libopk.opk_vdot(x::Ptr{opk_vector}, y::Ptr{opk_vector})::Cdouble
end

"""
    opk_vdot3(w, x, y)

Compute the inner product of three vectors.

This functions computes the sum of the componentwise product of the elements of three vectors. All three vectors must belong to the same vector space.

### Parameters
* `w`: - A vector.
* `x`: - Another vector.
* `y`: - Yet another vector.
### Returns
The sum of the componentwise product of the elements of three vectors.
"""
function opk_vdot3(w, x, y)
    @ccall libopk.opk_vdot3(w::Ptr{opk_vector}, x::Ptr{opk_vector}, y::Ptr{opk_vector})::Cdouble
end

"""
    opk_vnorm2(v)

Compute the L2 norm of a vector.

This functions computes the L2 norm, also known as the Euclidean norm, of a vector.

### Parameters
* `v`: - A vector.
### Returns
The Euclidean norm of the vector, that is the square root of the sum of its squared elements.
"""
function opk_vnorm2(v)
    @ccall libopk.opk_vnorm2(v::Ptr{opk_vector})::Cdouble
end

"""
    opk_vnorm1(v)

Compute the L1 norm of a vector.

This functions computes the L1 norm of a vector.

### Parameters
* `v`: - A vector.
### Returns
The L1 norm of the vector, that is the sum of the absolute values of its elements.
"""
function opk_vnorm1(v)
    @ccall libopk.opk_vnorm1(v::Ptr{opk_vector})::Cdouble
end

"""
    opk_vnorminf(v)

Compute the infinite norm of a vector.

This functions computes the infinite norm of a vector.

### Parameters
* `v`: - A vector.
### Returns
The infinite norm of the vector, that is the maximum absolute value of its elements.
"""
function opk_vnorminf(v)
    @ccall libopk.opk_vnorminf(v::Ptr{opk_vector})::Cdouble
end

"""
    opk_vproduct(dst, x, y)

Compute the elementwise product of two vectors.

All vectors must belong to the same vector space.

### Parameters
* `dst`: - The destination vector.
* `x`: - A vector.
* `y`: - Another vector.
"""
function opk_vproduct(dst, x, y)
    @ccall libopk.opk_vproduct(dst::Ptr{opk_vector}, x::Ptr{opk_vector}, y::Ptr{opk_vector})::Cvoid
end

"""
    opk_vaxpby(dst, alpha, x, beta, y)

Compute the linear combination of two vectors.

This functions stores in the destination vector `dst` the linear combination `alpha*x + beta*y` where `alpha` and `beta` are two scalars while `x` and `y` are two vectors. All vectors must belong to the same vector space.

### Parameters
* `dst`: - The destination vector.
* `alpha`: - The factor for the vector `x`.
* `x`: - A vector.
* `beta`: - The factor for the vector `y`.
* `y`: - Another vector.
"""
function opk_vaxpby(dst, alpha, x, beta, y)
    @ccall libopk.opk_vaxpby(dst::Ptr{opk_vector}, alpha::Cdouble, x::Ptr{opk_vector}, beta::Cdouble, y::Ptr{opk_vector})::Cvoid
end

"""
    opk_vaxpbypcz(dst, alpha, x, beta, y, gamma, z)

Compute the linear combination of three vectors.

This functions stores in the destination vector `dst` the linear combination `alpha*x + beta*y + gamma*z` where `alpha`, `beta` and `gamma` are three scalars while `x`, `y` and `z` are three vectors. All vectors must belong to the same vector space.

### Parameters
* `dst`: - The destination vector.
* `alpha`: - The factor for the vector `x`.
* `x`: - A vector.
* `beta`: - The factor for the vector `y`.
* `y`: - Another vector.
* `gamma`: - The factor for the vector `z`.
* `z`: - Yet another vector.
"""
function opk_vaxpbypcz(dst, alpha, x, beta, y, gamma, z)
    @ccall libopk.opk_vaxpbypcz(dst::Ptr{opk_vector}, alpha::Cdouble, x::Ptr{opk_vector}, beta::Cdouble, y::Ptr{opk_vector}, gamma::Cdouble, z::Ptr{opk_vector})::Cvoid
end

function opk_apply_direct(op, dst, src)
    @ccall libopk.opk_apply_direct(op::Ptr{opk_operator}, dst::Ptr{opk_vector}, src::Ptr{opk_vector})::opk_status
end

function opk_apply_adjoint(op, dst, src)
    @ccall libopk.opk_apply_adjoint(op::Ptr{opk_operator}, dst::Ptr{opk_vector}, src::Ptr{opk_vector})::opk_status
end

function opk_apply_inverse(op, dst, src)
    @ccall libopk.opk_apply_inverse(op::Ptr{opk_operator}, dst::Ptr{opk_vector}, src::Ptr{opk_vector})::opk_status
end

# typedef void opk_error_handler ( const char * message )
"""
` Error`

@{
"""
const opk_error_handler = Cvoid

function opk_get_error_handler()
    @ccall libopk.opk_get_error_handler()::Ptr{Cvoid}
end

"""
    opk_set_error_handler(handler)

Set the error handler.

### Parameters
* `handler`: - The new error handler or NULL to restore the default handler.
### Returns
The former error handler.
"""
function opk_set_error_handler(handler)
    @ccall libopk.opk_set_error_handler(handler::Ptr{Cvoid})::Ptr{Cvoid}
end

"""
    opk_error(reason)

Throw an error.

This function calls the current error handler. It is used in OptimPack library to throw errors corresponding to a misuse of the library. For instance, when one attempts to compute the dot product of two vectors which do not belong to the same vector space.

### Parameters
* `reason`: - The error message indicating the reason of the failure.
"""
function opk_error(reason)
    @ccall libopk.opk_error(reason::Cstring)::Cvoid
end

"""
    opk_task

Code returned by the reverse communication version of optimzation algorithms.

| Enumerator                | Note                                             |
| :------------------------ | :----------------------------------------------- |
| OPK\\_TASK\\_ERROR        | An error has ocurred.                            |
| OPK\\_TASK\\_START        | Caller must call `start` method.                 |
| OPK\\_TASK\\_COMPUTE\\_FG | Caller must compute f(x) and g(x).               |
| OPK\\_TASK\\_NEW\\_X      | A new iterate is available.                      |
| OPK\\_TASK\\_FINAL\\_X    | Algorithm has converged, solution is available.  |
| OPK\\_TASK\\_WARNING      | Algorithm terminated with a warning.             |
"""
@enum opk_task::Int32 begin
    OPK_TASK_ERROR = -1
    OPK_TASK_START = 0
    OPK_TASK_COMPUTE_FG = 1
    OPK_TASK_NEW_X = 2
    OPK_TASK_FINAL_X = 3
    OPK_TASK_WARNING = 4
end

"""
    opk_lnsrch_new_csrch(ftol, gtol, xtol)

Create a Moré and Thuente cubic line search.

Moré & Thuente cubic line search method is designed to find a step `stp` that satisfies the sufficient decrease condition (a.k.a. first Wolfe condition):

f(stp) ≤ f(0) + ftol⋅stp⋅f'(0)

and the curvature condition (a.k.a. second strong Wolfe condition):

abs(f'(stp)) ≤ gtol⋅abs(f'(0))

where `f(stp)` is the value of the objective function for a step `stp` along the search direction while `f'(stp)` is the derivative of this function.

The algorithm is described in:

- J.J. Moré and D.J. Thuente, "Line search algorithms with guaranteed sufficient decrease" in ACM Transactions on Mathematical Software, vol. 20, pp. 286–307 (1994).

### Parameters
* `ftol`: specifies the nonnegative tolerance for the sufficient decrease condition.
* `gtol`: specifies the nonnegative tolerance for the curvature condition.
* `xtol`: specifies a nonnegative relative tolerance for an acceptable step. The method exits with a warning if the relative size of the bracketting interval is less than `xtol`.
### Returns
A line search object.
"""
function opk_lnsrch_new_csrch(ftol, gtol, xtol)
    @ccall libopk.opk_lnsrch_new_csrch(ftol::Cdouble, gtol::Cdouble, xtol::Cdouble)::Ptr{opk_lnsrch}
end

"""
    opk_lnsrch_new_backtrack(ftol, amin)

Create a backtracking line search.

### Parameters
* `ftol`: - Parameter of the first Wolfe condition. Must be in the range (0,1/2); however a small value is recommended.
* `amin`: - Smallest parameter of the first Wolfe condition. Must be in the range (0,1); if larger of equal 1/2, a bisection step is always take (as in Armijo's rule).
### Returns
A line search object.
"""
function opk_lnsrch_new_backtrack(ftol, amin)
    @ccall libopk.opk_lnsrch_new_backtrack(ftol::Cdouble, amin::Cdouble)::Ptr{opk_lnsrch}
end

"""
    opk_lnsrch_new_nonmonotone(m, ftol, sigma1, sigma2)

Create a nonmonotone line search.

Nonmonotone line search is described in SPG2 paper:

> E.G. Birgin, J.M. Martínez, & M. Raydan, "Nonmonotone Spectral Projected > Gradient Methods on Convex Sets," SIAM J. Optim. **10**, 1196-1211 (2000).

The parameters used in the SPG2 paper: ~~~~~{.cpp} m = 10; ftol = 1E-4; sigma1 = 0.1; sigma2 = 0.9; ~~~~~

With `m = 1`, this line search method is equivalent to Armijo's line search except that it attempts to use quadratic interpolation rather than systematically use bisection to reduce the step size.

### Parameters
* `m`: - Number of previous steps to remember.
* `ftol`: - Parameter for the function reduction criterion.
* `sigma1`: - Lower steplength bound to trigger bissection.
* `sigma2`: - Upper steplength relative bound to trigger bissection.
### Returns
A new line search object.
"""
function opk_lnsrch_new_nonmonotone(m, ftol, sigma1, sigma2)
    @ccall libopk.opk_lnsrch_new_nonmonotone(m::opk_index, ftol::Cdouble, sigma1::Cdouble, sigma2::Cdouble)::Ptr{opk_lnsrch}
end

"""
    opk_lnsrch_start(ls, f0, df0, stp1, stpmin, stpmax)

Start a new line search.

```c++
 opk_lnsrch_use_deriv} for the definition).
 @param stp1   - The length of the first step to try (must be between
                 `stpmin` and `stpmax`).
 @param stpmin - The minimum allowed step length (must be nonnegative).
 @param stpmax - The maximum allowed step length (must be greater than
                 `stpmin`).
 @return The line search task, which is normally @link OPK_LNSRCH_SEARCH}.
         A different value indicates an error.


```

### Parameters
* `ls`: - The line search object.
* `f0`: - The function value at the start of the line search (that is, for a step of length 0).
* `df0`: - The directional derivative at the start of the line search. (see {
"""
function opk_lnsrch_start(ls, f0, df0, stp1, stpmin, stpmax)
    @ccall libopk.opk_lnsrch_start(ls::Ptr{opk_lnsrch}, f0::Cdouble, df0::Cdouble, stp1::Cdouble, stpmin::Cdouble, stpmax::Cdouble)::Cint
end

"""
    opk_lnsrch_iterate(ls, stp_ptr, f, df)

Check whether line search has converged or update the step size.

```c++
 opk_lnsrch_use_deriv} for the definition).
 @return The returned value is strictly negative to indicate an error; it is
 equal to zero when searching is in progress; it is strictly positive when
 line search has converged or cannot make any more progresses.


```

### Parameters
* `ls`: - The line search object.
* `stp_ptr`: - The address at which the step length is stored. On entry, it must be the current step length; on exit, it is updated with the next step to try (unless the line search is finished).
* `f`: - The function value for the current step length.
* `df`: - The directional derivative for the current step length (see {
"""
function opk_lnsrch_iterate(ls, stp_ptr, f, df)
    @ccall libopk.opk_lnsrch_iterate(ls::Ptr{opk_lnsrch}, stp_ptr::Ptr{Cdouble}, f::Cdouble, df::Cdouble)::Cint
end

"""
    opk_lnsrch_get_step(ls)

Get current step lenght.

### Parameters
* `ls`: - The line search object.
### Returns
Returned value should be >= 0; -1 is returned in case of error.
"""
function opk_lnsrch_get_step(ls)
    @ccall libopk.opk_lnsrch_get_step(ls::Ptr{opk_lnsrch})::Cdouble
end

"""
    opk_lnsrch_get_task(ls)

Get current line search pending task.

In case of error or warning, {

```c++
 opk_lnsrch_get_status} can be used to
 query more information.
 @param ls - The line search object.
 @return The current line search pending task.


```
"""
function opk_lnsrch_get_task(ls)
    @ccall libopk.opk_lnsrch_get_task(ls::Ptr{opk_lnsrch})::opk_lnsrch_task
end

"""
    opk_lnsrch_get_status(ls)

Get current line search status.

### Parameters
* `ls`: - The line search object.
### Returns
The current line search status.
"""
function opk_lnsrch_get_status(ls)
    @ccall libopk.opk_lnsrch_get_status(ls::Ptr{opk_lnsrch})::opk_status
end

"""
    opk_lnsrch_has_errors(ls)

Check whether line search has errors.

### Parameters
* `ls`: - The line search object.
### Returns
A boolean value, true if there are any errors.
"""
function opk_lnsrch_has_errors(ls)
    @ccall libopk.opk_lnsrch_has_errors(ls::Ptr{opk_lnsrch})::opk_bool
end

"""
    opk_lnsrch_has_warnings(ls)

Check whether line search has warnings.

### Parameters
* `ls`: - The line search object.
### Returns
A boolean value, true if there are any warnings.
"""
function opk_lnsrch_has_warnings(ls)
    @ccall libopk.opk_lnsrch_has_warnings(ls::Ptr{opk_lnsrch})::opk_bool
end

"""
    opk_lnsrch_converged(ls)

Check whether line search has converged.

### Parameters
* `ls`: - The line search object.
### Returns
A boolean value, true if line search has converged.
"""
function opk_lnsrch_converged(ls)
    @ccall libopk.opk_lnsrch_converged(ls::Ptr{opk_lnsrch})::opk_bool
end

"""
    opk_lnsrch_finished(ls)

Check whether line search has finished.

Note that line search termination may be due to convergence, possibly with warnings, or to errors.

### Parameters
* `ls`: - The line search object.
### Returns
A boolean value, true if line search has finished.
"""
function opk_lnsrch_finished(ls)
    @ccall libopk.opk_lnsrch_finished(ls::Ptr{opk_lnsrch})::opk_bool
end

"""
    opk_lnsrch_use_deriv(ls)

Check whether line search requires the directional derivative.

The directional derivative is the derivative with respect to the step size of the objective function during the line search. The directional derivative is given by: ~~~~~{.cpp} +/-d'.g(x0 +\\- alpha*d) ~~~~~ with `d` the search direction, `g(x)` the gradient of the objective function (the `+/-` signs are `+` if `d` is a descent direction and `-` if it is an ascent direction) and assuming that, during the line search, the variables change as: ~~~~~{.cpp} x = x0 +/- alpha*d ~~~~~ with `x0` the variables at the start of the line search and `alpha` the step size.

### Parameters
* `ls`: - The line search object.
### Returns
A boolean value, true if line search makes use of the directional derivative.
"""
function opk_lnsrch_use_deriv(ls)
    @ccall libopk.opk_lnsrch_use_deriv(ls::Ptr{opk_lnsrch})::opk_bool
end

"""
    opk_cstep(stx_ptr, fx_ptr, dx_ptr, sty_ptr, fy_ptr, dy_ptr, stp_ptr, fp, dp, brackt, stpmin, stpmax)

Moré & Thuente method to perform a cubic safeguarded step.
"""
function opk_cstep(stx_ptr, fx_ptr, dx_ptr, sty_ptr, fy_ptr, dy_ptr, stp_ptr, fp, dp, brackt, stpmin, stpmax)
    @ccall libopk.opk_cstep(stx_ptr::Ptr{Cdouble}, fx_ptr::Ptr{Cdouble}, dx_ptr::Ptr{Cdouble}, sty_ptr::Ptr{Cdouble}, fy_ptr::Ptr{Cdouble}, dy_ptr::Ptr{Cdouble}, stp_ptr::Ptr{Cdouble}, fp::Cdouble, dp::Cdouble, brackt::Ptr{opk_bool}, stpmin::Cdouble, stpmax::Cdouble)::opk_status
end

mutable struct opk_nlcg_ end

"""
Opaque type for non-linear conjugate gradient optimizers.
"""
const opk_nlcg = opk_nlcg_

"""
    opk_nlcg_options

Structure used to specify the settings of a NLCG optimizer.

The absolute threshold for the norm or the gradient for convergence are specified by the members `gatol` and `grtol` of this structure. The convergence of the non-linear convergence gradient (NLCG) method is defined by: ~~~~~{.cpp} ||g|| <= max(0, gatol, grtol*||ginit||) ~~~~~ where `||g||` is the Euclidean norm of the current gradient `g`, `||ginit||` is the Euclidean norm of the initial gradient `ginit` while `gatol` and `grtol` are absolute and relative thresholds.

During a line search, the step is constrained to be within `stpmin` and `stpmax` times the lenght of the first step. The relative bounds must be such that: ~~~~~{.cpp} 0 <= stpmin < stpmax ~~~~~

| Field        | Note                                                                                                             |
| :----------- | :--------------------------------------------------------------------------------------------------------------- |
| delta        | Relative size for a small step.                                                                                  |
| epsilon      | Threshold to accept descent direction.                                                                           |
| grtol        | Relative threshold for the norm or the gradient (relative to the norm of the initial gradient) for convergence.  |
| gatol        | Absolute threshold for the norm or the gradient for convergence.                                                 |
| stpmin       | Relative minimum step length.                                                                                    |
| stpmax       | Relative maximum step length.                                                                                    |
| fmin         | Minimal function value if provided.                                                                              |
| flags        | A bitwise combination of the non-linear conjugate gradient update method and options.                            |
| fmin\\_given | Minimal function value is provided?                                                                              |
"""
struct opk_nlcg_options
    delta::Cdouble
    epsilon::Cdouble
    grtol::Cdouble
    gatol::Cdouble
    stpmin::Cdouble
    stpmax::Cdouble
    fmin::Cdouble
    flags::Cuint
    fmin_given::opk_bool
end

"""
    opk_get_nlcg_default_options(opts)

Query default nonlinear conjugate gradient optimizer parameters.

### Parameters
* `opts`: - The address of the structure where to store the parameters.
"""
function opk_get_nlcg_default_options(opts)
    @ccall libopk.opk_get_nlcg_default_options(opts::Ptr{opk_nlcg_options})::Cvoid
end

"""
    opk_check_nlcg_options(opts)

Check nonlinear conjugate gradient optimizer parameters.

### Parameters
* `opts`: - The address of the structure with the parameters to check.
### Returns
A standard status.
"""
function opk_check_nlcg_options(opts)
    @ccall libopk.opk_check_nlcg_options(opts::Ptr{opk_nlcg_options})::opk_status
end

"""
    opk_new_nlcg_optimizer(opts, vspace, lnsrch)

Create a new optimizer instance for non-linear conjugate gradient method.

This function creates an optimizer instance for minimization by a non-linear conjugate gradient method over a given vector space. The returned instance must be unreferenced by calling the function `[`opk_drop_object`](@ref)()`, or th macro `[`OPK_DROP`](@ref)()` when no longer needed.

### Parameters
* `vspace`: The vector space of the unknowns.
* `lnsrch`: - Optional line search method to use; can be `NULL` to use a default one. Note that the optimizer will hold a reference to the line search object.
### Returns
The address of a new optimizer instance, or NULL in case of error. Global variable errno may be ENOMEM if there is not enough memory or EINVAL if one of the arguments is invalid or EFAULT if vspace is NULL.
"""
function opk_new_nlcg_optimizer(opts, vspace, lnsrch)
    @ccall libopk.opk_new_nlcg_optimizer(opts::Ptr{opk_nlcg_options}, vspace::Ptr{opk_vspace}, lnsrch::Ptr{opk_lnsrch})::Ptr{opk_nlcg}
end

function opk_start_nlcg(opt, x)
    @ccall libopk.opk_start_nlcg(opt::Ptr{opk_nlcg}, x::Ptr{opk_vector})::opk_task
end

function opk_iterate_nlcg(opt, x, f, g)
    @ccall libopk.opk_iterate_nlcg(opt::Ptr{opk_nlcg}, x::Ptr{opk_vector}, f::Cdouble, g::Ptr{opk_vector})::opk_task
end

"""
    opk_get_nlcg_step(opt)

Get the current step length.

This function retrieves the value of the current step size.

### Parameters
* `opt`: - The NLCG optimizer.
### Returns
The value of the current step size, should be strictly positive.
### See also
opk\\_get\\_nlcg\\_stpmax(), opk\\_set\\_nlcg\\_stpmin\\_and\\_stpmax().
"""
function opk_get_nlcg_step(opt)
    @ccall libopk.opk_get_nlcg_step(opt::Ptr{opk_nlcg})::Cdouble
end

function opk_get_nlcg_gnorm(opt)
    @ccall libopk.opk_get_nlcg_gnorm(opt::Ptr{opk_nlcg})::Cdouble
end

function opk_get_nlcg_evaluations(opt)
    @ccall libopk.opk_get_nlcg_evaluations(opt::Ptr{opk_nlcg})::opk_index
end

function opk_get_nlcg_iterations(opt)
    @ccall libopk.opk_get_nlcg_iterations(opt::Ptr{opk_nlcg})::opk_index
end

function opk_get_nlcg_restarts(opt)
    @ccall libopk.opk_get_nlcg_restarts(opt::Ptr{opk_nlcg})::opk_index
end

function opk_get_nlcg_task(opt)
    @ccall libopk.opk_get_nlcg_task(opt::Ptr{opk_nlcg})::opk_task
end

function opk_get_nlcg_status(opt)
    @ccall libopk.opk_get_nlcg_status(opt::Ptr{opk_nlcg})::opk_status
end

function opk_get_nlcg_name(buf, size, opt)
    @ccall libopk.opk_get_nlcg_name(buf::Cstring, size::Csize_t, opt::Ptr{opk_nlcg})::Csize_t
end

"""
    opk_get_nlcg_description(buf, size, opt)

Get description of nonlinear conjugate gradient method.

### Parameters
* `buf`: - A string buffer to copy the description (can be `NULL`).
* `size`: - The number of available bytes in `buf`.
* `opt`: - The optimizer.
### Returns
The minimum number of bytes required to store the description (including the terminating '\\0' character).
"""
function opk_get_nlcg_description(buf, size, opt)
    @ccall libopk.opk_get_nlcg_description(buf::Cstring, size::Csize_t, opt::Ptr{opk_nlcg})::Csize_t
end

function opk_get_nlcg_beta(opt)
    @ccall libopk.opk_get_nlcg_beta(opt::Ptr{opk_nlcg})::Cdouble
end

"""
    opk_bound_type

Type of bounds.

Variables can be bounded or unbounded from below and from above. The bounds are specified by two parameters: a type and an address. If the variables are unbounded, then the corresponding bound type is `OPK_BOUND_NONE` and the associated address must be `NULL`. If all variables have the same bound, it is more efficient to specify a scalar bound with type `OPK_BOUND_SCALAR_FLOAT` or `OPK_BOUND_SCALAR_DOUBLE` and the address of a single or double precision variable which stores the bound. It is also possible to specify a componentwise bound in three different ways depending how the bounds are stored. If the bounds are in an OptimPack vector (of the same vector space of the variables) use type `OPK_BOUND_VECTOR` and provide the address of the vector. If the bounds are in a conventional array (with the same number of elements as the variables), then the address of the array must be provided with type `OPK_BOUND_STATIC_FLOAT` or `OPK_BOUND_STATIC_DOUBLE` if the array will not be released while the bound is in use or `OPK_BOUND_VOLATILE_FLOAT` or `OPK_BOUND_VOLATILE_DOUBLE` otherwise.

| Enumerator                      | Note                                                                                               |
| :------------------------------ | :------------------------------------------------------------------------------------------------- |
| OPK\\_BOUND\\_NONE              | No-bound (associated value must be `NULL`).                                                        |
| OPK\\_BOUND\\_SCALAR\\_FLOAT    | Scalar bound (associated value is the address of a float).                                         |
| OPK\\_BOUND\\_SCALAR\\_DOUBLE   | Scalar bound (associated value is the address of a double).                                        |
| OPK\\_BOUND\\_STATIC\\_FLOAT    | Bounds are stored in a static array of float's (associated value is the address of the array).     |
| OPK\\_BOUND\\_STATIC\\_DOUBLE   | Bounds are stored in a static array of double's (associated value is the address of the array).    |
| OPK\\_BOUND\\_VOLATILE\\_FLOAT  | Bounds are stored in a volatile array of floats's (associated value is the address of the array).  |
| OPK\\_BOUND\\_VOLATILE\\_DOUBLE | Bounds are stored in a volatile array of double's (associated value is the address of the array).  |
| OPK\\_BOUND\\_VECTOR            | Vector bound (associated value is the address of an [`opk_vector`](@ref)).                         |
"""
@enum opk_bound_type::UInt32 begin
    OPK_BOUND_NONE = 0
    OPK_BOUND_SCALAR_FLOAT = 1
    OPK_BOUND_SCALAR_DOUBLE = 2
    OPK_BOUND_STATIC_FLOAT = 3
    OPK_BOUND_STATIC_DOUBLE = 4
    OPK_BOUND_VOLATILE_FLOAT = 5
    OPK_BOUND_VOLATILE_DOUBLE = 6
    OPK_BOUND_VECTOR = 7
end

"""
    opk_project_variables(dst, x, set)

Project the variables to the feasible set.

Given input variables `x`, the projection produces feasible output variables `dst` which belongs to the convex set. The input and output variables can be stored in the same vector (*i.e.* the method can be applied *in-place*).

The function `[`opk_can_project_variables`](@ref)()` can be used to determine whether this functionality is implemented by `set`.

### Parameters
* `dst`: - The output projected variables.
* `x`: - The input variables.
* `set`: - The convex set which implements the constraints.
### Returns
`OPK_SUCCESS` or `OPK_ILLEGAL_ADDRESS` if one argumeant has an invalid (`NULL`) address, `OPK_BAD_SPACE` if not all vectors and bounds belongs to the same space, `OPK_NOT_IMPLEMENTED` if this functionality is not implemented.
"""
function opk_project_variables(dst, x, set)
    @ccall libopk.opk_project_variables(dst::Ptr{opk_vector}, x::Ptr{opk_vector}, set::Ptr{opk_convexset})::opk_status
end

"""
    opk_can_project_variables(set)

Check whether the projection of the variables is implemented.

### Parameters
* `set`: - The convex set which implements the constraints.
### Returns
A boolean value, true if the projection of the variables is implemented by `set`.
"""
function opk_can_project_variables(set)
    @ccall libopk.opk_can_project_variables(set::Ptr{opk_convexset})::opk_bool
end

"""
    opk_orientation

Orientation of a search direction.

If orientation is `OPK_DESCENT` (or strictly positive), the search direction `d` is a descent direction and the variables are updated as: ~~~~~~~~~~{.c} x[i] + alpha*d[i] ~~~~~~~~~~ otherwise, `d` is considered as an ascent disrection and the variables are updated as: ~~~~~~~~~~{.c} x[i] - alpha*d[i] ~~~~~~~~~~

| Enumerator    | Note                       |
| :------------ | :------------------------- |
| OPK\\_ASCENT  | Ascent search direction.   |
| OPK\\_DESCENT | Descent search direction.  |
"""
@enum opk_orientation::Int32 begin
    OPK_ASCENT = -1
    OPK_DESCENT = 1
end

"""
    opk_project_direction(dst, x, set, d, orient)

Project a direction.

This function projects the direction `d` so that: ~~~~~~~~~~{.c} x + orient*alpha*d ~~~~~~~~~~ yields a feasible position for `alpha > 0` sufficiently small.

The function `[`opk_can_project_directions`](@ref)()` can be used to determine whether this functionality is implemented by `set`.

```c++
              #OPK_DESCENT} and {@link #OPK_ASCENT} can be used to specify
              the orientation.
 @return `OPK_SUCCESS` or `OPK_ILLEGAL_ADDRESS` if one argumeant has an
         invalid (`NULL`) address, `OPK_BAD_SPACE` if not all vectors
         and bounds belongs to the same space, `OPK_NOT_IMPLEMENTED` if
         this functionality is not implemented.


```

### Parameters
* `dst`: - The resulting projected direction.
* `x`: - The current variables.
* `set`: - The convex set which implements the constraints.
* `d`: - The direction to project.
* `orient`: - The orientation of the direction `d`. Strictly positive if `d` is a descent direction, strictly negative if `d` is an ascent direction. For convenience, the constants {
"""
function opk_project_direction(dst, x, set, d, orient)
    @ccall libopk.opk_project_direction(dst::Ptr{opk_vector}, x::Ptr{opk_vector}, set::Ptr{opk_convexset}, d::Ptr{opk_vector}, orient::opk_orientation)::opk_status
end

"""
    opk_can_project_directions(set)

Check whether projection of the search direction is implemented.

### Parameters
* `set`: - The convex set which implements the constraints.
### Returns
A boolean value, true if projection of the search direction is implemented by `set`.
"""
function opk_can_project_directions(set)
    @ccall libopk.opk_can_project_directions(set::Ptr{opk_convexset})::opk_bool
end

"""
    opk_get_free_variables(dst, x, set, d, orient)

Find the non-binding constraints.

The function `[`opk_can_get_free_variables`](@ref)()` can be used to determine whether this functionality is implemented by `set`.

```c++
              opk_project_direction}).
 @return `OPK_SUCCESS` or `OPK_ILLEGAL_ADDRESS` if one argumeant has an
         invalid (`NULL`) address, `OPK_BAD_SPACE` if not all vectors
         and bounds belongs to the same space, `OPK_NOT_IMPLEMENTED` if
         this functionality is not implemented.


```

### Parameters
* `dst`: - The resulting mask whose elements are set to 1 (resp. 0) if the corresponding variables are free (resp. binded).
* `x`: - The current variables.
* `set`: - The convex set which implements the constraints.
* `d`: - The search direction.
* `orient`: - The orientation of the search direction (see {
"""
function opk_get_free_variables(dst, x, set, d, orient)
    @ccall libopk.opk_get_free_variables(dst::Ptr{opk_vector}, x::Ptr{opk_vector}, set::Ptr{opk_convexset}, d::Ptr{opk_vector}, orient::opk_orientation)::opk_status
end

"""
    opk_can_get_free_variables(set)

Check whether determining the free variables is implemented.

### Parameters
* `set`: - The convex set which implements the constraints.
### Returns
A boolean value, true if determining the free variables is implemented by `set`.
"""
function opk_can_get_free_variables(set)
    @ccall libopk.opk_can_get_free_variables(set::Ptr{opk_convexset})::opk_bool
end

"""
    opk_get_step_limits(smin1, smin2, smax, x, set, d, orient)

Find the limits of the step size.

Along the search direction the new variables are computed as: ~~~~~~~~~~{.c} proj(x +/- alpha*d) ~~~~~~~~~~ where `proj` is the projection onto the feasible set, `alpha` is the step length and +/- is a plus for a descent direction and a minus otherwise. This method computes 3 specific step lengths: `smin1` which is the step length for the first bound reached by the variables along the search direction; `smin2` which is the step length for the first bound reached by the variables along the search direction and with a non-zero step length; `smax` which is the step length for the last bound reached by the variables along the search direction.

In other words, for any `0 <= alpha <= smin1`, no variables may overreach a bound, while, for any `alpha >= smax`, the projected variables are the same as those obtained with `alpha = smax`. If `d` has been properly projected (e.g. by {

```c++
 opk_project_direction}), then `smin1 = smin2` otherwise
 `0 <= smin1 <= smin2` and `smin2 > 0`.
 The function `opk_can_get_step_limits()` can be used to determine whether
 this functionality is implemented by `set`.
 @param smin1  - The address to store the value of `smin1` or `NULL`.
 @param smin2  - The address to store the value of `smin2` or `NULL`.
 @param smax   - The address to store the value of `smax` or `NULL`.
 @param x      - The current variables (assumed to be feasible).
 @param set - The convex set which implements the constraints.
 @param d      - The search direction.
 @param orient - The orientation of the search direction (see
                 {@link opk_project_direction}).
 @return `OPK_SUCCESS` or `OPK_ILLEGAL_ADDRESS` if one argumeant has an
         invalid (`NULL`) address, `OPK_BAD_SPACE` if not all vectors
         and bounds belongs to the same space, `OPK_NOT_IMPLEMENTED` if
         this functionality is not implemented.


```
"""
function opk_get_step_limits(smin1, smin2, smax, x, set, d, orient)
    @ccall libopk.opk_get_step_limits(smin1::Ptr{Cdouble}, smin2::Ptr{Cdouble}, smax::Ptr{Cdouble}, x::Ptr{opk_vector}, set::Ptr{opk_convexset}, d::Ptr{opk_vector}, orient::opk_orientation)::opk_status
end

"""
    opk_can_get_step_limits(set)

Check whether determining the step limits is implemented.

### Parameters
* `set`: - The convex set which implements the constraints.
### Returns
A boolean value, true if determining the step limits is implemented by `set`.
"""
function opk_can_get_step_limits(set)
    @ccall libopk.opk_can_get_step_limits(set::Ptr{opk_convexset})::opk_bool
end

"""
    opk_new_boxset(space, lower_type, lower, upper_type, upper)

Create a new box set.

A box set is a convex set with separable bounds on the variables.

### Parameters
* `space`: - The vector space of the variables. The convex set is a subset of this vector space.
* `lower_type`: - The type of the lower bound(s).
* `lower`: - The value of the lower bound(s).
* `upper_type`: - The type of the upper bound(s).
* `upper`: - The value of the upper bound(s).
### Returns
The address of a new box set, `NULL` in case of error (global variable `errno` can be used to figure out the reason of the failure).
"""
function opk_new_boxset(space, lower_type, lower, upper_type, upper)
    @ccall libopk.opk_new_boxset(space::Ptr{opk_vspace}, lower_type::opk_bound_type, lower::Ptr{Cvoid}, upper_type::opk_bound_type, upper::Ptr{Cvoid})::Ptr{opk_convexset}
end

mutable struct opk_vmlmb_ end

"""
Opaque type for a variable metric optimizer.
"""
const opk_vmlmb = opk_vmlmb_

"""
    opk_bfgs_scaling

Rules for scaling the inverse Hessian approximation.

| Enumerator                         | Note                                                 |
| :--------------------------------- | :--------------------------------------------------- |
| OPK\\_SCALING\\_NONE               | No-scaling.                                          |
| OPK\\_SCALING\\_OREN\\_SPEDICATO   | Scaling by: {  ```c++  gamma = (s'.y)/(y'.y)}   ```  |
| OPK\\_SCALING\\_BARZILAI\\_BORWEIN | Scaling by: {  ```c++  gamma = (s'.s)/(s'.y)}   ```  |
"""
@enum opk_bfgs_scaling::UInt32 begin
    OPK_SCALING_NONE = 0
    OPK_SCALING_OREN_SPEDICATO = 1
    OPK_SCALING_BARZILAI_BORWEIN = 2
end

"""
    opk_vmlmb_options

Structure used to store the settings of a VMLMB optimizer.

| Field         | Note                                                                                                                                 |
| :------------ | :----------------------------------------------------------------------------------------------------------------------------------- |
| delta         | Relative size for a small step.                                                                                                      |
| epsilon       | Threshold to accept descent direction.                                                                                               |
| grtol         | Relative threshold for the norm or the projected gradient (relative to the norm of the initial projected gradient) for convergence.  |
| gatol         | Absolute threshold for the norm or the projected gradient for convergence.                                                           |
| stpmin        | Relative minimum step length.                                                                                                        |
| stpmax        | Relative maximum step length.                                                                                                        |
| mem           | Maximum number of memorized steps.                                                                                                   |
| blmvm         | Emulate Benson & Moré BLMVM method?                                                                                                  |
| save\\_memory | Save some memory?                                                                                                                    |
"""
struct opk_vmlmb_options
    delta::Cdouble
    epsilon::Cdouble
    grtol::Cdouble
    gatol::Cdouble
    stpmin::Cdouble
    stpmax::Cdouble
    mem::opk_index
    blmvm::opk_bool
    save_memory::opk_bool
end

"""
    opk_get_vmlmb_default_options(opts)

Query default VMLMB optimizer parameters.

### Parameters
* `opts`: - The address of the structure where to store the parameters.
"""
function opk_get_vmlmb_default_options(opts)
    @ccall libopk.opk_get_vmlmb_default_options(opts::Ptr{opk_vmlmb_options})::Cvoid
end

"""
    opk_check_vmlmb_options(opts)

Check VMLMB optimizer parameters.

### Parameters
* `opts`: - The address of the structure with the parameters to check.
### Returns
A standard status.
"""
function opk_check_vmlmb_options(opts)
    @ccall libopk.opk_check_vmlmb_options(opts::Ptr{opk_vmlmb_options})::opk_status
end

"""
    opk_new_vmlmb_optimizer(opts, space, lnsrch, box)

Create a reverse communication optimizer implementing a limited memory quasi-Newton method.

The optimizer may account for bound constraints if argument `box` is non-`NULL`.

### Parameters
* `opts`: - The options to sue (can be `NULL` to use default options).
* `space`: - The space to which belong the variables.
* `lnsrch`: - Optional line search method to use; can be `NULL` to use a default one. Note that the optimizer will hold a reference to the line search object.
* `box`: - An optional box set implementing the bound constraints (can be `NULL`).
### Returns
The address of a new optimizer instance, or NULL in case of error. Global variable errno may be ENOMEM if there is not enough memory or EINVAL if one of the arguments is invalid or EFAULT if vspace is NULL.
"""
function opk_new_vmlmb_optimizer(opts, space, lnsrch, box)
    @ccall libopk.opk_new_vmlmb_optimizer(opts::Ptr{opk_vmlmb_options}, space::Ptr{opk_vspace}, lnsrch::Ptr{opk_lnsrch}, box::Ptr{opk_convexset})::Ptr{opk_vmlmb}
end

"""
    opk_vmlmb_method

The variants implemented by VMLMB.
"""
@enum opk_vmlmb_method::UInt32 begin
    OPK_LBFGS = 0
    OPK_VMLMB = 1
    OPK_BLMVM = 2
end

function opk_get_vmlmb_method(opt)
    @ccall libopk.opk_get_vmlmb_method(opt::Ptr{opk_vmlmb})::opk_vmlmb_method
end

function opk_get_vmlmb_method_name(opt)
    @ccall libopk.opk_get_vmlmb_method_name(opt::Ptr{opk_vmlmb})::Cstring
end

function opk_start_vmlmb(opt, x)
    @ccall libopk.opk_start_vmlmb(opt::Ptr{opk_vmlmb}, x::Ptr{opk_vector})::opk_task
end

function opk_iterate_vmlmb(opt, x, f, g)
    @ccall libopk.opk_iterate_vmlmb(opt::Ptr{opk_vmlmb}, x::Ptr{opk_vector}, f::Cdouble, g::Ptr{opk_vector})::opk_task
end

function opk_get_vmlmb_task(opt)
    @ccall libopk.opk_get_vmlmb_task(opt::Ptr{opk_vmlmb})::opk_task
end

function opk_get_vmlmb_status(opt)
    @ccall libopk.opk_get_vmlmb_status(opt::Ptr{opk_vmlmb})::opk_status
end

"""
    opk_get_vmlmb_description(buf, size, opt)

Get description of algorithm implemented by VMLMB.

### Parameters
* `buf`: - A string buffer to copy the description (can be `NULL`).
* `size`: - The number of available bytes in `buf`.
* `opt`: - The optimizer.
### Returns
The minimum number of bytes required to store the description (including the terminating '\\0' character).
"""
function opk_get_vmlmb_description(buf, size, opt)
    @ccall libopk.opk_get_vmlmb_description(buf::Cstring, size::Csize_t, opt::Ptr{opk_vmlmb})::Csize_t
end

function opk_get_vmlmb_evaluations(opt)
    @ccall libopk.opk_get_vmlmb_evaluations(opt::Ptr{opk_vmlmb})::opk_index
end

function opk_get_vmlmb_iterations(opt)
    @ccall libopk.opk_get_vmlmb_iterations(opt::Ptr{opk_vmlmb})::opk_index
end

function opk_get_vmlmb_restarts(opt)
    @ccall libopk.opk_get_vmlmb_restarts(opt::Ptr{opk_vmlmb})::opk_index
end

function opk_get_vmlmb_step(opt)
    @ccall libopk.opk_get_vmlmb_step(opt::Ptr{opk_vmlmb})::Cdouble
end

function opk_get_vmlmb_gnorm(opt)
    @ccall libopk.opk_get_vmlmb_gnorm(opt::Ptr{opk_vmlmb})::Cdouble
end

"""
    opk_get_vmlmb_mp(opt)

Get actual number of memorized steps.

### Parameters
* `opt`: - The VMLMB optimizer.
### Returns
The actual number of memorized steps which is in the range `[0,m]`, with `m` the maximum number of memorized steps.
"""
function opk_get_vmlmb_mp(opt)
    @ccall libopk.opk_get_vmlmb_mp(opt::Ptr{opk_vmlmb})::opk_index
end

"""
    opk_get_vmlmb_s(opt, j)

Get a given memorized variable change.

Variable metric methods store variable and gradient changes for the few last steps to measure the effect of the Hessian. Using pseudo-code notation the following `(s,y)` pairs are memorized: ~~~~~{.cpp} s[k-j] = x[k-j+1] - x[k-j] // variable change y[k-j] = g[k-j+1] - g[k-j] // gradient change ~~~~~ with `x[k]` and `g[k]` the variables and corresponding gradient at `k`-th iteration and `j=1,...,mp` the relative index of the saved pair.

### Parameters
* `opt`: - The VMLMB optimizer.
* `j`: - The index of the memorized pair to consider relative to the last one. Index `j` must be in the inclusive range `[0,mp]` with `mp` the actual number of saved corrections. The special case `j = 0` corresponds to the next saved pair (which will be overwritten at the end of the current iteration). The other cases, `j = 1,...,mp`, correspond to actually saved pairs. Because the algorithm may be automatically restarted or may try to save memory, the actual number of saved pairs `mp` may change between iterations and has to be retrieved each time a given save pair is queried.
### Returns
The variable difference `s[k-j]` where `k` is the current iteration number, with `m` the maximum number of memorized steps. `NULL` is returned if `j` is out of bounds.
### See also
[`opk_get_vmlmb_y`](@ref), opk\\_get\\_vmlmb\\_mb.
"""
function opk_get_vmlmb_s(opt, j)
    @ccall libopk.opk_get_vmlmb_s(opt::Ptr{opk_vmlmb}, j::opk_index)::Ptr{opk_vector}
end

"""
    opk_get_vmlmb_y(opt, j)

Get a given memorized gradient change.

### Parameters
* `opt`: - The VMLMB optimizer.
* `j`: - The index of the memorized pair to consider relative to the last one. Index `j` must be in the inclusive range `[0,mp]` with `mp` the actual number of saved corrections.
### Returns
The gradient difference `y[k-j]` where `k` is the current iteration number, with `m` the maximum number of memorized steps. `NULL` is returned if `j` is out of bounds.
### See also
[`opk_get_vmlmb_s`](@ref), opk\\_get\\_vmlmb\\_mb.
"""
function opk_get_vmlmb_y(opt, j)
    @ccall libopk.opk_get_vmlmb_y(opt::Ptr{opk_vmlmb}, j::opk_index)::Ptr{opk_vector}
end

"""
    opk_algorithm

Limited memory optimization algorithm.

| Enumerator              | Note                                                    |
| :---------------------- | :------------------------------------------------------ |
| OPK\\_ALGORITHM\\_NLCG  | Nonlinear conjugate gradient.                           |
| OPK\\_ALGORITHM\\_VMLMB | Limited memory variable metric (possibly with bounds).  |
"""
@enum opk_algorithm::UInt32 begin
    OPK_ALGORITHM_NLCG = 0
    OPK_ALGORITHM_VMLMB = 1
end

mutable struct opk_optimizer_ end

"""
Opaque structure for limited memory optimizer.
"""
const opk_optimizer = opk_optimizer_

"""
    opk_new_optimizer(algorithm, opts, type, n, lower_type, lower, upper_type, upper, lnschr)

Create a reverse communication optimizer implementing a limited memory optimization method.

This simple driver assumes that the variables and the gradient are stored as flat arrays of floating point values (`float` or `double`). The implemented algorithms are suitable for large problems and smooth (that is to say differentiable) objective functions.

Depending on the settings, optimization can be performed by an instance of the nonlinear conjugate gradient methods or an instance of the limited memory quasi-Newton (a.k.a. variable metric) methods. Simple bound constraints can be taken into account (providing the variable metric method is selected).

When no longer needed, the optimizer must be released with {

```c++
 opk_destroy_optimizer}.
 Typical usage is:
 ~~~~~{.cpp}
 const int n = 100000;
 const int type = OPK_FLOAT;
 double fx;
 float x[n];
 float gx[n];
 opk_optimizer* opt = opk_new_optimizer(OPK_ALGORITHM_VMLMB,
                                          NULL, type, n,
                                          OPK_BOUND_NONE, NULL,
                                          OPK_BOUND_NONE, NULL,
                                          NULL);
 task = opk_start(opt, x);
 for (;;) {
     if (task == OPK_TASK_COMPUTE_FG) {
          fx = f(x);
          for (i = 0; i < n; ++i) {
              gx[i] = ...;
          }
      } else if (task == OPK_TASK_NEW_X) {
          // a new iterate is available
          fprintf(stdout, "iter=%ld, f(x)=%g, |g(x)|=%g\\n",
                  opk_get_iterations(opt), fx,
                  opk_get_gnorm(opt));
      } else {
          break;
      }
      task = opk_iterate(opt, x, fx, gx);
 }
 if (task != OPK_TASK_FINAL_X) {
     fprintf(stderr, "some error occured (%s)",
             opk_get_reason(opk_get_status(opt)));
 }
 opk_destroy_optimizer(opt);
 ~~~~~
 @param algorithm - The limited memory algorithm to use.
 @param opts   - Address of structure with algorithm options (can be `NULL`).
 @param type   - The type of the variable limited memory algorithm to use
                 ({@link OPK_FLOAT} or {@link OPK_DOUBLE}).
 @param n      - The number of variables (`n` > 0).
 @param lower_type - The type of the lower bound.
 @param lower  - Optional lower bound for the variables.  Can be `NULL` if
                 there are no lower bounds and `lower_type` is {@link
                 OPK_BOUND_NONE}; otherwise must have the same type as the
                 variables.
 @param upper_type - The type of the upper bound.
 @param upper  - Optional upper bound for the variables.  Can be `NULL` if
                 there are no upper bounds and `upper_type` is {@link
                 OPK_BOUND_NONE}; otherwise must have the same type as the
                 variables.
 @param lnsrch - The line search method to use, can be `NULL` to use a default
                 line search which depends on the optimization algorithm.
 @return The address of a new optimizer instance, or `NULL` in case of error.
         In case of error, global variable `errno` may be `ENOMEM` if there
         is not enough memory or `EINVAL` if one of the arguments is invalid
         or `EFAULT` if the bounds are unexpectedly `NULL`.


```
"""
function opk_new_optimizer(algorithm, opts, type, n, lower_type, lower, upper_type, upper, lnschr)
    @ccall libopk.opk_new_optimizer(algorithm::opk_algorithm, opts::Ptr{Cvoid}, type::opk_eltype, n::opk_index, lower_type::opk_bound_type, lower::Ptr{Cvoid}, upper_type::opk_bound_type, upper::Ptr{Cvoid}, lnschr::Ptr{opk_lnsrch})::Ptr{opk_optimizer}
end

"""
    opk_destroy_optimizer(opt)

Destroy a reverse communication optimizer implementing a limited memory optimization method.

This function must be called when the optimizer is no longer in use. It reduces the reference count of the optimizer eventually freeing any associated ressources.

```c++
 #opk_new_optimizer}.


```

### Parameters
* `opt`: - An optimizer created by {
"""
function opk_destroy_optimizer(opt)
    @ccall libopk.opk_destroy_optimizer(opt::Ptr{opk_optimizer})::Cvoid
end

"""
    opk_start(opt, x)

Start the optimization with given initial variables.

```c++
 #opk_new_optimizer}.
 @param x    - The variables which must be of the same type (`float` or
               `double`) as assumed when creating the optimizer and have the
               correct number of elements.
 @return An integer indicating the next thing to do for the caller.
         Unless an error occured, it should be {@link OPK_TASK_COMPUTE_FG}.


```

### Parameters
* `opt`: - An optimizer created by {
"""
function opk_start(opt, x)
    @ccall libopk.opk_start(opt::Ptr{opk_optimizer}, x::Ptr{Cvoid})::opk_task
end

"""
    opk_iterate(opt, x, f, g)

Proceed with next optimization step.

Note that the variables must not be changed by the caller after calling {

```c++
 opk_start} and between calls to {@link opk_iterate}.  Arrays `x` and
 `g` must be of the same type (`float` or `double`) as assumed when creating
 the optimizer and have the correct number of elements.
 @param opt  - An optimizer created by {@link #opk_new_optimizer}.
 @param x    - The current variables.
 @param f    - The value of the objective function at the current variables.
 @param g    - The gradient of the objective function at the current
               variables.
 @return An integer indicating the next thing to do for the caller.


```
"""
function opk_iterate(opt, x, f, g)
    @ccall libopk.opk_iterate(opt::Ptr{opk_optimizer}, x::Ptr{Cvoid}, f::Cdouble, g::Ptr{Cvoid})::opk_task
end

"""
    opk_get_task(opt)

Get the current pending task.

```c++
 #opk_new_optimizer}.
 @return The current pending task.


```

### Parameters
* `opt`: - An optimizer created by {
"""
function opk_get_task(opt)
    @ccall libopk.opk_get_task(opt::Ptr{opk_optimizer})::opk_task
end

"""
    opk_get_status(opt)

Get the current optimizer status.

This function is useful to figure out which kind of problem occured when the pending task is {

```c++
 OPK_TASK_WARNING} or {@link OPK_TASK_ERROR}.
 @param opt  - An optimizer created by {@link #opk_new_optimizer}.
 @return The current optimizer status.


```
"""
function opk_get_status(opt)
    @ccall libopk.opk_get_status(opt::Ptr{opk_optimizer})::opk_status
end

function opk_get_evaluations(opt)
    @ccall libopk.opk_get_evaluations(opt::Ptr{opk_optimizer})::opk_index
end

function opk_get_iterations(opt)
    @ccall libopk.opk_get_iterations(opt::Ptr{opk_optimizer})::opk_index
end

function opk_get_restarts(opt)
    @ccall libopk.opk_get_restarts(opt::Ptr{opk_optimizer})::opk_index
end

function opk_get_name(buf, size, opt)
    @ccall libopk.opk_get_name(buf::Cstring, size::Csize_t, opt::Ptr{opk_optimizer})::Csize_t
end

function opk_get_description(buf, size, opt)
    @ccall libopk.opk_get_description(buf::Cstring, size::Csize_t, opt::Ptr{opk_optimizer})::Csize_t
end

function opk_get_step(opt)
    @ccall libopk.opk_get_step(opt::Ptr{opk_optimizer})::Cdouble
end

function opk_get_gnorm(opt)
    @ccall libopk.opk_get_gnorm(opt::Ptr{opk_optimizer})::Cdouble
end

@enum opk_fmin_task::Int32 begin
    OPK_FMIN_ERROR = -1
    OPK_FMIN_START = 0
    OPK_FMIN_FX = 1
    OPK_FMIN_NEWX = 2
    OPK_FMIN_CONVERGENCE = 3
end

"""
    opk_fmin(f, a, b, flags, maxeval, prec, out)

Search for the minimum of a function.

This function searches for the minimum `xmin` of the univariate function `f(x)`.

The algorithm requires an initial interval `(a,b)`. If the bit [`OPK_FMIN_BOUNDED_BY_A`](@ref) is set in `flags`, then the value `a` is a strict bound for the search. Similarly, if the bit [`OPK_FMIN_BOUNDED_BY_B`](@ref) is set in `flags`, then the value `b` is a strict bound for the search. If `a` and/or `b` are not exclusive bounds, their values are tried first by the algorithm.

If the bit [`OPK_FMIN_SMOOTH`](@ref) is set in `flags`, then the function is assumed to be smooth and Brent's algorithm [1] is used to find the minimum; otherwise, the golden section method is used.

The result is stored into array `out` as follows

- `out[0]` is the approximative solution `xmin` - `out[1]` is the lower bound `xlo` of the final interval - `out[2]` is the upper bound `xhi` of the final interval - `out[3]` is `f(xmin)` - `out[4]` is `f(xlo)` - `out[5]` is `f(xhi)` - `out[6]` is the number of function evaluations

Depending whether the input interval has exclusive bounds, the minimum number of function evaluations is between 1 and 3 whatever is the value of `maxeval`. If the minimum has not been bracketted, the result is:

- `out[0:2] = {u, v, w}` and - `out[3:5] = {f(u), f(v), f(w)}`

with `u`, `v`, and `w` the 3 last positions tried by the algorithm (in no particular order).

References: [1] Brent, R.P. 1973, "Algorithms for Minimization without Derivatives" (Englewood Cliffs, NJ: Prentice-Hall), Chapter 5.

### Parameters
* `f`: The function to minimize.
* `a`: The first point of the initial interval.
* `b`: The other point of the initial interval.
* `flags`: A bitwise combination of flags (see description).
* `maxeval`: If non-negative, the maximum number of function evaluations; otherwise, no limits.
* `prec`: The relative precision: the algorithm stop when the uncertainty is less or equal `prec` times the magnitude of the solution. If `prec` is strictly negative, then a default precision of `sqrt(epsilon)` is used with `epsilon` the machine relative precision.
* `out`: A 7-element array to store the result.
### Returns
0 on convergence, 1 if too many iterations but minimum was bracketted, 2 if too many iterations but minimum was *not* bracketted, and -1 on error (invalid input arguments).
### See also
[`opk_fmin_with_context`](@ref)().
"""
function opk_fmin(f, a, b, flags, maxeval, prec, out)
    @ccall libopk.opk_fmin(f::Ptr{Cvoid}, a::Cdouble, b::Cdouble, flags::Cuint, maxeval::Clong, prec::Cdouble, out::Ptr{Cdouble})::Cint
end

"""
    opk_fmin_with_context(f, a, b, flags, maxeval, prec, out, data)

Search for the minimum of a function.

This function is identical to [`opk_fmin`](@ref)() except that the user-defined function is called as `f(data,x)` to evaluate the function at `x`. The `data` argument is simply passed to the user-defined function and may be used to store any parameters (but the variable value) needed by the function.

Returns: Same value as <spf\\_fmin>.

### Parameters
* `f`: The function to minimize.
* `a`: The first point of the initial interval.
* `b`: The other point of the initial interval.
* `flags`: A bitwise combination of flags.
* `maxeval`: If non-negative, the maximum number of function evaluations; otherwise no limits.
* `prec`: The relative precision.
* `out`: A 7-element array to store the result.
* `data`: Anything needed by the user-defined function.
### See also
[`opk_fmin`](@ref)().
"""
function opk_fmin_with_context(f, a, b, flags, maxeval, prec, out, data)
    @ccall libopk.opk_fmin_with_context(f::Ptr{Cvoid}, a::Cdouble, b::Cdouble, flags::Cuint, maxeval::Clong, prec::Cdouble, out::Ptr{Cdouble}, data::Ptr{Cvoid})::Cint
end

"""
    opk_dgqt(n, a, lda, b, delta, rtol, atol, itmax, par_ptr, f_ptr, x, iter_ptr, z, wa1, wa2)

Computes a trust region step.

Given an `n` by `n` symmetric matrix `A`, an `n`-vector `b`, and a positive number `delta`, this subroutine determines a vector `x` which approximately minimizes the quadratic function:

f(x) = (1/2) x'.A.x + b'.x

subject to the Euclidean norm constraint

norm(x) <= delta.

This subroutine computes an approximation `x` and a Lagrange multiplier `par` such that either `par` is zero and

norm(x) <= (1 + rtol)*delta,

or `par` is positive and

abs(norm(x) - delta) <= rtol*delta.

If `xsol` is the exact solution to the problem, the approximation `x` satisfies

f(x) <= ((1 - rtol)^2)*f(xsol)

(where `^` means exponentiation).

* 1 - The function value `f(x)` has the relative accuracy specified by `rtol`.

* 2 - The function value `f(x)` has the absolute accuracy specified by `atol`.

* 3 - Rounding errors prevent further progress. On exit `x` is the best available approximation.

* 4 - Failure to converge after `itmax` iterations. On exit `x` is the best available approximation.

* MINPACK-2: [`opk_destsv`](@ref), [`opk_sgqt`](@ref).

* LAPACK: `opk_dpotrf`.

* Level 1 BLAS: [`opk_dasum`](@ref), [`opk_daxpy`](@ref), [`opk_dcopy`](@ref), [`opk_ddot`](@ref), [`opk_dnrm2`](@ref), [`opk_dscal`](@ref).

* Level 2 BLAS: [`opk_dtrmv`](@ref), [`opk_dtrsv`](@ref).

### History

* MINPACK-2 Project. July 1994. Argonne National Laboratory and University of Minnesota. Brett M. Averick, Richard Carter, and Jorge J. Moré

* C-version on 30 January 2006 by Éric Thiébaut (CRAL); `info` is the value returned by the function.

### Parameters
* `n`: - The order of `A`.
* `a`: - A real array of dimensions `lda` by `n`. On entry the full upper triangle of `a` must contain the full upper triangle of the symmetric matrix `A`. On exit the array contains the matrix `A`.
* `lda`: - The leading dimension of the array `a`.
* `b`: - A real array of dimension `n`. On entry `b` specifies the linear term in the quadratic. On exit `b` is unchanged.
* `delta`: - The bound on the Euclidean norm of `x`.
* `rtol`: - The relative accuracy desired in the solution. Convergence occurs if `f(x) <= ((1-rtol)^2)*f(xsol)` where `xsol` is the exact solution.
* `atol`: - The absolute accuracy desired in the solution. Convergence occurs when `norm(x) <= (1+rtol)*delta` and `max(-f(x),-f(xsol)) <= atol`.
* `itmax`: - The maximum number of iterations.
* `par_ptr`: - If non `NULL`, the address of an integer variable used to store the Lagrange multiplier. On entry `*par\\_ptr` is an initial estimate of the Lagrange multiplier for the constraint `norm(x) <= delta`. On exit `*par\\_ptr` contains the final estimate of the multiplier. If `NULL`, the initial Lagrange parameter is 0.
* `f_ptr`: - If non `NULL`, the address of a floating point variable used to store the function value. On entry `*f\\_ptr` needs not be specified. On exit `*f\\_ptr` is set to `f(x)` at the output `x`.
* `x`: - A real array of dimension `n`. On entry `x` need not be specified. On exit `x` is set to the final estimate of the solution.
* `iter_ptr`: - If non `NULL`, the address of an integer variable used to store the number of iterations.
* `z`: - A real work array of dimension `n`.
* `wa1`: - A real work array of dimension `n`.
* `wa2`: - A real work array of dimension `n`.
### Returns
The function returns one of the following integer values:
### See also

"""
function opk_dgqt(n, a, lda, b, delta, rtol, atol, itmax, par_ptr, f_ptr, x, iter_ptr, z, wa1, wa2)
    @ccall libopk.opk_dgqt(n::opk_index, a::Ptr{Cdouble}, lda::opk_index, b::Ptr{Cdouble}, delta::Cdouble, rtol::Cdouble, atol::Cdouble, itmax::opk_index, par_ptr::Ptr{Cdouble}, f_ptr::Ptr{Cdouble}, x::Ptr{Cdouble}, iter_ptr::Ptr{opk_index}, z::Ptr{Cdouble}, wa1::Ptr{Cdouble}, wa2::Ptr{Cdouble})::Cint
end

"""
    opk_sgqt(n, a, lda, b, delta, rtol, atol, itmax, par_ptr, f_ptr, x, iter_ptr, z, wa1, wa2)

Computes a trust region step.

This function is the single precision version of [`opk_dgqt`](@ref).

* MINPACK-2: [`opk_sestsv`](@ref), [`opk_dgqt`](@ref).

* LAPACK: `opk_spotrf`

* Level 1 BLAS: [`opk_sasum`](@ref), [`opk_saxpy`](@ref), [`opk_scopy`](@ref), [`opk_sdot`](@ref), [`opk_snrm2`](@ref), [`opk_sscal`](@ref).

* Level 2 BLAS: [`opk_strmv`](@ref), [`opk_strsv`](@ref).

### History

* MINPACK-2 Project. July 1994. Argonne National Laboratory and University of Minnesota. Brett M. Averick, Richard Carter, and Jorge J. Moré

* C-version on 30 January 2006 by Éric Thiébaut (CRAL); `info` is the value returned by the function.

### See also

"""
function opk_sgqt(n, a, lda, b, delta, rtol, atol, itmax, par_ptr, f_ptr, x, iter_ptr, z, wa1, wa2)
    @ccall libopk.opk_sgqt(n::opk_index, a::Ptr{Cfloat}, lda::opk_index, b::Ptr{Cfloat}, delta::Cfloat, rtol::Cfloat, atol::Cfloat, itmax::opk_index, par_ptr::Ptr{Cfloat}, f_ptr::Ptr{Cfloat}, x::Ptr{Cfloat}, iter_ptr::Ptr{opk_index}, z::Ptr{Cfloat}, wa1::Ptr{Cfloat}, wa2::Ptr{Cfloat})::Cint
end

"""
    opk_destsv(n, r, ldr, z)

Computes smallest singular value and corresponding vector from an upper triangular matrix.

Given an `n` by `n` upper triangular matrix `R`, this subroutine estimates the smallest singular value and the associated singular vector of `R`.

In the algorithm a vector `e` is selected so that the solution `y` to the system {

```c++
 R'.y = e} is large. The choice of sign for the components
 of `e` cause maximal local growth in the components of `y` as the forward
 substitution proceeds. The vector `z` is the solution of the system
 {@code R.z = y}, and the estimate `svmin` is {@code norm(y)/norm(z)} in the
 Euclidean norm.
 @param n - The order of `R`.
 @param r - A real array of dimension `ldr` by `n`.  On entry the full upper
            triangle must contain the full upper triangle of the matrix `R`.
            On exit `r` is unchanged.
 @param ldr - The leading dimension of `r`.
 @param z - A real array of dimension `n`.  On entry `z` need not be
            specified.  On exit `z` contains a singular vector associated
            with the estimate `svmin` such that {@code norm(R.z) = svmin}
            and {@code norm(z) = 1} in the Euclidean norm.
 @return
 This function returns `svmin`, an estimate for the smallest singular value
 of `R`.
 @see
  * MINPACK-2: `opk_dgqt`, `opk_sestsv`.
  * Level 1 BLAS: `opk_dasum`, `opk_daxpy`, `opk_dnrm2`, `opk_dscal`.
 ### History
  * MINPACK-2 Project. October 1993.
     Argonne National Laboratory
     Brett M. Averick and Jorge J. Moré.
  * C-version on 30 January 2006 by Éric Thiébaut (CRAL);
    `svmin` is the value returned by the function.


```
"""
function opk_destsv(n, r, ldr, z)
    @ccall libopk.opk_destsv(n::opk_index, r::Ptr{Cdouble}, ldr::opk_index, z::Ptr{Cdouble})::Cdouble
end

"""
    opk_sestsv(n, r, ldr, z)

Computes smallest singular value and corresponding vector from an upper triangular matrix.

This function is the single precision version of [`opk_destsv`](@ref).

* Level 1 BLAS: [`opk_sasum`](@ref), [`opk_saxpy`](@ref), [`opk_snrm2`](@ref), [`opk_sscal`](@ref).

### See also
* MINPACK-2: [`opk_sgqt`](@ref), [`opk_destsv`](@ref).
"""
function opk_sestsv(n, r, ldr, z)
    @ccall libopk.opk_sestsv(n::opk_index, r::Ptr{Cfloat}, ldr::opk_index, z::Ptr{Cfloat})::Cfloat
end

"""
    opk_blas_order

Matrix storage order.

| Enumerator               | Note                                           |
| :----------------------- | :--------------------------------------------- |
| OPK\\_BLAS\\_ROW\\_MAJOR | Matrix elements stored in row major order.     |
| OPK\\_BLAS\\_COL\\_MAJOR | Matrix elements stored in column major order.  |
"""
@enum opk_blas_order::UInt32 begin
    OPK_BLAS_ROW_MAJOR = 101
    OPK_BLAS_COL_MAJOR = 102
end

"""
    opk_blas_trans

Matrix transpose.

| Enumerator                | Note                      |
| :------------------------ | :------------------------ |
| OPK\\_BLAS\\_NO\\_TRANS   | Do not transpose matrix.  |
| OPK\\_BLAS\\_TRANS        | Transpose matrix.         |
| OPK\\_BLAS\\_CONJ\\_TRANS | Conjugate transpose.      |
"""
@enum opk_blas_trans::UInt32 begin
    OPK_BLAS_NO_TRANS = 111
    OPK_BLAS_TRANS = 112
    OPK_BLAS_CONJ_TRANS = 113
end

"""
    opk_blas_uplo

Part of a triangular matrix to use.

| Enumerator         | Note                        |
| :----------------- | :-------------------------- |
| OPK\\_BLAS\\_UPPER | Use upper triangular part.  |
| OPK\\_BLAS\\_LOWER | Use lower triangular part.  |
"""
@enum opk_blas_uplo::UInt32 begin
    OPK_BLAS_UPPER = 121
    OPK_BLAS_LOWER = 122
end

"""
    opk_blas_diag

Is diagonal unit?

| Enumerator              | Note                |
| :---------------------- | :------------------ |
| OPK\\_BLAS\\_NON\\_UNIT | Non unit diagonal.  |
| OPK\\_BLAS\\_UNIT       | Unit diagonal.      |
"""
@enum opk_blas_diag::UInt32 begin
    OPK_BLAS_NON_UNIT = 131
    OPK_BLAS_UNIT = 132
end

"""
    opk_blas_side

Side of matrix multiply.

| Enumerator         | Note             |
| :----------------- | :--------------- |
| OPK\\_BLAS\\_LEFT  | Left multiply.   |
| OPK\\_BLAS\\_RIGHT | Right multiply.  |
"""
@enum opk_blas_side::UInt32 begin
    OPK_BLAS_LEFT = 141
    OPK_BLAS_RIGHT = 142
end

"""
    opk_damax(n, x, incx)

Maximum absolute value of a double precision vector.

Get maximum absolute value of a vector. This is also the infinite norm of the vector.

### Parameters
* `n`: - The number of elements to consider in `x`.
* `x`: - A vector of `n*incx` values.
* `incx`: - The index increment (greater or equal 1).
### Returns
The maximum absolute value among the `n` elements of `x`.
### See also
[`opk_samax`](@ref), [`opk_idamax`](@ref).
"""
function opk_damax(n, x, incx)
    @ccall libopk.opk_damax(n::opk_index, x::Ptr{Cdouble}, incx::opk_index)::Cdouble
end

"""
    opk_samax(n, x, incx)

Maximum absolute value of a single precision vector.

### See also
[`opk_damax`](@ref), [`opk_isamax`](@ref).
"""
function opk_samax(n, x, incx)
    @ccall libopk.opk_samax(n::opk_index, x::Ptr{Cfloat}, incx::opk_index)::Cfloat
end

"""
    opk_dasum(n, x, incx)

Sum of the absolute values of a vector.

This function computes the sum of the absolute values of a vector. This is also the L-1 norm of the vector.

### Parameters
* `n`: - The number of elements to consider in `x`.
* `x`: - A vector of `n*incx` values.
* `incx`: - The index increment (greater or equal 1).
### Returns
Returns the sum of the absolute values of `x`.
### See also
[`opk_sasum`](@ref).
"""
function opk_dasum(n, x, incx)
    @ccall libopk.opk_dasum(n::opk_index, x::Ptr{Cdouble}, incx::opk_index)::Cdouble
end

"""
    opk_sasum(n, x, incx)

Sum of the absolute values of a vector.

### See also
[`opk_dasum`](@ref).
"""
function opk_sasum(n, x, incx)
    @ccall libopk.opk_sasum(n::opk_index, x::Ptr{Cfloat}, incx::opk_index)::Cfloat
end

"""
    opk_daxpy(n, a, x, incx, y, incy)

Linear combination of two vectors.

This function stores `a*x[ix] + y[iy]` into `y[iy]`. The index increments may be negative to consider elements in reverse order. The code is optimized for `a = +/-1` and `a = 0`.

### Parameters
* `n`: - The number of elements to consider in `x` and `y`.
* `a`: - The scalar factor.
* `x`: - A vector of `n*|incx|` values.
* `incx`: - The index increment for `x`.
* `y`: - A vector of `n*|incy|` values.
* `incy`: - The index increment for `y`.
### See also
[`opk_saxpy`](@ref).
"""
function opk_daxpy(n, a, x, incx, y, incy)
    @ccall libopk.opk_daxpy(n::opk_index, a::Cdouble, x::Ptr{Cdouble}, incx::opk_index, y::Ptr{Cdouble}, incy::opk_index)::Cvoid
end

"""
    opk_saxpy(n, a, x, incx, y, incy)

Linear combination of two vectors.

### See also
[`opk_daxpy`](@ref).
"""
function opk_saxpy(n, a, x, incx, y, incy)
    @ccall libopk.opk_saxpy(n::opk_index, a::Cfloat, x::Ptr{Cfloat}, incx::opk_index, y::Ptr{Cfloat}, incy::opk_index)::Cvoid
end

"""
    opk_dcopy(n, x, incx, y, incy)

Copy a vector into another one.

This function copies `n` elements of vectors `x` to vector `y`. The index increments may be negative to consider elements in reverse order.

### Parameters
* `n`: - The number of elements to consider in `x` and `y`.
* `x`: - A vector of `n*|incx|` values.
* `incx`: - The index increment for `x`.
* `y`: - A vector of `n*|incy|` values.
* `incy`: - The index increment for `y`.
### See also
[`opk_scopy`](@ref), [`opk_dswap`](@ref).
"""
function opk_dcopy(n, x, incx, y, incy)
    @ccall libopk.opk_dcopy(n::opk_index, x::Ptr{Cdouble}, incx::opk_index, y::Ptr{Cdouble}, incy::opk_index)::Cvoid
end

"""
    opk_scopy(n, x, incx, y, incy)

Copy a vector into another one.

### See also
[`opk_dcopy`](@ref), [`opk_sswap`](@ref).
"""
function opk_scopy(n, x, incx, y, incy)
    @ccall libopk.opk_scopy(n::opk_index, x::Ptr{Cfloat}, incx::opk_index, y::Ptr{Cfloat}, incy::opk_index)::Cvoid
end

"""
    opk_ddot(n, x, incx, y, incy)

Dot product of two vectors.

This function computes the dot product of two vectors. The index increments may be negative to consider elements in reverse order.

### Parameters
* `n`: - The number of elements to consider in `x` and `y`.
* `x`: - A vector of `n*|incx|` values.
* `incx`: - The index increment for `x`.
* `y`: - A vector of `n*|incy|` values.
* `incy`: - The index increment for `y`.
### Returns
The dot product of `x` by `y`.
### See also
[`opk_sdot`](@ref).
"""
function opk_ddot(n, x, incx, y, incy)
    @ccall libopk.opk_ddot(n::opk_index, x::Ptr{Cdouble}, incx::opk_index, y::Ptr{Cdouble}, incy::opk_index)::Cdouble
end

"""
    opk_sdot(n, x, incx, y, incy)

Dot product of two vectors.

### See also
[`opk_ddot`](@ref).
"""
function opk_sdot(n, x, incx, y, incy)
    @ccall libopk.opk_sdot(n::opk_index, x::Ptr{Cfloat}, incx::opk_index, y::Ptr{Cfloat}, incy::opk_index)::Cfloat
end

"""
    opk_dnrm2(n, x, incx)

Euclidean norm of a vector.

This function computes the Euclidean norm of a vector, avoiding overflows. This is also the L-2 norm of the vector.

sqrt(x[0]*x[0] + x[incx]*x[incx] + x[2*incx]*x[2*incx] + ...).

### Parameters
* `n`: - The number of elements to consider in `x`.
* `x`: - A vector of `n*incx` values.
* `incx`: - The index increment (greater or equal 1).
### Returns
The Euclidean norm of vector `x`:
### See also
[`opk_snrm2`](@ref).
"""
function opk_dnrm2(n, x, incx)
    @ccall libopk.opk_dnrm2(n::opk_index, x::Ptr{Cdouble}, incx::opk_index)::Cdouble
end

"""
    opk_snrm2(n, x, incx)

Euclidean norm of a vector.

### See also
[`opk_dnrm2`](@ref).
"""
function opk_snrm2(n, x, incx)
    @ccall libopk.opk_snrm2(n::opk_index, x::Ptr{Cfloat}, incx::opk_index)::Cfloat
end

"""
    opk_dscal(n, a, x, incx)

Scaling of a vector.

This function scales a vector by a scalar, the operation is done in place. The `n` elements of `x` get multiplied by `a`. Does nothing if `n` or `incx` are less than 1. The code is optimized for scalar `a = +/-1` and `a = 0`.

Parameters:

### Parameters
* `n`: - The number of elements to consider in `x`.
* `a`: - The scalar factor.
* `x`: - A vector of `n*incx` values.
* `incx`: - The index increment (greater or equal 1).
### See also
[`opk_sscal`](@ref).
"""
function opk_dscal(n, a, x, incx)
    @ccall libopk.opk_dscal(n::opk_index, a::Cdouble, x::Ptr{Cdouble}, incx::opk_index)::Cvoid
end

"""
    opk_sscal(n, a, x, incx)

Scaling of a vector.

### See also
[`opk_dscal`](@ref).
"""
function opk_sscal(n, a, x, incx)
    @ccall libopk.opk_sscal(n::opk_index, a::Cfloat, x::Ptr{Cfloat}, incx::opk_index)::Cvoid
end

"""
    opk_dsum(n, x, incx)

Sum of the values of a vector.

This function computes the sum of the values of a vector.

### Parameters
* `n`: - The number of elements to consider in `x`.
* `x`: - A vector of `n*incx` values.
* `incx`: - The index increment (greater or equal 1).
### Returns
Returns the sum of the values of `x`.
### See also
[`opk_ssum`](@ref), [`opk_dasum`](@ref).
"""
function opk_dsum(n, x, incx)
    @ccall libopk.opk_dsum(n::opk_index, x::Ptr{Cdouble}, incx::opk_index)::Cdouble
end

"""
    opk_ssum(n, x, incx)

Sum of the absolute values of a vector.

### See also
[`opk_dsum`](@ref), [`opk_sasum`](@ref).
"""
function opk_ssum(n, x, incx)
    @ccall libopk.opk_ssum(n::opk_index, x::Ptr{Cfloat}, incx::opk_index)::Cfloat
end

"""
    opk_dswap(n, x, incx, y, incy)

Exchanging contents of two vectors.

This function exchanges `n` elements of vectors `x` and `y`. The index increments may be negative to consider elements in reverse order.

### Parameters
* `n`: - The number of elements to consider in `x` and `y`.
* `x`: - A vector of `n*|incx|` values.
* `incx`: - The index increment for `x`.
* `y`: - A vector of `n*|incy|` values.
* `incy`: - The index increment for `y`.
### See also
[`opk_sswap`](@ref), [`opk_dcopy`](@ref).
"""
function opk_dswap(n, x, incx, y, incy)
    @ccall libopk.opk_dswap(n::opk_index, x::Ptr{Cdouble}, incx::opk_index, y::Ptr{Cdouble}, incy::opk_index)::Cvoid
end

"""
    opk_sswap(n, x, incx, y, incy)

Exchanging contents of two vectors.

### See also
[`opk_dswap`](@ref), [`opk_scopy`](@ref).
"""
function opk_sswap(n, x, incx, y, incy)
    @ccall libopk.opk_sswap(n::opk_index, x::Ptr{Cfloat}, incx::opk_index, y::Ptr{Cfloat}, incy::opk_index)::Cvoid
end

"""
    opk_dzero(n, x, incx)

Fill an array with zeros.

This function fills the `n` elements of vector `x` by step of `incx` with zeros.

### Parameters
* `n`: - The number of elements to consider in `x`.
* `x`: - A vector of `n*incx` values.
* `incx`: - The index increment (greater or equal 1).
### See also
[`opk_szero`](@ref).
"""
function opk_dzero(n, x, incx)
    @ccall libopk.opk_dzero(n::opk_index, x::Ptr{Cdouble}, incx::opk_index)::Cvoid
end

"""
    opk_szero(n, x, incx)

Fill an array with zeros.

This function fills the `n` elements of vector `x` by step of `incx` with zeros.

### Parameters
* `n`: - The number of elements to consider in `x`.
* `x`: - A vector of `n*incx` values.
* `incx`: - The index increment (greater or equal 1).
### See also
[`opk_dzero`](@ref).
"""
function opk_szero(n, x, incx)
    @ccall libopk.opk_szero(n::opk_index, x::Ptr{Cfloat}, incx::opk_index)::Cvoid
end

"""
    opk_idamax(n, x, incx)

Get index of maximum absolute value of a vector.

This function returns the index of the maximum absolute value of a vector. Following FORTRAN conventions, indices are 1-based. If `n = 0` or `incx = 0`, the returned value is 0.

### Parameters
* `n`: - The number of elements to consider in `x`.
* `x`: - A vector of `n*incx` values.
* `incx`: - The index increment (greater or equal 1).
### Returns
Returns the 1-based index of maximum absolute value of `x`. Returns 0 if `n` = 0 or `incx` = 0.
### See also
[`opk_isamax`](@ref), [`opk_damax`](@ref).
"""
function opk_idamax(n, x, incx)
    @ccall libopk.opk_idamax(n::opk_index, x::Ptr{Cdouble}, incx::opk_index)::opk_index
end

"""
    opk_isamax(n, x, incx)

Get index of maximum absolute value of a vector.

### See also
[`opk_idamax`](@ref), [`opk_samax`](@ref).
"""
function opk_isamax(n, x, incx)
    @ccall libopk.opk_isamax(n::opk_index, x::Ptr{Cfloat}, incx::opk_index)::opk_index
end

"""
    opk_dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

Matrix-vector operation.

This function performs one of the matrix-vector operations

y := alpha A.x + beta y,

or

y := alpha A'.x + beta y,

where `alpha` and `beta` are scalars, `x` and `y` are vectors and `A` is an `m` by `n` matrix.

Parameters:

### Parameters
* `trans`: - Transpose flag, specifies the matrix multiplication to be performed as follows: `OPK_BLAS_NO_TRANS` for <tt>A.x</tt>, or `OPK_BLAS_TRANS` for <tt>A'.x</tt>.
* `m`: - The number of rows of matrix `A`.
* `n`: - The number of columns of matrix `A`.
* `alpha`: - The first scalar parameter.
* `a`: - Matrix of dimensions `(dla,n)`.
* `lda`: - The leading dimension of `a`; `lda >= max(1,m)`.
* `x`: - The source vector of length at least `1+(n-1)*abs(incx)` when `trans = OPK\\_BLAS\\_NO\\_TRANS` or `1+(m-1)*abs(incx)` otherwise.
* `incx`: - The index increment for vector `x` (must be non-zero).
* `beta`: - The second scalar parameter.
* `y`: - The destination vector of length at least `1+(m-1)*abs(incy)` when `trans = OPK\\_BLAS\\_NO\\_TRANS` or `1+(n-1)*abs(incy)` otherwise.
* `incy`: - The index increment for vector `y` (must be non-zero).
### Returns
Non-zero result `k` means invalid `k`-th argument.
### See also
[`opk_sgemv`](@ref).
"""
function opk_dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall libopk.opk_dgemv(trans::opk_blas_trans, m::opk_index, n::opk_index, alpha::Cdouble, a::Ptr{Cdouble}, lda::opk_index, x::Ptr{Cdouble}, incx::opk_index, beta::Cdouble, y::Ptr{Cdouble}, incy::opk_index)::Cint
end

"""
    opk_sgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

Matrix-vector operation.

### See also
[`opk_dgemv`](@ref).
"""
function opk_sgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall libopk.opk_sgemv(trans::opk_blas_trans, m::opk_index, n::opk_index, alpha::Cfloat, a::Ptr{Cfloat}, lda::opk_index, x::Ptr{Cfloat}, incx::opk_index, beta::Cfloat, y::Ptr{Cfloat}, incy::opk_index)::Cint
end

"""
    opk_dtrmv(uplo, trans, diag, n, a, lda, x, incx)

Multiplication of a vector by a triangular matrix.

This function performs one of the matrix-vector operations:

x := A.x,

or

x := A'.x,

where `x` is an `n` element vector and `A` is an `n` by `n` unit, or non-unit, upper or lower triangular matrix.

\\result Non-zero result `k` means invalid `k`-th argument.

### Parameters
* `uplo`: - Specifies whether the matrix is an upper or lower triangular matrix as follows: `uplo = OPK\\_BLAS\\_UPPER` if `A` is an upper triangular matrix, and `uplo = OPK\\_BLAS\\_LOWER` if `A` is a lower triangular matrix.
* `trans`: - Specifies the operation to be performed as follows: if `trans = OPK\\_BLAS\\_NO\\_TRANS`, then <tt>x := A.x</tt>; otherwise if `trans = OPK\\_BLAS\\_TRANS` or `trans = OPK\\_BLAS\\_CONJ\\_TRANS`, then <tt>x := A'.x</tt>.
* `diag`: - Specifies whether or not `A` is unit triangular as follows: if `diag = OPK\\_BLAS\\_UNIT`, `A` is assumed to be unit triangular; else if `diag = OPK\\_BLAS\\_NON\\_UNIT`, `A` is not assumed to be unit triangular.
* `n`: - The order of the matrix `A`, must be at least zero.
* `a`: - An array of dimension `lda` by `n`. With `uplo = OPK\\_BLAS\\_UPPER`, the leading `n` by `n` upper triangular part of the array `A` must contain the upper triangular matrix and the strictly lower triangular part of `A` is not used. With `uplo = OPK\\_BLAS\\_LOWER`, the leading `n` by `n` lower triangular part of the array `A` must contain the lower triangular matrix and the strictly upper triangular part of `A` is not used. Note that when `diag = OPK\\_BLAS\\_UNIT`, the diagonal elements of `A` are not referenced either, but are assumed to be unity.
* `lda`: - The leading dimension of `A`, must be at least `max(n,1)`.
* `x`: - An array of at least `(1+(n-1)*abs(incx)` elements. Before entry, the incremented array must contain the `n` element vector `x`. On exit, it is overwritten with the tranformed vector `x`.
* `incx`: - The increment for the elements of `x`, must not be zero.
### See also
[`opk_strmv`](@ref), [`opk_dtrsv`](@ref).
"""
function opk_dtrmv(uplo, trans, diag, n, a, lda, x, incx)
    @ccall libopk.opk_dtrmv(uplo::opk_blas_uplo, trans::opk_blas_trans, diag::opk_blas_diag, n::opk_index, a::Ptr{Cdouble}, lda::opk_index, x::Ptr{Cdouble}, incx::opk_index)::Cint
end

"""
    opk_strmv(uplo, trans, diag, n, a, lda, x, incx)

Multiplication of a vector by a triangular matrix.

### See also
[`opk_dtrmv`](@ref), [`opk_strsv`](@ref).
"""
function opk_strmv(uplo, trans, diag, n, a, lda, x, incx)
    @ccall libopk.opk_strmv(uplo::opk_blas_uplo, trans::opk_blas_trans, diag::opk_blas_diag, n::opk_index, a::Ptr{Cfloat}, lda::opk_index, x::Ptr{Cfloat}, incx::opk_index)::Cint
end

"""
    opk_dtrsv(uplo, trans, diag, n, a, lda, x, incx)

Solves a tringular linear system of equations.

This function solves one of the matrix-vector operations:

A.x = b,

or

A'.x = b,

where `x` and `b` are `n` element vectors and `A` is an `n` by `n` unit, or non-unit, upper or lower triangular matrix.

No test for singularity or near-singularity is included in this routine. Such tests must be performed before calling this routine.

\\result Non-zero result `k` means invalid `k`-th argument.

### Parameters
* `uplo`: - Specifies whether the matrix is an upper or lower triangular matrix as follows: `uplo = OPK\\_BLAS\\_UPPER` if `A` is an upper triangular matrix, and `uplo = OPK\\_BLAS\\_LOWER` if `A` is a lower triangular matrix.
* `trans`: - Specifies the operation to be performed as follows: if `trans = OPK\\_BLAS\\_NO\\_TRANS`, then solves <tt>A.x = b</tt>; otherwise if `trans = OPK\\_BLAS\\_TRANS` or `trans = OPK\\_BLAS\\_CONJ\\_TRANS`, then solves <tt>A'.x = b</tt>.
* `diag`: - Specifies whether or not `A` is unit triangular as follows: if `diag = OPK\\_BLAS\\_UNIT`, `A` is assumed to be unit triangular; else if `diag = OPK\\_BLAS\\_NON\\_UNIT`, `A` is not assumed to be unit triangular.
* `n`: - The order of the matrix `A`, must be at least zero.
* `a`: - An array of dimension `lda` by `n`. With `uplo = OPK\\_BLAS\\_UPPER`, the leading `n` by `n` upper triangular part of the array `A` must contain the upper triangular matrix and the strictly lower triangular part of `A` is not used. With `uplo = OPK\\_BLAS\\_LOWER`, the leading `n` by `n` lower triangular part of the array `A` must contain the lower triangular matrix and the strictly upper triangular part of `A` is not used. Note that when `diag = OPK\\_BLAS\\_UNIT`, the diagonal elements of `A` are not referenced either, but are assumed to be unity.
* `lda`: - The leading dimension of `A`, must be at least `max(n,1)`.
* `x`: - An array of at least `(1+(n-1)*abs(incx)` elements. Before entry, the incremented array `x` must contain the `n` element vector `b`. On exit, `x` is overwritten with the solution vector `x`.
* `incx`: - The increment for the elements of `x`, must not be zero.
### See also
[`opk_strsv`](@ref), [`opk_dtrmv`](@ref).
"""
function opk_dtrsv(uplo, trans, diag, n, a, lda, x, incx)
    @ccall libopk.opk_dtrsv(uplo::opk_blas_uplo, trans::opk_blas_trans, diag::opk_blas_diag, n::opk_index, a::Ptr{Cdouble}, lda::opk_index, x::Ptr{Cdouble}, incx::opk_index)::Cint
end

"""
    opk_strsv(uplo, trans, diag, n, a, lda, x, incx)

Solves a tringular linear system of equations.

### See also
[`opk_dtrsv`](@ref), [`opk_strmv`](@ref).
"""
function opk_strsv(uplo, trans, diag, n, a, lda, x, incx)
    @ccall libopk.opk_strsv(uplo::opk_blas_uplo, trans::opk_blas_trans, diag::opk_blas_diag, n::opk_index, a::Ptr{Cfloat}, lda::opk_index, x::Ptr{Cfloat}, incx::opk_index)::Cint
end

"""
    opk_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

Performs a matrix-matrix operation.

This function performs one of the matrix-matrix operations

C := alpha op(A).op(B) + beta C,

where <tt>op(X) = X</tt> or <tt>op(X) = X'</tt>, `alpha` and `beta` are scalars, and `A`, `B` and `C` are matrices, with `op(A)` an `m` by `k` matrix, `op(B)` a `k` by `n` matrix and `C` an `m` by `n` matrix.

### Parameters
* `transa`: - Specifies the form of `op(A)` to be used in the matrix multiplication as follows: if `transa` = OPK\\_BLAS\\_NO\\_TRANS, then `op(A)` = `A`; otherwise if `trans = OPK\\_BLAS\\_TRANS` or `OPK_BLAS_CONJ_TRANS`, then <tt>op(A) = A'</tt>.
* `transb`: - Specifies the form of `op(B)` to be used in the matrix multiplication; see `transa`.
* `m`: - The number of rows of the matrix `op(A)` and of the matrix `C`, must be at least zero.
* `n`: - The number of columns of the matrix `op(B)` and of the matrix `C`, must be at least zero.
* `k`: - The number of columns of the matrix `op(A)` and the number of rows of the matrix `op(B)`, must be at least zero.
* `alpha`: - The scalar `alpha`.
* `a`: - An array of dimension `lda` by `ka`, where `ka` is `k` when `transa = OPK\\_BLAS\\_NO\\_TRANS`, and is `m` otherwise. Before entry with `transa = OPK\\_BLAS\\_NO\\_TRANS`, the leading `m` by `k` part of the array `a` must contain the matrix `A`, otherwise the leading `k` by `m` part of the array `a` must contain the matrix `A`.
* `lda`: - The first dimension of `a`. When `transa = OPK\\_BLAS\\_NO\\_TRANS` then `lda` must be at least `max(m,1)`, otherwise `lda` must be at least `max(k,1)`.
* `b`: - An array of dimension `ldb` by `kb`, where `kb` is `n` when `transb = OPK\\_BLAS\\_NO\\_TRANS`, and is `k` otherwise. Before entry with `transb = OPK\\_BLAS\\_NO\\_TRANS`, the leading `k` by `n` part of the array `b` must contain the matrix `B`, otherwise the leading `n` by `k` part of the array `b` must contain the matrix `B`.
* `ldb`: - The first dimension of `b`. When `transb = OPK\\_BLAS\\_NO\\_TRANS` then `ldb` must be at least `max(k,1)`, otherwise `ldb` must be at least `max(n,1)`.
* `beta`: - The scalar `beta`. When `beta` is supplied as zero then `C` need not be set on input.
* `c`: - An array of dimension `ldc` by `n`. Before entry, the leading `m` by `n` part of the array `c` must contain the matrix `C`, except when `beta` is zero, in which case `c` need not be set on entry. On exit, the array `c` is overwritten by the `m` by `n` matrix <tt>alpha op(A).op(B) + beta C</tt>.
* `ldc`: - The first dimension of `c`. `ldc` must be at least `max(m,1)`.
### Returns
Non-zero result `k` means invalid `k`-th argument.
### See also
[`opk_sgemm`](@ref).
"""
function opk_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    @ccall libopk.opk_dgemm(transa::opk_blas_trans, transb::opk_blas_trans, m::opk_index, n::opk_index, k::opk_index, alpha::Cdouble, a::Ptr{Cdouble}, lda::opk_index, b::Ptr{Cdouble}, ldb::opk_index, beta::Cdouble, c::Ptr{Cdouble}, ldc::opk_index)::Cint
end

"""
    opk_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

Performs a matrix-matrix operation.

### See also
[`opk_dgemm`](@ref).
"""
function opk_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    @ccall libopk.opk_sgemm(transa::opk_blas_trans, transb::opk_blas_trans, m::opk_index, n::opk_index, k::opk_index, alpha::Cfloat, a::Ptr{Cfloat}, lda::opk_index, b::Ptr{Cfloat}, ldb::opk_index, beta::Cfloat, c::Ptr{Cfloat}, ldc::opk_index)::Cint
end

"""
    opk_dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)

Performs a symmetric rank k operation.

This function performs one of the symmetric rank `k` operations

C := alpha A.A' + beta C,

or

C := alpha A'.A + beta C,

where `alpha` and `beta` are scalars, `C` is an `n` by `n` symmetric matrix and `A` is an `n` by `k` matrix in the first case and a `k` by `n` matrix in the second case.

### Parameters
* `uplo`: - Specifies whether the matrix is the upper or lower triangular part of the array `C` is to be referenced as follows. If `uplo = OPK\\_BLAS\\_UPPER`, only the upper triangular part of `C` is to be referenced. If `uplo = OPK\\_BLAS\\_LOWER`, only the lower triangular part of `C` is to be referenced.
* `trans`: - Specifies the operation to be performed as follows: <tt>C := alpha A.A' + beta C</tt>, if `trans = OPK\\_BLAS\\_NO\\_TRANS`; or <tt>C := alpha A'.A + beta C</tt>, if `trans = OPK\\_BLAS\\_TRANS` or `OPK_BLAS_CONJ_TRANS`.
* `n`: - The order of the matrix `C`; `N` must be at least zero.
* `k`: - On entry with `trans = OPK\\_BLAS\\_NO\\_TRANS`, `k` specifies the number of columns of the matrix `A`, and on entry with `trans` = OPK\\_BLAS\\_TRANS` or `OPK_BLAS_CONJ_TRANS`, `k` specifies the number of rows of the matrix `A`. `k` must be at least zero.
* `alpha`: - The scalar `alpha`.
* `a`: - Real array of dimension `lda` by `ka`, where `ka` is `k` when `trans = OPK\\_BLAS\\_NO\\_TRANS`, and is `n` otherwise. Before entry with `trans = OPK\\_BLAS\\_NO\\_TRANS`, the leading `n` by `k` part of the array `a` must contain the matrix `A`, otherwise the leading `k` by `n` part of the array `a` must contain the matrix `A`.
* `lda`: - The leading dimension of `a`. When `trans = OPK\\_BLAS\\_NO\\_TRANS`, then `lda` must be at least `max(n,1)`, otherwise `LDA` must be at least `max(k,1)`.
* `beta`: - The scalar `beta`.
* `c`: - Real array of dimension `ldc` by `n`. Before entry with `uplo = OPK\\_BLAS\\_UPPER`, the leading `n` by `n` upper triangular part of the array `c` must contain the upper triangular part of the symmetric matrix and the strictly lower triangular part of `c` is not referenced. On exit, the upper triangular part of the array `c` is overwritten by the upper triangular part of the updated matrix. Before entry with `uplo = OPK\\_BLAS\\_LOWER`, the leading `n` by `n` lower triangular part of the array `c` must contain the lower triangular part of the symmetric matrix and the strictly upper triangular part of `c` is not referenced. On exit, the lower triangular part of the array `c` is overwritten by the lower triangular part of the updated matrix.
* `ldc`: - The leading dimension of `c`. `ldc` must be at least `max(n,1)`.
### Returns
Non-zero result `k` means invalid `k`-th argument.
### See also
[`opk_ssyrk`](@ref).
"""
function opk_dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
    @ccall libopk.opk_dsyrk(uplo::opk_blas_uplo, trans::opk_blas_trans, n::opk_index, k::opk_index, alpha::Cdouble, a::Ptr{Cdouble}, lda::opk_index, beta::Cdouble, c::Ptr{Cdouble}, ldc::opk_index)::Cint
end

"""
    opk_ssyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)

Performs a symmetric rank k operation.

### See also
[`opk_dsyrk`](@ref).
"""
function opk_ssyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
    @ccall libopk.opk_ssyrk(uplo::opk_blas_uplo, trans::opk_blas_trans, n::opk_index, k::opk_index, alpha::Cfloat, a::Ptr{Cfloat}, lda::opk_index, beta::Cfloat, c::Ptr{Cfloat}, ldc::opk_index)::Cint
end

"""
    opk_dpotf2(uplo, n, a, lda)

Cholesky factorization of a real symmetric positive definite matrix.

This function computes the Cholesky factorization of a real symmetric positive definite matrix `A`.

The factorization has the form <tt>A = U'.U</tt>, if <tt>uplo = OPK\\_BLAS\\_UPPER</tt>, or <tt>A = L.L'</tt>, if <tt>uplo = OPK\\_BLAS\\_LOWER</tt> where `U` is an upper triangular matrix and `L` is lower triangular.

This is the unblocked version of the algorithm, calling Level 2 BLAS.

### Parameters
* `uplo`: - Specifies whether the upper or lower triangular part of the symmetric matrix `A` is stored. `uplo` is one of: `OPK_BLAS_UPPER` or `OPK_BLAS_LOWER`.
* `n`: - The order of the matrix `A`, must be greater of equal zero.
* `a`: - Input/output array of dimension `lda` by `n`. On entry, `a` contains the symmetric matrix `A`. If <tt>uplo = OPK\\_BLAS\\_UPPER</tt>, the leading `n` by `n` upper triangular part of `a` contains the upper triangular part of the matrix `A`, and the strictly lower triangular part of `a` is not referenced. If <tt>uplo = OPK\\_BLAS\\_LOWER</tt>, the leading `n` by `n` lower triangular part of `a` contains the lower triangular part of the matrix `A`, and the strictly upper triangular part of `a` is not referenced. On successful exit (with returned value 0), `a` contains the factor `U` or `L` from the Cholesky factorization <tt>A = U'.U</tt> or <tt>A = L.L'</tt>.
* `lda`: - The leading dimension of the array `a`. <tt>lda >= max(n,1)</tt>.
### Returns
The returned value, says `info` is: * `info = 0` - successful exit. * `info < 0` - the `k`-th argument had an illegal value, with `k = -info`. * `info > 0` - the leading minor of order `k` is not positive definite, and the factorization could not be completed, with `k = info`.
### See also
[`opk_spotf2`](@ref).
"""
function opk_dpotf2(uplo, n, a, lda)
    @ccall libopk.opk_dpotf2(uplo::opk_blas_uplo, n::opk_index, a::Ptr{Cdouble}, lda::opk_index)::opk_index
end

"""
    opk_spotf2(uplo, n, a, lda)

Cholesky factorization of a real symmetric positive definite matrix.

### See also
[`opk_dpotf2`](@ref).
"""
function opk_spotf2(uplo, n, a, lda)
    @ccall libopk.opk_spotf2(uplo::opk_blas_uplo, n::opk_index, a::Ptr{Cfloat}, lda::opk_index)::opk_index
end

"""
    opk_cg_state

Possible values of conjugate gradient state.

| Enumerator              | Note                                                     |
| :---------------------- | :------------------------------------------------------- |
| OPK\\_CG\\_ERROR        | error                                                    |
| OPK\\_CG\\_START        | start with no initial variable (all zero)                |
| OPK\\_CG\\_RESTART      | start or restart with initial variables                  |
| OPK\\_CG\\_NEWX         | new step taken                                           |
| OPK\\_CG\\_PRECOND      | caller has to to compute z = Q.r                         |
| OPK\\_CG\\_AP           | caller has to to compute q = A.p                         |
| OPK\\_CG\\_FINISH       | convergence or rounding errors prevent further progress  |
| OPK\\_CG\\_NON\\_CONVEX | non-positive definitiveness has been detected            |
| OPK\\_CG\\_TRUNCATED    | step truncated at trust region boundary                  |
"""
@enum opk_cg_state::Int32 begin
    OPK_CG_ERROR = -1
    OPK_CG_START = 0
    OPK_CG_RESTART = 1
    OPK_CG_NEWX = 2
    OPK_CG_PRECOND = 3
    OPK_CG_AP = 4
    OPK_CG_FINISH = 5
    OPK_CG_NON_CONVEX = 6
    OPK_CG_TRUNCATED = 7
end

"""
    opk_dplcg(n, p, q, r, x, z, rho, state)

Preconditioned linear conjugate gradient (double precision).

Iteratively solve a linear system <tt>A.x = b</tt> where `A` is a symmetric positive definite matrix by preconditioned conjugate gradient method with reverse communication.

- `OPK_CG_NEWX`: `x` stores the current solution, `r` stores the current residuals, the change in parameters is equal to <tt>rho[2]*p</tt> and the corresponding residuals change is equal to <tt>-rho[2]*q</tt>. The caller can decide to terminate the iterations, to pursue the iterations or to restart the algorithm.

- `OPK_CG_AP`: The caller must compute the matrix vector multiplication <tt>q = A.p</tt> (*i.e.*, store into vector `q` the result of multiplying vector `p` by the matrix `A`) and call the conjugate gradient routine again.

- `OPK_CG_PRECOND`: The caller must compute and store the preconditioned residuals into `z` (given the current residuals which are in `r`) and call the conjugate gradient routine again.

- `OPK_CG_FINISH`: No further progress are possible either because the exact solution has been found or because rounding errors prevent further progress.

- `OPK_CG_ERROR`: An error has been detected by the algorithm: invalid parameter or non-positive definitiveness of the matrix `A` or of the preconditioner.

For consistency reasons, `OPK_CG_NEWX` is returned after internal initialization, *i.e.*, `x` is just `x0` and `r` is just <tt>r = b - A.x0</tt>.

### Example

In the following piece of code, we assume that the arrays `a` and `c` store respectively the coefficients of the matrix `A` and of the preconditioner.

for (i = 0; i < n; ++i) { x[i] = ...; // initial set of parameters } for (i = 0; i < n; ++i) { r[i] = b[i]; // copy b into r } state = OPK\\_CG\\_RESTART; for (;;) { [`opk_dplcg`](@ref)(n, p, q, r, x, z, rho, &state); if (state == OPK\\_CG\\_AP) { for (i = 0; i < n; ++i) { double s = 0.0; for (j = 0; j < n; ++j) { s += a[i*n + j]*p[j]; } q[i] = s; } } else if (state == OPK\\_CG\\_PRECOND) { for (i = 0; i < n; ++i) { double s = 0.0; for (j = 0; j < n; ++j) { s += c[i*n + j]*r[j]; } z[i] = s; } } else if (state == OPK\\_CG\\_NEWX) { double s = 0.0; for (i = 0; i < n; ++i) { s += r[i]*r[i]; } if (s < 1e-3) { break; // convergence } } else { // unexpected (must be an error) fprintf(stderr, "error in PLCG"); break; } }

### Parameters
* `n`: - The number of variables.
* `p`: - A work array of size `n`; see parameter `state` for explanations.
* `q`: - A work array of size `n`; on return with `state` set to `OPK_CG_AP`, the caller must compute the matrix vector multiplication <tt>q=A.p</tt> (i.e. store into vector `q` the result of multiplying vector `p` by the matrix `A`); on return with `state` set to `OPK_CG_NEWX`, `q` is filled with the unscaled residuals step.
* `r`: - An array of size `n`; on initialization or restart (with `state` set to `OPK_CG_START` or `OPK_CG_RESTART`), `r` stores the right hand side vector `b`; otherwise, `r` stores the current residuals.
* `x`: - An array of size `n` to store the unknowns. On initialization with given parameter values or restart, i.e. with `state` set to `OPK_CG_RESTART`, `x` stores the initial unknowns `x0`; on initial entry with `state` set to `OPK_CG_START`, the contents of `x` is ignored (`x` will be filled with zeros); otherwise, `x` stores the current solution.
* `z`: - Optional work array of size `n`; on return with `state` set to `OPK_CG_PRECOND`, the caller must compute and store the preconditioned residuals into `z` given the current residuals which are in `r`. If no preconditioning is to be used, `z` must be set to `NULL` (in which case, a `state` with value `OPK_CG_PRECOND` will never get returned). If not `NULL`, `z` can be the same array as `q` to save memory.
* `rho`: - A 4-element array used to store the dot product <tt>r'.z</tt> of the current and previous iterations and to store some parameters of the algorithm. <tt>rho[0]</tt> is the former value of <tt>rho[1]</tt>; <tt>rho[1]</tt> is the dot product of the residuals `r` by the (preconditioned) residuals `z`; <tt>rho[2]</tt> is the optimal step size (`alpha`); <tt>rho[3]</tt> is the weight of the former search direction (`beta`).
* `state`: - Address of integer variable to store the current stage of the algorithm (must not be altered by the caller between calls except to restart the algorithm). On initialization or restart of the conjugate gradient iterations, `state` must be set to `OPK_CG_START` or `OPK_CG_RESTART` and `r` must be set to the right hand side vector, that is: <tt>r = b</tt>; if `state` is `OPK_CG_START`, `x` is filled with zeros by the PLCG routine; if `state` is `OPK_CG_RESTART`, `x` must be set with some initial values of the parameters, that is: <tt>x = x0</tt>. Upon return of the function, the value stored in `state` is one of:
### See also
[`opk_dlcg`](@ref), [`opk_splcg`](@ref).
"""
function opk_dplcg(n, p, q, r, x, z, rho, state)
    @ccall libopk.opk_dplcg(n::opk_index, p::Ptr{Cdouble}, q::Ptr{Cdouble}, r::Ptr{Cdouble}, x::Ptr{Cdouble}, z::Ptr{Cdouble}, rho::Ptr{Cdouble}, state::Ptr{opk_cg_state})::Cvoid
end

"""
    opk_splcg(n, p, q, r, x, z, rho, state)

Preconditioned linear conjugate gradient (single precision).

### See also
[`opk_slcg`](@ref)(), [`opk_dplcg`](@ref)() for the meaning of the arguments.
"""
function opk_splcg(n, p, q, r, x, z, rho, state)
    @ccall libopk.opk_splcg(n::opk_index, p::Ptr{Cfloat}, q::Ptr{Cfloat}, r::Ptr{Cfloat}, x::Ptr{Cfloat}, z::Ptr{Cfloat}, rho::Ptr{Cfloat}, state::Ptr{opk_cg_state})::Cvoid
end

"""
    opk_dlcg(n, p, q, r, x, rho, state)

Linear conjugate gradient (double precision).

This function implements reverse communication linear conjugate gradient without predconditioner for double precision floating point. See [`opk_dplcg`](@ref)() for the meaning of the arguments and [`opk_slcg`](@ref)() for a single precision version.

### See also
[`opk_slcg`](@ref), [`opk_dplcg`](@ref).
"""
function opk_dlcg(n, p, q, r, x, rho, state)
    @ccall libopk.opk_dlcg(n::opk_index, p::Ptr{Cdouble}, q::Ptr{Cdouble}, r::Ptr{Cdouble}, x::Ptr{Cdouble}, rho::Ptr{Cdouble}, state::Ptr{opk_cg_state})::Cvoid
end

"""
    opk_slcg(n, p, q, r, x, rho, state)

Linear conjugate gradient (single precision).

This function implements reverse communication linear conjugate gradient without predconditioner for single precision floating point. See [`opk_dplcg`](@ref)() for the meaning of the arguments and [`opk_dlcg`](@ref)() for a double precision version.

### See also
[`opk_dlcg`](@ref)(), [`opk_dplcg`](@ref)().
"""
function opk_slcg(n, p, q, r, x, rho, state)
    @ccall libopk.opk_slcg(n::opk_index, p::Ptr{Cfloat}, q::Ptr{Cfloat}, r::Ptr{Cfloat}, x::Ptr{Cfloat}, rho::Ptr{Cfloat}, state::Ptr{opk_cg_state})::Cvoid
end

"""
    opk_dtrcg(n, p, q, r, x, z, delta, rho, state)

Trust region conjugate gradient (double precision).

This function implements a reverse communication version of a trust region linear conjugate gradient algorithm with optional preconditioning. The trust-region (or truncated) conjugate gradient method is due to Steihaug (see References below). This version is for double precision variables, see [`opk_strcg`](@ref)() for a single precision version. See [`opk_dplcg`](@ref)() for an example of using reverse communication.

- `OPK_CG_NEWX`: `x` stores the current solution, `r` stores the current residuals, the change in parameters is equal to <tt>rho[2]*p</tt> and the corresponding residuals change is equal to <tt>-rho[2]*q</tt>. The caller can decide to terminate the iterations, to pursue the iterations or to restart the algorithm.

- `OPK_CG_PRECOND`: The caller must compute and store the preconditioned residuals into `z` (given the current residuals which are in `r`) and call the conjugate gradient routine again.

- `OPK_CG_AP`: The caller must compute the matrix vector multiplication <tt>q=A.p</tt> (i.e., store into vector `q` the result of multiplying vector `p` by the matrix `A`) and call the conjugate gradient routine again.

- `OPK_CG_FINISH`: No further progress are possible either because the exact solution has been found or because rounding errors prevent further progress. The Euclidean norm of `x` is less than `delta`.

- `OPK_CG_NON_CONVEX`: A further conjugate gradient step cannot be taken because non positive definitiveness of the preconditioner has been detected. The Euclidean norm of `x` is less than `delta`.

- `OPK_CG_TRUNCATED`: A truncated gradient step has been taken. The Euclidean norm of `x` is equal to `delta`. No further iterations should be taken (unless with a larger `delta` and `state` set to `OPK_CG_RESTART`).

- `OPK_CG_ERROR`: An error has been detected by the algorithm: invalid parameter, or corrupted workspace `rho`.

For consistency reasons, `OPK_CG_NEWX` is returned after internal initialization, i.e., `x` is just `x0` and `r` is just <tt>r = b - A.x0<tt>.

It is a bad idea to restart the algorithm with an initial `x` with norm greater than `delta`, in this case, `state` is set to `OPK_CG_TRUNCATED` and the initial `x` is rescaled to have a Euclidean norm equals to `delta`.

### References

- J. Nocedal and S. J. Wright, "Numerical Optimization", Springer Verlag, 2006.

- T. Steihaug, "The conjugate gradient method and trust regions in large scale optimization", SIAM Journal on Numerical Analysis, vol. **20**, pp. 626-637, 1983.

### Parameters
* `n`: - The number of variables.
* `p`: - A work array of size `n`; see parameter `state` for explanations.
* `q`: - A work array of size `n`; on return with `state` set to `OPK_CG_AP`, the caller must compute the matrix vector multiplication <tt>q=A.p</tt> (i.e., store into vector `q` the result of multiplying vector `p` by the matrix `A`); on return with `state` set to `OPK_CG_NEWX`, `q` is filled with the unscaled residuals step.
* `r`: - An array of size `n`; on initialization or restart (i.e., with `state` set to `OPK_CG_START` or `OPK_CG_RESTART`), `r` stores the right hand side vector `b`; otherwise, `r` stores the current residuals.
* `x`: - An array of size `n` to store the unknowns. On initialization with given parameter values or restart, i.e. with `state` set to `OPK_CG_RESTART`, `x` stores the initial unknowns `x0`; on initial entry with `state` set to `OPK_CG_START`, the contents of `x` is ignored (`x` will be filled with zeroes); otherwise, `x` stores the current solution.
* `z`: - Optional work array of size `n`; on return with `state` set to `OPK_CG_PRECOND`, the caller must compute and store the preconditioned residuals into `z` given the current residuals which are in `r`. If no preconditioning is to be used, `z` must be set to `NULL` (in which case, a `state` with value `OPK_CG_PRECOND` will never get returned).
* `delta`: - The maximum length (Euclidean norm) of vector `x`. This value must not be changed once the algorithm is started.
* `rho`: - A 5-element array used to store the dot product <tt>r'.z</tt> of the current and previous iterations and to store some parameters of the algorithm. <tt>rho[0]</tt> is the former value of <tt>rho[1]</tt>; <tt>rho[1]</tt> is the dot product of the residuals `r` by the (preconditioned) residuals `z`; <tt>rho[2]</tt> is the optimal step size (`alpha`); <tt>rho[3]</tt> is the weight of the former search direction (`beta`); <tt>rho[4]</tt> is the Euclidean norm of `x`.
* `state`: - Address of integer variable to store the current stage of the algorithm (must not be altered by the caller between calls except to restart the algorithm). On initialization or restart of the conjugate gradient iterations, `state` must be set to `OPK_CG_START` or `OPK_CG_RESTART` and `r` must be set to the right hand side vector, that is: <tt>r=b</tt>; if `state` is `OPK_CG_START`, `x` is filled with zeroes by the conjugate gradient routine; if `state` is `OPK_CG_RESTART`, `x` must be set with some initial values of the parameters, that is: <tt>x=x0</tt>. Upon return of the function, the value stored in `state` is one of:
### See also
[`opk_strcg`](@ref), [`opk_dplcg`](@ref).
"""
function opk_dtrcg(n, p, q, r, x, z, delta, rho, state)
    @ccall libopk.opk_dtrcg(n::opk_index, p::Ptr{Cdouble}, q::Ptr{Cdouble}, r::Ptr{Cdouble}, x::Ptr{Cdouble}, z::Ptr{Cdouble}, delta::Cdouble, rho::Ptr{Cdouble}, state::Ptr{opk_cg_state})::Cvoid
end

"""
    opk_strcg(n, p, q, r, x, z, delta, rho, state)

Trust region conjugate gradient (single precision).

This function implements reverse communication trust region linear conjugate gradient with optional predconditioning. This version is for single precision variables, see [`opk_strcg`](@ref)() for a double precision version and for explanations.

### See also
[`opk_dtrcg`](@ref), [`opk_dplcg`](@ref).
"""
function opk_strcg(n, p, q, r, x, z, delta, rho, state)
    @ccall libopk.opk_strcg(n::opk_index, p::Ptr{Cfloat}, q::Ptr{Cfloat}, r::Ptr{Cfloat}, x::Ptr{Cfloat}, z::Ptr{Cfloat}, delta::Cfloat, rho::Ptr{Cfloat}, state::Ptr{opk_cg_state})::Cvoid
end

"""
    opk_allocate_object(finalize, nbytes)

Allocate a new (empty) object.

This function allocates enough bytes of memory to store a type derived from the basic object type, initializes the basic type and zero-fill the extra bytes of allocated memory.

The caller of this function holds a reference on the returned object. When the object is no longer needed by the caller, he/she has to call `[`opk_drop_object`](@ref)()` to release this reference.

When the object is no longer in use (its last reference has been dropped), it is effectively destroyed by first applying the `finalize()` method on it (unless it is NULL) and then freeing the dynamic memory where the object was stored.

The typical usage consists in building a derived type as follows: ~~~~~~~~~~{.c} typedef struct { [`opk_object`](@ref) base; // base type (must be the first member) ...; // other members } sub\\_type; ~~~~~~~~~~ and provide a destructor and a constructor like: ~~~~~~~~~~{.c} static void finalize\\_sub\\_type([`opk_object`](@ref)* self) { ...; // do whatever cleanup is needed }

sub\\_type* create\\_sub\\_type(args...) { sub\\_type* obj = (sub\\_type*)[`opk_allocate_object`](@ref)(finalize\\_sub\\_type, sizeof(sub\\_type)); ...; // any other initialization return obj; } ~~~~~~~~~~ where `finalize\\_sub\\_type()` is in charge of releasing any specific ressources of the object. Note that this function must be able to deal with a partially initialized object, although this is simplified by the fact that the non-basic parts of the object are intially zero-filled. The memory allocated by `[`opk_allocate_object`](@ref)()` is automatically freed and must not be freed by the `finalize()` method.

The object model of OptimPack is very simple but has some drawbacks (C is not C++): casts are needed to force the proper type of an object pointer, memory management by reference counting forbid circular references.

### Parameters
* `finalize`: - If not NULL, method to call when the object is no longer referenced. This method is in charge of freeing object ressources specific to the sub-type. This method is *never* called with a NULL argument.
* `nbytes`: - Number of bytes to allocate for the object. This value may be adjusted so that at least a basic object can be stored.
### Returns
A new basic object with one reference set or NULL in case of error.
"""
function opk_allocate_object(finalize, nbytes)
    @ccall libopk.opk_allocate_object(finalize::Ptr{Cvoid}, nbytes::Csize_t)::Ptr{opk_object}
end

function opk_allocate_vector_space(vops, nvariables, nbytes)
    @ccall libopk.opk_allocate_vector_space(vops::Ptr{opk_vspace_operations}, nvariables::opk_index, nbytes::Csize_t)::Ptr{opk_vspace}
end

"""
    opk_allocate_vector(vspace, nbytes)

Allocate a new vector object.

This utility function is designed for constructors of vector sub-types, that is the `create()` method of a vector space table of operations. To create vectors, the end-users use the `[`opk_vcreate`](@ref)()` function.

This function allocates dynamic memory and instanciates it with as a basic vector type. The storage size is adjusted to be at least sufficient for a basic vector of type [`opk_vector`](@ref). As part of its initialization, the returned vector holds a reference on its vector space. The returned vector is managed as any other OptimPack object; in particular, the function `[`opk_drop_object`](@ref)()` or the macro `[`OPK_DROP`](@ref)()` has to be called to drop the reference on the vector.

### Parameters
* `vspace`: - The owner of the vector.
* `nbytes`: - The minimum number of bytes to allocate.
### Returns
A new vector object instantiated as a basic vector type; `NULL` in case of error.
"""
function opk_allocate_vector(vspace, nbytes)
    @ccall libopk.opk_allocate_vector(vspace::Ptr{opk_vspace}, nbytes::Csize_t)::Ptr{opk_vector}
end

"""
    opk_allocate_convexset(space, finalize, projvar, projdir, freevar, steplim, nbytes)

Allocate a new instance of a convex set.
"""
function opk_allocate_convexset(space, finalize, projvar, projdir, freevar, steplim, nbytes)
    @ccall libopk.opk_allocate_convexset(space::Ptr{opk_vspace}, finalize::Ptr{Cvoid}, projvar::Ptr{Cvoid}, projdir::Ptr{Cvoid}, freevar::Ptr{Cvoid}, steplim::Ptr{Cvoid}, nbytes::Csize_t)::Ptr{opk_convexset}
end

function opk_allocate_line_search(ops, size)
    @ccall libopk.opk_allocate_line_search(ops::Ptr{opk_lnsrch_operations}, size::Csize_t)::Ptr{opk_lnsrch}
end

function opk__set_nlcg_status(opt, status)
    @ccall libopk.opk__set_nlcg_status(opt::Ptr{opk_nlcg}, status::opk_status)::Cvoid
end

function opk__set_vmlmb_status(opt, status)
    @ccall libopk.opk__set_vmlmb_status(opt::Ptr{opk_vmlmb}, status::opk_status)::Cvoid
end

function opk_allocate_operator(ops, inpspace, outspace, size)
    @ccall libopk.opk_allocate_operator(ops::Ptr{opk_operator_operations}, inpspace::Ptr{opk_vspace}, outspace::Ptr{opk_vspace}, size::Csize_t)::Ptr{opk_operator}
end

# Skipping MacroDefinition: OPK_STATUS_ ( a , b , c ) b = a ,

# Skipping MacroDefinition: const OPK_STATUS_LIST_ = ((((((((((((((((((((((((((((((((((((((((((((((((((((((((OPK_STATUS_(0, OPK_SUCCESS, "Success"))(OPK_STATUS_))(1, OPK_INVALID_ARGUMENT, "Invalid argument"))(OPK_STATUS_))(2, OPK_INSUFFICIENT_MEMORY, "Insufficient memory"))(OPK_STATUS_))(3, OPK_ILLEGAL_ADDRESS, "Illegal address"))(OPK_STATUS_))(4, OPK_NOT_IMPLEMENTED, "Not implemented"))(OPK_STATUS_))(5, OPK_CORRUPTED_WORKSPACE, "Corrupted workspace"))(OPK_STATUS_))(6, OPK_BAD_SPACE, "Bad variable space"))(OPK_STATUS_))(7, OPK_OUT_OF_BOUNDS_INDEX, "Out of bounds index"))(OPK_STATUS_))(8, OPK_NOT_STARTED, "Line search not started"))(OPK_STATUS_))(9, OPK_NOT_A_DESCENT, "Not a descent direction"))(OPK_STATUS_))(10, OPK_STEP_CHANGED, "Step changed"))(OPK_STATUS_))(11, OPK_STEP_OUTSIDE_BRACKET, "Step outside bracket"))(OPK_STATUS_))(12, OPK_STPMIN_GT_STPMAX, "Lower step bound larger than upper bound"))(OPK_STATUS_))(13, OPK_STPMIN_LT_ZERO, "Minimal step length less than zero"))(OPK_STATUS_))(14, OPK_STEP_LT_STPMIN, "Step lesser than lower bound"))(OPK_STATUS_))(15, OPK_STEP_GT_STPMAX, "Step greater than upper bound"))(OPK_STATUS_))(16, OPK_FTOL_TEST_SATISFIED, "Convergence within variable tolerance"))(OPK_STATUS_))(17, OPK_GTOL_TEST_SATISFIED, "Convergence within function tolerance"))(OPK_STATUS_))(18, OPK_XTOL_TEST_SATISFIED, "Convergence within gradient tolerance"))(OPK_STATUS_))(19, OPK_STEP_EQ_STPMAX, "Step blocked at upper bound"))(OPK_STATUS_))(20, OPK_STEP_EQ_STPMIN, "Step blocked at lower bound"))(OPK_STATUS_))(21, OPK_ROUNDING_ERRORS_PREVENT_PROGRESS, "Rounding errors prevent progress"))(OPK_STATUS_))(22, OPK_NOT_POSITIVE_DEFINITE, "Operator is not positive definite"))(OPK_STATUS_))(23, OPK_BAD_PRECONDITIONER, "Preconditioner is not positive definite"))(OPK_STATUS_))(24, OPK_INFEASIBLE_BOUNDS, "Box set is infeasible"))(OPK_STATUS_))(25, OPK_WOULD_BLOCK, "Variables cannot be improved (would block)"))(OPK_STATUS_))(26, OPK_UNDEFINED_VALUE, "Undefined value"))(OPK_STATUS_))(27, OPK_TOO_MANY_EVALUATIONS, "Too many evaluations"))(OPK_STATUS_))(28, OPK_TOO_MANY_ITERATIONS, "Too many iterations")

const OPK_NLCG_FLETCHER_REEVES = 1

const OPK_NLCG_HESTENES_STIEFEL = 2

const OPK_NLCG_POLAK_RIBIERE_POLYAK = 3

const OPK_NLCG_FLETCHER = 4

const OPK_NLCG_LIU_STOREY = 5

const OPK_NLCG_DAI_YUAN = 6

const OPK_NLCG_PERRY_SHANNO = 7

const OPK_NLCG_HAGER_ZHANG = 8

const OPK_NLCG_POWELL = 1 << 8

const OPK_NLCG_SHANNO_PHUA = 1 << 9

const OPK_NLCG_DEFAULT = (OPK_NLCG_POLAK_RIBIERE_POLYAK | OPK_NLCG_POWELL) | OPK_NLCG_SHANNO_PHUA

const OPK_FMIN_BOUNDED_BY_A = 1

const OPK_FMIN_BOUNDED_BY_B = 2

const OPK_FMIN_SMOOTH = 4

end # module
