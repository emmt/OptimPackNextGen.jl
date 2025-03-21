# User visible changes in `OptimPackNextGen` package

## Version 0.4.3

- Faster use of `Zygote` and extend to version 0.7 of `Zygote`.

## Version 0.4.2

- Fix loading of `Zygote` extension with Julia ≤ 1.8.

## Version 0.4.1

- In Powell's methods (COBYLA, NEWUOA, and BOBYQA):
  - Preserve scaling factors from being garbage collected while calling the C
    code.
  - Variables, bounds, and scaling factors can be specified in more flexible
    ways than dense vector of C-double values.

## Version 0.4.0

- New methods `Brent.fmax` and `Brent.fmaxbrkt`.

- Methods for univariate functions `Brent.fzero`, `Brent.fmin`,
  `Brent.fminbrkt`, `BraDi.minimize`, `BraDi.maximize` `Step.minimize`,
  `Step.maximize`, and `Step.search` have changed as follows:
  - `x` and `f(x)` may have units.
  - The floating-point type for computations is not `Float64` by default but
    determined from the types of the specified numerical arguments.
  - The number of function calls has been appended to the tuple returned by
    these methods.
  - The keyword `period` of `BraDi.minimize` and `BraDi.maximize` has been
    replaced by `peridoc` which is a Boolean, the period being given by the
    distance between the extreme values of `x`.
  - `Brent.fzero` yields the 5-tuple `(x,fx,lo,hi,nf)` with `x` the estimated
    solution, `fx = f(x)` the corresponding function value, `lo` and `hi` the lower
    and upper bounds for the solution, and `nf` the number of calls to `f`.
  - `Step.minimize`, `Step.maximize`, `Step.search`, `BraDi.minimize`, and
    `BraDi.maximize` return a 5-tuple `(xm,fm,lo,hi,nf)` with `xm` the position
    of the global optimum, `fm = f(xm)` the corresponding function value, `lo`
    and `hi` the lower and upper bounds for the exact solution, and `nf` the number
    of function calls. This is similar to `fmin` and `fmax`.
  - `Step.minimize`, `Step.maximize`, and `Step.search` no longer have a
    `maxeval` keyword. The number of function evaluations thus only depends on
    the requested accuracy for the solution.

- The STEP method for finding a global minimum or maximum of an univariate
  function `f(x)` has been improved in many respects:
  - `x` and `f(x)` may have units.
  - The numerical precision used by the algorithm is automatically defined by
    the floating-point types of `x` and `f(x)`.
  - Speed-up computations (by a factor of two) by simplifying the storage of
    trials.
  - Keywords have different names: `atol` and `rtol` specify absolute and
    relative tolerances for the precision of the solution, `aboost` and
    `rboost` specify absolute and relative boost parameters to define the
    function value to aim at.


## Version 0.3.1

- Fix missing methods to handle upper bounds.


## Version 0.3.0

- Automatic-differentiation now possible with
  [`Zygote`](https://github.com/FluxML/Zygote.jl).

- Fix termination of SPG when the function value no longer change.

- Keyword `verb` in `vmlmb` and `spg` can be an integer to print information
  every `verb` iteration.

- Additional points can be specified in `fmin`.  This replaces unexported
  methods `fmin0`, `fmin1`, `fmin2`, and `fmin3` which have been suppressed.


## Version 0.1

- Use [LazyAlgebra](https://github.com/emmt/LazyAlgebra.jl) package for
  vectorized operations and linear conjugate gradient.
