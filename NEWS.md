# User visible changes in `OptimPackNextGen` package

- Methods for univariate functions `Brent.fzero`, `Brent.fmin`,
  `Brent.fminbrkt`, `BraDi.minimize`, and `BraDi.maximize` have changed as
  follows:
  - `x` and `f(x)` may have units.
  - The floating-point type for computations is not `Float64` by default but
    determined from the types of the specified numerical arguments.
  - The number of function calls has been appended to the tuple returned by
    these methods.


## Version 0.4.0

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
