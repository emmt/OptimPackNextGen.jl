# User visible changes in `OptimPackNextGen` package

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
