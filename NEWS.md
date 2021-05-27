# User visible changes in OptimPackNextGen

## Branch master
- Automatic-differentiation now possible with
  [`Zygote`](https://github.com/FluxML/Zygote.jl).
- Fix termination of SPG when the function value no longer change.
- Keyword `verb` in `vmlmb` and `spg` can be an integer to print information
  every `verb` iteration.

## Branch v0.1
- Use [LazyAlgebra](https://github.com/emmt/LazyAlgebra.jl) package for
  vectorized operations and linear conjugate gradient.
