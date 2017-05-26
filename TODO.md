* Brent.fzero: yield the endpoints of the bracket.

* Brent.fzero and Brent.fmin: automatic bracketting of the solution:

> If the interval to consider is not bounded or only left/right bounded, the
> idea is to find a suitable interval (A,B) where at least one minimum must
> exists (if the function is continue) and start Brent's algorithm with correct
> values for X, FX, ... (in order to save some function evaluations).

* Add BigFloat to the tests.

* Uniformize the tests for automatic check.

* Add verbose mode for `conjgrad`.

* Write doc. for `bobyqa`.

* Automatically build dependencies.

* Reorganize things:
  - OptimPack.jl -> OptimPackLib.jl = Julia wrapper to use OptimPack library;
  - OptimPackNextGen.jl -> OptimPack.jl = pure Julia version of large scale
    optimization methods;
  - OptimPackNextGen.jl/Powell -> OptimPackPowell.jl = Julia wrapper to use
    Powell methods in OptimPack library;
