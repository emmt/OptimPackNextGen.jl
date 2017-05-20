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
