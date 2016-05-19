# TiPi.jl

**TiPi** is a **T**oolkit for **I**nverse **P**roblems and **I**maging in
Julia.  One of the main objectives of TiPi is to solve image reconstruction
problems, so TiPi is designed to deal with large number of unknowns
(*e.g.*, as many as pixels in an image).

The documentation is split in several parts:

* [Cost Functions](doc/cost.md) describes how to implement objective functions
  which are to be minimized to solve a given inverse problem;

* [Algebra](doc/algebra.md) describes linear operators and operations
  on "vectors" (that is to say the "variables" of the problem to solve);
