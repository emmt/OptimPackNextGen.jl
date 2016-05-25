module Test

using TiPi.Algebra

import TiPi.Algebra: apply_direct, apply_adjoint

abstract VectorSpace

type SpaceE <: VectorSpace; name::AbstractString; end
type SpaceF <: VectorSpace; name::AbstractString; end
type SpaceG <: VectorSpace; name::AbstractString; end
type SpaceH <: VectorSpace; name::AbstractString; end

type Operator{E,F} <: LinearOperator{E,F}
    name::AbstractString
end

type BogusOperator{E,F} <: LinearOperator{E,F}
    name::AbstractString
end

import Base: show

show{E,F}(io::IO, A::Operator{E,F}) = print(io, A.name)
show{E,F}(io::IO, A::BogusOperator{E,F}) = print(io, A.name)
show{T<:VectorSpace}(io::IO, x::T) = print(io, x.name)

apply_direct{E,F}(A::Operator{E,F}, x::F) = E(A.name * "*" * x.name)
apply_adjoint{E,F}(A::Operator{E,F}, x::E) = F(A.name * "'*" * x.name)

apply_direct{E,F}(A::BogusOperator{E,F}, x::F) = F(A.name * "*" * x.name)
apply_adjoint{E,F}(A::BogusOperator{E,F}, x::E) = E(A.name * "'*" * x.name)

function runtests()
    A = Operator{SpaceE,SpaceF}("A")
    B = Operator{SpaceF,SpaceG}("B")
    C = Operator{SpaceG,SpaceH}("C")
    D = Operator{SpaceH,SpaceG}("D")
    E = Operator{SpaceG,SpaceF}("E")
    F = Operator{SpaceF,SpaceE}("F")
    I = Identity(SpaceE)
    w = SpaceE("w")
    x = SpaceF("x")
    y = SpaceG("y")
    z = SpaceH("z")

    println(A*x)
    println(A(x))
    println(A'*w)
    println(A'*I*w)
    println(A'(w))
    println(A(B(y)))
    println(A*B*y)

    println((A*B)'(w))
    println((A*B)'*w)

    println(A(B(C(z))))
    println(A*B*C*z)
    println(A(B(D'(z))))
    println(A*B*D'*z)
    println(A(E'(C(z))))
    println(A*E'*C*z)
    println(F'(E'(C(z))))
    println(F'*E'*C*z)

    println((A*B*C)'*w)
    println((A*B*C)''*z)

    Q = A*B*D'
    println(Q*z)

    Z = BogusOperator{SpaceE,SpaceF}("Z");
    #println(Z*x)
    #println(Z'*w)
    nothing
end

end
