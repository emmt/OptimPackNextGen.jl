module Test

using TiPi
importall TiPi.Algebra

abstract VectorSpace

for space in (:SpaceE, :SpaceF, :SpaceG, :SpaceH)
    @eval type $space <: VectorSpace
        name::AbstractString
        $space(name::AbstractString) = new(preserve(name))
    end
end

type TestOperator{E,F} <: LinearOperator{E,F}
    name::AbstractString
    TestOperator(name::AbstractString) = new(preserve(name))
end

type BogusOperator{E,F} <: LinearOperator{E,F}
    name::AbstractString
    BogusOperator(name::AbstractString) = new(preserve(name))
end

function preserve(str::ASCIIString)
    str = strip(str)
    if length(str) == 0 || (ismatch(r"[-+*/\\']", str) && (str[1] != '(' || str[end] != ')'))
        return "("*str*")"
    else
        return str
    end
end

import Base: show

show{E,F}(io::IO, A::TestOperator{E,F}) = print(io, A.name)
show{E,F}(io::IO, A::BogusOperator{E,F}) = print(io, A.name)
show{T<:VectorSpace}(io::IO, x::T) = print(io, x.name)

apply_direct{E,F}(A::TestOperator{E,F}, x::F) = E(A.name * "*" * x.name)
apply_adjoint{E,F}(A::TestOperator{E,F}, x::E) = F(A.name * "'*" * x.name)

vcreate{E<:VectorSpace}(x::E) = E("")

vcopy!{E<:VectorSpace}(dst::E, src::E) = (dst.name = src.name; dst)

vcombine{E<:Union{TestOperator,BogusOperator,VectorSpace}}(alpha::Real, A::E, beta::Real, B::E) =
    E(string(alpha,"*",A.name," + ",beta,"*",B.name))

apply_direct{E,F}(A::BogusOperator{E,F}, x::F) = F(A.name * "*" * x.name)
apply_adjoint{E,F}(A::BogusOperator{E,F}, x::E) = E(A.name * "'*" * x.name)

function runtests()
    A = TestOperator{SpaceE,SpaceF}("A")
    U = TestOperator{SpaceE,SpaceF}("U")
    B = TestOperator{SpaceF,SpaceG}("B")
    C = TestOperator{SpaceG,SpaceH}("C")
    D = TestOperator{SpaceH,SpaceG}("D")
    E = TestOperator{SpaceG,SpaceF}("E")
    F = TestOperator{SpaceF,SpaceE}("F")


   # I = Identity(SpaceE)
    w = SpaceE("w")
    x = SpaceF("x")
    u = SpaceF("u")
    y = SpaceG("y")
    z = SpaceH("z")

    M = A'*A
    println("M' === M with M = A'*A? ", M' === M)
    println(M*x)
    println(M*u)

    M = A'*A + 42*F*F'
    println("M' === M with M = A'*A + 3*F*F'? ", M' === M)
    println(M*x)

    K = 3F
    #println(K*x)
    K = A + A
    #println(K*x)
    K = F'
    #println(K*x)

    K = A + 3F'
    println(K)
    println(K*x)
    println(2K*x)

    println(((2F)' + 7A*(2A')*A)*x)

    println(((2F)' + (2U*A')'*A'' + 11A)*x)

    M = 2F'F + (3U*A')'
    N = F*M
    println(M*w)
    println(N*w)

    println(A*x)
    println(A(x))
    println(A(B(y)))
    println(A*B*y)

    println(A'(w))
    println(A'*w)
    #println(A'*I*w)

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

runtests()

end # module Test
