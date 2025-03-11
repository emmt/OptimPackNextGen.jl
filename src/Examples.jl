module  Examples



function rosenbrock_fg!(x::Array{T,1}, gx::Array{T,1}) where {T<:Real}
	local c1::T = 1
	local c2::T = 2
	local c10::T = 10
	local c200::T = 200
	x1 = x[1:2:end]
	x2 = x[2:2:end]
	t1 = c1 .- x1
	t2 = c10*(x2 - x1.*x1)
	g2 = c200*(x2 - x1.*x1)
	gx[1:2:end] = -c2*(x1 .* g2 + t1)
	gx[2:2:end] = g2
	return sum(t1.*t1) + sum(t2.*t2)
end

function rosenbrock_f(x::Array{T,1}) where {T<:Real}
	local c1::T = 1
	local c2::T = 2
	local c10::T = 10
	local c200::T = 200
	x1 = x[1:2:end]
	x2 = x[2:2:end]
	t1 = c1 .- x1
	t2 = c10*(x2 - x1.*x1)
	return sum(t1.*t1) + sum(t2.*t2)
end

end