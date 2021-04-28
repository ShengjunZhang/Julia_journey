"""
This is used to demonstrate macro in Julia.

Author: Daniel Zhang
Copyright (c) 2021
Email: zsjcameron@gmail.com
"""

macro ILike(name::String)
    return :(println("I like ", $name))
end

macro my_add(a::Real, b::Real)
    return :($a + $b)
end

println(@my_add(1, 2))
@ILike("Daniel")

function something(x::T) where T <: Real
    return x.^2
end

println(something(2))