"""
This is used to demonstrate functions ending with an excalmation mark in Julia.

Author: Daniel Zhang
Copyright (c) 2021
Email: zsjcameron@gmail.com
"""

#=

Here we consider a fucntion that has exactly same fucntion as fill!(arg1, arg2) 
fucntion from Base.

=#

function my_fill!(v :: T where T <: Vector, number :: Real) 
    v[:] = [number for i in 1:length(v)]
    return v
end

function my_fill_no_ex(v :: T where T <: Vector, number :: Real) 
    v = [number for i in 1:length(v)]
    return v
end

v = [1.0, 1]

# Physically the same v
println("The output of v === my_fill!(v, 2) is ", v === my_fill!(v, 2))

# Physically not the same v
println("The output of v === my_fill_no_ex(v, 3) is ", 
        v === my_fill_no_ex(v, 3))

u = my_fill_no_ex(v, 3)
println("The output of u = my_fill_no_ex(v, 3) is ", u)
println("v is ", v)