"""
This is a demo of gradient descent in Julia
in Julia.

Author: Daniel Zhang
Copyright (c) 2021
Email: zsjcameron@gmail.com
"""

f(x) = -sin(x[1] ^2 /2 -x[2]^2/4 +3) * cos(2x[1] + 1 - exp(x[2]))
g(x) = x^2
function df(x)
    a1 = x[1]^2/2 - x[2]^2/4 + 3
    a2 = 2x[1] + 1 - exp(x[2])
    b1 = cos(a1)*cos(a2)
    b2 = sin(a1)*sin(a2)
    return -[x[1]*b1 - 2b2, -x[2]/2*b1 + exp(x[2])*b2]
end

function dg(x)
    return 2x
end

maxiter = 500
α = 0.1
# x0 = 8
x0 = [0, 0.5]
tol = 1e-3
x_ = x0
optimal = -0.2072853680186296
cost = []
x = []
for i = 1: maxiter
    gradient = df(x_)
    # println(i, gradient)
    if sqrt(sum(gradient.^2)) < tol
        break
    end
    x_ -=  α*gradient
    push!(x, x_)
    push!(cost, f(x_) - optimal)
    # println(i, cost[i])
end
# println(cost)

using PyPlot
PyPlot.clf()
PyPlot.plot(cost)
ax = gca()
ax.set_yscale("log")
display(gcf())
