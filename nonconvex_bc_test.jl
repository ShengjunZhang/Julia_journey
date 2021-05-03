"""
This is a test of nonconvex binary classification in Julia.

Reference:
    Linear Convergence of First- and Zeroth-Order Primal-Dual Algorithms for
    Distributed Nonconvex Optimization
    Xinlei Yi, Shengjun Zhang, Tao Yang, Tianyou Chai, Karl H. Johansson
    https://arxiv.org/pdf/1912.12110.pdf

Author: Daniel Zhang
Copyright (c) 2021
Email: zsjcameron@gmail.com
"""

using ProgressBars
# Cost function
function cf(x, λ, α, z, y, batch_size, n_agents)
    temp = -y*x'*z
    return 1/(batch_size*n_agents)*(log(1+exp(temp[1]))) .+ 1/(n_agents)* 
           ((λ*α*x.^2)./((1 .+ α*x.^2)))
end

# Gradinet
function gc(x, λ, α, z, y, batch_size, n_agents)
    temp = y*x'*z
    return 1/(batch_size*n_agents) * (-y*z) / (1+exp(temp[1])) .+ (1/n_agents)*
           ((2*λ*α*x)./((1 .+ α*x.^2).^2))
end

# Generate data

dim             = 10                       # problem dimension
batch_size      = 5                       # batch size
n_agents        = 50                       # agent number
total_num_data  = batch_size * n_agents    # total number of data

λ = 0.001
α = 1

using Random
features = randn(MersenneTwister(2021), Float64, (dim, total_num_data))
labels = rand(1:2, (1, total_num_data))
labels[labels .== 2] .= -1

iter_max = 5000
α_ss = 0.9

#=
Centralized Stochastic gradient descent, use all the data to perform, 
i.e. 2500 data.
=#

x_init = ones(dim, 1).+2
cost = []
x_ = x_init
x = []
for i in ProgressBar(1: iter_max)
    # println(iter, "iteration: $i")
    # for i = 1: 2500
    #     grad =  gc(x_, λ, α, features[:, i], labels[i], 1, n_agents)
    #     # println(grad)
    # end
    idx = rand(1:total_num_data, (1, 1))
    grad = 0
    for i in idx
        grad = grad .+ gc(x_, λ, α, features[:, i], labels[i], 1, n_agents)
    end
    # grad =  gc(x_, λ, α, features[:, idx[1]], labels[idx[1]], 1, n_agents)
    x_ -=  α_ss*grad
    push!(x, x_)
    cost_temp = 0
    for i = 1: 250
        cost_temp = cost_temp .+ cf(x_, λ, α, features[:, i], labels[i], 1, n_agents)
        # println(grad)
    end
    push!(cost, sum(cost_temp)/250)
end

using PyPlot
PyPlot.clf()
PyPlot.plot(cost)
ax = gca()
# ax.set_yscale("log")
xlabel("Iterations")
ylabel("Cost")
display(gcf())
