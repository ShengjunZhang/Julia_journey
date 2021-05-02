"""
This is a demo of EXTRA in Julia.

Reference: 
[1] EXTRA: An Exact First-Order Algorithm for Decentralized Consensus 
           Optimization 
    Wei Shi, Qing Ling, Gang Wu, and Wotao Yin
    https://epubs.siam.org/doi/10.1137/14096668X

Author: Daniel Zhang
Copyright (c) 2021
Email: zsjcameron@gmail.com
"""

"""
    Problem formulation and gradient 1/2Σ(x^2)
"""

function cost(x)::Real
    return 0.5 * sum(x .^ 2)
end

function grad(x)::Array
    return  x
end

function reshape_grad(grad_val, dimension)
    n = size(grad_val)[1]
    grad_vec = []
    for i = 1:dimension
        for j = 1: n
            push!(grad_vec, grad_val[j][i])
        end
    end
    return grad_vec
end

"""
    Perform the EXTRA algorithm
"""

function EXTRA(x_init, W_comm, W_tilde_comm, iteration, optimal, α0, n_agents, dimension)
    x = x_init[:]
    g_init = [grad(x_init[i, :]) for i = 1 : n_agents]
    ∇ = reshape_grad(g_init, dimension)
    history_cost = [sum(cost(x_init[i, :]) for i = 1 : n_agents)]
    y = zeros(size(x)[1], 1)
    for i = 1: iteration
        # α = α0 / (5 + i) with α0 = 1
        # α = α0 / sqrt(i) with α0 = 0.1
        α = α0
        x = W_comm * x - α * (∇ + y)
        y = y + W_tilde_comm * x
        if i % 100 == 0
            println(∇)
        end    
        x_cur = reshape(x, :, dimension)
        push!(history_cost, 
            sum(cost(x_cur[i, :]) - optimal for i = 1 : n_agents))
        cur_grad = [grad(x_cur[i, :]) for i = 1 : n_agents]
        ∇ = reshape_grad(cur_grad, dimension)
    end
    return history_cost
end


using PyPlot

"""
    Graph package and settings
"""

include("graph.jl")

n_agents, connection = 50, 0.4
error, Adj, degree, xy = randomgraph(n_agents, connection)
W = metropolis_hastings(incidence(Adj))
W_tilde = Matrix{Float64}(I, n_agents, n_agents) - W

dimension = 10
optimal = 0
W_comm = kron(W, Matrix{Float64}(I, dimension, dimension))
W_tilde_comm = kron(W_tilde, Matrix{Float64}(I, dimension, dimension))
iteration = 500
α0 = 0.1
x_init = rand(MersenneTwister(2021), Float64, (n_agents, dimension))
history_cost_EXTRA = EXTRA(x_init, W_comm, W_tilde_comm, iteration, optimal, α0, n_agents, dimension)

PyPlot.clf()
PyPlot.plot((history_cost_EXTRA/n_agents).^2)
ax = gca()
ax.set_yscale("log")
xlabel("Iterations")
ylabel("Residual")
legend(["EXTRA"])
display(gcf())