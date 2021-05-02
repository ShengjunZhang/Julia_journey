"""
This is a demo of distributed optimization algorithms in Julia.

Author: Daniel Zhang
Copyright (c) 2021
Email: zsjcameron@gmail.com
"""

using PyPlot
include("graph.jl")
include("disopt_alg.jl")

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

n_agents, connection = 50, 0.4
error, Adj, degree, xy = randomgraph(n_agents, connection)
W = metropolis_hastings(incidence(Adj))
W_tilde = (Matrix{Float64}(I, n_agents, n_agents) + W) ./ 2

dimension = 10
optimal = 0
W_comm = kron(W, Matrix{Float64}(I, dimension, dimension))
W_tilde_comm = kron(W_tilde, Matrix{Float64}(I, dimension, dimension))
iteration = 1000
α0 = 0.02
α0_extra = 0.5

x_init = rand(MersenneTwister(2021), Float16, (n_agents, dimension))
history_cost_DGD, frob_norm_DGD = DGD(x_init, W_comm, iteration, optimal, α0, n_agents, dimension)
history_cost_EXTRA, frob_norm_EXTRA = EXTRA(x_init, W_comm, W_tilde_comm, iteration, optimal, α0_extra, n_agents, dimension)

PyPlot.clf()
PyPlot.plot(frob_norm_DGD / norm(x_init[:]), color = "#274862")
PyPlot.plot(frob_norm_EXTRA / norm(x_init[:]), color = "#e6b33d")
ax = gca()
ax.set_yscale("log")
xlabel("Iterations")
ylabel("Residual")
legend(["Distributed (Sub)gradient Methods", "EXTRA"])
display(gcf())
