"""
This is a collection of distributed optimization algorithms in Julia.

Author: Daniel Zhang
Copyright (c) 2021
Email: zsjcameron@gmail.com
"""

""" EXTRA

Reference: 
EXTRA: An Exact First-Order Algorithm for Decentralized Consensus 
       Optimization 
Wei Shi, Qing Ling, Gang Wu, and Wotao Yin
https://epubs.siam.org/doi/10.1137/14096668X

"""

function EXTRA(x_init, W_comm, W_tilde_comm, iteration, optimal, α0, n_agents, dimension)
    x = x_init[:]
    g_init = [grad(x_init[i, :]) for i = 1 : n_agents]
    ∇ = reshape_grad(g_init, dimension)
    history_cost = [sum(cost(x_init[i, :]) for i = 1 : n_agents)]
    y = zeros(size(x)[1], 1)
    frob_norm = [norm(x)]
    for i = 1: iteration
        # α = α0 / (5 + i) with α0 = 1
        # α = α0 / sqrt(i) with α0 = 0.1
        α = α0
        x = W_comm * x - α * (∇ + y)
        y = y + W_tilde_comm * x
        x_cur = reshape(x, :, dimension)
        push!(history_cost, 
            sum(cost(x_cur[i, :]) - optimal for i = 1 : n_agents))
        push!(frob_norm, norm(x))
        cur_grad = [grad(x_cur[i, :]) for i = 1 : n_agents]
        ∇ = reshape_grad(cur_grad, dimension)
    end
    return history_cost, frob_norm
end

""" DGD

Reference: 
Distributed Subgradient Methods for Multi-Agent Optimization, 
Angelia Nedic´ and Asuman Ozdaglar, 
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4749425

"""

function DGD(x_init, W_comm, iteration, optimal, α0, n_agents, dimension)
    x = x_init[:]
    g_init = [grad(x_init[i, :]) for i = 1 : n_agents]
    ∇ = reshape_grad(g_init, dimension)
    history_cost = [sum(cost(x_init[i, :]) for i = 1 : n_agents)]
    frob_norm = [norm(x)]
    for i = 1: iteration
        α = α0
        x = W_comm * x - α * ∇
        x_cur = reshape(x, :, dimension)
        push!(history_cost, 
            sum(cost(x_cur[i, :]) - optimal for i = 1 : n_agents))
        push!(frob_norm, norm(x))
        cur_grad = [grad(x_cur[i, :]) for i = 1 : n_agents]
        ∇ = reshape_grad(cur_grad, dimension)
    end
    return history_cost, frob_norm
end