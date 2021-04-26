"""

This is used to generate a random graph and its associated matrices
in Julia.

Author: Daniel Zhang
Copyright (c) 2021
Email: zsjcameron@gmail.com

"""

using Random
using LinearAlgebra

"""
This function is used to generate a random graph based on 
    Erdos-Renyi model.
# Arguments
- 'n_agents::Int64 > 2': the number of agents or nodes in a graph
- 'connection::Float64 (0, 1)': the probability of connection
"""
function randomgraph(n_agents::Int64, connection::Float64)
    rng = MersenneTwister(2021)
    loop = 1;
    count = 0;
    error = 0;
    while (loop & count < 10)
        xy = rand(rng, Float64, (n_agents, 2))
        Md = ((xy[:,1]*ones(1,n_agents)-ones(n_agents,1)*xy[:,1]').^2 +
                 (xy[:,2]*ones(1,n_agents)-ones(n_agents,1)*xy[:,2]').^2).^(0.5)
        # println(Md)
        # Adjacency matrix
        Identity = Matrix{Float64}(I, n_agents, n_agents)
        A = ( ( Md + 2 *connection * Identity ) .< connection) * Identity
        # println(A)
        degree = A * ones(n_agents, 1)
        L = diagm(degree[:]) - A
        if (rank(L) == n_agents - 1)
            loop = 0
            return error, A, degree, xy
            break 
        end
        count += 1
    end

    if (count == 10)
        error = 1
    end

end

"""
This function is used to compute the incidence matrix
# Arguments
- 'Adjacency::Array{Float64, 2}': the adjacency matrix of a given graph
"""

function incidence(Adjacency::Array{Float64, 2})
    n_agents = size(Adjacency)[1]
    edge = Int(sum(sum(Adjacency)) / 2)
    A = zeros(n_agents, edge)
    l = 0
    for i = 1: n_agents - 1
        for j = i + 1: n_agents
            if Adjacency[i, j] > 0.5
                l += 1
                A[i, l] = 1
                A[j, l] = -1
            end
        end
    end
    return A
end

"""
This function is used to compute the Metropolis-Hastings weights matrix
# Arguments
- 'Incidence::Array{Float64, 2}': the incidence matrix of a given graph
"""
function metropolis_hastings(Incidence::Array{Float64, 2})
    n, _ = size(Incidence)
    W = zeros(n, n)
    Lunw = Incidence * Incidence'
    Lunw = abs.(Lunw)
    for i = 1: n - 1
        for j = i + 1: n
            if Lunw[i, j] == 1
                W[i,j] = 1/(1 + max(Lunw[i,i], Lunw[j,j]))
            end
        end
    end
    W = W + W'
    sum_row_W = sum(W, dims = 1)
    for i = 1: n
        W[i,i] = 1 - sum_row_W[i]
    end
    return W
end


n_agents, connection = 50, 0.4
error, Adj, degree, xy = randomgraph(n_agents, connection)
W = metropolis_hastings(incidence(Adj))

println(Adj)
