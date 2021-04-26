"""
This is used to import "graph.jl" and use nx package from python to draw.

Author: Daniel Zhang
Copyright (c) 2021
Email: zsjcameron@gmail.com
"""

using PyCall
plt = pyimport("matplotlib.pyplot")
nx = pyimport("networkx")

include("graph.jl")

n_agents, connection = 50, 0.4
error, Adj, degree, xy = randomgraph(n_agents, connection)
W = metropolis_hastings(incidence(Adj))
L = laplacian_matrix(Adj, degree)

"""
This function is used to plot a graph using networkx package from python
# Arguments
- 'Adj::Array{Float64, 2}': the adjacency matrix of a given graph
"""

function plot_graph(Adj::Array{Float64, 2})
    G = nx.Graph()
    m = size(Adj)[1]

    for i = 1 : m
        for j = 1 : m
            if Adj[i, j] ≠ 0
                G.add_edge(i, j)
            end
        end
    end
    nx.draw(G, node_color = "#274862", 
            node_size = 500, 
            width = 0.5, 
            font_size = 9, 
            font_weight="bold",
            with_labels = true, 
            edge_color = "grey", 
            font_color = "#e6b33d")
    plt.show()
end

plot_graph(Adj)

