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

# Evaluate but nonsense because of the nature of data in this case.
function eval_bc(x_opt, features)
    temp = x_opt'*features
    return log(1 + exp(temp[1]))
end

# Generate data

dim             = 50                       # problem dimension
batch_size      = 700                       # batch size
n_agents        = 10                       # agent number
total_num_data  = batch_size * n_agents    # total number of data

λ = 0.001
α = 1

using Random
features = randn(MersenneTwister(2021), Float64, (dim, total_num_data))
labels = rand(1:2, (1, total_num_data))
labels[labels .== 2] .= -1


training_num = 5000
validation_num = 1000
testing_num = 1000

training_features = features[:, 1:training_num]
training_labels = labels[1: training_num]

validation_features = features[:, training_num+1: training_num+validation_num]
validation_labels = labels[training_num+1: training_num+validation_num]

testing_features = features[:, training_num+validation_num+1: end]
testing_labels = labels[training_num+validation_num+1: end]


iter_max = 5000
α_ss = 0.05

#=
Centralized Stochastic gradient descent, use all the data to perform, 
i.e. 2500 data.
=#

x_init = zeros(dim, 1).+2
cost = []
x_ = x_init
x = []
for i in ProgressBar(1: iter_max)
    idx = rand(1:training_num, (1, 1))
    grad = 0
    for i in idx
        grad = grad .+ gc(x_, λ, α, training_features[:, i], training_labels[i], 1, 1)
    end
    x_ -=  α_ss*grad
    push!(x, x_)
    # Validation
    cost_temp = 0
    for i = 1: validation_num
        cost_temp = cost_temp .+ cf(x_, λ, α, validation_features[:, i], validation_labels[i], 1, 1)
        # println(grad)
    end
    push!(cost, sum(cost_temp)/total_num_data)
end

function eval_bc(x_opt, features)
    temp = x_opt'*features
    return log(1 + exp(temp[1]))
end

# Accuracy
x_opt = x[end]
count = 0
for i = 1: testing_num
    temp = eval_bc(x_opt, testing_features[:, i])
    if temp >= 0
        temp = 1
    else
        temp = -1
    end
    # println(temp)
    if temp == testing_labels[i]
        count += 1
        continue
    end
end

acc = count / testing_num

using PyPlot
PyPlot.clf()
PyPlot.plot(cost)
ax = gca()
# ax.set_yscale("log")
xlabel("Iterations")
ylabel("Validation Loss")
display(gcf())

