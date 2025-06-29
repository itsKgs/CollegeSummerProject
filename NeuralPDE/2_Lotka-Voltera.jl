using NeuralPDE, Lux, Plots, OrdinaryDiffEq, Distributions, Random

function lotka_volterra(u, p, t)
    # Model parameters.
    α, β, γ, δ = p
    # Current state.
    x, y = u

    # Evaluate differential equations.
    dx = (α - β * y) * x # prey
    dy = (δ * x - γ) * y # predator

    return [dx, dy]
end

# initial-value problem.
u0 = [1.0, 1.0]
p = [1.5, 1.0, 3.0, 1.0]
tspan = (0.0, 4.0)
prob = ODEProblem(lotka_volterra, u0, tspan, p)

# Solve using OrdinaryDiffEq.jl solver
dt = 0.01
solution = solve(prob, Tsit5(); saveat = dt)

# Dataset creation for parameter estimation (30% noise)
time = solution.t
solu = solution.u
u = hcat(solution.u...)

x = u[1, :] + (u[1, :]) .* (0.3 .* randn(length(u[1, :])))
y = u[2, :] + (u[2, :]) .* (0.3 .* randn(length(u[2, :])))
dataset = [x, y, time]

# Plotting the data which will be used
plot(time, x, label = "noisy x")
plot!(time, y, label = "noisy y")
plot!(solution, labels = ["x" "y"])


# Neural Networks must have 2 outputs as u -> [dx,dy] in function lotka_volterra()

chain = Chain(
    Dense(1, 6, tanh),
    Dense(6, 6, tanh),
    Dense(6, 2)
)

alg = BNNODE(chain;
    dataset = dataset,
    draw_samples = 1000,
    l2std = [0.1, 0.1],
    phystd = [0.1, 0.1],
    priorsNNw = (0.0, 3.0),
    param = [
        Normal(1, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(0, 2)],
    progress = false
)

sol_pestim = solve(prob, alg; saveat = dt)

# plotting solution for x,y for chain
plot(time, sol_pestim.ensemblesol[1], label = "estimated x")
plot!(time, sol_pestim.ensemblesol[2], label = "estimated y")

# comparing it with the original solution
plot!(solution, labels = ["true x" "true y"])