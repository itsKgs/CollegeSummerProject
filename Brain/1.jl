using NeuralPDE, ModelingToolkit, Optimization, OptimizationOptimisers, Lux
using CSV, DataFrames, Plots

# Step 1: Load EEG time-series data
data = CSV.read("C:\\Users\\kunwa\\Documents\\Programming\\Project\\Brain\\chb01_01_filtered.csv", DataFrame)
t_data = data.time
u_data = data."FP1-F7"

# Step 2: Define the PDE ∂u/∂t = f(t)
@parameters t
@variables u(..)
Dt = Differential(t)

eq = Dt(u(t)) ~ 0  # We will fit this using data

# Step 3: Domain and data points
t_min, t_max = minimum(t_data), maximum(t_data)
domains = [t ∈ IntervalDomain(t_min, t_max)]

# Step 4: Neural network approximation for u(t)
input = [t]
target = [u(t)]

chain = Lux.Chain(Dense(1, 16, tanh), Dense(16, 1))
strategy = NeuralPDE.GridTraining([100])

discretization = PhysicsInformedNN(chain, strategy)

# Step 5: Set up and solve the PINN problem
@named pde_system = PDESystem(
    [eq],       # equations
    [],         # boundary conditions
    domains,
    input,
    target
)
prob = discretize(pde_system, discretization)

# Fit to data
function loss_func(p)
    predicted = [prob.phi(ti, p)[1] for ti in t_data]
    return sum(abs2, predicted .- u_data)
end

optprob = OptimizationProblem(prob.f, prob.u0, prob.p)
#opt_state = Optimization.OptimizationProblem(opt_prob, prob.init_params)
res = solve(optprob, Adam(0.01), maxiters=500)


# Plot result
t_plot = collect(range(t_min, t_max, length=300))
u_pred = [prob.phi(ti, res.u)[1] for ti in t_plot]

plot(t_data, u_data, label="EEG Data", legend=:bottomright)
plot!(t_plot, u_pred, label="PINN Prediction", lw=2)