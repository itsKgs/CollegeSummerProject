using NeuralPDE, ModelingToolkit, DomainSets
using Lux, Optimization, OptimizationOptimJL
using CSV, DataFrames, Plots

# -------------------------------
# Step 1: Load FEniCS-generated Data
# -------------------------------
df = CSV.read("C:\\Users\\kunwa\\Documents\\Programming\\Julia\\FEniCS\\Navier_Stokes_Channel\\2D_navier_fenics_data.csv", DataFrame)
X_data = Matrix(df[:, ["x", "y", "t"]])  # Input: Nx3
Y_data = Matrix(df[:, ["u", "v", "p"]])  # Output: Nx3

# -------------------------------
# Step 2: Define PDE (Navier-Stokes)
# -------------------------------
@parameters x y t
@variables u(..) v(..) p(..)

Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
DxDy = Dx * Dy

μ = 1.0
ρ = 1.0

# PDE system
eqs = [
    ρ * Dt(u(x, y, t)) + ρ * (u(x, y, t) * Dx(u(x, y, t)) + v(x, y, t) * Dy(u(x, y, t))) - (μ * (2 * Dxx(u(x, y, t)) + Dyy(u(x, y, t)) + DxDy(v(x, y, t))) - Dx(p(x, y, t))) ~ 0, 
    ρ * Dt(v(x, y, t)) + ρ * (u(x, y, t) * Dx(v(x, y, t)) + v(x, y, t) * Dy(v(x, y, t))) - ((μ * (2 * Dxx(v(x, y, t)) + Dyy(v(x, y, t)) + DxDy(u(x, y, t))) - Dy(p(x, y, t)))) ~ 0, 
    Dx(u(x, y, t)) + Dy(v(x, y, t)) ~ 0
]


# -------------------------------
# Step 3: Define Domain
# -------------------------------
domains = [
    x ∈ IntervalDomain(0.0, 1.0),
    y ∈ IntervalDomain(0.0, 1.0),
    t ∈ IntervalDomain(0.0, 1.0)
]

# Boundary Comditions
bcs = [
    u(x, y, 0) ~ 0.0,
    v(x, y, 0) ~ 0.0,
    p(x, y, 0) ~ 0.0,

    u(1, y, t) ~ 0.0,     
    v(0, y, t) ~ 0.0,    

    u(0, y, t) ~ 0.0,
    v(1, y, t) ~ 0.0,
    u(x, 0, t) ~ 0.0, v(x, 0, t) ~ 0.0, # No slip condition
    u(x, 1, t) ~ 0.0, v(x, 1, t) ~ 0.0, # No slip condition

    p(0, y, t) ~ 8.0,
    p(1, y, t) ~ 0.0,

    #p(x, 0, t) ~ 0.0,
    #p(x, 1, t) ~ 0.0,
    Dt(u(x, y, 0)) ~ 0.0

]


@named pde_system = PDESystem(eqs, bcs, domains, [x, y, t], [u(x, y, t), v(x, y, t), p(x, y, t)])

# -------------------------------
# Step 4: Define Neural Network
# -------------------------------
nn = Lux.Chain(
    Lux.Dense(3, 64, tanh),
    Lux.Dense(64, 64, tanh),
    Lux.Dense(64, 64, tanh),
    Lux.Dense(64, 3)
)

using NeuralPDE: DataDrivenTraining
strategy = DataDrivenTraining(X_data, Y_data; data_loss_weight=1.0, pde_loss_weight=1.0)
discretization = PhysicsInformedNN(nn, strategy)
prob = discretize(pde_system, discretization)

# -------------------------------
# Step 5: Setup PDE System & PINN Problem
# -------------------------------
prob = discretize(pde_system, discretization)

# -------------------------------
# Step 6: Add Supervised Data Loss (FEniCS)
# -------------------------------
data_points = [[X_data[i, 1], X_data[i, 2], X_data[i, 3]] for i in 1:size(X_data, 1)]
add_data!(prob, data_points, Y_data)

# -------------------------------
# Step 8: Train the PINN
# -------------------------------
callback = function (p, l)
    println("Loss: ", l)
    return false
end

opt = OptimizationOptimJL.BFGS()
res = solve(prob, opt, callback=callback, maxiters=500)

# -------------------------------
# Step 9: Visualize Predicted u(x, y) at t = 0.5
# -------------------------------
x_vals = range(0, 1, length=50)
y_vals = range(0, 1, length=50)
t_fixed = 0.5

u_pred = [res([x, y, t_fixed], u)[1] for x in x_vals, y in y_vals]
v_pred = [res([x, y, t_fixed], v)[1] for x in x_vals, y in y_vals]
p_pred = [res([x, y, t_fixed], p)[1] for x in x_vals, y in y_vals]

# u-velocity
heatmap(x_vals, y_vals, u_pred,
    title = "Predicted u(x, y) at t = 0.5",
    xlabel = "x", ylabel = "y", c = :thermal, aspect_ratio = 1)

# Optionally add more plots:
# heatmap(x_vals, y_vals, v_pred, title="v(x, y)")
# heatmap(x_vals, y_vals, p_pred, title="p(x, y)")
