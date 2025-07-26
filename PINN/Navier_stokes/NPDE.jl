using NeuralPDE, ModelingToolkit, DomainSets
using Lux, Optimization, OptimizationOptimJL, OptimizationOptimisers
using ComponentArrays, Random, Plots
using MAT, GLMakie

@parameters t x y
@variables u(..) v(..) p(..)

Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

@parameters λ1, λ2

# PDE system
eqs = [
    Dt(u(t, x, y)) + λ1 * (u(t, x, y) * Dx(u(t, x, y)) + v(t, x, y) * Dy(u(t, x, y))) - λ2 * (Dxx(u(t, x, y)) + Dyy(u(t, x, y))) + Dx(p(t, x, y)) ~ 0, 
    Dt(v(t, x, y)) + λ2 * (u(t, x, y) * Dx(v(t, x, y)) + v(t, x, y) * Dy(v(t, x, y))) - λ2 * (Dxx(v(t, x, y)) + Dyy(v(t, x, y))) + Dy(p(t, x, y)) ~ 0, 
    Dx(u(t, x, y)) + Dy(v(t, x, y)) ~ 0
]

domains = [
    x ∈ Interval(1.0, 8.0),
    y ∈ Interval(-2.0, 2.0),
    t ∈ Interval(0.0, 20.0)
]

bcs = []  # no boundary condition

chain = Chain(
    Dense(3, 20, tanh),
    Dense(20, 20, tanh),
    Dense(20, 20, tanh),
    Dense(20, 20, tanh),
    Dense(20, 20, tanh),
    Dense(20, 20, tanh),
    Dense(20, 20, tanh),
    Dense(20, 20, tanh),
    Dense(20, 20, tanh),
    Dense(20, 20, tanh),
    Dense(20, 3)
)


mat = matread("C:\\Users\\kunwa\\Documents\\Programming\\Julia\\PINN\\Navier_stokes\\cylinder_nektar_wake.mat")
x_star = mat["X_star"] # # N x 2;  spatial coordinates (x,y) of N points
p_star = mat["p_star"] # N x T; pressure at N points over T time steps
t_star = mat["t"] # T x 1; time vector of length T
u_star = mat["U_star"]  # N x 2 x T; velocity components (u,v) at N points over T time steps

N = size(x_star, 1)
T = size(t_star, 1)

# Rearranging: Expand spatial and temporal dimensions
XX = repeat(x_star[:, 1:1], 1, T)  # N x T (repeat x-coords across time)
YY = repeat(x_star[:, 2:2], 1, T)  # N x T (repeat y-coords across time)
TT = repeat(t_star', N, 1)         # N x T (repeat times across space)

# Extract velocity and pressure
UU = u_star[:, 1, :]  # N x T (u-component)
VV = u_star[:, 2, :]  # N x T (v-component)
PP = p_star           # N x T

# Flatten all arrays into NT x 1 column vectors
x_data = vec(XX)
minimum(x_data), maximum(x_data)  # Check min/max of x
y_data = vec(YY)
minimum(y_data), maximum(y_data)  # Check min/max of y
t_data = vec(TT)
minimum(t_data), maximum(t_data)  # Check min/max of t

u_data = vec(UU)
v_data = vec(VV)
p_data = vec(PP)


# Plot using GLMakie
fig = Figure(resolution = (800, 600))
ax = Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="t", title="Space-Time Grid (t, x, y)")

GLMakie.scatter!(ax, x_data, y_data, t_data, markersize=2, color=:blue)

display(fig)

N1 = 5000
Random.seed!(123)
inds = rand(1:length(x_data), N1)
xtyt = hcat(t_data[inds], x_data[inds], y_data[inds])
u_vals = u_data[inds]
v_vals = v_data[inds]

function additional_loss(phi, θ, p)
    ŷ = phi(xtyt, θ)  # predict (u, v, p)
    u_pred = Array(ŷ[1])
    v_pred = Array(ŷ[2])
    return sum((u_pred .- u_vals).^2) + sum((v_pred .- v_vals).^2)
end

@named pde = PDESystem(eqs, domains, bcs, [t, x, y], [u, v, p], [λ1, λ2], defaults=Dict(λ1 => 1.0, λ2 => 1.0))

discretization = PhysicsInformedNN(chain, QuadratureTraining(; abstol = 1e-6, reltol = 1e-6, batch = 200), param_estim = true,
        additional_loss = additional_loss)

@time prob = discretize(pde, discretization)


# Add data loss (Raissi's additional loss)
add_data!(prob, u(t, x, y) ~ u_vals, [xtyt[i, :] for i in 1:N1])
add_data!(prob, v(t, x, y) ~ v_vals, [xtyt[i, :] for i in 1:N1])

#Callback function
global  iter = 0
losses = Float64[]
callback = function (p, l)
    global iter += 1
    push!(losses, l)
    if isnan(l)
        println("NaN detected at iteration $iter")
        return true  # stops training
    end
    #println("pde_losses: ", map(l_ -> l_(p.u), pde_inner_loss_functions))
    #println("bcs_losses: ", map(l_ -> l_(p.u), bcs_inner_loss_functions))
    println("Iteration: $iter, loss is: $l")
    return false
end

# Training
opt = Optimization.OptimizationFunction(prob, Optimization.AutoZygote())
params = prob |> NeuralPDE.initial_params

data_prob = Optimization.OptimizationProblem(opt, params)
res = Optimization.solve(data_prob, ADAM(0.001), maxiters = 3000)

res = Optimization.solve(prob, Optim.BFGS(linesearch = BackTracking()); callback=callback, maxiters=10000)

plot(1:length(losses), losses, xlabel="Iteration", ylabel="Loss", title="Training Loss")