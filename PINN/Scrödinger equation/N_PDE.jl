using NeuralPDE, ModelingToolkit, DomainSets
using Lux, Optimization, OptimizationOptimJL, OptimizationOptimisers
using Plots
using LuxCUDA, Random, ComponentArrays

@parameters t x
@variables u(..) v(..)

Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# Domain limits
x_min = -5.0
x_max = 5.0

t_min = 0.0
t_max = π/2


# 2. PDE system
eqs =[
    Dt(u(t, x)) + 0.5 * Dxx(v(t, x)) + v(t, x) * ((u(t, x))^2 + (v(t, x))^2) ~ 0.0,
    Dt(v(t, x)) - 0.5 * Dxx(u(t, x)) - u(t, x) * ((u(t, x))^2 + (v(t, x))^2) ~ 0.0
]

# 3. Domains
domains = [
    x ∈ Interval(x_min, x_max),
    t ∈ Interval(t_min, t_max)
]

# 4. Initial conditions
h0 = x -> 2 * sech(x)

# 5. Boundary and initial conditions 
bcs = [
    u(0, x) ~ h0(x),
    v(0, x) ~ 0.0,

    u(t, -5) ~ u(t, 5),
    Dx(u(t, -5)) ~ Dx(u(t, 5)),

    v(t, -5) ~ v(t, 5),
    Dx(v(t, -5)) ~ Dx(v(t, 5))
]

@named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])

# Neural network
chain = Chain(
    Dense(2, 100, tanh),
    Dense(100, 100, tanh),
    Dense(100, 100, tanh),
    Dense(100, 100, tanh),
    Dense(100, 100, tanh),
    Dense(100, 2)
)

dx = 0.04   
dt = 0.01
strategy = GridTraining([dx, dt])
#rng = Random.default_rng()
#Random.seed!(rng, 0)
#ps, st = Lux.setup(rng, chain) |> f64

#strategy = QuasiRandomTraining(500) 

#const gpud = gpu_device()
#ps = Lux.setup(Random.default_rng(), chain)[1]
#ps = ps |> ComponentArray |> gpud .|> Float64

discretization = PhysicsInformedNN(chain, strategy) #; init_params=ps
prob = discretize(pdesys, discretization)
symprob = symbolic_discretize(pdesys, discretization)

# Print initial loss
initial_loss = prob.f(prob.u0, nothing)
println("Initial loss = ", initial_loss)

pde_inner_loss_functions = symprob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = symprob.loss_functions.bc_loss_functions

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

opt1 = OptimizationOptimisers.Adam(1e-3)
res1 = Optimization.solve(prob, opt1; callback=callback, maxiters=500)

plot(1:length(losses), losses, xlabel="Iteration", ylabel="Loss", title="Training Loss")


prob = remake(prob, u0 = res1.u)
global  iter = 0
losses = Float64[]
opt = OptimizationOptimJL.LBFGS()
res = Optimization.solve(prob, opt; callback=callback, maxiters = 100)

plot(1:length(losses), losses, xlabel="Iteration", ylabel="Loss", title="Training Loss")

phi = discretization.phi
# Extract trained params
trained_params = res1.u
#trained_params = res.u |> cpu_device()

using MAT

mat_data = matread("C:\\Users\\kunwa\\Documents\\Programming\\Julia\\PINN\\Scrödinger equation\\NLS.mat")
x_data = mat_data["x"]  # spatial grid
x_vec = vec(x_data)  # ensure it's a vector
t_data = mat_data["tt"]   # time points
t_vec = vec(t_data)  # ensure it's a vector
u_data = mat_data["uu"]   # solution matrix (size: length(t) × length(x))

#discrete_x = range(-5.0, 5.0, length=256)
#discrete_t = range(0.0, π/2, length=159)

Nx = length(x_vec)
Nt = length(t_vec)

# Allocate arrays (time-major: Nt × Nx)
U_pred = zeros(Float64, Nx, Nt)
V_pred = zeros(Float64, Nx, Nt)

for (ti, tt) in enumerate(t_vec)
    for (xi, xx) in enumerate(x_vec)
        uv = phi([xx, tt], trained_params)
        U_pred[xi, ti] = uv[1]
        V_pred[xi, ti] = uv[2]
    end
end

#u_pred = [phi([t, x], trained_params)[1] for t in t_data, x in x_data]  # predicted u(t,x)
#v_pred = [phi([t, x], trained_params)[2] for t in t_data, x in x_data]  # predicted v(t,x)

#u_predict = [U_pred[xi, ti] for ti in 1:Nx, xi in 1:Nt]
#v_predict = [V_pred[xi, ti] for ti in 1:Nx, xi in 1:Nt]

U_pred
V_pred
# Complex wave function ψ = u + i v
ψ_pred = ComplexF64.(U_pred, V_pred)
absψ_pred = abs.(ψ_pred)
absu_data = abs.(u_data)  # absolute values of the solution

mkpath("PINN/Scrödinger equation")

anim = @animate for i in eachindex(t_vec)
    plot(x_vec, absψ_pred[:, i], label="|ψ(t=$(t_vec[i]), x)|", xlabel="x", ylabel="|ψ|",
         title="Time: $(t_vec[i])", legend=:topright)
    plot!(x_vec, absu_data[:, i], label="|ψ_data(t=$(t_vec[i]), x)|", xlabel="x", ylabel="|ψ|",
             title="Time: $(t_vec[i])", legend=:topright)
end
gif(anim, "PINN/Scrödinger equation/schrodinger_abs_N_PDE.gif", fps = 3)

τ = 0.79
i = findmin(abs.(t_vec .- τ))[2]  
@show length(x_vec)
@show size(u_data)
# nearest saved time index
plt = plot(x_vec, absψ_pred[:, i];
     xlabel="x", ylabel="|ψ|",
     title="t ≈ $(round(t_vec[i], digits=3))",
     label="|ψ|")
plot!(plt, x_vec, abs.(u_data[:, i]), label="MAT |ψ|", lw=2, ls=:dash)


savefig(plt, "PINN/Scrödinger equation/schrodinger_abs_t ≈ $(round(t_vec[i], digits=3))_N_PDE.png")

# Create a heatmap
plt = heatmap(
    t_vec, x_vec, absψ_pred',  # transpose to match (x vs t)
    xlabel = "t", ylabel = "x",
    title = "|ψ(t,x)|",
    colorbar_title = "|ψ|",
    color = :blues,          
    aspect_ratio = :auto
)


savefig(plt, "PINN/Scrödinger equation/schrodinger_abs_heatmap_N_PDE.png")