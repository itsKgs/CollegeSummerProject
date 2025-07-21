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
    Dense(2, 10, sin),
    Dense(10, 10, sin),
    #Dense(64, 64, tanh),
    #Dense(64, 64, tanh),
    Dense(10, 2)
)

#dx = 0.05   
#dt = 0.01
#strategy = GridTraining([dx, dt])

strategy = QuasiRandomTraining(2000) 

const gpud = gpu_device()
ps = Lux.setup(Random.default_rng(), chain)[1]
ps = ps |> ComponentArray |> gpud .|> Float64

discretization = PhysicsInformedNN(chain, strategy; init_params = ps)
prob = discretize(pdesys, discretization)
symprob = symbolic_discretize(pdesys, discretization)

# Print initial loss
initial_loss = prob.f(prob.u0, nothing)
println("Initial loss = ", initial_loss)

#Callback function
global  iter = 0
callback = function (p, l)
    global iter += 1
    if isnan(l)
        println("NaN detected at iteration $iter")
        return true  # stops training
    end
    println("Iteration: $iter, loss is: $l")
    return false
end

opt1 = OptimizationOptimisers.Adam(1e-10)

res1 = Optimization.solve(prob, opt1; callback=callback, maxiters=100)

opt = OptimizationOptimJL.LBFGS()
res = Optimization.solve(prob, opt; callback=callback, maxiters = 15)

phi = discretization.phi
# Extract trained params
trained_params = res.u
trained_params = res.u |> cpu_device()

discrete_x = range(-5.0, 5.0, length=100)
discrete_t = range(0.0, π/2, length=50)

Nx = length(discrete_x)
Nt = length(discrete_t)

# Allocate arrays (time-major: Nt × Nx)
U_pred = zeros(Float64, Nt, Nx)
V_pred = zeros(Float64, Nt, Nx)


for (ti, tt) in enumerate(discrete_t)
    for (xi, xx) in enumerate(discrete_x)
        uv = Array(phi([tt, xx], trained_params))
        U_pred[ti, xi] = uv[1]
        V_pred[ti, xi] = uv[2]
    end
end
# Example: predict u and v at a grid of (t, x) points
# (for 1D x, so no y dimension)
# If you want a 2D array of predictions for u:
u_predict = [U_pred[ti, xi] for ti in 1:Nt, xi in 1:Nx]
v_predict = [V_pred[ti, xi] for ti in 1:Nt, xi in 1:Nx]


# Complex wave function ψ = u + i v
ψ_pred = ComplexF64.(U_pred, V_pred)
absψ_pred = abs.(ψ_pred)

anim = @animate for i in eachindex(discrete_t)
    plot(discrete_x, absψ_pred[i, :], label="|ψ(t=$(discrete_t[i]), x)|", xlabel="x", ylabel="|ψ|",
         title="Time: $(discrete_t[i])", legend=:topright)
end
gif(anim, "schrodinger_abs.gif", fps = 2)

τ = 0.79
i = findmin(abs.(discrete_t .- τ))[2]  # nearest saved time index
plot(discrete_x, absψ_pred[i, :];
     xlabel="x", ylabel="|ψ|",
     title="t ≈ $(round(discrete_t[i], digits=3))",
     label="|ψ|")


# Create a heatmap
heatmap(
    discrete_t, discrete_x, absψ_pred',  # transpose to match (x vs t)
    xlabel = "t", ylabel = "x",
    title = "|ψ(t,x)|",
    colorbar_title = "|ψ|",
    color = :blues,          # similar colormap
    aspect_ratio = :auto
)
