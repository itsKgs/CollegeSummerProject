using NeuralPDE, ModelingToolkit, DomainSets
using Lux, Optimization, OptimizationOptimJL, OptimizationOptimisers
using Plots
using LuxCUDA, Random, ComponentArrays

@parameters x y z t
@variables T(..)

Dt = Differential(t)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dzz = Differential(z)^2

# Domain limits
x_min = t_min = y_min = z_min = 0.0
x_max = 1.0
y_max = 1.0
z_max = 0.5
t_max = 1.0

k = 1.0
ρ = 1.0
C_p = 505.0
D = 1/(ρ * C_p)
#D = 1.0
a = 0.1
b = 0.1
c = 0.05
q = 200 * 0.3
#v = 0.1
v = 0.5


# Initial conditions
T0(x, y, z, t) = 300.0 

Q(x, y, z, t) = ((6 * sqrt(3) * q)/(a * b * c * π * sqrt(π))) * exp(-3 * ((((x - v * t)^2)/(a^2)) + (((y - 0.5)^2)/(b^2)) + (((z)^2)/(c^2))))
#Q(x, y, z, t) = 1000
# 2. PDE system

eqs = [
    Dt(T(x, y, z, t)) ~ D * (k * (Dxx(T(x, y, z, t)) + Dyy(T(x, y, z, t)) + Dzz(T(x, y, z, t))) + Q(x, y, z, t))
]

# 3. Domains
domains = [
    x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max),
    z ∈ Interval(z_min, z_max),
    t ∈ Interval(t_min, t_max)
]

# 4. Boundary and initial conditions (periodic in x and y)

bcs = [
    T(x, y, z, 0) ~ 300.0,
    T(0, y, z, t) ~ 300.0,
    T(1, y, z, t) ~ 300.0,
    T(x, 0, z, t) ~ 300.0,
    T(x, 1, z, t) ~ 300.0,
    T(x, y, 0, t) ~ 300.0,
    T(x, y, 0.5, t) ~ 300.0
] 

@named pdesys = PDESystem(eqs, bcs, domains, [x, y, z, t], [T(x, y, z, t)])

# Neural network
chain = Chain(
    Dense(4, 64, tanh),
    Dense(64, 64, tanh),
    Dense(64, 1)
)

dx = 0.05   
dy = 0.05 
dz = 0.025 
dt = 0.05
strategy = GridTraining([dx, dy, dz, dt])

const gpud = gpu_device()
ps = Lux.setup(Random.default_rng(), chain)[1]
ps = ps |> ComponentArray |> gpud .|> Float64

discretization = PhysicsInformedNN(chain, strategy; init_params = ps)
prob = discretize(pdesys, discretization)
symprob = symbolic_discretize(pdesys, discretization)

#Callback function
global  iter = 0
callback = function (p, l)
    global iter =  iter + 1
    println("Iteration: $iter, loss is: $l")
    return false
end

#opt1 = OptimizationOptimisers.Adam(0.001)
#res1 = Optimization.solve(prob, opt1; callback=callback, maxiters=1000)

opt = OptimizationOptimJL.BFGS()

res = Optimization.solve(prob, opt; callback, maxiters = 15)

phi = discretization.phi

xs = range(0.0, 1.0, length=50)
ys = range(0.0, 1.0, length=50)
ts = range(0.0, 1.0, length=30)
zs = 0.1
T_pred = [first(Array(phi([x, y, 0.1, 0.1], res.u)))[1] for x in xs for y in ys]



anim_T = @animate for t in ts
    T_pred = [first(Array(phi([x, y, zs, t], res.u)))[1] for x in xs for y in ys]
    heatmap(xs, ys, T_pred',
        xlabel="x", ylabel="y",
        title="T(x, y, z=$(round(zs, digits=2)), t=$(round(t, digits=2)))",
        c=:thermal, size=(800, 600) 
    ) 
end

gif(anim_T, "pinn_heatmap_z$(round(zs, digits=2)).gif", fps=5)

zs = 0.1
using Printf 
anim_T = @animate for t in ts
    T_pred = [first(Array(phi([x, y, zs, t], res.u))) for x in xs for y in ys]
    title= @sprintf("T(x, y, z=%.2f, t=%.2f)", zs, t)
    plot(xs, ys, T_pred, st = :surface, label = "", title = title) 
end
gif(anim_T, "pinn_surface_z$(round(zs, digits=2)).gif", fps=5)

using Printf  # ← Add this at the top of your file

zs = 0.1
anim_T = @animate for t in ts
    T_pred = [first(Array(phi([x, y, zs, t], res.u))) for x in xs, y in ys]
    ttl = @sprintf("T(x, y, z=%.2f, t=%.2f)", zs, t)
    surface(xs, ys, reshape(T_pred, :, length(ys)),
            xlabel="x", ylabel="y", zlabel="T",
            title=ttl, c=:thermal, legend=false)
end

gif(anim_T, "pinn_surface_z$(round(zs, digits=2)).gif", fps=5)


zs = 0.2
anim_T = @animate for t in ts
    T_pred = [phi([x, y, zs, t], res.u)[1] for x in xs, y in ys]
    heatmap(xs, ys, T_pred',
        xlabel="x", ylabel="y",
        title="T(x, y, z=$(round(zs, digits=4)), t=$(round(t, digits=4)))",
        c=:thermal, size=(800, 600) 
    ) 
end

gif(anim_T, "pinn_heatmap_z$(round(zs, digits=4)).gif", fps=6)

zs = 0.3
anim_T = @animate for t in ts
    T_pred = [phi([x, y, zs, t], res.u)[1] for x in xs, y in ys]
    heatmap(xs, ys, T_pred',
        xlabel="x", ylabel="y",
        title="T(x, y, z=$(round(zs, digits=4)), t=$(round(t, digits=4)))",
        c=:thermal, size=(800, 600) 
    ) 
end

gif(anim_T, "pinn_heatmap_z$(round(zs, digits=4)).gif", fps=10)

zs = 0.4
anim_T = @animate for t in ts
    T_pred = [phi([x, y, zs, t], res.u)[1] for x in xs, y in ys]
    heatmap(xs, ys, T_pred',
        xlabel="x", ylabel="y",
        title="T(x, y, z=$(round(zs, digits=4)), t=$(round(t, digits=4)))",
        c=:thermal, size=(800, 600) 
    ) 
end

gif(anim_T, "pinn_heatmap_z$(round(zs, digits=4)).gif", fps=10)

zs = 0.5
anim_T = @animate for t in ts
    T_pred = [phi([x, y, zs, t], res.u)[1] for x in xs, y in ys]
    heatmap(xs, ys, T_pred',
        xlabel="x", ylabel="y",
        title="T(x, y, z=$(round(zs, digits=4)), t=$(round(t, digits=4)))",
        c=:thermal, size=(800, 600) 
    ) 
end

gif(anim_T, "pinn_heatmap_z$(round(zs, digits=4)).gif", fps=1)

@show phi([0.0005, 0.0005, 0.00005, 0.005], res.u)[1]
@show phi([0.0005, 0.0005, 0.00010, 0.005], res.u)[1]
@show phi([0.0005, 0.0005, 0.00015, 0.005], res.u)[1]