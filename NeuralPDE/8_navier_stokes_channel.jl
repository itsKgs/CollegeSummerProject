using MethodOfLines, NeuralPDE, Lux, OptimizationOptimJL, ModelingToolkit, DomainSets, OptimizationOptimisers, Optimization
using Plots, LineSearches

@parameters x y t
@variables u(..) v(..) p(..)

# Define the independent variable
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
DxDy = Dx * Dy

μ = 1.0
ρ = 1.0

# Domain limits
x_min = y_min = t_min = 0.0
x_max = y_max = t_max = 1.0


# PDE system
eqs = [
    ρ * Dt(u(x, y, t)) + ρ * (u(x, y, t) * Dx(u(x, y, t)) + v(x, y, t) * Dy(u(x, y, t))) - (μ * (2 * Dxx(u(x, y, t)) + Dyy(u(x, y, t)) + DxDy(v(x, y, t))) - Dx(p(x, y, t))) ~ 0, 
    ρ * Dt(v(x, y, t)) + ρ * (u(x, y, t) * Dx(v(x, y, t)) + v(x, y, t) * Dy(v(x, y, t))) - ((μ * (2 * Dxx(v(x, y, t)) + Dyy(v(x, y, t)) + DxDy(u(x, y, t))) - Dy(p(x, y, t)))) ~ 0, 
    Dx(u(x, y, t)) + Dy(v(x, y, t)) ~ 0
]

# Domains
domains = [
    x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max),
    t ∈ Interval(t_min, t_max)
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


@named pdesys = PDESystem(eqs, bcs, domains, [x, y, t], [u(x, y, t), v(x, y, t), p(x, y, t)])

# Neural network
chain = Chain(
    Dense(3, 64, tanh),
    Dense(64, 64, tanh),
    Dense(64, 3)
)

dx = 0.05
dy = 0.05
dt = 0.05


#strategy = QuadratureTraining(; batch=200, abstol=1e-6, reltol=1e-6)
strategy = GridTraining([dx, dy, dt])
discretization = PhysicsInformedNN(chain, strategy)

prob = discretize(pdesys, discretization)

#Callback function
global  iter = 0
callback = function (p, l)
    global iter =  iter + 1
    println("Iteration: $iter, loss is: $l")
    return false
end


opt = OptimizationOptimJL.BFGS()
#opt = LBFGS(linesearch = BackTracking())
#opt = OptimizationOptimisers.ADAM(0.01)

res = Optimization.solve(prob, opt; maxiters = 500, callback)

#opt2 = LBFGS()
#res2 = Optimization.solve(prob, opt2; maxiters=500, u0=res1.u)

gr()
mkpath("NeuralPDE/Navier_stokes_channel")

phi = discretization.phi  # the neural network that maps (x, y, t) → [u, v, p]

xs = ys = range(0.0, 1.0; length=50)  # spatial grid
ts = range(0.0, 1.0; length=20)       # time steps for animation

# Predict u, v, and p at Each (x, y, t)
function predict_field(phi, res, xs, ys, t)
    u_vals = [phi([x, y, t], res.u)[1] for x in xs, y in ys]
    v_vals = [phi([x, y, t], res.u)[2] for x in xs, y in ys]
    p_vals = [phi([x, y, t], res.u)[3] for x in xs, y in ys]
    return u_vals, v_vals, p_vals
end
u_vals, v_vals, p_vals = predict_field(phi, res, xs, ys, 0.5)
u_vals
v_vals
p_vals

# pressure
anim = @animate for t in ts
    u_vals, v_vals, p_vals = predict_field(phi, res, xs, ys, t)
    p_pred = reshape(p_vals, length(xs), length(ys))
    heatmap(
        xs, ys, p_pred',
        xlabel = "x", ylabel = "y",
        title = "Pressure at t = $(round(t, digits=2))",
        clims = (minimum(p_vals), maximum(p_vals)),
        c = :thermal, aspect_ratio = 1
    )
end

gif(anim, "NeuralPDE/Navier_stokes_channel/navier_stokes_pressure_heatmap.gif", fps=5)

# u(x, y, t)
anim_u = @animate for t in ts
    u_vals, v_vals, p_vals = predict_field(phi, res, xs, ys, t)
    u_pred = reshape(u_vals, length(xs), length(ys))
    heatmap(xs, ys, u_pred',
        xlabel="x", ylabel="y",
        title="u-component Velocity at t = $(round(t, digits=2))",
        c=:blues, clims = (minimum(u_vals), maximum(u_vals)), size=(800, 600) 
    )  
end

gif(anim_u, "NeuralPDE/Navier_stokes_channel/velocity_u_component_heatmap.gif", fps=1)


# Surface Animation for u(x, y, t)
anim_u = @animate for t in ts
    u_vals, v_vals, p_vals = predict_field(phi, res, xs, ys, t)
    u_pred = reshape(u_vals, length(xs), length(ys))
    surface(xs, ys, u_pred',
        xlabel="x", ylabel="y", zlabel="u(x,y)",
        title="u-component Velocity at t = $(round(t, digits=2))",
        c=:blues, clims = (minimum(u_vals), maximum(u_vals)), size=(800, 600)
    )  
end

gif(anim_u, "NeuralPDE/Navier_stokes_channel/velocity_u_component.gif", fps=1)

# v(x, y, t)
anim_u = @animate for t in ts
    u_vals, v_vals, p_vals = predict_field(phi, res, xs, ys, t)
    v_pred = reshape(v_vals, length(xs), length(ys))
    heatmap(xs, ys, v_pred',
        xlabel="x", ylabel="y",
        title="u-component Velocity at t = $(round(t, digits=2))",
        c=:blues, clims = (minimum(u_vals), maximum(u_vals)), size=(800, 600) 
    )  
end

gif(anim_u, "NeuralPDE/Navier_stokes_channel/velocity_v_component_heatmap.gif", fps=1)


# Surface Animation for v(x, y, t)
anim_u = @animate for t in ts
    u_vals, v_vals, p_vals = predict_field(phi, res, xs, ys, t)
    v_pred = reshape(v_vals, length(xs), length(ys))
    surface(xs, ys, v_pred',
        xlabel="x", ylabel="y", zlabel="u(x,y)",
        title="u-component Velocity at t = $(round(t, digits=2))",
        c=:blues, clims = (minimum(u_vals), maximum(u_vals)), size=(800, 600)
    )  
end

gif(anim_u, "NeuralPDE/Navier_stokes_channel/velocity_v_component.gif", fps=1)


#Surface Animation of Velocity Magnitude
function predict_velocity_magnitude(phi, res, xs, ys, t)
    [sqrt(phi([x, y, t], res.u)[1]^2 + phi([x, y, t], res.u)[2]^2) for x in xs, y in ys]
end

anim_mag = @animate for t in ts
    mag = predict_velocity_magnitude(phi, res, xs, ys, t)
    mag_pred = reshape(mag, length(xs), length(ys))
    surface(xs, ys, mag_pred',
        xlabel="x", ylabel="y", zlabel="|u(x,y)|",
        title="Velocity Magnitude at t = $(round(t, digits=2))",
        c=:plasma, clims=(0, 1.5), size=(800, 600),
    )
end

gif(anim_mag, "NeuralPDE/Navier_stokes_channel/velocity_surface.gif", fps=5)

# Quiver Plot Animation (Velocity Vector Field)
anim = @animate for t in ts
    u_vals, v_vals, p_vals = predict_field(phi, res, xs, ys, t)
    u_pred = reshape(u_vals, length(xs), length(ys))
    v_pred = reshape(v_vals, length(xs), length(ys))
    quiver(xs, ys, quiver=(u_pred', v_pred'),
        xlabel="x", ylabel="y",
        title="Velocity Field at t = $(round(t, digits=2))",
        aspect_ratio=1, linealpha=0.8, size=(700, 600)
    )
end

gif(anim, "NeuralPDE/Navier_stokes_channel/velocity_quiver.gif", fps=1)
