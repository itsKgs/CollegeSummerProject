using MethodOfLines, NeuralPDE, Lux, OptimizationOptimJL, ModelingToolkit, DomainSets, OptimizationOptimisers, Optimization
using Plots, LineSearches, Optimisers
using LuxCUDA, Random, ComponentArrays

@parameters x y
@variables u(..) v(..) p(..)

# Define the independent variable
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

ν = 0.01

# Domain limits
x_min = y_min = 0.0
x_max = y_max = 1.0


# PDE system
#eqs = [
#    ρ * Dt(u(x, y, t)) + ρ * (u(x, y, t) * Dx(u(x, y, t)) + v(x, y, t) * Dy(u(x, y, t))) - (μ * (2 * Dxx(u(x, y, t)) + Dyy(u(x, y, t)) + DxDy(v(x, y, t))) - Dx(p(x, y, t))) ~ 0, 
#    ρ * Dt(v(x, y, t)) + ρ * (u(x, y, t) * Dx(v(x, y, t)) + v(x, y, t) * Dy(v(x, y, t))) - ((μ * (2 * Dxx(v(x, y, t)) + Dyy(v(x, y, t)) + DxDy(u(x, y, t))) - Dy(p(x, y, t)))) ~ 0, 
#    Dx(u(x, y, t)) + Dy(v(x, y, t)) ~ 0
#]

eqs = [
    u(x, y) * Dx(u(x, y)) + v(x, y) * Dy(u(x, y)) - ν * (Dxx(u(x, y)) + Dyy(u(x, y))) + Dx(p(x, y)) ~ 0, 
    u(x, y) * Dx(v(x, y)) + v(x, y) * Dy(v(x, y)) - ν * (Dxx(v(x, y)) + Dyy(v(x, y))) + Dy(p(x, y)) ~ 0, 
    Dx(u(x, y)) + Dy(v(x, y)) ~ 0
]

# Domains
domains = [
    x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max)
    #t ∈ Interval(t_min, t_max)
]

# Boundary Conditions
bcs = [
    u(0, y) ~ 0.0,  u(1, y) ~ 0.0,  u(x, 0) ~ 0.0,  u(x, 1) ~ 1.0,  # Lid: U = 1
    v(0, y) ~ 0.0,  v(1, y) ~ 0.0,  v(x, 0) ~ 0.0,  v(x, 1) ~ 0.0
]


@named pdesys = PDESystem(eqs, bcs, domains, [x, y], [u(x, y), v(x, y), p(x, y)])

# Neural network
chain = Chain(
    Dense(2, 20, tanh),
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

#chain = Chain(
#    Dense(3, 10, tanh), Dropout(0.1),
#    Dense(10, 10, tanh), Dropout(0.1),
#    Dense(10, 10, tanh), Dropout(0.1),
#    Dense(10, 3)
#)

dx = 0.04
dy = 0.04

#strategy = QuadratureTraining(; batch=200, abstol=1e-6, reltol=1e-6)
strategy = GridTraining([dx, dy])

#const gpud = gpu_device()
#ps = Lux.setup(Random.default_rng(), chain)[1]
#ps = ps |> ComponentArray |> gpud .|> Float64

discretization = PhysicsInformedNN(chain, strategy) #; init_params=ps
prob = discretize(pdesys, discretization)
symprob = symbolic_discretize(pdesys, discretization)

optim = Optimization.OptimizationFunction(prob.f, Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optim, prob.u0)

# Print initial loss
initial_loss = optprob.f(optprob.u0, nothing)
println("Initial loss = ", initial_loss)

pde_inner_loss_functions = symprob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = symprob.loss_functions.bc_loss_functions

#Callback function
global  iter = 0
losses = Float64[]
callback = function (p, l)
    global iter += 1

    #println("pde_losses: ", map(l_ -> l_(p.u), pde_inner_loss_functions))
    #println("bcs_losses: ", map(l_ -> l_(p.u), bcs_inner_loss_functions))
    println("Iteration: $iter, loss is: $l")

    # Print every 100 iterations
    if iter % 100 == 0
        push!(losses, l)
        #println("Iteration: $iter, loss is: $l")
        # Optional: print PDE and BC losses here if needed
        # println("pde_losses: ", map(l_ -> l_(p.u), pde_inner_loss_functions))
        # println("bcs_losses: ", map(l_ -> l_(p.u), bcs_inner_loss_functions))
    end

    return false
end

opt1 = OptimizationOptimisers.Adam(1e-3)
res1 = Optimization.solve(optprob, opt1; callback=callback, maxiters=10000)

#res = Optimization.solve(optprob, Optim.BFGS(); callback=callback, maxiters=10000)

plot(1:length(losses), losses, xlabel="Iteration", ylabel="Loss", title="Training Loss")

global  iter = 0
losses = Float64[]
prob1 = remake(optprob, u0 = res1.u)
#opt2 = LBFGS(linesearch = BackTracking())
res2 = Optimization.solve(prob1, Optim.BFGS(); callback = callback, maxiters = 10000)
plot(1:length(losses), losses, xlabel="Iteration", ylabel="Loss", title="Training Loss")

prob2 = remake(optprob, u0 = res2.u)
opt = LBFGS(linesearch = BackTracking())
res = Optimization.solve(prob2, opt; maxiters = 5000, callback)

#opt2 = LBFGS()
#res2 = Optimization.solve(prob, opt2; maxiters=500, u0=res1.u)

gr()
mkpath("NeuralPDE/Navier_stokes_channel")

phi = discretization.phi  # the neural network that maps (x, y, t) → [u, v, p]


## Plot solution
#xs = ys = range(0, 1, length=50)
#xv, yv = [x for x in xs], [y for y in ys]
#u_vals = [first(phi([x, y], res2.u)) for x in xs, y in ys]
#v_vals = [phi([x, y], res2.u)[2] for x in xs, y in ys]
#
## Quiver Plot
#heatmap(xs, ys, u_vals', xlabel="x", ylabel="y", title="U velocity", color=:viridis)

# Create a grid for plotting
nx, ny = 30, 30
xs = range(0, 1, length=nx)
ys = range(0, 1, length=ny)

xv = repeat(xs, inner=ny)
yv = repeat(ys, outer=nx)
#println("xv: ", xv)
#println("yv: ", yv)

# First plot the grid points
p = scatter(xv, yv, markersize=1.5, label="", title="Velocity Field (Grid + Quiver)",
            xlabel="x", ylabel="y", aspect_ratio=1, size=(800, 800))

# Evaluate velocity at each grid point
u_vals = [first(phi([x, y], res2.u)) for (x, y) in zip(xv, yv)]
v_vals = [phi([x, y], res2.u)[2] for (x, y) in zip(xv, yv)]


# Reshape into matrices for quiver
u_mat = reshape(u_vals, ny, nx)
v_mat = reshape(v_vals, ny, nx)

scale_factor = 0.09  # Adjust as needed

u_mat_scaled = u_mat .* scale_factor
v_mat_scaled = v_mat .* scale_factor

# Plot heatmap of speed
# Compute speed (magnitude of velocity)
#speed_mat = sqrt.(u_mat.^2 .+ v_mat.^2)
#
#p = heatmap(xs, ys, speed_mat, color=:plasma, colorbar=true,
#            xlabel="x", ylabel="y", title="Velocity Magnitude + Quiver", size=(800, 800),
#            aspect_ratio=1)
#
#
# Overlay the quiver plot
quiver(p, xv, yv, quiver=(u_mat, v_mat), color=:blue, label="",
       quiver_width=0.002, quiver_headwidth=0.01, quiver_headlength=0.00002, quiver_alpha=0.8)

# Display the plot
display(p)



#xs = ys = range(0.0, 1.0; length=50)  # spatial grid
#ts = range(0.0, 1.0; length=20)       # time steps for animation
#
## Predict u, v, and p at Each (x, y, t)
#function predict_field(phi, res2, xs, ys, t)
#    u_vals = [phi([x, y, t], res2.u)[1] for x in xs, y in ys]
#    v_vals = [phi([x, y, t], res2.u)[2] for x in xs, y in ys]
#    p_vals = [phi([x, y, t], res2.u)[3] for x in xs, y in ys]
#    return u_vals, v_vals, p_vals
#end
#u_vals, v_vals, p_vals = predict_field(phi, res2, xs, ys, 0.5)
#u_vals
#v_vals
#p_vals
#
## pressure
#anim = @animate for t in ts
#    u_vals, v_vals, p_vals = predict_field(phi, res, xs, ys, t)
#    p_pred = reshape(p_vals, length(xs), length(ys))
#    heatmap(
#        xs, ys, p_pred',
#        xlabel = "x", ylabel = "y",
#        title = "Pressure at t = $(round(t, digits=2))",
#        clims = (minimum(p_vals), maximum(p_vals)),
#        c = :thermal, aspect_ratio = 1
#    )
#end
#
#gif(anim, "NeuralPDE/Navier_stokes_channel/navier_stokes_pressure_heatmap.gif", fps=5)
#
## u(x, y, t)
#anim_u = @animate for t in ts
#    u_vals, v_vals, p_vals = predict_field(phi, res, xs, ys, t)
#    u_pred = reshape(u_vals, length(xs), length(ys))
#    heatmap(xs, ys, u_pred',
#        xlabel="x", ylabel="y",
#        title="u-component Velocity at t = $(round(t, digits=2))",
#        c=:blues, clims = (minimum(u_vals), maximum(u_vals)), size=(800, 600) 
#    )  
#end
#
#gif(anim_u, "NeuralPDE/Navier_stokes_channel/velocity_u_component_heatmap.gif", fps=1)
#
#
## Surface Animation for u(x, y, t)
#anim_u = @animate for t in ts
#    u_vals, v_vals, p_vals = predict_field(phi, res, xs, ys, t)
#    u_pred = reshape(u_vals, length(xs), length(ys))
#    surface(xs, ys, u_pred',
#        xlabel="x", ylabel="y", zlabel="u(x,y)",
#        title="u-component Velocity at t = $(round(t, digits=2))",
#        c=:blues, clims = (minimum(u_vals), maximum(u_vals)), size=(800, 600)
#    )  
#end
#
#gif(anim_u, "NeuralPDE/Navier_stokes_channel/velocity_u_component.gif", fps=1)
#
## v(x, y, t)
#anim_u = @animate for t in ts
#    u_vals, v_vals, p_vals = predict_field(phi, res, xs, ys, t)
#    v_pred = reshape(v_vals, length(xs), length(ys))
#    heatmap(xs, ys, v_pred',
#        xlabel="x", ylabel="y",
#        title="u-component Velocity at t = $(round(t, digits=2))",
#        c=:blues, clims = (minimum(u_vals), maximum(u_vals)), size=(800, 600) 
#    )  
#end
#
#gif(anim_u, "NeuralPDE/Navier_stokes_channel/velocity_v_component_heatmap.gif", fps=1)
#
#
## Surface Animation for v(x, y, t)
#anim_u = @animate for t in ts
#    u_vals, v_vals, p_vals = predict_field(phi, res, xs, ys, t)
#    v_pred = reshape(v_vals, length(xs), length(ys))
#    surface(xs, ys, v_pred',
#        xlabel="x", ylabel="y", zlabel="u(x,y)",
#        title="u-component Velocity at t = $(round(t, digits=2))",
#        c=:blues, clims = (minimum(u_vals), maximum(u_vals)), size=(800, 600)
#    )  
#end
#
#gif(anim_u, "NeuralPDE/Navier_stokes_channel/velocity_v_component.gif", fps=1)
#
#
##Surface Animation of Velocity Magnitude
#function predict_velocity_magnitude(phi, res, xs, ys, t)
#    [sqrt(phi([x, y, t], res.u)[1]^2 + phi([x, y, t], res.u)[2]^2) for x in xs, y in ys]
#end
#
#anim_mag = @animate for t in ts
#    mag = predict_velocity_magnitude(phi, res, xs, ys, t)
#    mag_pred = reshape(mag, length(xs), length(ys))
#    surface(xs, ys, mag_pred',
#        xlabel="x", ylabel="y", zlabel="|u(x,y)|",
#        title="Velocity Magnitude at t = $(round(t, digits=2))",
#        c=:plasma, clims=(0, 1.5), size=(800, 600),
#    )
#end
#
#gif(anim_mag, "NeuralPDE/Navier_stokes_channel/velocity_surface.gif", fps=5)
#
## Quiver Plot Animation (Velocity Vector Field)
#anim = @animate for t in ts
#    u_vals, v_vals, p_vals = predict_field(phi, res, xs, ys, t)
#    u_pred = reshape(u_vals, length(xs), length(ys))
#    v_pred = reshape(v_vals, length(xs), length(ys))
#    quiver(xs, ys, quiver=(u_pred', v_pred'),
#        xlabel="x", ylabel="y",
#        title="Velocity Field at t = $(round(t, digits=2))",
#        aspect_ratio=1, linealpha=0.8, size=(700, 600)
#    )
#end
#
#gif(anim, "NeuralPDE/Navier_stokes_channel/velocity_quiver.gif", fps=1)
