
using ModelingToolkit, DifferentialEquations, MethodOfLines, DomainSets, LinearAlgebra

# Define parameters
@parameters t x y
@variables u(..) v(..) p(..) u_star(..) v_star(..)
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

#parameters
ν = 0.01  # Kinematic viscosity
dt = 0.01  # Time step size
ρ = 1.0  # Density

# Step 1: Intermediate velocity without pressure
momentum_eqs = [
    Dt(u_star(t, x, y)) ~ -u(t, x, y)*Dx(u(t, x, y)) - v(t, x, y)*Dy(u(t, x, y)) + ν*(Dxx(u(t, x, y)) + Dyy(u(t, x, y))),
    Dt(v_star(t, x, y)) ~ -u(t, x, y)*Dx(v(t, x, y)) - v(t, x, y)*Dy(v(t, x, y)) + ν*(Dxx(v(t, x, y)) + Dyy(v(t, x, y)))
]

# Step 2: Pressure Poisson equation
pressure_eq = [
    Dxx(p(t, x, y)) + Dyy(p(t, x, y)) ~ (ρ / dt) * (Dx(u_star(t, x, y)) + Dy(v_star(t, x, y)))
]

# Step 3: Velocity correction step
projection_eqs = [
    u(t, x, y) ~ u_star(t, x, y) - dt * Dx(p(t, x, y)),
    v(t, x, y) ~ v_star(t, x, y) - dt * Dy(p(t, x, y))
]

# Define domain
domains = [t ∈ IntervalDomain(0.0, 1.0), x ∈ IntervalDomain(0.0, 1.0), y ∈ IntervalDomain(0.0, 1.0)]

# Boundary and initial conditions
bcs_momentum = [
    #u(0, x, y) ~ 0.0,
    #v(0, x, y) ~ 0.0,

    u_star(t, 0, y) ~ 0.0, v_star(t, 0, y) ~ 0.0, 
    u_star(t, 1, y) ~ 0.0, v_star(t, 1, y) ~ 0.0,

    u_star(t, x, 0) ~ 0.0, v_star(t, x, 0) ~ 0.0,
    u_star(t, x, 1) ~ 1.0, v_star(t, x, 1) ~ 0.0
  
    #u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0,
    #u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,  
    #v(t, 0, y) ~ 0.0, v(t, 1, y) ~ 0.0,
    #v(t, x, 0) ~ 0.0, v(t, x, 1) ~ 0.0
]

# Combine equations
#eqs = vcat(momentum_eqs, pressure_eq, projection_eqs)

# Step 1: Solve for intermediate velocity
@named momentum_pde = PDESystem(momentum_eqs, bcs_momentum, domains, [t, x, y], [u_star(t, x, y), v_star(t, x, y)])

# Spatial discretization (Method of Lines)
dx = 0.05
dy = 0.05
discretization = MOLFiniteDifference([x => dx, y => dy], t)

# Discretize PDE system
@time prob_m = discretize(momentum_pde, discretization)

# Solve system
@time sol_m = solve(prob_m, Rodas5(), saveat=0.1)


# Step 2: Solve Pressure Poisson Equation
bcs_pressure = [
    p(0, x, y) ~ 0.0,
    Dx(p(t, 0, y)) ~ 0.0, Dx(p(t, 1, y)) ~ 0.0,
    Dy(p(t, x, 0)) ~ 0.0, Dy(p(t, x, 1)) ~ 0.0
]

@named pressure_pde = PDESystem(pressure_eq, bcs_pressure, domains, [t, x, y], [p(t, x, y)])
prob_p = discretize(pressure_pde, discretization)
sol_p = solve(prob_p, Rodas5(), saveat=0.1)


# Step 3: Final corrected velocity
bcs_projection = [
    u(0, x, y) ~ 0.0,
    v(0, x, y) ~ 0.0,
    #u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0,
    #u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    #v(t, 0, y) ~ 0.0, v(t, 1, y) ~ 0.0,
    #v(t, x, 0) ~ 0.0, v(t, x, 1) ~ 0.0
]

@named projection_pde = PDESystem(projection_eqs, bcs_projection, domains, [t, x, y], [u(t, x, y), v(t, x, y)])
projection_prob = discretize(projection_pde, discretization)       
projection_sol = solve(projection_prob, Tsit5(), dt=dt_val, saveat=0.01)

discrete_x = sol[x]
discrete_y = sol[y]
discrete_t = sol[t]

solu = sol[u(x, y, t)]
solv = sol[v(x, y, t)]
solp = sol[p(x, y, t)]

gr(size=(800, 600))

# Animate p(x,y,t) as Heatmap
x_vals = range(0, 1, length=N)
y_vals = range(0, 1, length=N)
t_vals = sol.t
pressure = sol[p(x, y, t)]

# pressure
anim = @animate for i in eachindex(t_vals)
    heatmap(
        x_vals, y_vals, pressure[:, :, i],
        xlabel = "x", ylabel = "y",
        title = "Pressure at t = $(round(t_vals[i], digits=2))",
        clims = (minimum(pressure), maximum(pressure)),
        c = :thermal, aspect_ratio = 1#, camera = (30 + i, 45)
    )
end

gif(anim, "navier_stokes_pressure_heatmap.gif", fps=5)


# Surface Animation for u(x, y, t)
anim_u = @animate for i in eachindex(t_vals)
    surface(
        x_vals, y_vals, solu[:, :, i],
        xlabel = "x", ylabel = "y", zlabel = "u(x,y)",
        title = "u-component Velocity at t = $(round(t_vals[i], digits=2))",
        c = :blues, camera = (30 + i, 45),
        legend = false
    )
end

gif(anim_u, "velocity_u_component.gif", fps=1)

#Surface Animation for v(x, y, t)
anim_v = @animate for i in eachindex(t_vals)
    surface(
        x_vals, y_vals, solv[:, :, i],
        xlabel = "x", ylabel = "y", zlabel = "v(x,y)",
        title = "v-component Velocity at t = $(round(t_vals[i], digits=2))",
        c = :greens, camera = (30 + i, 45),
        legend = false
    )
end

gif(anim_v, "velocity_v_component.gif", fps=5)

#Surface Animation of Velocity Magnitude
u_vals = solu
v_vals = solv
velocity_mag = sqrt.(u_vals.^2 .+ v_vals.^2)

anim = @animate for i in eachindex(t_vals)
    surface(
        x_vals, y_vals, velocity_mag[:, :, i],
        xlabel = "x", ylabel = "y", zlabel = "|u|",
        title = "Velocity Magnitude at t = $(round(t_vals[i], digits=2))",
        c = :viridis, camera = (30 + i, 45),
        legend = false
    )
end

gif(anim, "velocity_surface.gif", fps=5)

# 2D meshgrid for plotting
# Quiver Plot Animation (Velocity Vector Field)
X, Y = [x for x in x_vals, y in y_vals], [y for x in x_vals, y in y_vals]

plot(X, Y)

anim = @animate for i in eachindex(t_vals)
    quiver(
        X, Y,
        quiver=(solu[:, :, i]', solv[:, :, i]'),
        title = "Velocity Field at t = $(round(t_vals[i], digits=2))",
        xlabel = "x", ylabel = "y",
        aspect_ratio = 1,
        linealpha=0.8
    )
end

gif(anim, "velocity_quiver.gif", fps=1)

