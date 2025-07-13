using ModelingToolkit, DomainSets, DifferentialEquations, MethodOfLines, Plots, Sundials

@parameters x y t
@variables u(..) v(..) p(..)

μ = 0.01
ρ = 1.0

# Define the independent variable
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
DxDy = Dx * Dy

# Domain limits
x_min = y_min = t_min = 0.0
x_max = y_max = t_max = 1.0

# Initial conditions
#u0(x, y, t) = 0
#v0(x, y, t) = 0
#p0(x, y, t) = 0

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

    u(x, 1, t) ~ 1.0,  v(x, 1, t) ~ 0.0,      # Top lid (moving right)
    u(x, 0, t) ~ 0.0,  v(x, 0, t) ~ 0.0,      # Bottom
    u(0, y, t) ~ 0.0,  v(0, y, t) ~ 0.0,      # Left wall
    u(1, y, t) ~ 0.0,  v(1, y, t) ~ 0.0,       # Right wall
                     
    Dt(u(x, y, 0)) ~ 0.0,
]

@named pdesys = PDESystem(eqs, bcs, domains, [x, y, t], [u(x, y, t), v(x, y, t), p(x, y, t)])

# Discretize using MethodOfLines (MOL)
N = 16
discretization = MOLFiniteDifference([x => N, y => N], t, approx_order=2)

# Convert PDE to ODE system
@time prob = discretize(pdesys, discretization)

# Solve using a stiff ODE solver
sol = solve(prob, Rodas5(), saveat=0.05)


discrete_x = sol[x]
discrete_y = sol[y]
discrete_t = sol[t]

solu = sol[u(x, y, t)]
solv = sol[v(x, y, t)]
solp = sol[p(x, y, t)]


gr(size=(800, 600))
mkpath("Navier_stokes_channel")

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

gif(anim, "Navier_stokes_channel/navier_stokes_pressure_heatmap.gif", fps=5)


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

gif(anim_u, "Navier_stokes_channel/velocity_u_component.gif", fps=1)

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

gif(anim_v, "Navier_stokes_channel/velocity_v_component.gif", fps=5)

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

gif(anim, "Navier_stokes_channel/velocity_surface.gif", fps=5)

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

gif(anim, "Navier_stokes_channel/velocity_quiver.gif", fps=1)