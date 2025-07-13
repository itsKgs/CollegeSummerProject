using ModelingToolkit, DomainSets, DifferentialEquations, MethodOfLines, Plots, Sundials

@parameters x y t
@variables u(..) v(..) p(..)

μ = 1.0
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

# Define the equations

# ϵ components (symmetric gradient)
#ϵx(u, v) = Dx(u(x, y, t))
#ϵy(u, v) = Dy(v(x, y, t))
#ϵxy(u, v) = ϵyx(u, v) = 0.5 * (Dy(u(x, y, t)) + Dx(v(x, y, t)))

## Stress (σ) components
#σx = 2 * μ * ϵx(u, v) - p(x, y, t)
#σy = 2 * μ * ϵy(u, v) - p(x, y, t)
#σxy = σyx = μ * (Dy(u(x, y, t)) + Dx(v(x, y, t))) 
#
## divergence of stress (∇.σ) components
#∇σx = Dx(σx) + Dy(σxy)
#∇σy = Dx(σyx) + Dy(σy)
#
## Nonlinear advection terms(u.∇)u
#adv_u = u(x, y, t) * Dx(u(x, y, t)) + v(x, y, t) * Dy(u(x, y, t))
#adv_v = u(x, y, t) * Dx(v(x, y, t)) + v(x, y, t) * Dy(v(x, y, t))

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
    Dt(u(x, y, 0)) ~ 0.0,

    #Dx(p(x, 0, t)) ~ 0.0,
    #Dy(p(x, 1, t)) ~ 0.0 
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
mkpath("MethodOfLines_jl/Navier_stokes_channel")

# Animate p(x,y,t) as Heatmap
x_vals = range(0, 1, length=N)
y_vals = range(0, 1, length=N)
t_vals = sol.t

# pressure
anim = @animate for i in eachindex(t_vals)
    heatmap(
        x_vals, y_vals, solp[:, :, i],
        xlabel = "x", ylabel = "y",
        title = "Pressure at t = $(round(t_vals[i], digits=2))",
        clims = (minimum(solp), maximum(solp)),
        c = :thermal, aspect_ratio = 1
    )
end

gif(anim, "MethodOfLines_jl/Navier_stokes_channel/navier_stokes_pressure_heatmap.gif", fps=5)

anim = @animate for i in eachindex(t_vals)
    surface(
        x_vals, y_vals, solp[:, :, i],
        xlabel = "x", ylabel = "y", zlabel = "P(x, y, t)",
        title = "Pressure at t = $(round(t_vals[i], digits=2))",
        clims = (minimum(solp), maximum(solp)),
        c = :thermal, aspect_ratio = 1, camera = (30 + i, 45)
    )
end

gif(anim, "MethodOfLines_jl/Navier_stokes_channel/navier_stokes_pressure_surface.gif", fps=5)


# Surface Animation for u(x, y, t)
anim_u = @animate for i in eachindex(t_vals)
    heatmap(
        x_vals, y_vals, solu[:, :, i],
        xlabel = "x", ylabel = "y",
        title = "u-component Velocity at t = $(round(t_vals[i], digits=2))",
        c = :blues, 
    )
end

gif(anim_u, "MethodOfLines_jl/Navier_stokes_channel/velocity_u_component_heatmap.gif", fps=1)


anim_u = @animate for i in eachindex(t_vals)
    surface(
        x_vals, y_vals, solu[:, :, i],
        xlabel = "x", ylabel = "y", zlabel = "u(x,y)",
        title = "u-component Velocity at t = $(round(t_vals[i], digits=2))",
        c = :blues, camera = (30 + i, 45),
        legend = true
    )
end

gif(anim_u, "MethodOfLines_jl/Navier_stokes_channel/velocity_u_component.gif", fps=1)

#Surface Animation for v(x, y, t)

anim_v = @animate for i in eachindex(t_vals)
    heatmap(
        x_vals, y_vals, solv[:, :, i],
        xlabel = "x", ylabel = "y", zlabel = "v(x,y)",
        title = "v-component Velocity at t = $(round(t_vals[i], digits=2))",
        c = :greens,
        legend = true
    )
end

gif(anim_v, "MethodOfLines_jl/Navier_stokes_channel/velocity_v_component_heatmap.gif", fps=1)

anim_v = @animate for i in eachindex(t_vals)
    surface(
        x_vals, y_vals, solv[:, :, i],
        xlabel = "x", ylabel = "y", zlabel = "v(x,y)",
        title = "v-component Velocity at t = $(round(t_vals[i], digits=2))",
        c = :greens, camera = (30 + i, 45),
        legend = true
    )
end

gif(anim_v, "MethodOfLines_jl/Navier_stokes_channel/velocity_v_component.gif", fps=1)

#Surface Animation of Velocity Magnitude
u_vals = solu
v_vals = solv
velocity_mag = sqrt.(u_vals.^2 .+ v_vals.^2)

anim = @animate for i in eachindex(t_vals)
    heatmap(
        x_vals, y_vals, velocity_mag[:, :, i],
        xlabel = "x", ylabel = "y",
        title = "Velocity Magnitude at t = $(round(t_vals[i], digits=2))",
        c = :viridis,
        legend = true
    )
end

gif(anim, "MethodOfLines_jl/Navier_stokes_channel/velocity_surface_heatmap.gif", fps=5)


anim = @animate for i in eachindex(t_vals)
    surface(
        x_vals, y_vals, velocity_mag[:, :, i],
        xlabel = "x", ylabel = "y", zlabel = "|u|",
        title = "Velocity Magnitude at t = $(round(t_vals[i], digits=2))",
        c = :viridis, camera = (30 + i, 45),
        legend = false
    )
end

gif(anim, "MethodOfLines_jl/Navier_stokes_channel/velocity_surface.gif", fps=5)

# 2D meshgrid for plotting
# Quiver Plot Animation (Velocity Vector Field)
X, Y = [x for x in x_vals, y in y_vals], [y for x in x_vals, y in y_vals]

anim = @animate for i in eachindex(t_vals)
    quiver(
        X, Y,
        quiver=(solu[:, :, i], solv[:, :, i]),
        title = "Velocity Field at t = $(round(t_vals[i], digits=2))",
        xlabel = "x", ylabel = "y",
        aspect_ratio = 1,
        linealpha=0.8
    )
end

gif(anim, "MethodOfLines_jl/Navier_stokes_channel/velocity_quiver.gif", fps=1)