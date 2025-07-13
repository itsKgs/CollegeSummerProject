using ModelingToolkit, MethodOfLines, DomainSets, DifferentialEquations
using Plots

# Method of Manufactured Solutions: exact solution
u_exact = (x, y, t) -> 16 * sin.(π * x) .* sin.(π * y) .* exp(-2 * π^2 * t) # All ops broadcasted


@parameters x y t
@variables u(..)

Dt = Differential(t)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# Domain limits
x_min = t_min = y_min = 0.0
x_max = y_max = 1.0
t_max = 0.2

k = 1.0 

# Initial conditions
u0(t, x, y) = 16*sin(π*x)*sin(π*y)

# 2. PDE system
eqs = [
    Dt(u(t, x, y)) ~ k*(Dxx(u(t, x, y)) + Dyy(u(t, x, y)))
]

# 3. Domains
domains = [
    x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max),
    t ∈ Interval(t_min, t_max)
]

# 4. Boundary and initial conditions (periodic in x and y)

bcs = [
    u(0, x, y) ~ 16*sin(π*x)*sin(π*y),
    u(t, 0, y) ~ 0.0,
    u(t, 1, y) ~ 0.0,
    u(t, x, 0) ~ 0.0,
    u(t, x, 1) ~ 0.0
] 

@named pdesys = PDESystem(eqs, bcs, domains, [x, y,t], [u(t, x, y)])

# 5. Discretize using MethodOfLines (MOL)
N = 32
#dx = 0.1
#dy = 0.05
discretization = MOLFiniteDifference([x => N, y => N], t, approx_order=2)

# 6. Convert PDE to ODE system
@time prob = discretize(pdesys, discretization)

# 7. Solve using a stiff ODE solver
sol = solve(prob, FBDF(), abstol=1e-8, reltol=1e-8, saveat=0.01)

discrete_x = sol[x]
discrete_y = sol[y]
discrete_t = sol[t]

solu = sol[u(t, x, y)]


anim = @animate for i in eachindex(discrete_t)
    heatmap(discrete_x, discrete_y, solu[i, :, :], 
        title="Numerical Solution at t=$(round(discrete_t[i], digits=3))",
        xlabel="x", ylabel="y",
        c=:thermal, clims=(0, 16))
end

gif(anim, "2D_heat_equation_animation.gif", fps=5)


# Animation
anim = @animate for i in eachindex(discrete_t)
    surface(
        discrete_x, discrete_y, solu[i, :, :],
        xlabel = "x axis (mm)", ylabel = "y axis (mm)", zlabel = "Temperature (°C)",
        title = "Temperature Profile at t = $(round(discrete_t[i], digits=2))",
        c=:viridis, clims = (0, 200),
        camera = (60, 45),
        legend = true
    )
end

gif(anim, "3D_temperature_surface.gif", fps = 1)

# Create 2D meshgrid to evaluate exact solution
X = repeat(discrete_x', N, 1)  # x grid repeated across rows
Y = repeat(discrete_y, 1, N)   # y grid repeated across columns

# 10. Plot both as heatmaps
anim = @animate for i in eachindex(discrete_t)
    u_ex = u_exact.(X, Y, discrete_t[i])

    # Plot side-by-side heatmaps
    p1 = heatmap(discrete_x, discrete_y, solu[i, :, :],
        title = "Numerical at t = $(round(discrete_t[i], digits=2))",
        xlabel = "x", ylabel = "y", c = :thermal, clims = (0, 16), aspect_ratio = 1)

    p2 = heatmap(discrete_x, discrete_y, u_ex,
        title = "Exact at t = $(round(discrete_t[i], digits=2))",
        xlabel = "x", ylabel = "y", c = :thermal, clims = (0, 16), aspect_ratio = 1)

    plot(p1, p2, layout = (1, 2), size = (900, 400))
end

# Save the animation
gif(anim, "2D_HeatEqn_heatmap_numerical_vs_exact.gif", fps = 5)

# Create meshgrid for broadcasting exact solution
X = repeat(discrete_x', N, 1)
Y = repeat(discrete_y, 1, N)

anim = @animate for i in eachindex(discrete_t)
    u_ex = u_exact.(X, Y, discrete_t[i])          # exact solution at current time

    azimuth = 30 + 2*i  # slowly rotating view

    plt1 = surface(discrete_x, discrete_y, solu[i, :, :],
        title = "Numerical at t = $(round(discrete_t[i], digits=2))",
        xlabel = "x", ylabel = "y", zlabel = "u(x,y,t)",
        c = :thermal, clims = (0, 16), camera = (azimuth, 45),
        legend = false)

    plt2 = surface(discrete_x, discrete_y, u_ex,
        title = "Exact at t = $(round(discrete_t[i], digits=2))",
        xlabel = "x", ylabel = "y", zlabel = "u(x,y,t)",
        c = :thermal, clims = (0, 16), camera = (azimuth, 45),
        legend = false)

    plot(plt1, plt2, layout = (1, 2), size = (1000, 400))
end

gif(anim, "2D_HeatEqn_heat_equation_exact_vs_numerical.gif", fps = 2)


