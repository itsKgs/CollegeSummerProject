using ModelingToolkit, MethodOfLines, DomainSets, DifferentialEquations
using Plots

# Method of Manufactured Solutions: exact solution
u_exact = (x, y, z, t) -> 6 * sin.(π * x) .* sin.(π * y) .* sin.(π * z) .* exp(-3 * π^2 * t) # All ops broadcasted


@parameters x y z t
@variables u(..)

Dt = Differential(t)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dzz = Differential(z)^2

# Domain limits
x_min = t_min = y_min = y_min = 0.0
x_max = y_max = y_max = 1.0
t_max = 0.2

k = 1.0

# Initial conditions
u0(t, x, y, z) = 6*sin(π*x)*sin(π*y)*sin(π*z)

# 2. PDE system
eqs = [
    Dt(u(t, x, y, z)) ~ k*(Dxx(u(t, x, y, z)) + Dyy(u(t, x, y, z)) + Dzz(u(t, x, y, z)))
]

# 3. Domains
domains = [
    x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max),
    z ∈ Interval(y_min, y_max),
    t ∈ Interval(t_min, t_max)
]

# 4. Boundary and initial conditions (periodic in x and y)

bcs = [
    u(0, x, y, z) ~ 16*sin(π*x)*sin(π*y)*sin(π*z),
    u(t, 0, y, z) ~ 0.0,
    u(t, 1, y, z) ~ 0.0,
    u(t, x, 0, z) ~ 0.0,
    u(t, x, 1, z) ~ 0.0,
    u(t, x, y, 0) ~ 0.0,
    u(t, x, y, 1) ~ 0.0
] 

@named pdesys = PDESystem(eqs, bcs, domains, [x, y, z, t], [u(t, x, y, z)])

# 5. Discretize using MethodOfLines (MOL)
N = 16
#dx = 0.1
#dy = 0.05
discretization = MOLFiniteDifference([x => N, y => N, z => N], t, approx_order=2)

# 6. Convert PDE to ODE system
@time prob = discretize(pdesys, discretization; simplify=false)

# 7. Solve using a stiff ODE solver
sol = solve(prob, TRBDF2(), saveat=0.01, abstol=1e-6, reltol=1e-6)

discrete_x = sol[x]
discrete_y = sol[y]
discrete_z = sol[z]
discrete_t = sol[t]

solu = sol[u(t, x, y, z)]

# Choose fixed z and t values
z_val = 0.5
t_val = 0.2

# Find nearest indices
z_idx = findfirst(z -> abs(z - z_val) < 0.01, discrete_z)
t_idx = findfirst(t -> abs(t - t_val) < 0.01, discrete_t)

println(discrete_z)  # List of z values
println(discrete_t)  # List of t values


# Use actual available values
z_val = discrete_z[5]   # pick from printed list
t_val = discrete_t[4]   # pick from printed list

# Find indices safely
z_idx = findfirst(z -> abs(z - z_val) < 1e-5, discrete_z)
t_idx = findfirst(t -> abs(t - t_val) < 1e-5, discrete_t)

# Check if found
if isnothing(z_idx) || isnothing(t_idx)
    error("Chosen z or t value not found. Pick valid ones from discrete_z/t.")
end

# Get 2D slice of solution
u_slice = solu[t_idx, :, :, z_idx]

# Plot
using Plots
heatmap(discrete_x, discrete_y, u_slice',
    xlabel = "x", ylabel = "y",
    title = "Heatmap at z = $(round(z_val, digits=3)), t = $(round(t_val, digits=3))",
    c = :thermal, aspect_ratio = 1)


using Plots

# 1. Select middle z index robustly
z_index = length(discrete_z) ÷ 2
z_val = discrete_z[z_index]

# 2. Animate over time
anim = @animate for i in eachindex(discrete_t)
    heatmap(
        discrete_x, discrete_y, solu[i, :, :, z_index]',
        title = "u(x, y, z=$(round(z_val, digits=2)), t=$(round(discrete_t[i], digits=2)))",
        xlabel = "x", ylabel = "y",
        clims = (0, 64), c = :thermal, aspect_ratio = 1
    )
end

# 3. Save GIF
gif(anim, "3D_heat_eq_z$(round(z_val, digits=2))_heatmap.gif", fps = 1)



# Safely get index closest to z = 0.5
z_index = argmin(abs.(discrete_z .- 0.9))
z_val = discrete_z[z_index]  # Actual value at that index

# Create animation
anim = @animate for i in eachindex(discrete_t)
    heatmap(
        discrete_x, discrete_y, solu[i, :, :, z_index],
        title = "u(x, y, z=$(round(z_val, digits=3)), t=$(round(discrete_t[i], digits=2)))",
        xlabel = "x", ylabel = "y", clims=(0, 64), c=:thermal,
        aspect_ratio=1
    )
end

gif(anim, "3D_heat_eq_z$(round(z_val, digits=2))_heatmap.gif", fps=5)

anim = @animate for i in eachindex(discrete_t)
    surface(
        discrete_x, discrete_y, solu[i, :, :, z_index],
        xlabel = "x", ylabel = "y", zlabel = "u(x,y)",
        title = "Surface at z=0.5, t=$(round(discrete_t[i], digits=2))",
        c = :viridis, clims = (0, 64),
        camera = (30 + i*2, 45), legend = false
    )
end

gif(anim, "3D_heat_eq_z0.5_surface.gif", fps = 5)


X = repeat(discrete_x', N, 1)
Y = repeat(discrete_y, 1, N)
Z_val = discrete_z[z_index]


anim = @animate for i in eachindex(discrete_t)
    u_ex = u_exact.(X, Y, Z_val, discrete_t[i])          # exact solution at current time

    azimuth = 30 + 2*i  # slowly rotating view

    plt1 = surface(discrete_x, discrete_y, solu[i, :, :, z_index],
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

gif(anim, "3D_HeatEqn_heat_equation_exact_vs_numerical.gif", fps = 2)


anim = @animate for i in eachindex(discrete_t)
    u_ex = u_exact.(X, Y, Z_val, discrete_t[i])

    p1 = heatmap(discrete_x, discrete_y, solu[i, :, :, z_index],
        title = "Numerical at t = $(round(discrete_t[i], digits=2))",
        xlabel = "x", ylabel = "y", c = :thermal, clims = (0, 64), aspect_ratio = 1)

    p2 = heatmap(discrete_x, discrete_y, u_ex,
        title = "Exact at t = $(round(discrete_t[i], digits=2))",
        xlabel = "x", ylabel = "y", c = :thermal, clims = (0, 64), aspect_ratio = 1)

    plot(p1, p2, layout = (1, 2), size = (1000, 400))
end

gif(anim, "3D_heat_eq_numerical_vs_exact.gif", fps = 5)


# Animation: plot all z-slices for each time step
anim = @animate for i in eachindex(discrete_t)
    plots = []

    for k in eachindex(discrete_z)
        p = heatmap(
            discrete_x, discrete_y, solu[i, :, :, k],
            title = "t = $(round(discrete_t[i], digits=2)), z = $(round(discrete_z[k], digits=2))",
            xlabel = "x", ylabel = "y", c = :thermal, clims = (0, 64),
            aspect_ratio = 1, framestyle = :box
        )
        push!(plots, p)
    end

    plot(plots..., layout = (ceil(Int, sqrt(length(discrete_z))), :), size = (1000, 900))
end

gif(anim, "3D_heat_all_zslices.gif", fps = 2)