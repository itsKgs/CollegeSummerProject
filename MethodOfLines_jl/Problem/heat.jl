using ModelingToolkit, MethodOfLines, DomainSets, DifferentialEquations
using Plots


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
z_max = 0.3
t_max = 1.0

k = 1.0
ρ = 1.0
C_p = 700.0
D = 1/(ρ * C_p)
a = 0.2
b = 0.2
c = 0.05
q = 1.0
v = 1.0


# Initial conditions
T0(x, y, z, t) = 300.0 

Q(x, y, z, t) = ((6 * sqrt(3) * q)/(a * b * c * π * sqrt(π))) * exp(-3 * (((x^2)/(a^2)) + ((y^2)/(b^2)) + (((z - v * t)^2)/(c^2))))

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
    T(x, y, 0.3, t) ~ 300.0
] 

@named pdesys = PDESystem(eqs, bcs, domains, [x, y, z, t], [T(x, y, z, t)])

# 5. Discretize using MethodOfLines (MOL)
N_x = 16
N_y = 16
N_z = 7
discretization = MOLFiniteDifference([x => N_x, y => N_y, z => N_z], t, approx_order=2)

# 6. Convert PDE to ODE system
@time prob = discretize(pdesys, discretization)

# 7. Solve using a stiff ODE solver
sol = solve(prob, TRBDF2(), saveat=0.05, abstol=1e-6, reltol=1e-6)

discrete_x = sol[x]
discrete_y = sol[y]
discrete_z = sol[z]
discrete_t = sol[t]

solu = sol[T(x, y, z, t)]

gr(size=(800, 600)) 

mkpath("MethodOfLines_jl/Problem/heat")

# Safely get index closest to z = 0.0
z_index = argmin(abs.(discrete_z .- 0.0))
z_val = discrete_z[z_index]  # Actual value at that index

# Create animation
anim = @animate for i in eachindex(discrete_t)
    heatmap(
        discrete_x, discrete_y, solu[:, :, z_index, i],
        title = "T(x, y, z=$(round(z_val, digits=3)), t=$(round(discrete_t[i], digits=2)))",
        xlabel = "x", ylabel = "y", c=:thermal,
        aspect_ratio=1
    )
end

gif(anim, "MethodOfLines_jl/Problem/heat/heat_eq_z$(round(z_val, digits=2))_heatmap.gif", fps=5)


anim = @animate for i in eachindex(discrete_t)
    surface(
        discrete_x, discrete_y, transpose(solu[:, :, z_index, i]),
        xlabel = "x", ylabel = "y", zlabel = "T(x, y)",
        title = "T(x, y, z=$(round(z_val, digits=3)), t=$(round(discrete_t[i], digits=2)))",
        c = :viridis,
        camera = (30 + i*2, 45),  # smooth rotating view
        legend = true
    )
end

gif(anim, "MethodOfLines_jl/Problem/heat/heat_eqn_z$(round(z_val, digits=2))_surface_anim.gif", fps=5)



# Safely get index closest to z = 0.5
z_index = argmin(abs.(discrete_z .- 0.5))
z_val = discrete_z[z_index]  # Actual value at that index

# Create animation
anim = @animate for i in eachindex(discrete_t)
    heatmap(
        discrete_x, discrete_y, solu[:, :, z_index, i],
        title = "T(x, y, z=$(round(z_val, digits=3)), t=$(round(discrete_t[i], digits=2)))",
        xlabel = "x", ylabel = "y", c=:thermal,
        aspect_ratio=1
    )
end

gif(anim, "MethodOfLines_jl/Problem/heat/heat_eq_z$(round(z_val, digits=2))_heatmap.gif", fps=5)


anim = @animate for i in eachindex(discrete_t)
    surface(
        discrete_x, discrete_y, transpose(solu[:, :, z_index, i]),
        xlabel = "x", ylabel = "y", zlabel = "T(x, y)",
        title = "T(x, y, z=$(round(z_val, digits=3)), t=$(round(discrete_t[i], digits=2)))",
        c = :viridis, 
        camera = (30 + i*2, 45),  # smooth rotating view
        legend = true
    )
end

gif(anim, "MethodOfLines_jl/Problem/heat/heat_eqn_z$(round(z_val, digits=2))_surface_anim.gif", fps=5)


# Safely get index closest to z = 1.0
z_index = argmin(abs.(discrete_z .- 1.1))
z_val = discrete_z[z_index]  # Actual value at that index

# Create animation
anim = @animate for i in eachindex(discrete_t)
    heatmap(
        discrete_x, discrete_y, solu[:, :, z_index, i],
        title = "T(x, y, z=$(round(z_val, digits=3)), t=$(round(discrete_t[i], digits=2)))",
        xlabel = "x", ylabel = "y", clims=(0, 64), c=:thermal,
        aspect_ratio=1
    )
end

gif(anim, "MethodOfLines_jl/Problem/heat/heat_eq_z$(round(z_val, digits=2))_heatmap.gif", fps=5)


anim = @animate for i in eachindex(discrete_t)
    surface(
        discrete_x, discrete_y, transpose(solu[:, :, z_index, i]),
        xlabel = "x", ylabel = "y", zlabel = "T(x, y)",
        title = "T(x, y, z=$(round(z_val, digits=3)), t=$(round(discrete_t[i], digits=2)))",
        c = :viridis, clims = (0, 64),
        camera = (30 + i*2, 45),  # smooth rotating view
        legend = true
    )
end

gif(anim, "MethodOfLines_jl/Problem/heat/heat_eqn_z$(round(z_val, digits=2))_surface_anim.gif", fps=5)


# Safely get index closest to z = 1.5
z_index = argmin(abs.(discrete_z .- 1.5))
z_val = discrete_z[z_index]  # Actual value at that index

# Create animation
anim = @animate for i in eachindex(discrete_t)
    heatmap(
        discrete_x, discrete_y, solu[i, :, :, z_index],
        title = "T(x, y, z=$(round(z_val, digits=3)), t=$(round(discrete_t[i], digits=2)))",
        xlabel = "x", ylabel = "y", clims=(0, 64), c=:thermal,
        aspect_ratio=1
    )
end

gif(anim, "MethodOfLines_jl/Problem/heat/heat_eq_z$(round(z_val, digits=2))_heatmap.gif", fps=5)

anim = @animate for i in eachindex(discrete_t)
    surface(
        discrete_x, discrete_y, transpose(solu[i, :, :, z_index]),
        xlabel = "x", ylabel = "y", zlabel = "T(x, y)",
        title = "Surface at z=$(round(z_val, digits=2)), t=$(round(discrete_t[i], digits=2))",
        c = :viridis, clims = (0, 64),
        camera = (30 + i*2, 45),  # smooth rotating view
        legend = true
    )
end

gif(anim, "MethodOfLines_jl/Problem/heat/heat_eqn_z$(round(z_val, digits=2))_surface_anim.gif", fps=5)

# Safely get index closest to z = 2.0
z_index = argmin(abs.(discrete_z .- 2.0))
z_val = discrete_z[z_index]  # Actual value at that index

# Create animation
anim = @animate for i in eachindex(discrete_t)
    heatmap(
        discrete_x, discrete_y, solu[i, :, :, z_index],
        title = "T(x, y, z=$(round(z_val, digits=3)), t=$(round(discrete_t[i], digits=2)))",
        xlabel = "x", ylabel = "y", clims=(0, 64), c=:thermal,
        aspect_ratio=1
    )
end

gif(anim, "MethodOfLines_jl/Problem/heat/heat_eq_z$(round(z_val, digits=2))_heatmap.gif", fps=5)

anim = @animate for i in eachindex(discrete_t)
    surface(
        discrete_x, discrete_y, transpose(solu[i, :, :, z_index]),
        xlabel = "x", ylabel = "y", zlabel = "T(x, y)",
        title = "Surface at z=$(round(z_val, digits=2)), t=$(round(discrete_t[i], digits=2))",
        c = :viridis, clims = (0, 64),
        camera = (30 + i*2, 45),  # smooth rotating view
        legend = true
    )
end

gif(anim, "MethodOfLines_jl/Problem/heat/heat_eqn_z$(round(z_val, digits=2))_surface_anim.gif", fps=5)

# Safely get index closest to z = 3.0
z_index = argmin(abs.(discrete_z .- 3.0))
z_val = discrete_z[z_index]  # Actual value at that index

# Create animation
anim = @animate for i in eachindex(discrete_t)
    heatmap(
        discrete_x, discrete_y, solu[i, :, :, z_index],
        title = "T(x, y, z=$(round(z_val, digits=3)), t=$(round(discrete_t[i], digits=2)))",
        xlabel = "x", ylabel = "y", clims=(0, 64), c=:thermal,
        aspect_ratio=1
    )
end

gif(anim, "MethodOfLines_jl/Problem/heat/heat_eq_z$(round(z_val, digits=2))_heatmap.gif", fps=5)

anim = @animate for i in eachindex(discrete_t)
    surface(
        discrete_x, discrete_y, transpose(solu[i, :, :, z_index]),
        xlabel = "x", ylabel = "y", zlabel = "T(x, y)",
        title = "Surface at z=$(round(z_val, digits=2)), t=$(round(discrete_t[i], digits=2))",
        c = :viridis, clims = (0, 64),
        camera = (30 + i*2, 45),  # smooth rotating view
        legend = true
    )
end

gif(anim, "MethodOfLines_jl/Problem/heat/heat_eqn_z$(round(z_val, digits=2))_surface_anim.gif", fps=5)

# Safely get index closest to z = 2.5
z_index = argmin(abs.(discrete_z .- 2.5))
z_val = discrete_z[z_index]  # Actual value at that index

# Create animation
anim = @animate for i in eachindex(discrete_t)
    heatmap(
        discrete_x, discrete_y, solu[i, :, :, z_index],
        title = "T(x, y, z=$(round(z_val, digits=3)), t=$(round(discrete_t[i], digits=2)))",
        xlabel = "x", ylabel = "y", clims=(0, 64), c=:thermal,
        aspect_ratio=1
    )
end

gif(anim, "MethodOfLines_jl/Problem/heat/heat_eq_z$(round(z_val, digits=2))_heatmap.gif", fps=5)

anim = @animate for i in eachindex(discrete_t)
    surface(
        discrete_x, discrete_y, transpose(solu[i, :, :, z_index]),
        xlabel = "x", ylabel = "y", zlabel = "T(x, y)",
        title = "Surface at z=$(round(z_val, digits=2)), t=$(round(discrete_t[i], digits=2))",
        c = :viridis, clims = (0, 64),
        camera = (30 + i*2, 45),  # smooth rotating view
        legend = true
    )
end

gif(anim, "MethodOfLines_jl/Problem/heat/heat_eqn_z$(round(z_val, digits=2))_surface_anim.gif", fps=5)




using WriteVTK # Ensure output directory exists

# This creates .vti files and links them via a .pvd time series file
paraview_collection("heat_solution") do pvd
    for i in eachindex(discrete_t)
        # Extract 3D slice at this time step
        T_i = solu[i, :, :, :]  # 3D data: (Nx, Ny, Nz)

        # Name the file for this timestep (e.g., timestep_001.vti)
        vtk_filename = "heat_solution_t$(i)"

        # Write the .vti file
        vtk_grid(vtk_filename, discrete_x , discrete_y, discrete_z) do vtk
            vtk["Temperature"] = T_i
            pvd[discrete_t[i]] = vtk  # Add time to .pvd collection
        end
    end
end

