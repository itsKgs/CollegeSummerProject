using ModelingToolkit, MethodOfLines, DomainSets, DifferentialEquations
using Plots

@parameters x t
@variables T(..)

Dt = Differential(t)
Dxx = Differential(x)^2

# Domain limits
x_min = t_min = 0.0
x_max = 10.0
t_max = 600.0

k = 1.0

# Initial conditions
T0(t, x) = 27

# 2. PDE system
eqs = [
    Dt(T(t, x)) ~ k*Dxx(T(t, x))
]

# 3. Domains
domains = [
    x ∈ Interval(x_min, x_max),
    t ∈ Interval(t_min, t_max)
]

# 4. Boundary and initial conditions (periodic in x and y)

bcs = [
    T(0, x) ~ 27,
    T(t, 0) ~ 200,
    T(t, 10.0) ~ 500
] 

@named pdesys = PDESystem(eqs, bcs, domains, [x, t], [T(t, x)])

# 5. Discretize using MethodOfLines (MOL)
#N = 64
#dx = 0.1
#dy = 0.05
discretization = MOLFiniteDifference([x => 0.1], t, approx_order=2)

# 6. Convert PDE to ODE system
@time prob = discretize(pdesys, discretization)

# 7. Solve using a stiff ODE solver
sol = solve(prob, Tsit5(), saveat=0.01)

discrete_x = sol[x]
discrete_t = sol[t]

solu = sol[T(t, x)]

plt = plot()

for i in eachindex(discrete_t)
    plot!(discrete_x, solu[i, :], label="Numerical, t=$(discrete_t[i])", legend=true)
    #scatter!(discrete_x, u_exact(discrete_x, discrete_t[i]), label="Exact, t=$(discrete_t[i])")
end
plt

# Create animation
anim = @animate for i in eachindex(discrete_t)
    plot(discrete_x, solu[i, :], 
        linewidth=2, 
        label="Numerical, t=$(round(discrete_t[i], digits=2))",
        xlabel="x", 
        ylabel="u(x,t)",
        title="1D Heat Equation: Numerical vs Exact Solution",
        ylims=(minimum(solu)-0.5, maximum(solu)+0.5),
        legend=:topright)

end

# Save as GIF
gif(anim, "FDM_1D_heat_equation_animation.gif", fps=1)


anim = @animate for i in eachindex(discrete_t)
    p1 = plot(discrete_x, solu[i, :], label="u, t=$(discrete_t[i])"; legend=true, xlabel="x",ylabel="u(x,t)",ylims=(minimum(solu)-0.5, maximum(solu)+0.5))
    scatter!(discrete_x, u_exact(discrete_x, discrete_t[i]), 
             label="Exact, t=$(round(discrete_t[i], digits=2))",
             color=:red,
             markersize=2)
    plot(p1)
end
gif(anim, "FDM_1D_Heat_equation_animation2.gif",fps=10) 


