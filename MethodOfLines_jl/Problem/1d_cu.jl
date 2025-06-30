using ModelingToolkit, MethodOfLines, DomainSets, DifferentialEquations
using Plots
using CUDA

# Enable full GPU vectorization (no slow scalar operations)
CUDA.allowscalar(false)

# Manufactured exact solution for comparison
u_exact = (x, t) -> 6 * sin.(π * x) * exp(-2 * π^2 * t)

# Parameters and variables
@parameters x t
@variables u(..)

# Differential operators
Dt = Differential(t)
Dxx = Differential(x)^2

# Domain limits
x_min = t_min = 0.0
x_max = 1.0
t_max = 1.0

k = 2.0

# PDE equation
eqs = [
    Dt(u(t, x)) ~ k * Dxx(u(t, x))
]

# Domain
domains = [
    x ∈ Interval(x_min, x_max),
    t ∈ Interval(t_min, t_max)
]

# Boundary and initial conditions
bcs = [
    u(0, x) ~ 6 * sin(π * x),
    u(t, 0) ~ 0.0,
    u(t, 1) ~ 0.0
]

@named pdesys = PDESystem(eqs, bcs, domains, [x, t], [u(t, x)])

# Discretization in space (Method of Lines)
N = 16
discretization = MOLFiniteDifference([x => N], t, approx_order=2)

# Discretize the PDE into an ODEProblem
@time prob = discretize(pdesys, discretization)

# Check for CUDA GPU and solve accordingly
if CUDA.functional()
    println("✅ CUDA GPU is available. Solving on GPU...")
    prob_gpu = remake(prob; u0 = CuArray(prob.u0))
    sol = solve(prob_gpu, Tsit5(), saveat=0.02)
else
    error("CUDA GPU not available. Aborting — this code is intended to run only on GPU.")
end

#sol = solve(prob, Tsit5(), saveat=0.02)

# Extract solution
discrete_x = sol[x]
discrete_t = sol[t]
solu = sol[u(t, x)]

# Plot and animate
plt = plot()
for i in eachindex(discrete_t)
    plot!(discrete_x, solu[i, :], label="Numerical, t=$(discrete_t[i])", legend=true)
    scatter!(discrete_x, u_exact(discrete_x, discrete_t[i]), label="Exact, t=$(discrete_t[i])")
end
display(plt)

# Animation: 1D heat equation evolution
anim = @animate for i in eachindex(discrete_t)
    plot(discrete_x, solu[i, :], 
         linewidth=2, 
         label="Numerical, t=$(round(discrete_t[i], digits=2))",
         xlabel="x", 
         ylabel="u(x,t)",
         title="1D Heat Equation: Numerical vs Exact Solution",
         ylims=(minimum(solu)-0.5, maximum(solu)+0.5),
         legend=:topright)
    
    scatter!(discrete_x, u_exact(discrete_x, discrete_t[i]), 
             label="Exact, t=$(round(discrete_t[i], digits=2))",
             color=:red,
             markersize=3)
end
gif(anim, "1D_heat_equation_gpu.gif", fps=10)

