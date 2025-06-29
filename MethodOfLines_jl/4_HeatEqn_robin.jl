using ModelingToolkit, MethodOfLines, DomainSets, DifferentialEquations
using Plots

# Method of Manufactured Solutions: exact solution
u_exact = (x,t) -> exp.(-t) * sin.(x)


@parameters x t
@variables u(..)

Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# Domain limits
x_min = -1.0
t_min = 0.0
x_max = t_max = 1.0

# Initial conditions
u0(t, x) = sin(x)

# 2. PDE system
eqs = [
    Dt(u(t, x)) ~ Dxx(u(t, x))
]

# 3. Domains
domains = [
    x ∈ Interval(x_min, x_max),
    t ∈ Interval(t_min, t_max)
]

# 4. Boundary and initial conditions (periodic in x and y)

bcs = [
    u(0, x) ~ sin(x),
    u(t, -1.0) + 3 * Dx(u(t, -1.0)) ~ exp(-t) * (sin(-1.0) + 3 * cos(-1.0)),
    u(t, 1.0) + Dx(u(t, 1.0)) ~ exp(-t) * (sin(1.0) + cos(1.0))
] 

@named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x)])

# 5. Discretize using MethodOfLines (MOL)
#N = 32
dx = 0.05
#dy = 0.05
discretization = MOLFiniteDifference([x => dx], t, approx_order=2)

# 6. Convert PDE to ODE system
@time prob = discretize(pdesys, discretization)

# 7. Solve using a stiff ODE solver
sol = solve(prob, Tsit5(), saveat=0.2)

discrete_x = sol[x]
discrete_t = sol[t]

solu = sol[u(t, x)]

plt = plot()

for i in eachindex(discrete_t)
    plot!(discrete_x, solu[i, :], label="Numerical, t=$(discrete_t[i])")
    scatter!(discrete_x, u_exact(discrete_x, discrete_t[i]), label="Exact, t=$(discrete_t[i])")
end
plt