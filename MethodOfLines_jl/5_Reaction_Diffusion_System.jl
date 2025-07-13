using ModelingToolkit, MethodOfLines, DomainSets, DifferentialEquations
using Plots

@parameters t x
@parameters Dn, Dp
@variables u(..) v(..)

Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# Domain limits
x_min = t_min = 0.0
x_max = t_max = 1.0

# Initial conditions
u0(t, x) = sin((π*x)/2)
v0(t, x) = sin((π*x)/2)

# 2. PDE system
eqs = [
    Dt(u(t, x)) ~ Dn * Dxx(u(t, x)) + u(t, x)*v(t, x),
    Dt(v(t, x)) ~ Dp * Dxx(v(t, x)) - u(t, x)*v(t, x)
]

# 3. Domains
domains = [
    x ∈ Interval(x_min, x_max),
    t ∈ Interval(t_min, t_max)
]

# 4. Boundary and initial conditions (periodic in x and y)

bcs = [
    u(0, x) ~ sin((π*x)/2),
    v(0, x) ~ sin((π*x)/2),

    u(t, 0) ~ 0.0,
    v(t, 0) ~ 0.0,

    Dx(u(t, 1)) ~ 0.0,
    Dx(v(t, 1)) ~ 0.0

] 

@named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)], [Dn, Dp], defaults = Dict(Dn => 0.5, Dp => 2.0))

# 5. Discretize using MethodOfLines (MOL)
#N = 32
dx = 0.1
#dy = 0.05
discretization = MOLFiniteDifference([x => dx], t, approx_order=2)

# 6. Convert PDE to ODE system
@time prob = discretize(pdesys, discretization)

# 7. Solve using a stiff ODE solver
sol = solve(prob, Tsit5())

discrete_x = sol[x]
discrete_t = sol[t]

solu = sol[u(t, x)]
solv = sol[v(t, x)]


anim = @animate for i in eachindex(discrete_t)
    p1 = plot(discrete_x, solu[i, :], label="u, t=$(discrete_t[i])"; legend=false, xlabel="x",ylabel="u",ylim=[0,1])
    p2 = plot(discrete_x, solv[i, :], label="v, t=$(discrete_t[i])"; legend=false, xlabel="x", ylabel="v",ylim=[0, 1])
    plot(p1, p2)
end
gif(anim, "5_Reaction_Diffusion_System.gif",fps=10) 