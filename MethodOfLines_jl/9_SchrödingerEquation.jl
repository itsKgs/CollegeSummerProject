using MethodOfLines, DifferentialEquations, Plots, DomainSets, ModelingToolkit

@parameters t x
@variables  u(..), v(..)

Dt = Differential(t)
Dxx = Differential(x)^2

# Domain limits
x_min = t_min = 0.0
x_max = t_max = 1.0

V(x) = 0.0

# 2. PDE system
eqs = [
    Dt(u(t,x)) ~ -Dxx(v(t,x)) + V(x)*v(t,x),
    Dt(v(t,x)) ~  Dxx(u(t,x)) - V(x)*u(t,x)
]

# 3. Domains
domains = [
    x ∈ Interval(x_min, x_max),
    t ∈ Interval(t_min, t_max)
]

# 4. Initial conditions
ψ0 = x -> sin(2*π*x) 

# 5. Boundary and initial conditions 
# Initial conditions at t=0
bcs = [
    u(0,x) ~ ψ0(x),
    v(0,x) ~ 0.0,
    # Dirichlet walls at x=0,1 for full ψ => u=v=0 at boundaries
    u(t,0) ~ 0.0,
    u(t,1) ~ 0.0,
    v(t,0) ~ 0.0,
    v(t,1) ~ 0.0,
]

@named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])

# 5. Discretize using MethodOfLines (MOL)
N=32
@time discretization = MOLFiniteDifference([x => N], t, approx_order=2)

prob = discretize(pdesys, discretization;)

sol = solve(prob, TRBDF2(), saveat = 0.01)

discrete_x = sol[x]
discrete_t = sol[t]

discrete_u = sol[u(t, x)]
discrete_v = sol[v(t, x)]

anim = @animate for i in eachindex(discrete_t)
    u = discrete_u[i, :]
    v = discrete_v[i, :]
    plot(discrete_x, u, ylim = (-1.5, 1.5), title = "t = $(discrete_t[i])", xlabel = "x", ylabel = "u(t,x)", label = ["re(u)" "im(u)"], legend = :topleft)
    plot!(discrete_x, v, ylim = (-1.5, 1.5), title = "t = $(discrete_t[i])", xlabel = "x", ylabel = "v(t,x)", label = ["re(v)" "im(v)"], legend = :topleft)
end
gif(anim, "schroedinger.gif", fps = 10)


