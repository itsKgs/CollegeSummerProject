using MethodOfLines, DifferentialEquations, Plots, DomainSets, ModelingToolkit

@parameters t x
@variables ψ(..)

Dt = Differential(t)
Dxx = Differential(x)^2

# Domain limits
x_min = t_min = 0.0
x_max = t_max = 1.0

V(x) = 0.0

# 2. PDE system
eqs = [
   im * Dt(ψ(t, x)) ~ -Dxx(ψ(t, x)) + V(x)*ψ(t, x)
]

# 3. Domains
domains = [
    x ∈ Interval(x_min, x_max),
    t ∈ Interval(t_min, t_max)
]

# 4. Initial conditions
ψ0 = x -> sin(2*π*x) + 0im

# 5. Boundary and initial conditions 
bcs = [
    ψ(0, x) ~ sin(2*π*x) + 0im,

    ψ(t, 0) ~ 0.0 + 0im,
    ψ(t, 1) ~ 0.0 + 0im
]

@named pdesys = PDESystem(eqs, bcs, domains, [t, x], [ψ(t, x)])

# 5. Discretize using MethodOfLines (MOL)
N=100
@time discretization = MOLFiniteDifference([x => N], t, approx_order=2)

prob = discretize(pdesys, discretization;)

sol = solve(prob, TRBDF2(), saveat = 0.01)

discrete_x = sol[x]
discrete_t = sol[t]

discrete_ψ = sol[ψ(t, x)]

anim = @animate for i in eachindex(disct)
    u = discrete_ψ[i, :]
    plot(discrete_x, [real.(u), imag.(u)], ylim = (-1.5, 1.5), title = "t = $(discrete_t[i])", xlabel = "x", ylabel = "ψ(t,x)", label = ["re(ψ)" "im(ψ)"], legend = :topleft)
end
gif(anim, "schroedinger.gif", fps = 10)


