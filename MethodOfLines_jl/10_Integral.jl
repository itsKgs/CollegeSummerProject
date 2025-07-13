using MethodOfLines, ModelingToolkit, DifferentialEquations, DomainSets, Plots

@parameters t, x
@variables u(..) cumuSum(..)

Dt = Differential(t)
Dx = Differential(x)

xmin = tmin = 0.0
xmax = 2.0pi
tmax = 2.0

# Integral limits are defined with DomainSets.ClosedInterval

Ix = Integral(x in DomainSets.ClosedInterval(xmin, x)) # basically cumulative sum from 0 to x

eq = [
    cumuSum(t, x) ~ Ix(u(t, x)), # Note wrapping the argument to the derivative with an auxiliary variable
    Dt(u(t, x)) + 2 * u(t, x) + 5 * Dx(cumuSum(t, x)) ~ 1
]

bcs = [
    u(0.0, x) ~ cos(x), 
    Dx(u(t, xmin)) ~ 0.0, 
    Dx(u(t, xmax)) ~ 0
]

domains = [
    t ∈ Interval(0.0, 2.0), 
    x ∈ Interval(xmin, xmax)
]

@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x), cumuSum(t, x)])

discretization = MOLFiniteDifference([x => 30], t, approx_order=2)

prob = MethodOfLines.discretize(pde_system, discretization)
sol = solve(prob, QNDF(), saveat = 0.1);

solu = sol[u(t, x)]

plot(sol[x], transpose(solu))