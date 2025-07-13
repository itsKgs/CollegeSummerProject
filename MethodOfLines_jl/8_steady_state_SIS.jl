using DifferentialEquations, ModelingToolkit, MethodOfLines, DomainSets, Plots

@parameters t x
@parameters dS dI brn ϵ
@variables S(..) I(..)

Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# Define functions
function γ(x)
    y = x + 1.0
    return y
end

function ratio(x, brn, ϵ)
    y = brn + ϵ * sin(2 * pi * x)
    return y
end

# Domain limits
x_min = t_min = 0.0
x_max = 1.0
t_max = 10.0

# 2. PDE system
eqs = [
    Dt(S(t, x)) ~ dS * Dxx(S(t, x) - (ratio(x, brn, ϵ)*S(t, x)*I(t, x))/(S(t, x) + I(t, x)) + γ(x)*I(t, x)),
    Dt(I(t, x)) ~ dI * Dxx(I(t, x)) + (ratio(x, brn, ϵ)*S(t, x)*I(t, x))/(S(t, x) + I(t, x)) - γ(x)*I(t, x)
]

# 3. Domains
domains = [
    x ∈ Interval(x_min, x_max),
    t ∈ Interval(t_min, t_max)
]

# 4. Boundary and initial conditions 

bcs = [
    S(0, x) ~ 0.9 + 0.1 * sin(2 * π * x)
    I(0, x) ~ 0.1 + 0.1 * cos(2 * π * x)

    Dx(S(t, 0)) ~ 0.0
    Dx(S(t, 1)) ~ 0.0
    Dx(I(t, 0)) ~ 0.0
    Dx(I(t, 1)) ~ 0.0
]

@named pdesys = PDESystem(eqs, bcs, domains, [t, x], [S(t, x), I(t, x)], [dS, dI, brn, ϵ], defaults=Dict(dS => 0.5, dI => 0.1, brn => 3, ϵ => 0.1))

# 5. Discretize using MethodOfLines (MOL)
dx = 0.01
discretization = MOLFiniteDifference([x => dx], t, approx_order=2)

prob = discretize(pdesys, discretization)

# Solving SIS reaction diffusion model
sol = solve(prob, Tsit5(), saveat=0.2);

# Retriving the results
discrete_x = sol[x]
discrete_t = sol[t]
S_solution = sol[S(t, x)]
I_solution = sol[I(t, x)]

p = surface(discrete_x, discrete_t, S_solution, xlabel="x", ylabel="t", zlabel="S", title="S(t,x)")
display(p)

# Change the elliptic problem to steady state problem of reaction diffusion equation.
steadystateprob = SteadyStateProblem(prob)
steadystate = solve(steadystateprob, DynamicSS(Tsit5()))

function episize!(dS, dI)
    newprob = remake(prob, p=[dS => dS, dI => dI, brn => 3, ϵ => 0.1])
    steadystateprob = SteadyStateProblem(newprob)
    state = solve(steadystateprob, DynamicSS(Tsit5()))
    y = sum(state[100:end]) / 99
    return y
end
episize!(exp(1.0),exp(0.5))
