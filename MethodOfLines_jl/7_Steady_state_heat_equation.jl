using ModelingToolkit, MethodOfLines, DomainSets, NonlinearSolve
using Plots

@parameters x y
@variables u(..)


Dy = Differential(y)
Dx = Differential(x)
Dyy = Differential(y)^2
Dxx = Differential(x)^2

# Domain limits
x_min = y_min = 0.0
x_max = y_max = 1.0

# Initial conditions
u0(x, y) = 0
u1(x, y) = y
v0(x, y) = 0
v1(x, y) = x

# 2. PDE system
eqs = [
    Dxx(u(x, y)) + Dyy(u(x, y)) ~ 0
]

# 3. Domains
domains = [
    x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max)
]


# 4. Boundary and initial conditions 

bcs = [
    u(0, y) ~ 0.0,
    u(x, 0) ~ 0.0,

    u(1, y) ~ y,
    u(x, 1) ~ x,

] 


@named pdesys = PDESystem(eqs, bcs, domains, [x, y], [u(x, y)])

# 5. Discretize using MethodOfLines (MOL)
dx = 0.1
dy = 0.1
discretization = MOLFiniteDifference([x => dx, y => dy], nothing, approx_order=2)

prob = discretize(pdesys, discretization)

sol = NonlinearSolve.solve(prob, NewtonRaphson())

solu = sol[u(x, y)]


heatmap(sol[x], sol[y], solu, xlabel="x values", ylabel="y values", title="Steady State Heat Equation")