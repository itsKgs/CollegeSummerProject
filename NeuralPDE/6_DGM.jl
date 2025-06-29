using NeuralPDE
using ModelingToolkit, Optimization, OptimizationOptimisers
using Distributions
using ModelingToolkit: Interval, infimum, supremum
using MethodOfLines, OrdinaryDiffEq
using Plots

@parameters x t
@variables u(..)


Dt = Differential(t)
Dx = Differential(x)
Dxx = Dx^2
α = 0.05

# Burger's equation
eq = Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) - α * Dxx(u(t, x)) ~ 0

# boundary conditions
bcs = [
    u(0.0, x) ~ -sin(π * x),
    u(t, -1.0) ~ 0.0,
    u(t, 1.0) ~ 0.0
]

domains = [t ∈ Interval(0.0, 1.0), 
           x ∈ Interval(-1.0, 1.0)
]

# MethodOfLines, for FD solution
dx = 0.01
order = 2
discretization = MOLFiniteDifference([x => dx], t, saveat = 0.01)

@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

prob = discretize(pde_system, discretization)

sol = solve(prob, Tsit5())
ts = sol[t]
xs = sol[x]

u_MOL = sol[u(t, x)]

# NeuralPDE, using Deep Galerkin Method
strategy = QuasiRandomTraining(256, minibatch = 32)

discretization = DeepGalerkin(2, 1, 50, 5, tanh, tanh, identity, strategy)
@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
prob = discretize(pde_system, discretization)

callback = function (p, l)
    #(p.iter % 20 == 0) && println("$(p.iter) => $l")
    println("Iteration $(p.iter): Loss = $l")
    return false
end

opt = OptimizationOptimisers.Adam(0.0001)
res = solve(prob, opt; maxiters = 500, callback = callback)
prob = remake(prob, u0 = res.u)
res = solve(prob, opt; maxiters = 500, callback = callback)
phi = discretization.phi

u_predict = [first(phi([t, x], res.minimizer)) for t in ts, x in xs]

diff_u = abs.(u_predict .- u_MOL)
tgrid = 0.0:0.01:1.0
xgrid = -1.0:0.01:1.0

p1 = plot(tgrid, xgrid, u_MOL', linetype = :contourf, title = "FD");
p2 = plot(tgrid, xgrid, u_predict', linetype = :contourf, title = "predict");
p3 = plot(tgrid, xgrid, diff_u', linetype = :contourf, title = "error");
plot(p1, p2, p3)