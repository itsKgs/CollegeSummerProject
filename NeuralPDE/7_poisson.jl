using NeuralPDE, Lux, Optimization, OptimizationOptimJL, LineSearches, Plots
using  ModelingToolkit: Interval 

@parameters x, y
@variables u(..)

Dxx = Differential(x)^2
Dyy = Differential(y)^2

#2D PDE
eq = [
    Dxx(u(x, y)) + Dyy(u(x, y)) ~ - sin(π * x) * sin(π * y)
]

# Space and time domains
domains = [
    x ∈ Interval(0.0, 1.0),
    y ∈ Interval(0.0, 1.0)
]

#Boundary Comnditions

bcs = [
    u(0, y) ~ 0.0,
    u(1, y) ~ 0.0,
    u(x, 0) ~ 0.0,
    u(x, 1) ~ 0.0
]

# Neural network
dim = 2 # number of dimensions
chain = Chain(
    Dense(dim, 16, σ),
    Dense(16, 16, σ),
    Dense(16, 1)
)

discretization = PhysicsInformedNN(chain, QuadratureTraining(; batch=200, abstol=1e-6, reltol=1e-6))

@named pdsys = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

prob = discretize(pdsys, discretization)

#Callback function
global  iter = 0
callback = function (p, l)
    global iter =  iter + 1
    println("Iteration: $iter, loss is: $l")
    return false
end

# Optimizer
opt = LBFGS(linesearch = BackTracking())
res = solve(prob, opt, callback=callback, maxiters = 500)
phi = discretization.phi


dx = 0.05
xs, ys = [infimum(d.domain):(dx / 10):supremum(d.domain) for d in domains]
analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)

u_predict = reshape([first(phi([x, y], res.u)) for x in xs for y in ys],(length(xs), length(ys)))
u_real = reshape([analytic_sol_func(x, y) for x in xs for y in ys],(length(xs), length(ys)))
diff_u = abs.(u_predict .- u_real)

p1 = plot(xs, ys, u_real, linetype = :contourf, title = "analytic");
p2 = plot(xs, ys, u_predict, linetype = :contourf, title = "predict");
p3 = plot(xs, ys, diff_u, linetype = :contourf, title = "error");
plot(p1, p2, p3)