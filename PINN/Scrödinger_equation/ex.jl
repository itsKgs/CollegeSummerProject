using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, DomainSets, Plots

@parameters x t
@variables u(..) v(..)
Dx = Differential(x)
Dt = Differential(t)

# Split into real and imaginary parts
eqs = [
    Dt(u(x,t)) ~ -0.5*Dx(Dx(v(x,t)))-v(x,t)*(u(x,t)^2 + v(x,t)^2),
    Dt(v(x,t)) ~  0.5*Dx(Dx(u(x,t))) + u(x,t)*(u(x,t)^2 + v(x,t)^2)
]

# Domain
domains = [x ∈ IntervalDomain(-5.0, 5.0),
           t ∈ IntervalDomain(0.0, π/2)]

# Initial and boundary conditions
ics = [
    u(x,0) ~ 2*sech(x),#in(π*x),
    v(x,0) ~ 0.0
]
bcs = [
    u(-5,t) ~ u(5,t), Dx(u(-5,t)) ~ Dx(u(5,t)),
    v(-5,t) ~ v(5,t), Dx(v(-5,t)) ~ Dx(v(5,t))
]

# Combine all
pdesys = PDESystem(eqs, bcs ∪ ics, domains, [x, t], [u(x, t), v(x, t)]; name = :schrodinger)
dim = 2  # (x, t)

chain = Chain(Dense(dim, 32, tanh),
              Dense(32, 32, tanh),
              Dense(32, 2))  # output: [u, v]

strategy = NeuralPDE.GridTraining([100, 100])
discretization = PhysicsInformedNN(chain, strategy)
prob = discretize(pdesys, discretization)
optim = Optimization.OptimizationFunction(prob.f, Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optim, prob.u0)
res = Optimization.solve(optprob, Optim.BFGS(); maxiters=1000)
phi = discretization.phi
xs = range(-5, 5, length=100)
ts = range(0, π/2, length=100)
us = [phi([x,t], res.u)[1] for x in xs, t in ts]
vs = [phi([x,t], res.u)[2] for x in xs, t in ts]

heatmap(xs, ts, abs.(us .+ im .* vs)', xlabel="x", ylabel="t", title="|ψ(x,t)|", colorbar_title="Magnitude")
savefig("schrodingertest_heatmap.png")
anim = @animate for i in 1:length(ts)
    #plot(xs, us[:, i], label="Re(ψ)", title="t = $(ts[i])", xlabel="x", ylabel="ψ(x,t)")
    #plot!(xs, vs[:, i], label="Im(ψ)")
    plot(xs, abs.(us[:, i] .+ im .* vs[:, i]), label="|ψ|", linestyle=:dash)
end
gif(anim, "schrodingertest.gif", fps=10)