using MethodOfLines, OrdinaryDiffEq, Plots, DomainSets, ModelingToolkit
using Pkg
Pkg.status()

@parameters t, x
@variables ψ(..) ψ0(..) 

Dt = Differential(t)
Dxx = Differential(x)^2

xmin = 0
xmax = 1

V(x) = 0.0

eq = [im*Dt(ψ(t,x)) ~ Dxx(ψ(t,x)) + V(x)*ψ(t,x)] # You must enclose complex equations in a vector, even if there is only one equation

ψ0(x) = sin(2pi*x)
@initial_conditions ic = [ψ(0,x) ~ ψ0(x)]

bcs = [ψ(0,x) ~ ψ0(x),
    ψ(t,xmin) ~ 0,
    ψ(t,xmax) ~ 0]

domains = [t ∈ Interval(0, 1), x ∈ Interval(xmin, xmax)]

@named sys = PDESystem(eq, bcs, domains, [t, x], [ψ(t,x)])

disc = MOLFiniteDifference([x => 100], t)

prob = discretize(sys, disc)

sol = solve(prob, TRBDF2(), saveat = 0.01)

discx = sol[x]
disct = sol[t]

discψ = sol[ψ(t, x)]
anim = @animate for i in 1:length(disct)
    u = discψ[i, :]
    plot(discx, [real.(u), imag.(u)], ylim = (-1.5, 1.5), title = "t = $(disct[i])", xlabel = "x", ylabel = "ψ(t,x)", label = ["re(ψ)" "im(ψ)"], legend = :topleft)
end
gif(anim, "schroedinger.gif", fps = 10)