using DifferentialEquations
using Plots

# u' = 0.98u ; u(0) = 1.0 ; t=0 to t=1.0

f(u, p, t) = 0.98u
u0 = 1.0
tspan = (0.0, 1.0)

prob = ODEProblem(f, u0, tspan)

sol = solve(prob)

plot(sol)

plot(sol, linewidth=2, linecolor=:blue, linestyle=:solid, title="Solution of linear ODEProblem", ylabel="u(t)", xlabel="Time (t)", label="u(t)")
plot!(sol.t, t->1.0*exp(0.98t), lw=:2, ls=:dash, label="True Solution")

sol.t
sol.u

[t+u for (t, u) in tuples(sol)]
sol

sol(0.45)

# Add a point at t = 0.45
plot!([0.45], [sol(0.45)], seriestype=:scatter, label="Point at t=0.45", markersize=6, color=:red)

sol = solve(prob, abstol=1e-8, reltol=1e-8)
plot!(sol, linecolor=:green, label="After applyn reltol")

sol = solve(prob, saveat=0.1)


sol = solve(prob, saveat=[0.2, 0.7, 0.9], save_start=true, save_end=false)
sol = solve(prob, dense=false)

sol = solve(prob, save_everystep=false)

sol = solve(prob, alg_hints=[:stiff])

sol = solve(prob, Tsit5(), reltol=1e-6)