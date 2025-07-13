using  DifferentialEquations
using ModelingToolkit
using Plots

function lotka_volterra!(du, u, p, t)
    du[1] = p[1]*u[1] - p[2]*u[1]*u[2]
    du[2] = -p[3]*u[2] + p[4]*u[1]*u[2]
end

#lv! = @ode_def lotka_volterra begin
#    dx = a*x - b*x*y
#    dy = -c*y + d*x*y
#end a b c d

u0 = [1.0, 1.0]
p = (1.5, 1.0, 3.0, 1,0)
tspan = (0.0, 10.0)
prob = ODEProblem(lotka_volterra!, u0, tspan, p)
sol = solve(prob)
plot(sol)

lotka_volterra!.Jex