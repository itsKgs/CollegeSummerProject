# System of Equation 
using  DifferentialEquations
using Plots

function lorenz!(du, u, p, t)
    σ,ρ,β = p

    du[1] = σ*(u[2] - u[1])
    du[2] = u[1]*( ρ - u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
end

u0 = [1.0, 0.0, 0.0]

p = (10, 28, 8/3)

tspan = (0.0, 100)
prob = ODEProblem(lorenz!, u0, tspan, p)


sol = solve(prob)

plot(sol)

sol.t
sol.t[10] # 10 element in array of time
sol[10]
sol[2, 10] # it is the value of second variable at time 10.

A = convert(Array, sol)

x_vals = A[1, :]  # row 1: x(t)
y_vals = A[2, :]  # row 2: y(t)
z_vals = A[3, :]  # row 3: z(t)

plot(sol.t, A[1, :], label="x(t)")
plot!(sol.t, A[2, :], label="y(t)")
plot!(sol.t, A[3, :], label="z(t)")


plot(sol.t, A[2, :], label="y(t)")
plot(sol.t, A[3, :], label="z(t)")

plot(sol, vars=(1, 2, 3), xlabel="x(t)", ylabel="y(t)", zlabel="z(t)", title="Lorenz Attractor", legend=true)
plot(sol, vars=(1, 2, 3), xlabel="x(t)", ylabel="y(t)", zlabel="z(t)", title="Lorenz Attractor", legend=true, denseplot=false)
plot(sol, vars=(1, 2, 3), xlabel="x(t)", ylabel="y(t)", zlabel="z(t)", title="Lorenz Attractor", legend=true, denseplot=false, plotdensity=1000)

plot(sol, vars=(0,2)) # 0-> time and 2->y(t)
plot(sol, vars=(1,2))