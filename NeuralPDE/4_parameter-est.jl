using NeuralPDE, OrdinaryDiffEq, Lux, Random, OptimizationOptimJL, LineSearches, Plots

function lv(u, p, t)
    u₁, u₂ = u
    α, β, γ, δ = p
    du₁ = α * u₁ - β * u₁ * u₂
    du₂ = δ * u₁ * u₂ - γ * u₂
    return [du₁, du₂]
end

tspan = (0.0, 5.0)
u0 = [5.0, 5.0]
p = [1.0, 1.0, 1.0, 1.0]
prob = ODEProblem(lv, u0, tspan, p)

#solution = solve(prob)
#
#plot(solution)

true_p = [1.5, 1.0, 3.0, 1.0]
prob_data = remake(prob, p = true_p)
sol_data = solve(prob_data, Tsit5(), saveat = 0.01)

t_ = sol_data.t

u_ = reduce(hcat, sol_data.u)

rng = Random.default_rng()
#rand(rng)          # generates a random Float64 between 0 and 1
#rand(rng, 1:10)    # generates a random integer from 1 to 10
#rand(rng, 3)       # generates a vector of 3 random numbers

Random.seed!(rng, 0)
#println(rand(rng))  # always gives the same value
#println(rand(rng))  # same again on every run

n = 15
chain = Chain(
    Dense(1, n, σ), 
    Dense(n, n, σ), 
    Dense(n, n, σ), 
    Dense(n, 2)
)

ps, st = Lux.setup(rng, chain) |> f64

additional_loss(phi, θ) = sum(abs2, phi(t_, θ) .- u_) / size(u_, 2) #Mean Squared Error

opt = LBFGS(linesearch = BackTracking())
alg = NNODE(chain, opt, ps; strategy = WeightedIntervalTraining([0.7, 0.2, 0.1], 500), param_estim = true, additional_loss)

sol = solve(prob, alg, verbose = true, abstol = 1e-8, maxiters = 5000, saveat = t_)

plot(sol, labels = ["u1_pinn" "u2_pinn"])
plot!(sol_data, labels = ["u1_data" "u2_data"])

sol.k.u.p