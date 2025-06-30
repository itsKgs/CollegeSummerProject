using  Test, OptimizationOptimisers, NeuralOperators, NeuralPDE, Random, Statistics, Lux, Plots

# Define the parametric ODE equation
equation = (u, p, t) -> p[1] * cos(p[2] * t) + p[3]
tspan = (0.0, 1.0)
u0 = 1.0

prob = ODEProblem(equation, u0, tspan)

# Set the number of parameters for the ODE
num_params = 3

# Define the DeepONet architecture for the PINO
deeponet = NeuralOperators.DeepONet(
    Chain(
        Dense(num_params => 10, Lux.tanh_fast), 
        Dense(10 => 10, Lux.tanh_fast), 
        Dense(10 => 10)
    ),
    Chain(
        Dense(1 => 10, Lux.tanh_fast), 
        Dense(10 => 10, Lux.tanh_fast),
        Dense(10 => 10, Lux.tanh_fast)
    )
)

# Define the bounds for the parameters
bounds = [(1.0, π), (1.0, 2.0), (2.0, 3.0)]
number_of_parameter_samples = 50

# Define the training strategy
strategy = StochasticTraining(20)

# Define the optimizer
opt = OptimizationOptimisers.Adam(0.03)

# Define `PINNODE`
alg = PINOODE(deeponet, opt, bounds, num_params; strategy = strategy)

# Solve the ODE problem using the PINOODE algorithm
sol = solve(prob, alg, verbose = true, maxiters = 4000)


function get_trainset(bounds, tspan, number_of_parameters, dt)
    p_ = [range(start = b[1], length = number_of_parameters, stop = b[2]) for b in bounds]
    p = vcat([collect(reshape(p_i, 1, size(p_i, 1))) for p_i in p_]...)
    t_ = collect(tspan[1]:dt:tspan[2])
    t = collect(reshape(t_, 1, size(t_, 1), 1))
    (p, t)
end

# Compute the ground truth solution for each parameter
ground_solution = (u0, p, t) -> u0 + p[1] / p[2] * sin(p[2] * t) + p[3] * t

function ground_solution_f(p, t)
    reduce(hcat,
        [[ground_solution(u0, p[:, i], t[j]) for j in axes(t, 2)] for i in axes(p, 2)])
end

# generate the solution with new parameters for test the model
(p, t) = get_trainset(bounds, tspan, 50, 0.025)

# compute the ground truth solution
ground_solution_ = ground_solution_f(p, t)

# predict the solution with the PINO model
predict = sol.interp(p, t)

# calculate the errors between the ground truth solution and the predicted solution
errors = ground_solution_ - predict

# calculate the mean error and the standard deviation of the errors
mean_error = mean(errors)

# calculate the standard deviation of the errors
std_error = std(errors)

p, t = get_trainset(bounds, tspan, 100, 0.01)

ground_solution_ = ground_solution_f(p, t)
predict = sol.interp(p, t)

errors = ground_solution_ - predict
mean_error = mean(errors)
std_error = std(errors)

# Plot the predicted solution and the ground truth solution as a filled contour plot
# predict, represents the predicted solution for each parameter value and time
plot(predict, linetype = :contourf)
plot!(ground_solution_, linetype = :contourf)