using NeuralPDE, ModelingToolkit, Optimization, OptimizationOptimisers, Lux
using CSV, DataFrames

@parameters t
@variables u(..)
Dt = Differential(t)

eq = Dt(u(t)) ~ 0

domains = [t ∈ IntervalDomain(0.0, 1.0)]

input = [t]
target = [u(t)]
chain = Lux.Chain(Dense(1, 8, tanh), Dense(8, 1))
strategy = GridTraining([50])
discretization = PhysicsInformedNN(chain, strategy)

@named pde_system = PDESystem(
    [eq],       # equations
    [],         # boundary conditions
    domains,
    input,
    target
)
prob = discretize(pde_system, discretization)  # ✅ This line should succeed
