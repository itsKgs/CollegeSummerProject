using Flux
using Statistics
using Plots
using Flux: gradient
using Zygote: gradient

NNODE = Chain(
    x -> [x],
    Dense(1 => 32, tanh),
    Dense(32 => 1),
    first
)

NNODE(1)

g(t) = t * NNODE(t) + 1f0

function dgdt(t)
    gradient(g, t)[1]
end

loss(_) = mean(abs2(dgdt(t) - cos(2π * t)) for t in 0:0.01:1f0)

opt = ADAM(0.001)
opt_state = Flux.setup(opt, NNODE)

println("Initial loss = ", loss(()))
for iter in 1:5000
    grads = gradient(loss, NNODE)
    Flux.update!(opt_state, NNODE, grads[1])  # <-- HERE extract gradient for model only

    if iter % 500 == 0
        println("Iteration $iter: loss = ", loss(()))
    end
end