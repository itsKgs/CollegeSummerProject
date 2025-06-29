using Flux
using Statistics

NNODE = Chain(x -> [x], # Take in a scalar and transform it into an array
           Dense(1 => 32,tanh),
           Dense(32 => 1),
           first) # Take first value, i.e. return a scalar
NNODE(1.0)

g(t) = t*NNODE(t) + 1f0

ϵ = sqrt(eps(Float32))
loss() = mean(abs2(((g(t+ϵ)-g(t))/ϵ) - cos(2π*t)) for t in 0:1f-2:1f0)

println(loss())

opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 500 == 0
    display(loss())
  end
end
display(loss())
#Flux.train!(loss, NNODE, data, opt; cb=cb)


# 4. Set up optimizer with proper learning rate
opt = Optimisers.Adam(0.001f0)  # More robust than Descent
state = Optimisers.setup(opt, NNODE)

# 5. Training loop with gradient checking
for epoch in 1:5000
    grads = gradient(NNODE) do m
        loss()
    end
    
    state, NNODE = Optimisers.update(state, NNODE, grads[1])
    
    if epoch % 100 == 0
        current_loss = loss()
        println("Epoch $epoch, Loss: $current_loss")
        # Early stopping if loss stops changing
        if epoch > 1000 && abs(current_loss - loss()) < 1e-6
            println("Converged at epoch $epoch")
            break
        end
    end
end

# Final evaluation
println("Final loss: ", loss())