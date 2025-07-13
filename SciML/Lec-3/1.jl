using Flux
NN = Chain(Dense(10 => 32,tanh),
           Dense(32 => 32,tanh),
           Dense(32 => 5))
loss(nothing) = sum(abs2,sum(abs2,NN(rand(10)).-1) for i in 1:100)
loss(())

NN[1].weight # The W matrix of the first layer

p = Flux.params(NN)

opt = Flux.Optimise.Adam(0.001)

#Flux.train!(loss, p, Iterators.repeated((), 10000), ADAM(0.1))
Flux.train!(loss, p, Iterators.repeated((), 10000), opt)

loss()
