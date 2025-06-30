using ModelingToolkit, MethodOfLines, DomainSets, DifferentialEquations

@parameters x y t
@variables u(..) v(..)

Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# Laplacian operator (scalar Laplace)
∇²(u) = Dxx(u) + Dyy(u)

# Forcing function
brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.

# Domain limits
x_min = y_min = t_min = 0.0
x_max = y_max = 1.0
t_max = 11.5

# Diffusion coefficient
α = 10.0

# Initial conditions
u0(x, y, t) = 22 * (y * (1 - y))^(3/2)
v0(x, y, t) = 27 * (x * (1 - x))^(3/2)

# 2. PDE system
eqs = [
    Dt(u(x, y, t)) ~ 1 + v(x, y, t) * u(x, y, t)^2 - 4.4 * u(x, y, t) + α * ∇²(u(x, y, t)) + brusselator_f(x, y, t),
    Dt(v(x, y, t)) ~ 3.4 * u(x, y, t) - v(x, y, t) * u(x, y, t)^2 + α * ∇²(v(x, y, t))
]

# 3. Domains
domains = [
    x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max),
    t ∈ Interval(t_min, t_max)
]

# 4. Boundary and initial conditions (periodic in x and y)

bcs = [u(x,y,0) ~ u0(x,y,0),
       u(0,y,t) ~ u(1,y,t),
       u(x,0,t) ~ u(x,1,t),

       v(x,y,0) ~ v0(x,y,0),
       v(0,y,t) ~ v(1,y,t),
       v(x,0,t) ~ v(x,1,t)
] 

@named pdesys = PDESystem(eqs, bcs, domains, [x, y, t], [u(x, y, t), v(x, y, t)])

# 5. Discretize using MethodOfLines (MOL)
N = 32
#dx = 0.05
#dy = 0.05
discretization = MOLFiniteDifference([x => N, y => N], t, approx_order=2)


# 6. Convert PDE to ODE system
@time prob = discretize(pdesys,discretization)

# 7. Solve using a stiff ODE solver
sol = solve(prob, TRBDF2(), saveat=0.1)

discrete_x = sol[x]
discrete_y = sol[y]
discrete_t = sol[t]

solu = sol[u(x, y, t)]
solv = sol[v(x, y, t)]

using Plots

#for u
anim = @animate for k in eachindex(discrete_t)
    heatmap(solu[2:end, 2:end, k], title="$(discrete_t[k])") # 2:end since end = 1, periodic condition
end
gif(anim, "Brusselator2Dsol_u.gif", fps = 2)

#for v
anim = @animate for k in eachindex(discrete_t)
    heatmap(solv[2:end, 2:end, k], title="$(discrete_t[k])")
end
gif(anim, "Brusselator2Dsol_v.gif", fps = 8)


