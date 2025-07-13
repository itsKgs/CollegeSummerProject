using ModelingToolkit, MethodOfLines, DomainSets, DifferentialEquations
using Plots, CUDA


@parameters x y z t
@variables T(..)

Dt = Differential(t)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dzz = Differential(z)^2

# Domain limits
x_min = t_min = y_min = z_min = 0.0
x_max = 10.0
y_max = 10.0
z_max = 3.0
t_max = 2.0

k = 1.0
ρ = 1.0
C_p = 700.0
D = 1/(ρ * C_p)
a = 2.0
b = 2.0
c = 0.5
q = 1.0
v = 1.0


# Initial conditions
T0(t, x, y, z) = 27.0 

Q(t, x, y, z) = ((6 * sqrt(3) * q)/(a * b * c * π * sqrt(π))) * exp(-3 * (((x^2)/(a^2)) + ((y^2)/(b^2)) + (((z - v * t)^2)/(c^2))))

# 2. PDE system

eqs = [
    Dt(T(t, x, y, z)) ~ D * (k * (Dxx(T(t, x, y, z)) + Dyy(T(t, x, y, z)) + Dzz(T(t, x, y, z))) + Q(t, x, y, z))
]

# 3. Domains
domains = [
    x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max),
    z ∈ Interval(z_min, z_max),
    t ∈ Interval(t_min, t_max)
]

# 4. Boundary and initial conditions (periodic in x and y)

bcs = [
    T(0, x, y, z) ~ 27.0,
    T(t, 0, y, z) ~ 27.0,
    T(t, 10, y, z) ~ 27.0,
    T(t, x, 0, z) ~ 27.0,
    T(t, x, 10, z) ~ 27.0,
    T(t, x, y, 0) ~ 27.0,
    T(t, x, y, 3) ~ 27.0
] 

@named pdesys = PDESystem(eqs, bcs, domains, [x, y, z, t], [T(t, x, y, z)])

# 5. Discretize using MethodOfLines (MOL)
N_x = 6
N_y = 6
N_z = 7
discretization = MOLFiniteDifference([x => N_x, y => N_y, z => N_z], t, approx_order=2)

# 6. Convert PDE to ODE system
@time prob = discretize(pdesys, discretization; simplify=false)
u0 = CuArray(prob.u0)

f_gpu = ODEFunction(prob.f)
# 9. Move the problem to GPU

#prob_gpu = remake(prob; u0 = CuArray(prob.u0))

prob_gpu = ODEProblem(f_gpu, u0, prob.tspan, Float64[])
# 10. Solve on GPU (use non-stiff solver unless needed)

@allowscalar sol = solve(prob_gpu, Tsit5(), saveat=0.05, abstol=1e-6, reltol=1e-6)

# 7. Solve using a stiff ODE solver
#sol = solve(prob, TRBDF2(), saveat=0.05, abstol=1e-6, reltol=1e-6)

discrete_x = sol[x]
discrete_y = sol[y]
discrete_z = sol[z]
discrete_t = sol[t]

solu = sol[T(t, x, y, z)]