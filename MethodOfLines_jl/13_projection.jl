using ModelingToolkit, MethodOfLines, DifferentialEquations, Plots

# Parameters
Lx, Ly = 1.0, 1.0
Nx, Ny = 16, 16
ν = 0.01
x = range(0, Lx, length = Nx)
y = range(0, Ly, length = Ny)
dx = step(x)
dy = step(y)
tspan = 1.0


@parameters x y t
@variables u(..) v(..)
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Dx^2
Dyy = Dy^2

# Define the equations
eqs_u_star = [
    Dt(u(x, y, t)) ~ -u(x, y, t)*Dx(u(x, y, t)) - v(x, y, t)*Dy(u(x, y, t)) + ν*(Dxx(u(x, y, t)) + Dyy(u(x, y, t))),
    Dt(v(x, y, t)) ~ -u(x, y, t)*Dx(v(x, y, t)) - v(x, y, t)*Dy(v(x, y, t)) + ν*(Dxx(v(x, y, t)) + Dyy(v(x, y, t))),
]

# Domains
domains = [
    x ∈ IntervalDomain(0, Lx),
    y ∈ IntervalDomain(0, Ly),
    t ∈ IntervalDomain(0, tspan)
]

# Initial conditions
u0(x, y) = 0.0
v0(x, y) = 0.0
bcs = [
    u(x, y, 0) ~ 0.0,
    v(x, y, 0) ~ 0.0,

    u(0, y, t) ~ 0.0, u(Lx, y, t) ~ 0.0,
    u(x, 0, t) ~ 0.0, u(x, Ly, t) ~ 1.0,  # Lid-driven condition
    v(0, y, t) ~ 0.0, v(Lx, y, t) ~ 0.0,
    v(x, 0, t) ~ 0.0, v(x, Ly, t) ~ 0.0
]

# Build PDE system
@named pdesys_star = PDESystem(eqs_u_star, bcs, domains, [x, y, t], [u(x, y, t), v(x, y, t)])
disc_star = MOLFiniteDifference([x => dx, y => dy], t, approx_order=2)
prob_star = discretize(pdesys_star, disc_star)
sol_star = solve(prob_star, Tsit5(); saveat = 0.01)
sol_star[u(x, y, t)]
sol_star[v(x, y, t)]
sol_star.t

#@variables p(..)
#eqs_p = [
#    0 ~ Dxx(p(x, y, t)) + Dyy(p(x, y, t)) - (1/0.01) * (Dx(u(x, y, t)) + Dy(v(x, y, t)))
#]
#
## BCs for pressure (Neumann on all sides)
#bcs_p = [
#    Dx(p(0, y, t)) ~ 0.0, Dx(p(Lx, y, t)) ~ 0.0,
#    Dy(p(x, 0, t)) ~ 0.0, Dy(p(x, Ly, t)) ~ 0.0,
#    p(x, y, 0) ~ 0.0  # Initial condition
#]
#
## PDE system and solve
#@named pdesys_p = PDESystem(eqs_p, bcs_p, domains, [x, y, t], [p(x, y, t)])
#disc_p = MOLFiniteDifference([x => dx, y => dy], t, approx_order=2)
#prob_p = discretize(pdesys_p, disc_p)
#sol_p = solve(prob_p, Rodas5(); saveat = 0.01)

# Pressure PDE definition (without time)
@parameters x y
@variables p(..)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Dx^2
Dyy = Dy^2
dummy_source(x, y) = 0.0  # placeholder

eqs_p = [
    0 ~ Dxx(p(x, y)) + Dyy(p(x, y)) - dummy_source(x, y)
]
bcs_p = [
    Dx(p(0, y)) ~ 0.0, Dx(p(Lx, y)) ~ 0.0,
    Dy(p(x, 0)) ~ 0.0, Dy(p(x, Ly)) ~ 0.0
]
domains_p = [x ∈ IntervalDomain(0, Lx), y ∈ IntervalDomain(0, Ly)]

# Loop to solve pressure at each time step and correct velocity
pressure_solutions = []
u_corrected = []
v_corrected = []

for (i, tᵢ) in enumerate(sol_star.t)
    u_star = sol_star[u(x, y, tᵢ)]
    v_star = sol_star[v(x, y, tᵢ)]

    # Compute divergence of u*, v*
    div = zeros(Nx, Ny)
    for ix in 2:Nx-1, iy in 2:Ny-1
        ∂u_∂x = (u_star[ix+1, iy] - u_star[ix-1, iy]) / (2dx)
        ∂v_∂y = (v_star[ix, iy+1] - v_star[ix, iy-1]) / (2dy)
        div[ix, iy] = ∂u_∂x + ∂v_∂y
    end

    # Interpolate divergence as source
    function source_term(x, y)
        xi = clamp(round(Int, x/dx + 1), 1, Nx)
        yi = clamp(round(Int, y/dy + 1), 1, Ny)
        return (ρ / Δt) * div[xi, yi]
    end

    eqs_pressure = [0 ~ Dxx(p(x, y)) + Dyy(p(x, y)) - source_term(x, y)]
    @named pdesys_p = PDESystem(eqs_pressure, bcs_p, domains_p, [x, y], [p(x, y)])
    disc = MOLFiniteDifference([x => dx, y => dy])
    prob_p = discretize(pdesys_p, disc)
    sol_p = solve(prob_p, Rodas5())
    push!(pressure_solutions, sol_p)

    # Correct velocities
    p_vals = [sol_p[p(x, y)] for x in x_vals, y in y_vals]
    u_new = copy(u_star)
    v_new = copy(v_star)

    for ix in 2:Nx-1, iy in 2:Ny-1
        ∂p_∂x = (p_vals[ix+1, iy] - p_vals[ix-1, iy]) / (2dx)
        ∂p_∂y = (p_vals[ix, iy+1] - p_vals[ix, iy-1]) / (2dy)
        u_new[ix, iy] -= Δt * ∂p_∂x / ρ
        v_new[ix, iy] -= Δt * ∂p_∂y / ρ
    end

    push!(u_corrected, u_new)
    push!(v_corrected, v_new)
end
