using ModelingToolkit, MethodOfLines, DifferentialEquations, LinearAlgebra, SparseArrays, Plots

# Parameters
ν = 0.01         # Viscosity
Δt = 0.001       # Time step
t_end = 1.0      # End time
nx = 16          # Grid points in x
ny = 16          # Grid points in y
L = 1.0          # Domain size
dx = L / (nx - 1)
dy = L / (ny - 1)

# Symbolic variables
@parameters t x y
@variables u(..) v(..)
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# Momentum (advection + diffusion) system (no pressure)
eqs = [
    Dt(u(t, x, y)) ~ -u(t, x, y)*Dx(u(t, x, y)) - v(t, x, y)*Dy(u(t, x, y)) + ν*(Dxx(u(t, x, y)) + Dyy(u(t, x, y))),
    Dt(v(t, x, y)) ~ -u(t, x, y)*Dx(v(t, x, y)) - v(t, x, y)*Dy(v(t, x, y)) + ν*(Dxx(v(t, x, y)) + Dyy(v(t, x, y))),
]

domains = [x ∈ (0.0, L), y ∈ (0.0, L), t ∈ (0.0, t_end)]

# Boundary conditions
bcs = [
    u(0, x, y) ~ 0.0, v(0, x, y) ~ 0.0,
    u(t, 0.0, y) ~ 0.0, u(t, L, y) ~ 0.0, u(t, x, 0.0) ~ 0.0, u(t, x, L) ~ 1.0,
    v(t, 0.0, y) ~ 0.0, v(t, L, y) ~ 0.0, v(t, x, 0.0) ~ 0.0, v(t, x, L) ~ 0.0,
]

@named pdesys = PDESystem(eqs, bcs, domains, [t, x, y], [u(t, x, y), v(t, x, y)])
discretization = MOLFiniteDifference([x => dx, y => dy], t, approx_order=2)
prob = discretize(pdesys, discretization)

# Initialize u, v
u_now = zeros(nx, ny); u_now[:, end] .= 1.0  # Top lid
v_now = zeros(nx, ny)

function chorin_step!(u::Matrix{Float64}, v::Matrix{Float64},
                      prob::ODEProblem, t::Float64, Δt::Float64, ν::Float64,
                      dx::Float64, dy::Float64,
                      u_sym, v_sym, x_sym, y_sym)

    # Flatten the input velocity fields
    u_vec = vec(u)
    v_vec = vec(v)

    # Step 1: Set initial conditions for momentum solve
    u0_new = copy(prob.u0)

    # Assumes first half of u0 is u, second half is v — adapt if layout differs
    N = length(u_vec)
    u0_new[1:N] .= u_vec
    u0_new[N+1:2N] .= v_vec

    # Solve intermediate velocity (u*, v*) from momentum equations
    prob_mom = remake(prob; u0=u0_new, tspan=(t, t + Δt))
    sol_mom = solve(prob_mom, Tsit5(), reltol=1e-6, abstol=1e-6)

    # Get predicted velocity
    u_star = sol_mom(t + Δt)[1:N]
    v_star = sol_mom(t + Δt)[N+1:2N]

    # [Placeholder for pressure solve + projection step]

    # For now, return reshaped intermediate velocities
    u_new = reshape(u_star, size(u))
    v_new = reshape(v_star, size(v))

    return u_new, v_new
end

# Time-stepping loop
tcur = 0.0
while tcur < t_end
    u_now, v_now = chorin_step!(u_now, v_now, prob, tcur, Δt, ν, dx, dy, u, v, x, y)
    tcur += Δt
end

println("✅ Simulation complete.")

# Visualization
heatmap(u_now', title="Final u-velocity", xlabel="x", ylabel="y", aspect_ratio=1)
