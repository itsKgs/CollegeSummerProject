###############################################################################
# Physics‑Informed Neural Network (PINN) – Moving Gaussian Laser Source        #
# 3‑D transient heat conduction in a thin layer (additive‑manufacturing demo)  #
###############################################################################
# Governing equation                                                           #
#   ∂T/∂t = D ∇²T + Θ(x,y,z,t)                                                #
# where D = k /(ρ cₚ)  and Θ = Q /(ρ cₚ)                                       #
# Q is a planar Gaussian heat flux that moves with the laser scan velocity     #
###############################################################################

using NeuralPDE, ModelingToolkit, DomainSets
using Flux, Optimization, OptimizationOptimisers, OptimizationOptimJL
using Random, Plots

###############################################################################
# 1. Physical / process parameters                                            #
###############################################################################
const P       = 200.0        # laser power [W]
const V       = 0.8          # scan speed [m/s]
const σ       = 13.75e-6     # beam 1‑σ radius [m]
const A_abs   = 0.3          # absorptivity [–]

const ρ       = 7910.0       # density [kg/m³]
const cₚ      = 505.0        # specific heat [J/(kg·K)]
const k       = 21.5         # thermal conductivity [W/(m·K)]
const T₀      = 300.0        # ambient temperature [K]
const ε       = 1.0e-5       # depth spread of planar source [m]

const D       = k / (ρ*cₚ)   # thermal diffusivity [m²/s] ≈ 5.37×10⁻⁶

###############################################################################
# 2. Computational domain                                                     #
###############################################################################
Lx  = 1.0e-3    # half‑width in x (±1 mm)
Ly  = 1.0e-3    # half‑width in y (±1 mm)
Lz  = 5.0e-4    # layer thickness  (0–0.5 mm)
Tf  = 2.0e-2    # final time       (20 ms)

@parameters x y z t
@variables  T(..)
Dx  = Differential(x)
Dy  = Differential(y)
Dz  = Differential(z)
Dt  = Differential(t)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dzz = Differential(z)^2

# ---------------- Moving Gaussian heat source (Θ) ----------------------------
function Θ(x,y,z,t)
    coeff = 2 * A_abs * P / (π^(3/2) * σ^2 * ε * ρ * cₚ)
    ξ²    = ((x - V*t)^2 + y^2) / σ^2 + (z^2) / ε^2
    return coeff * exp(-ξ²)
end

# ---------------- PDE definition --------------------------------------------
eqs = [ Dt(T(x,y,z,t)) ~ D * (Dxx(T(x,y,z,t)) +
                                Dyy(T(x,y,z,t)) +
                                Dzz(T(x,y,z,t))) + Θ(x,y,z,t) ]

# ---------------- Domains ----------------------------------------------------
domains = [ x ∈ Interval(-Lx,  Lx),
            y ∈ Interval(-Ly,  Ly),
            z ∈ Interval( 0.0, Lz),
            t ∈ Interval( 0.0, Tf) ]

# ---------------- Initial & boundary conditions -----------------------------
# IC: uniform ambient temperature
bcs = [ T(x,y,z,0)           ~ T₀,
        Dz(T(x,y,0,t))       ~ 0,          # insulated top surface (∂T/∂z = 0 at z = 0)
        T(x,y,Lz,t)          ~ T₀,         # fixed bottom (Dirichlet)
        T(-Lx,y,z,t)         ~ T₀,         # side walls – ambient Dirichlet
        T( Lx,y,z,t)         ~ T₀,
        T(x,-Ly,z,t)         ~ T₀,
        T(x, Ly,z,t)         ~ T₀ ]

@named pdesys = PDESystem(eqs, bcs, domains, [x,y,z,t], [T(x,y,z,t)])

###############################################################################
# 3. PINN discretisation                                                      #
###############################################################################

# Network: 4‑input  →  4 hidden layers (tanh)  →  1 output
chain = Chain(Dense(4, 64, tanh),
              Dense(64, 64, tanh),
              Dense(64, 64, tanh),
              Dense(64, 64, tanh),
              Dense(64, 1))

# Collocation grid spacing
Δx, Δy, Δz, Δt = 2.0e-4, 2.0e-4, 1.0e-4, 1.0e-3
strategy = GridTraining([Δx, Δy, Δz, Δt])

discretization = PhysicsInformedNN(chain, strategy)
prob = discretize(pdesys, discretization)

###############################################################################
# 4. Training schedule: Adam  →  LBFGS (fixed grid)                           #
###############################################################################

cb  = (p,l)->(println("loss = $(l)" ); false)

rng = Random.default_rng()

# ---- phase 1: Adam (stochastic) -------------------------------------------
opt_adam = OptimizationOptimisers.Adam(0.001)
res      = Optimization.solve(prob, opt_adam;
                             maxiters = 4000,
                             callback = cb,
                             rng = rng)

# ---- phase 2: LBFGS (deterministic, fixed collocation) --------------------
# Freeze the collocation points before switching optimisers
strategy_fixed = FixedTraining()
discretization.training_strategy[] = strategy_fixed
prob_fixed = discretize(pdesys, discretization)

opt_lbfgs = OptimizationOptimJL.LBFGS()
res       = Optimization.solve(prob_fixed, opt_lbfgs;
                              u0        = res.u,   # warm‑start from Adam
                              maxiters  = 800,
                              callback  = cb,
                              rng       = rng)

###############################################################################
# 5. Helper: Temperature sampling & optional animation                        #
###############################################################################
φ = discretization.phi   # surrogate: (coords, params) → temperature

"""Return a Matrix size(length(yvec), length(xvec)) with T(y,x) at (z,t)."""
function sample_temperature(xvec, yvec, zval, tval)
    return [ φ([x, y, zval, tval], res.u)[1]  for y in yvec, x in xvec ]
end

# ---------------------------------------------------------------------------
# Example usage (comment out or adapt for batch runs):
# xs = range(-Lx, Lx; length = 101)
# ys = range(-Ly, Ly; length = 101)
# zs = ε
# ts = range(0.0, Tf; length = 40)
# anim = @animate for τ in ts
#     Tslice = sample_temperature(xs, ys, zs, τ)
#     heatmap(xs, ys, Tslice', aspect_ratio = :equal,
#             xlabel = "x [m]", ylabel = "y [m]",
#             title  = "T(z=$(round(zs;digits=6)) m, t=$(round(τ;digits=3)) s)",
#             c = :thermal, clims = (T₀, 2000), size = (650,550))
# end
# gif(anim, "moving_laser_pinn.gif", fps = 10)
###############################################################################
