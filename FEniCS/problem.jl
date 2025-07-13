# FEniCS.jl 2D Simulation of Melting with Mass/Energy Source
# This version uses a custom C++ Expression to define the
# complex, non-linear enthalpy-temperature relationship.

using FEniCS

# --- 1. Parameters ---

# Time stepping
T_final = 10.0
num_steps = 200
dt = T_final / num_steps
k_dt = Constant(dt)

# Physical parameters (SI units for steel)
val_Cp = 700.0      # Specific Heat [J/kg.K]
val_T_s = 1673.15   # Solidus Temperature [K]
val_T_l = 1773.15   # Liquidus Temperature [K]
val_L = 2.7e5       # Latent Heat [J/kg]
val_T_cold = 300.0  # Initial/Boundary temperature [K]

rho = Constant(7850.0)    # Density [kg/m^3]
mu = Constant(0.005)      # Viscosity [Pa.s]
k_therm = Constant(45.0)  # Thermal Conductivity [W/m.K]
Cp = Constant(val_Cp)
T_s = Constant(val_T_s)
T_l = Constant(val_T_l)
L = Constant(val_L)
g = Constant((0.0, -9.81))

# Calculate enthalpy reference values as Julia numbers for the C++ Expression
val_h_s = val_Cp * val_T_s
val_h_l = val_h_s + val_L

# --- 2. Source Term Modeling (Simplified Gaussian Blob) ---

source_params = Dict(:mag => 5.0, :cx => 0.5, :cy => 0.8, :r => 0.05)
V_s_velocity = (0.0, -0.1)
source_cpp_code = "mag * exp(-(pow(x[0] - cx, 2) + pow(x[1] - cy, 2)) / (2 * r * r))"

m_dot_s = Expression(source_cpp_code, degree=2; source_params...)
h_dot_s = m_dot_s * (Cp * T_l + L)
F_dot_s = m_dot_s / rho

# --- 3. Mesh, Function Spaces, and Functions ---

nx, ny = 64, 64
mesh = UnitSquareMesh(nx, ny)

V = VectorFunctionSpace(mesh, "P", 2) # Velocity
Q = FunctionSpace(mesh, "P", 1)       # Pressure
H = FunctionSpace(mesh, "P", 1)       # Enthalpy
F = FunctionSpace(mesh, "P", 1)       # Volume Fraction

# Trial and Test Functions
u, v = TrialFunction(V), TestFunction(V)
p, q = TrialFunction(Q), TestFunction(Q)
h, phi = TrialFunction(H), TestFunction(H)
F_var, psi = TrialFunction(F), TestFunction(F)

# Solution Functions
u_, u_n, u_star = FeFunction(V), FeFunction(V), FeFunction(V)
p_, p_n = FeFunction(Q), FeFunction(Q)
h_, h_n = FeFunction(H), FeFunction(H)
F_, F_n = FeFunction(F), FeFunction(F)
T = FeFunction(H) # For visualization of Temperature

# --- 4. Boundary and Initial Conditions ---

const bottom_wall = "near(x[1], 0.0) && on_boundary"
const all_walls = "on_boundary"
bc_u = [DirichletBC(V, Constant((0.0, 0.0)), all_walls)]

h_cold_const = Constant(val_Cp * val_T_cold)
bc_h = [DirichletBC(H, h_cold_const, bottom_wall)]

# Set initial values
assign(u_n, Constant((0.0, 0.0)))
assign(p_n, Constant(0.0))
assign(h_n, project(h_cold_const, H))
assign(F_n, Constant(0.0))

# --- 5. Enthalpy-Temperature Relation (The Core Fix) ---

# This C++ code defines a complete class that inherits from dolfin::Expression
# THE FIX: The "class" keyword MUST be on the same line as the opening """.
# This prevents a leading newline character in the string.
T_cpp_code = """class TempExpr : public Expression
{
public:
  // A handle to the FEniCS Function for enthalpy (h_n)
  std::shared_ptr<const Function> h_func;

  // Member variables for physical constants
  double h_s, h_l, T_s, T_l, Cp, L;

  TempExpr() : Expression() {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    // Create a temporary array to hold the result of evaluating h_func at point x
    Array<double> h_at_x(1);
    h_func->eval(h_at_x, x);
    double hh = h_at_x[0]; // Extract the scalar value

    // Perform the conditional logic to convert enthalpy to temperature
    values[0] = hh < h_s ? hh / Cp :
                (hh < h_l ? T_s + (hh - h_s) * (T_l - T_s) / L :
                            T_l + (hh - h_l) / Cp);
  }
};
"""
# Create an instance of our custom C++ Expression
T_expr = Expression(T_cpp_code, degree=2)

# Set its properties. 'h_func' is the handle to our Julia-side FEniCS function `h_n`.
# The others are the constants we calculated earlier.
T_expr.h_func = h_n
T_expr.h_s = val_h_s; T_expr.h_l = val_h_l; T_expr.T_s = val_T_s
T_expr.T_l = val_T_l; T_expr.Cp = val_Cp; T_expr.L = val_L


# --- 6. Variational Forms (Chorin Projection Scheme) ---

V_s = Constant(V_s_velocity)

# Tentative velocity (u_star)
a1 = (1/k_dt)*dot(u, v)*dx + dot(dot(u_n, nabla_grad(u)), v)*dx + (mu/rho)*inner(grad(u), grad(v))*dx
L1 = (1/k_dt)*dot(u_n, v)*dx - (1/rho)*dot(grad(p_n), v)*dx + dot(g, v)*dx + (m_dot_s/rho)*dot(V_s - u_n, v)*dx

# Pressure correction (p_)
a_p = dot(grad(p), grad(q))*dx
L_p = m_dot_s*q*dx - (rho/k_dt)*div(u_star)*q*dx

# Velocity correction (u_)
a_u = dot(u, v)*dx
L_u = dot(u_star, v)*dx - (k_dt/rho)*dot(grad(p_ - p_n), v)*dx

# Enthalpy/Energy equation (h_)
# The thermal diffusion term now correctly uses the C++ Expression T_expr
a2 = (1/k_dt)*h*phi*dx + dot(u_, nabla_grad(h))*phi*dx + (m_dot_s/rho)*h*phi*dx
L2 = (1/k_dt)*h_n*phi*dx - (k_therm/rho)*dot(grad(T_expr), grad(phi))*dx + (1/rho)*h_dot_s*phi*dx

# Volume Fraction (F_)
a3 = (1/k_dt)*F_var*psi*dx + dot(u_, nabla_grad(F_var))*psi*dx
L3 = (1/k_dt)*F_n*psi*dx + F_dot_s*psi*dx

# --- 7. Time-Stepping Loop ---
mkpath("julia_melt_2d_pvd")
vtkfile_u = File("julia_melt_2d_pvd/velocity.pvd")
vtkfile_p = File("julia_melt_2d_pvd/pressure.pvd")
vtkfile_T = File("julia_melt_2d_pvd/temperature.pvd")
vtkfile_F = File("julia_melt_2d_pvd/vof.pvd")

global t = 0.0
for step in 1:num_steps
    global t += dt
    println("Time step $step / $num_steps, Time = $(round(t, digits=3))s")

    # Solve the system of equations
    lvsolve(a1, L1, u_star, bc_u)
    lvsolve(a_p, L_p, p_, [])
    lvsolve(a_u, L_u, u_, bc_u)

    # Before solving for enthalpy, T_expr is already correctly pointing to h_n,
    # which holds the value from the previous time step. This is what we want.
    lvsolve(a2, L2, h_, bc_h)

    # Update T for visualization only. We must project T_expr onto T.
    assign(T, project(T_expr, H))

    # Solve for Volume Fraction and clamp values between 0 and 1
    lvsolve(a3, L3, F_, [])
    F_.vector()[:] = map(x -> clamp(x, 0.0, 1.0), F_.vector()[:])

    # Save to file for visualization
    u_.rename("velocity", "u"); p_.rename("pressure", "p"); T.rename("temperature", "T"); F_.rename("vof", "F")
    vtkfile_u << (u_, t); vtkfile_p << (p_, t); vtkfile_T << (T, t); vtkfile_F << (F_, t)

    # Update previous-step solutions for the next iteration
    assign(u_n, u_); assign(p_n, p_); assign(h_n, h_); assign(F_n, F_)
end

println("Simulation complete.")